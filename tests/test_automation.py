"""Automated ingestion: queued uploads, replacement, deletion, retries, website monitoring and webhooks.

Covers the two required real-world scenarios end to end (against the isolated copy of the real knowledge base):
  Scenario 1 - an admin uploads a new product document; it is processed in the background and a user's question is
               immediately answered from it, with a source citation.
  Scenario 2 - a monitored website article changes; only that page is re-processed, outdated chunks are removed
               and the updated information is served.
External HTTP and DNS are replaced with in-process fakes (no network).
"""
from __future__ import annotations

import hashlib
import hmac
import json
import socket

import pytest

from ingestion import automation, jobs, registry, web

NEW_PRODUCT_DOC = b"""# Sapphirus VerifyBox Product Sheet

## Overview
VerifyBox is a Sapphirus hardware kiosk that lets employers verify LearningChain certificates offline at campus
placement drives.

## Key Features
- Offline verification of LearningChain QR codes without internet access, synchronised every 24 hours.
- Supports up to 400 verifications per hour per kiosk.
- Prints a signed verification receipt for each certificate checked.
"""


def _ask(question: str):
    from backend.security.auth import AuthenticatedUser
    from backend.services.chat_service import ChatService

    return ChatService(llm=None).answer(question, AuthenticatedUser("t_public", {"public"}), persist=False)


def _chunks_for(doc_id: str) -> list:
    from backend.retrieval.kb_index import get_index_manager

    idx = get_index_manager().get("products")
    return [c for c in idx.chunks if c.metadata.get("doc_id") == doc_id]


@pytest.fixture()
def public_dns(monkeypatch):
    """Resolve every host to a public address so SSRF validation runs without network access."""
    monkeypatch.setattr(web.socket, "getaddrinfo",
                        lambda host, port, **kw: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port))])


# ----------------------------------------------------------------------------- Scenario 1
def test_scenario1_upload_is_indexed_in_background_and_answerable(workspace):
    job, created = automation.stage_upload("admin", "products", "VerifyBox_Product_Sheet.md", NEW_PRODUCT_DOC)
    assert created and job["status"] == "queued"
    assert jobs.get_job(job["id"])["status"] == "queued"          # nothing indexed before the worker runs
    assert automation.run_pending() >= 1
    done = jobs.get_job(job["id"])
    assert done["status"] == "succeeded", done["error"]
    assert done["result"]["status"] == "added" and done["result"]["chunks"] >= 1

    resp = _ask("How many verifications per hour does the VerifyBox kiosk support?")
    assert resp.answer_type not in ("refusal", "out_of_scope"), resp.answer
    assert "400" in resp.answer
    assert any("VerifyBox" in c.doc_name for c in resp.citations)


def test_duplicate_upload_is_idempotent(workspace):
    job1, _ = automation.stage_upload("admin", "products", "VerifyBox_Product_Sheet.md", NEW_PRODUCT_DOC)
    job2, created2 = automation.stage_upload("admin", "products", "VerifyBox_Product_Sheet.md", NEW_PRODUCT_DOC)
    assert not created2 and job2["id"] == job1["id"]               # same content queued once
    automation.run_pending()
    assert jobs.get_job(job1["id"])["result"]["status"] in ("unchanged", "added", "updated")
    docs = [d for d in registry.list_documents("products") if d["doc_name"].lower().startswith("sapphirus verifybox")]
    assert len(docs) == 1                                          # no duplicate registry record


def test_replace_updates_in_place_and_removes_stale_chunks(workspace):
    doc = next(d for d in registry.list_documents("products") if "verifybox" in d["source_path"].lower())
    old_version = doc["version"]
    v2 = NEW_PRODUCT_DOC.replace(b"400 verifications per hour", b"650 verifications per hour")
    job, _ = automation.stage_upload("admin", "products", "VerifyBox_Product_Sheet.md", v2,
                                     replace_doc_id=doc["doc_id"])
    automation.run_pending()
    assert jobs.get_job(job["id"])["status"] == "succeeded"
    new = registry.get_document(doc["doc_id"])
    assert new["version"] == old_version + 1
    texts = " ".join(c.text for c in _chunks_for(doc["doc_id"]))
    assert "650 verifications" in texts and "400 verifications" not in texts
    assert len([d for d in registry.list_documents("products") if "verifybox" in d["source_path"].lower()]) == 1


def test_delete_removes_vectors(workspace):
    doc = next(d for d in registry.list_documents("products") if "verifybox" in d["source_path"].lower())
    job, _ = automation.enqueue_delete("admin", doc["doc_id"])
    automation.run_pending()
    assert jobs.get_job(job["id"])["status"] == "succeeded"
    assert _chunks_for(doc["doc_id"]) == []
    assert registry.get_document(doc["doc_id"])["status"] != "active"
    resp = _ask("How many verifications per hour does the VerifyBox kiosk support?")
    assert "650" not in resp.answer


# ----------------------------------------------------------------------------- retries / recovery
def test_failed_job_retries_then_dead_then_manual_retry_without_duplicates(workspace, monkeypatch):
    from ingestion.pipeline import IngestionPipeline

    calls = {"n": 0}
    real = IngestionPipeline.ingest_file

    def flaky(self, *a, **kw):
        calls["n"] += 1
        raise RuntimeError("simulated embedding outage")

    monkeypatch.setattr(IngestionPipeline, "ingest_file", flaky)
    doc = NEW_PRODUCT_DOC.replace(b"VerifyBox", b"RetryBox")
    job, _ = automation.stage_upload("admin", "products", "RetryBox.md", doc)
    for _ in range(5):
        automation.run_pending()
    state = jobs.get_job(job["id"])
    assert state["status"] == "dead" and state["attempts"] == state["max_attempts"] and calls["n"] == state["attempts"]

    monkeypatch.setattr(IngestionPipeline, "ingest_file", real)
    jobs.retry(job["id"])
    automation.run_pending()
    assert jobs.get_job(job["id"])["status"] == "succeeded"
    jobs.retry(job["id"])                      # retrying a finished job again must not duplicate anything
    automation.run_pending()
    assert len([d for d in registry.list_documents("products") if "retrybox" in d["source_path"].lower()]) == 1


def test_stale_running_job_is_requeued(workspace):
    from datetime import datetime, timedelta, timezone

    from backend.db.models import IngestionJob
    from backend.db.session import db_session

    job, _ = jobs.enqueue("sync_kb", "company", {"full": False}, dedupe_key="test:stale")
    claimed = jobs.claim_next("crashed-worker")
    assert claimed and claimed["id"] == job["id"]
    with db_session() as s:
        s.get(IngestionJob, job["id"]).locked_at = datetime.now(timezone.utc) - timedelta(hours=2)
    assert jobs.requeue_stale(lease_minutes=30) >= 1
    assert jobs.get_job(job["id"])["status"] == "queued"
    automation.run_pending()
    assert jobs.get_job(job["id"])["status"] == "succeeded"


# ----------------------------------------------------------------------------- Scenario 2 (website change)
PAGE_V1 = """<html><head><title>LearningChain Pricing Notes</title></head><body><nav>Home | Products</nav>
<article><h1>LearningChain Bulk Plans</h1><h2>Bulk issuance</h2>
<p>The LearningChain Campus plan allows bulk issuance of up to 5,000 certificates per batch for universities.</p>
<p>Batches are processed within two business days after upload by the institution admin.</p></article>
<footer>Copyright</footer></body></html>"""
PAGE_V2 = PAGE_V1.replace("5,000 certificates per batch", "8,000 certificates per batch")


def test_scenario2_website_change_detected_and_reindexed(workspace, public_dns, monkeypatch):
    pages = {"html": PAGE_V1, "status": 200}

    def fake_fetch(url, etag="", last_modified="", **kw):
        if pages["status"] == 404:
            raise ValueError(f"HTTP 404 for {url}")
        return web.FetchResult(url, 200, "text/html", pages["html"])

    monkeypatch.setattr(web, "fetch", fake_fetch)
    src = automation.add_source("https://learningchain.in/bulk-plans/", "products", "page")

    automation.enqueue_crawl(src["id"], "test")
    automation.run_pending()
    first = next(s for s in automation.list_sources() if s["id"] == src["id"])
    assert first["last_status"] == "changed"
    assert "5,000" in _ask("How many certificates per batch does the LearningChain Campus plan allow?").answer

    # unchanged page -> skipped (no re-index, version unchanged)
    path = automation._page_path("products", src["url"])
    doc_id = registry.get_document_by_path(str(path)) if hasattr(registry, "get_document_by_path") else None
    from ingestion.pipeline import IngestionPipeline
    doc_id = IngestionPipeline().doc_id_for("products", path)
    v_before = registry.get_document(doc_id)["version"]
    automation.enqueue_crawl(src["id"], "test")
    automation.run_pending()
    assert next(s for s in automation.list_sources() if s["id"] == src["id"])["last_status"] == "unchanged"
    assert registry.get_document(doc_id)["version"] == v_before

    # page edited on the website -> only this document re-processed, outdated chunks removed
    pages["html"] = PAGE_V2
    automation.enqueue_crawl(src["id"], "test")
    automation.run_pending()
    assert next(s for s in automation.list_sources() if s["id"] == src["id"])["last_status"] == "changed"
    assert registry.get_document(doc_id)["version"] == v_before + 1
    texts = " ".join(c.text for c in _chunks_for(doc_id))
    assert "8,000" in texts and "5,000" not in texts
    answer = _ask("How many certificates per batch does the LearningChain Campus plan allow?")
    assert "8,000" in answer.answer and "5,000" not in answer.answer
    assert any(c.url == "https://learningchain.in/bulk-plans/" for c in answer.citations)

    # page removed from the website -> its chunks are deleted
    pages["status"] = 404
    automation.enqueue_crawl(src["id"], "test")
    automation.run_pending()
    assert next(s for s in automation.list_sources() if s["id"] == src["id"])["last_status"] == "removed"
    assert _chunks_for(doc_id) == []


def test_ssrf_urls_rejected(workspace, public_dns):
    for url in ("http://127.0.0.1/", "https://10.0.0.5/x", "https://evil.example.com/", "file:///etc/passwd",
                "https://sapphirus.in:8443/", "https://user:pass@sapphirus.in/"):
        with pytest.raises(web.UnsafeURLError):
            automation.add_source(url, "products", "page")


def test_ssrf_dns_rebinding_to_private_ip_rejected(workspace, monkeypatch):
    monkeypatch.setattr(web.socket, "getaddrinfo",
                        lambda host, port, **kw: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("169.254.169.254", port))])
    with pytest.raises(web.UnsafeURLError):
        web.validate_url("https://sapphirus.in/page/")


# ----------------------------------------------------------------------------- API + webhook
def test_async_upload_api_returns_202_and_job(workspace):
    from fastapi.testclient import TestClient

    from backend.main import app

    with TestClient(app) as client:
        h = {"Authorization": "Bearer admin-token-for-tests-0001"}
        r = client.post("/api/v1/admin/documents", headers=h, data={"kb": "products"},
                        files={"file": ("ApiBox.md", NEW_PRODUCT_DOC.replace(b"VerifyBox", b"ApiBox"), "text/markdown")})
        assert r.status_code == 202, r.text
        job_id = r.json()["job"]["id"]
        assert client.get(f"/api/v1/admin/jobs/{job_id}", headers=h).json()["status"] == "queued"
        automation.run_pending()
        assert client.get(f"/api/v1/admin/jobs/{job_id}", headers=h).json()["status"] == "succeeded"
        bad = client.post("/api/v1/admin/documents", headers=h, data={"kb": "products"},
                          files={"file": ("x.exe", b"MZ\x90\x00", "application/octet-stream")})
        assert bad.status_code == 400
        anon = client.post("/api/v1/admin/documents", data={"kb": "products"},
                           files={"file": ("y.md", b"# hi\n\ntext", "text/markdown")})
        assert anon.status_code in (401, 403)


def test_webhook_requires_valid_signature(workspace, public_dns, monkeypatch):
    from fastapi.testclient import TestClient

    from backend.main import app

    monkeypatch.setattr(web, "fetch", lambda url, *a, **k: web.FetchResult(url, 200, "text/html", PAGE_V2))
    src = automation.add_source("https://learningchain.in/webhook-page/", "products", "page")
    body = json.dumps({"url": src["url"]}).encode()
    good = "sha256=" + hmac.new(b"test-webhook-secret", body, hashlib.sha256).hexdigest()
    with TestClient(app) as client:
        assert client.post("/api/v1/webhooks/content-updated", content=body).status_code == 401
        assert client.post("/api/v1/webhooks/content-updated", content=body,
                           headers={"X-Sapphirus-Signature": "sha256=deadbeef"}).status_code == 401
        ok = client.post("/api/v1/webhooks/content-updated", content=body, headers={"X-Sapphirus-Signature": good})
        assert ok.status_code == 202 and ok.json()["queued"][0]["source_id"] == src["id"]


def test_health_endpoints(workspace):
    from fastapi.testclient import TestClient

    from backend.main import app

    with TestClient(app) as client:
        assert client.get("/health/live").json()["status"] == "alive"
        ready = client.get("/health/ready")
        assert ready.status_code == 200, ready.json()
        assert ready.json()["checks"]["database"]["ok"]
