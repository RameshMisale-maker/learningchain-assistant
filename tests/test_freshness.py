"""Phase 3: document freshness tracking - seen / changed, missed crawls, tombstone, sitemap rules from config."""
from __future__ import annotations

import pytest

URL = "https://learningchain.in/freshness-test-page"
PAGE = ("# Freshness test page\n\nThe Quorvex certificate archive keeps every issued credential for twelve years. "
        "Administrators can export the archive from the Reports area at any time.\n")


@pytest.fixture()
def tracking(workspace):
    from config.settings import get_settings

    settings = get_settings()
    settings.doc_tracking_enabled, settings.review_queue_enabled = True, True
    yield settings
    settings.doc_tracking_enabled, settings.review_queue_enabled = False, False


def _source(url=URL, kb="products", created_by="admin", source_type="page") -> int:
    from backend.db.models import WebSource
    from backend.db.session import db_session
    with db_session() as s:
        src = WebSource(url=url, kb=kb, product_id="learningchain", source_type=source_type, enabled=True,
                        created_by=created_by, interval_minutes=1440)
        s.add(src)
        s.flush()
        return src.id


def _crawl(source_id, monkeypatch, markdown=None, error=None):
    from ingestion import automation, web
    from ingestion.pipeline import IngestionPipeline

    def fake_scrape(url, etag="", last_modified=""):
        if error:
            raise ValueError(error)
        return web.ScrapeResult(url=url, page=web.ExtractedPage(title="Freshness test page", markdown=markdown))
    monkeypatch.setattr(web, "scrape_page", fake_scrape)
    return automation.handle_crawl({"id": 99, "payload": {"source_id": source_id, "force": False}}, IngestionPipeline())


def _doc(url=URL):
    from ingestion import automation
    from ingestion.pipeline import IngestionPipeline
    from backend.db.models import Document
    from backend.db.session import db_session
    path = automation._page_path("products", url)
    with db_session() as s:
        d = s.get(Document, IngestionPipeline().doc_id_for("products", path))
        return None if d is None else {"status": d.status, "missed": d.missed_crawls, "seen": d.last_seen_at,
                                       "changed": d.last_changed_at, "url": d.source_url, "tomb": d.tombstoned_at}


def test_url_rules_from_product_config(workspace):
    from backend.products import get_product
    from ingestion.freshness import url_allowed

    company = get_product("sapphirus")
    assert url_allowed("https://sapphirus.in/services/", company)
    assert not url_allowed("https://sapphirus.in/category/news/", company)          # exclude_paths
    assert not url_allowed("https://evil.example.com/page", company)                # allowed_domains
    assert url_allowed("https://www.sapphirus.in/about/", company)                  # sub-domain of an allowed one


def test_seen_then_two_misses_tombstone(tracking, monkeypatch):
    from backend.services import review

    sid = _source()
    out = _crawl(sid, monkeypatch, PAGE)
    assert out["status"] == "changed"
    first = _doc()
    assert first["status"] == "active" and first["url"] == URL and first["seen"] and first["missed"] == 0
    assert _crawl(sid, monkeypatch, PAGE)["status"] == "unchanged"
    assert _doc()["seen"] >= first["seen"]

    assert _crawl(sid, monkeypatch, error="HTTP 404 for page")["status"] == "missing"
    assert _doc()["status"] == "active" and _doc()["missed"] == 1                     # one miss: content kept
    assert _crawl(sid, monkeypatch, PAGE)["status"] in ("unchanged", "changed")
    assert _doc()["missed"] == 0                                                       # back again: counter reset

    _crawl(sid, monkeypatch, error="HTTP 404 for page")
    assert _crawl(sid, monkeypatch, error="HTTP 410 gone")["status"] == "tombstoned"
    gone = _doc()
    assert gone["status"] == "deleted" and gone["tomb"]
    item = next(i for i in review.list_items(source="stale_doc") if i["details"].get("url") == URL)
    review.resolve(item["id"], "test", "dismiss")


def test_flag_off_keeps_immediate_delete(workspace, monkeypatch):
    url = URL + "-legacy"
    sid = _source(url)
    _crawl(sid, monkeypatch, PAGE.replace("Quorvex", "Legacyvex"))
    assert _doc(url)["status"] == "active"
    assert _crawl(sid, monkeypatch, error="HTTP 404 for page")["status"] == "removed"
    assert _doc(url)["status"] == "deleted"


def test_changed_page_flags_approved_answers_and_drops_cache(tracking, monkeypatch):
    from backend.services import response_cache, review
    from ingestion.freshness import on_content_changed

    review.upsert_golden("learningchain", "products", "FRESH1", "How long is the Quorvex archive kept?",
                         "Twelve years.", source_url=URL + "-golden")
    calls = []
    monkeypatch.setattr(tracking, "response_cache_enabled", True)
    monkeypatch.setattr(response_cache, "invalidate", lambda product_id=None: calls.append(product_id) or 0)
    on_content_changed("products", URL + "-golden")
    assert calls == ["learningchain"]
    item = next(i for i in review.list_items(source="stale_doc") if i["details"].get("qid") == "FRESH1")
    review.resolve(item["id"], "test", "dismiss")
    from sqlalchemy import delete

    from backend.db.models import GoldenQA
    from backend.db.session import db_session
    with db_session() as s:
        s.execute(delete(GoldenQA).where(GoldenQA.qid == "FRESH1"))


def test_sitemap_filters_and_counts_dropped_pages(tracking, monkeypatch):
    from ingestion import automation, web
    from ingestion.pipeline import IngestionPipeline

    page = "https://learningchain.in/sitemap-page-a"
    page_sid = _source(page, created_by="sitemap")
    _crawl(page_sid, monkeypatch, PAGE.replace("Quorvex", "Sitemapvex"))
    sitemap_sid = _source("https://learningchain.in/sitemap.xml", source_type="sitemap")
    xml = "<urlset><url><loc>https://learningchain.in/other</loc></url></urlset>"
    monkeypatch.setattr(web, "fetch", lambda url, *a, **k: web.FetchResult(url=url, status=200,
                                                                         content_type="application/xml", text=xml))
    added = []
    monkeypatch.setattr(automation, "add_source", lambda url, kb, *a, **k: added.append(url))
    out = automation.handle_crawl({"id": 7, "payload": {"source_id": sitemap_sid}}, IngestionPipeline())
    assert added == ["https://learningchain.in/other"]
    assert out["dropped_pages"]["missing"] >= 1 and _doc(page)["missed"] == 1


def test_stale_documents_admin_endpoint(tracking):
    from fastapi.testclient import TestClient

    from backend.main import app

    with TestClient(app) as c:
        r = c.get("/api/v1/admin/documents/stale", headers={"Authorization": "Bearer admin-token-for-tests-0001"})
        assert r.status_code == 200 and r.json()["enabled"] is True
