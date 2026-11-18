"""Automated knowledge-base updates: background job handlers, website monitoring and the ingestion worker.

Workflows
  Upload    : admin uploads -> validated + original stored (data/uploads) -> job queued (HTTP 202) -> worker moves the
              file into the KB folder, extracts, chunks, embeds, indexes -> audit log + notification.
  Replace   : same document id, new version; new chunks are written before stale ones are removed (no gap).
  Delete    : job removes vectors + registry entry; the file is archived.
  Website   : registered sources are checked on a schedule (and on webhook calls); unchanged content (same hash)
              is skipped; changed content is re-indexed; removed pages (404/410) have their chunks deleted.

The worker runs inside the API process (single writer for the embedded vector store); see docs for scaling.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from sqlalchemy import select

from backend.db import repository
from backend.db.models import WebSource
from backend.db.session import db_session
from config.settings import get_settings
from ingestion import jobs, registry, sources, web
from ingestion.pipeline import IngestionPipeline

logger = logging.getLogger(__name__)

SOURCE_TYPES = ("page", "sitemap", "zendesk", "faq")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _aware(dt: datetime | None) -> datetime | None:
    return dt.replace(tzinfo=timezone.utc) if dt is not None and dt.tzinfo is None else dt


# ============================================================================ uploads (enqueue side)
def staging_dir() -> Path:
    path = get_settings().data_dir / "uploads"
    path.mkdir(parents=True, exist_ok=True)
    return path


def stage_upload(username: str, kb: str, filename: str, content: bytes, roles: list[str] | None = None,
                 category: str | None = None, is_golden: bool = False, replace_doc_id: str = "") -> tuple[dict, bool]:
    """Validate and store the original file, then queue ingestion. Returns (job, created)."""
    from backend.security.upload_validator import validate_upload

    pipeline = IngestionPipeline()
    pipeline.kb_cfg(kb)                       # raises for an unknown KB
    valid = validate_upload(filename, content)
    digest = hashlib.sha256(content).hexdigest()
    staged = staging_dir() / f"{uuid.uuid4().hex}__{valid.filename}"
    staged.write_bytes(content)
    payload = {"staged_path": str(staged), "filename": valid.filename, "roles": roles, "category": category,
               "is_golden": is_golden, "replace_doc_id": replace_doc_id, "sha256": digest}
    target = replace_doc_id or valid.filename
    job, created = jobs.enqueue("ingest_upload", kb, payload, dedupe_key=f"upload:{kb}:{target}:{digest}",
                                requested_by=username)
    if not created:
        staged.unlink(missing_ok=True)        # identical upload already queued/running
    repository.write_audit(username, "document_upload_queued", f"{kb}/{valid.filename}", outcome="queued",
                           details={"job_id": job["id"], "sha256": digest, "replace_doc_id": replace_doc_id})
    return job, created


def enqueue_delete(username: str, doc_id: str) -> tuple[dict, bool]:
    doc = registry.get_document(doc_id)
    if not doc or doc["status"] != "active":
        raise KeyError(doc_id)
    return jobs.enqueue("delete_document", doc["kb"], {"doc_id": doc_id}, dedupe_key=f"delete:{doc_id}",
                        requested_by=username)


def enqueue_sync(username: str, kb: str, full: bool = False) -> tuple[dict, bool]:
    return jobs.enqueue("sync_kb", kb, {"full": full}, dedupe_key=f"sync:{kb}:{full}", requested_by=username)


# ============================================================================ web sources
def source_to_dict(src: WebSource) -> dict[str, Any]:
    iso = lambda d: d.isoformat() if d else None  # noqa: E731
    return {"id": src.id, "url": src.url, "kb": src.kb, "source_type": src.source_type, "enabled": src.enabled,
            "interval_minutes": src.interval_minutes, "content_hash": src.content_hash[:16],
            "last_checked_at": iso(src.last_checked_at), "last_changed_at": iso(src.last_changed_at),
            "last_status": src.last_status, "last_error": src.last_error, "created_by": src.created_by}


def add_source(url: str, kb: str, source_type: str = "page", interval_minutes: int | None = None,
               created_by: str = "admin", enabled: bool = True) -> dict:
    if source_type not in SOURCE_TYPES:
        raise ValueError(f"source_type must be one of {SOURCE_TYPES}")
    IngestionPipeline().kb_cfg(kb)
    url = web.validate_url(url)               # SSRF check (domain allow-list + public IPs only)
    with db_session() as s:
        existing = s.scalar(select(WebSource).where(WebSource.url == url))
        if existing:
            existing.kb, existing.source_type, existing.enabled = kb, source_type, enabled
            if interval_minutes:
                existing.interval_minutes = interval_minutes
            return source_to_dict(existing)
        src = WebSource(url=url, kb=kb, source_type=source_type, enabled=enabled, created_by=created_by,
                        interval_minutes=interval_minutes or get_settings().web_monitor_interval_minutes)
        s.add(src)
        s.flush()
        return source_to_dict(src)


def list_sources() -> list[dict]:
    with db_session() as s:
        return [source_to_dict(x) for x in s.scalars(select(WebSource).order_by(WebSource.id)).all()]


def update_source(source_id: int, **fields: Any) -> dict:
    with db_session() as s:
        src = s.get(WebSource, source_id)
        if src is None:
            raise KeyError(source_id)
        for key in ("enabled", "interval_minutes", "kb"):
            if fields.get(key) is not None:
                setattr(src, key, fields[key])
        return source_to_dict(src)


def remove_source(source_id: int, delete_content: bool = True, username: str = "admin") -> dict:
    with db_session() as s:
        src = s.get(WebSource, source_id)
        if src is None:
            raise KeyError(source_id)
        info = source_to_dict(src)
        s.delete(src)
    if delete_content and info["source_type"] == "page":
        path = _page_path(info["kb"], info["url"])
        doc = registry.get_document(IngestionPipeline().doc_id_for(info["kb"], path)) if path.exists() else None
        if doc and doc["status"] == "active":
            enqueue_delete(username, doc["doc_id"])
    return info


def enqueue_crawl(source_id: int, requested_by: str = "scheduler", force: bool = False) -> tuple[dict, bool]:
    with db_session() as s:
        src = s.get(WebSource, source_id)
        if src is None:
            raise KeyError(source_id)
        kb = src.kb
    return jobs.enqueue("crawl_source", kb, {"source_id": source_id, "force": force},
                        dedupe_key=f"crawl:{source_id}", requested_by=requested_by)


def schedule_due_sources() -> int:
    """Queue a crawl for every enabled source whose interval has elapsed (idempotent via dedupe_key)."""
    now, queued = _now(), 0
    with db_session() as s:
        due = [x.id for x in s.scalars(select(WebSource).where(WebSource.enabled.is_(True))).all()
               if not x.last_checked_at or _aware(x.last_checked_at) + timedelta(minutes=x.interval_minutes) <= now]
    for source_id in due:
        _, created = enqueue_crawl(source_id)
        queued += int(created)
    return queued


def find_sources_for_url(url: str) -> list[int]:
    """Webhook helper: sources whose URL equals or contains the changed page URL (e.g. a help-centre article)."""
    with db_session() as s:
        rows = s.scalars(select(WebSource).where(WebSource.enabled.is_(True))).all()
        exact = [r.id for r in rows if r.url.rstrip("/") == url.rstrip("/")]
        if exact:
            return exact
        host = re.sub(r"^https?://", "", url).split("/")[0].lower()
        return [r.id for r in rows if r.source_type in ("zendesk", "sitemap") and host in r.url.lower()]


# ============================================================================ job handlers
def _kb_dir(pipeline: IngestionPipeline, kb: str) -> Path:
    return pipeline.primary_upload_dir(kb).parent if pipeline.primary_upload_dir(kb).name == "uploads" \
        else pipeline.primary_upload_dir(kb)


def _page_path(kb: str, url: str) -> Path:
    pipeline = IngestionPipeline()
    return _kb_dir(pipeline, kb) / "web" / f"{web.slug_for(url)}.md"


def _write_if_changed(path: Path, text: str) -> bool:
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
    return True


def handle_ingest_upload(job: dict, pipeline: IngestionPipeline) -> dict:
    from backend.retrieval.kb_index import get_index_manager

    p, kb = job["payload"], job["kb"]
    staged = Path(p["staged_path"])
    if not staged.exists():
        raise FileNotFoundError(f"Staged upload missing: {staged.name}")
    content = staged.read_bytes()
    if hashlib.sha256(content).hexdigest() != p["sha256"]:
        raise ValueError("Staged upload is corrupted (hash mismatch)")
    replace_id = p.get("replace_doc_id") or ""
    if replace_id:
        doc = registry.get_document(replace_id)
        if not doc or doc["status"] != "active":
            raise KeyError(f"Document {replace_id} no longer exists")
        target, roles, golden = pipeline._abs(doc["source_path"]), doc["access_roles"], doc["doc_type"] == "golden_qa"
    elif p.get("is_golden"):
        target, roles, golden = pipeline._abs(pipeline.kb_cfg(kb)["golden_files"][0]), p.get("roles"), True
    else:
        target, roles, golden = pipeline.primary_upload_dir(kb) / p["filename"], p.get("roles"), False
    new_file = not target.exists()
    if target.exists():
        archive = get_settings().data_dir / "archive" / kb
        archive.mkdir(parents=True, exist_ok=True)
        stamp = _now().strftime("%Y%m%dT%H%M%S")
        archive.joinpath(f"{stamp}__{target.name}").write_bytes(target.read_bytes())
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".uploading")
    tmp.write_bytes(content)
    tmp.replace(target)
    result = pipeline.ingest_file(kb, target, uploaded_by=job["requested_by"], roles=roles,
                                  category=p.get("category") or None, golden=golden, force=bool(replace_id))
    get_index_manager().invalidate(kb)
    if result.status == "failed":
        if new_file:
            target.unlink(missing_ok=True)    # no orphan file for a later sync to pick up
        raise RuntimeError(result.message or "ingestion failed")
    staged.unlink(missing_ok=True)
    return result.to_dict()


def handle_delete(job: dict, pipeline: IngestionPipeline) -> dict:
    from backend.retrieval.kb_index import get_index_manager

    doc_id = job["payload"]["doc_id"]
    doc = registry.get_document(doc_id)
    if not doc or doc["status"] != "active":
        return {"doc_id": doc_id, "status": "already_deleted"}       # idempotent
    result = pipeline.delete_document(doc_id, job["requested_by"], archive_file=True)
    get_index_manager().invalidate(doc["kb"])
    return result.to_dict()


def handle_sync(job: dict, pipeline: IngestionPipeline) -> dict:
    from backend.retrieval.kb_index import get_index_manager

    kb = job["kb"]
    results = pipeline.reindex_kb(kb, job["requested_by"]) if job["payload"].get("full") else \
        pipeline.sync_kb(kb, job["requested_by"])
    get_index_manager().invalidate(kb)
    failed = [r.to_dict() for r in results if r.status == "failed"]
    if failed:
        raise RuntimeError(f"{len(failed)} file(s) failed: " + "; ".join(f["source"] for f in failed)[:500])
    return {"results": [r.status for r in results]}


def _ingest_paths(pipeline: IngestionPipeline, kb: str, paths: list[Path], user: str, golden: set[Path]) -> list[dict]:
    out = []
    for path in paths:
        res = pipeline.ingest_file(kb, path, uploaded_by=user, golden=path in golden)
        out.append(res.to_dict())
        if res.status == "failed":
            raise RuntimeError(f"{path.name}: {res.message}")
    return out


def handle_crawl(job: dict, pipeline: IngestionPipeline) -> dict:
    from backend.retrieval.kb_index import get_index_manager

    source_id, force = job["payload"]["source_id"], job["payload"].get("force", False)
    with db_session() as s:
        src = s.get(WebSource, source_id)
        if src is None:
            return {"status": "source_removed"}
        url, kb, stype, etag, last_mod, old_hash = (src.url, src.kb, src.source_type, src.etag, src.last_modified,
                                                    src.content_hash)
    status, new_hash, details, new_etag, new_lm = "unchanged", old_hash, {}, etag, last_mod
    try:
        if stype == "page":
            path = _page_path(kb, url)
            try:
                scraped = web.scrape_page(url, "" if force else etag, "" if force else last_mod)
            except ValueError as exc:
                if re.search(r"HTTP (404|410)", str(exc)):     # page removed -> remove its chunks
                    doc = registry.get_document(pipeline.doc_id_for(kb, path)) if path.exists() else None
                    if doc and doc["status"] == "active":
                        pipeline.delete_document(doc["doc_id"], "website-monitor", archive_file=True)
                    status, details = "removed", {"note": str(exc)}
                    raise _Done() from None
                raise
            if scraped.not_modified:
                raise _Done()
            new_etag, new_lm = scraped.etag, scraped.last_modified
            page = scraped.page
            if len(page.markdown) < 80:
                raise ValueError("No main content could be extracted from the page")
            new_hash = page.content_hash
            if new_hash == old_hash and path.exists() and not force:
                raise _Done()
            _write_if_changed(path, web.page_to_document(page, scraped.url))
            details = {"scraper": scraped.backend,
                       "ingest": _ingest_paths(pipeline, kb, [path], "website-monitor", set())}
            status = "changed"
        elif stype == "zendesk":
            export = sources.zendesk_export(url.split("/hc/")[0], (url.split("/hc/")[1].strip("/") or "en-gb")
                                            if "/hc/" in url else "en-gb")
            files = sources.zendesk_to_markdown(export, url)
            new_hash = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
            if new_hash == old_hash and not force:
                raise _Done()
            folder = _kb_dir(pipeline, kb)
            changed = [folder / name for name, text in files.items() if _write_if_changed(folder / name, text) or force]
            details = {"articles": len(export["articles"]), "files_changed": [p.name for p in changed],
                       "ingest": _ingest_paths(pipeline, kb, changed, "website-monitor", set())}
            status = "changed" if changed else "unchanged"
        elif stype == "faq":
            faqs = sources.parse(web.fetch_html(url))
            if not faqs:
                raise ValueError("No FAQ entries found - the page layout may have changed")
            golden_cfg = pipeline.kb_cfg(kb).get("golden_files") or []
            golden_path = pipeline._abs(golden_cfg[0]) if golden_cfg else None
            doc_path = _kb_dir(pipeline, kb) / "LearningChain_00_Website_FAQ.md"
            new_hash = hashlib.sha256(json.dumps(faqs).encode()).hexdigest()
            if new_hash == old_hash and not force:
                raise _Done()
            changed = []
            if _write_if_changed(doc_path, sources.faq_document(faqs, url)) or force:
                changed.append(doc_path)
            if golden_path is not None:
                existing = golden_path.read_text(encoding="utf-8") if golden_path.exists() else ""
                if _write_if_changed(golden_path, sources.faq_golden_csv(faqs, url, existing)) or force:
                    changed.append(golden_path)
            details = {"faqs": len(faqs), "ingest": _ingest_paths(pipeline, kb, changed, "website-monitor",
                                                                  {golden_path} if golden_path else set())}
            status = "changed" if changed else "unchanged"
        elif stype == "sitemap":
            fetched = web.fetch(url)
            locs = re.findall(r"<loc>\s*([^<\s]+)\s*</loc>", fetched.text)
            added = 0
            for loc in locs[:500]:
                try:
                    web.validate_url(loc, resolve=False)
                except web.UnsafeURLError:
                    continue
                if loc.endswith(".xml"):
                    continue
                with db_session() as s:
                    exists = s.scalar(select(WebSource.id).where(WebSource.url == loc))
                if not exists:
                    add_source(loc, kb, "page", created_by="sitemap")
                    added += 1
            new_hash = hashlib.sha256("\n".join(sorted(locs)).encode()).hexdigest()
            status, details = ("changed" if added else "unchanged"), {"urls": len(locs), "new_pages": added}
    except _Done:
        pass
    except Exception as exc:
        _update_source(source_id, "failed", old_hash, etag, last_mod, str(exc))
        raise
    finally:
        get_index_manager().invalidate(kb)   # every outcome (changed / removed / partial) is visible immediately
    _update_source(source_id, status, new_hash, new_etag, new_lm, "")
    return {"source_id": source_id, "url": url, "status": status, **details}


class _Done(Exception):
    """Internal: crawl finished early (unchanged / not modified / removed)."""


def _update_source(source_id: int, status: str, content_hash: str, etag: str, last_modified: str, error: str) -> None:
    with db_session() as s:
        src = s.get(WebSource, source_id)
        if src is None:
            return
        src.last_checked_at, src.last_status, src.last_error = _now(), status, error[:500]
        src.etag, src.last_modified = etag or "", last_modified or ""
        if status in ("changed", "removed"):
            src.last_changed_at = _now()
        if status != "failed":
            src.content_hash = content_hash or ""


HANDLERS: dict[str, Callable[[dict, IngestionPipeline], dict]] = {
    "ingest_upload": handle_ingest_upload, "delete_document": handle_delete, "sync_kb": handle_sync,
    "crawl_source": handle_crawl,
}


def notify(job: dict, ok: bool, message: str) -> None:
    """Record the outcome for administrators (audit log + job status in the admin console). If
    ADMIN_NOTIFY_WEBHOOK_URL is set (e.g. a Teams/Slack incoming webhook), post a short message there too."""
    repository.write_audit(job.get("requested_by") or "system", f"ingestion_{'succeeded' if ok else 'failed'}",
                           f"job:{job['id']}:{job['job_type']}", outcome="success" if ok else "failure",
                           details={"kb": job.get("kb"), "message": message[:500]})
    hook = get_settings().admin_notify_webhook_url
    if hook and (not ok or job["job_type"] == "ingest_upload"):
        try:
            import httpx
            httpx.post(hook, json={"text": f"[Sapphirus assistant] ingestion job {job['id']} ({job['job_type']}) "
                                           f"{'succeeded' if ok else 'FAILED'}: {message[:300]}"}, timeout=10)
        except Exception:
            logger.warning("Admin notification webhook failed", exc_info=True)


def process_job(job: dict, pipeline: IngestionPipeline | None = None) -> dict:
    """Run one claimed job; record success / failure (with retry scheduling)."""
    pipeline = pipeline or IngestionPipeline()
    handler = HANDLERS.get(job["job_type"])
    started = time.perf_counter()
    try:
        if handler is None:
            raise ValueError(f"Unknown job type {job['job_type']}")
        result = handler(job, pipeline)
        result["duration_ms"] = round((time.perf_counter() - started) * 1000, 1)
        jobs.complete(job["id"], result)
        notify(job, True, json.dumps(result, default=str)[:300])
        return {"status": "succeeded", **result}
    except Exception as exc:
        logger.exception("Ingestion job %s (%s) failed", job["id"], job["job_type"])
        state = jobs.fail(job["id"], f"{type(exc).__name__}: {exc}")
        if state["status"] == "dead":
            notify(job, False, f"{type(exc).__name__}: {exc}")
        return {"status": state["status"], "error": str(exc)}


def run_pending(max_jobs: int = 100) -> int:
    """Process due jobs synchronously (used by tests and the CLI)."""
    done = 0
    pipeline = IngestionPipeline()
    while done < max_jobs:
        job = jobs.claim_next()
        if job is None:
            break
        process_job(job, pipeline)
        done += 1
    return done


# ============================================================================ background worker
class IngestionWorker(threading.Thread):
    """Polls the job queue; also schedules website checks. One per API process (single vector-store writer)."""

    def __init__(self) -> None:
        super().__init__(name="ingestion-worker", daemon=True)
        self._stop_event = threading.Event()
        self.last_schedule = 0.0
        self.processed = 0

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        s = get_settings()
        logger.info("Ingestion worker started (%s)", jobs.WORKER_ID)
        jobs.requeue_stale()
        pipeline = IngestionPipeline()
        while not self._stop_event.is_set():
            try:
                if s.web_monitor_enabled and time.monotonic() - self.last_schedule > 60:
                    self.last_schedule = time.monotonic()
                    queued = schedule_due_sources()
                    if queued:
                        logger.info("Website monitor queued %d crawl(s)", queued)
                job = jobs.claim_next()
                if job is None:
                    self._stop_event.wait(s.ingest_poll_seconds)
                    continue
                process_job(job, pipeline)
                self.processed += 1
            except Exception:
                logger.exception("Ingestion worker loop error")
                self._stop_event.wait(5)
        logger.info("Ingestion worker stopped")


_worker: IngestionWorker | None = None


def start_worker() -> IngestionWorker | None:
    global _worker
    if not get_settings().ingest_worker_enabled:
        return None
    if _worker is None or not _worker.is_alive():
        _worker = IngestionWorker()
        _worker.start()
    return _worker


def stop_worker() -> None:
    if _worker is not None:
        _worker.stop()
        _worker.join(timeout=10)


def worker_status() -> dict:
    return {"enabled": get_settings().ingest_worker_enabled, "alive": bool(_worker and _worker.is_alive()),
            "processed": _worker.processed if _worker else 0, "worker_id": jobs.WORKER_ID}
