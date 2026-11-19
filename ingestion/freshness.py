"""Document freshness tracking for crawled pages (DOC_TRACKING_ENABLED).

* Every crawl of a page updates its document row: `source_url`, `last_seen_at`, `last_changed_at`, `last_crawl_id`.
* A page that is missing (HTTP 404/410, or dropped from its sitemap) is NOT deleted at once - a temporary error must
  not empty the knowledge base. `missed_crawls` counts consecutive misses; after TOMBSTONE_AFTER_MISSES the document
  is tombstoned: its chunks are removed, the row keeps `tombstoned_at`, and a `stale_doc` review item is created.
* When a page's content changes, cached answers of its product are dropped and every approved answer that cites the
  page (golden_qa.source_url) gets a `stale_doc` review item so a person re-checks it.
* `seed_sources_from_config()` registers each product's `sources.sitemaps` (config/products/<id>.yaml) as monitored
  sitemap sources, crawled every CRAWL_INTERVAL_MINUTES (default daily); sitemap pages are filtered by the product's
  `include_paths` / `exclude_paths` (regular expressions on the URL path) and `allowed_domains`.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from config.settings import get_settings

logger = logging.getLogger(__name__)


def enabled() -> bool:
    return get_settings().doc_tracking_enabled


def _now() -> datetime:
    return datetime.now(timezone.utc)


def url_allowed(url: str, product) -> bool:
    """Product crawl rules: allowed domains, include paths (any must match, if set), exclude paths (none may)."""
    parsed = urlparse(url)
    host, path = (parsed.hostname or "").lower(), parsed.path or "/"
    domains = [d.lower() for d in product.sources.allowed_domains]
    if domains and not any(host == d or host.endswith("." + d) for d in domains):
        return False
    if product.sources.include_paths and not any(re.search(p, path) for p in product.sources.include_paths):
        return False
    return not any(re.search(p, path) for p in product.sources.exclude_paths)


def mark_seen(pipeline, kb: str, path: Path, url: str, changed: bool, crawl_id: str = "") -> None:
    from backend.db.models import Document
    from backend.db.session import db_session

    with db_session() as s:
        doc = s.get(Document, pipeline.doc_id_for(kb, path))
        if doc is None:
            return
        now = _now()
        doc.source_url, doc.last_seen_at, doc.missed_crawls, doc.last_crawl_id = url[:600], now, 0, crawl_id[:64]
        doc.tombstoned_at = None
        if changed or doc.last_changed_at is None:
            doc.last_changed_at = now
    if changed:
        on_content_changed(kb, url)


def mark_missing(pipeline, kb: str, path: Path, url: str, reason: str) -> str:
    """One more miss for this page; tombstone it after TOMBSTONE_AFTER_MISSES. Returns missing | tombstoned | unknown."""
    from backend.db.models import Document
    from backend.db.session import db_session

    doc_id = pipeline.doc_id_for(kb, path)
    with db_session() as s:
        doc = s.get(Document, doc_id)
        if doc is None or doc.status != "active":
            return "unknown"
        doc.missed_crawls = (doc.missed_crawls or 0) + 1
        doc.source_url = doc.source_url or url[:600]
        misses, product_id = doc.missed_crawls, doc.product_id
        tombstone = misses >= get_settings().tombstone_after_misses
        if tombstone:
            doc.tombstoned_at = _now()
    if not tombstone:
        logger.warning("Page %s missing (%s) - miss %d of %d before it is removed", url, reason, misses,
                       get_settings().tombstone_after_misses)
        return "missing"
    pipeline.delete_document(doc_id, "website-monitor", archive_file=True)
    logger.warning("Page %s tombstoned after %d missed crawls (%s)", url, misses, reason)
    _review("stale_doc", f"Source page removed: {url}", product_id,
            {"url": url, "doc_id": doc_id, "reason": reason, "kb": kb, "missed_crawls": misses})
    _invalidate_cache(product_id)
    return "tombstoned"


def on_content_changed(kb: str, url: str) -> None:
    """A page changed: drop cached answers of its product and re-check approved answers that cite it."""
    from sqlalchemy import select

    from backend.db.models import GoldenQA
    from backend.db.session import db_session
    from backend.products import product_for_kb

    product_id = product_for_kb(kb).product_id
    _invalidate_cache(product_id)
    with db_session() as s:
        cited = [(g.qid, g.question) for g in s.scalars(select(GoldenQA).where(GoldenQA.source_url == url,
                                                                            GoldenQA.status == "active"))]
    for qid, question in cited:
        _review("stale_doc", question, product_id, {"url": url, "qid": qid, "reason": "source page changed"})


def _invalidate_cache(product_id: str) -> None:
    if not get_settings().response_cache_enabled:
        return
    try:
        from backend.services.response_cache import invalidate

        invalidate(product_id)
    except Exception:
        logger.exception("Could not invalidate the response cache of %s", product_id)


def _review(source: str, question: str, product_id: str, details: dict[str, Any]) -> None:
    if not get_settings().review_queue_enabled:
        return
    try:
        from backend.services.review import add_item

        add_item(source, question, "", None, product_id or None, details)
    except Exception:
        logger.exception("Could not create a review item for %s", details.get("url"))


def sitemap_missing(pipeline, kb: str, sitemap_url: str, present: set[str], page_path) -> dict[str, int]:
    """Pages added from this sitemap that it no longer lists count a miss (tombstoned after N misses)."""
    from sqlalchemy import select

    from backend.db.models import WebSource
    from backend.db.session import db_session

    with db_session() as s:
        pages = [(x.id, x.url) for x in s.scalars(select(WebSource).where(WebSource.kb == kb,
                                                                          WebSource.source_type == "page",
                                                                          WebSource.created_by == "sitemap"))]
    counts = {"missing": 0, "tombstoned": 0}
    host = urlparse(sitemap_url).hostname
    for source_id, url in pages:
        if url in present or urlparse(url).hostname != host:
            continue
        outcome = mark_missing(pipeline, kb, page_path(kb, url), url, f"not in sitemap {sitemap_url}")
        if outcome in counts:
            counts[outcome] += 1
        if outcome == "tombstoned":
            with db_session() as s:
                src = s.get(WebSource, source_id)
                if src is not None:
                    src.enabled = False          # stop crawling a page that no longer exists
    return counts


def seed_sources_from_config() -> list[dict]:
    """Register every product's configured sitemaps as monitored sources (idempotent)."""
    from backend.products import load_products
    from ingestion.automation import add_source

    added = []
    interval = get_settings().crawl_interval_minutes
    for product in load_products().values():
        for sitemap in product.sources.sitemaps:
            try:
                added.append(add_source(sitemap, product.kb, "sitemap", interval_minutes=interval,
                                        created_by="product-config"))
            except Exception as exc:
                logger.warning("Could not register sitemap %s of %s: %s", sitemap, product.product_id, exc)
    return added


def stale_documents(kb: str | None = None, product_id: str | None = None) -> list[dict[str, Any]]:
    """Crawled documents with misses or tombstones (admin view)."""
    from sqlalchemy import or_, select

    from backend.db.models import Document
    from backend.db.session import db_session

    iso = lambda d: d.isoformat() if d else None  # noqa: E731
    with db_session() as s:
        stmt = select(Document).where(or_(Document.missed_crawls > 0, Document.tombstoned_at.is_not(None)))
        if kb:
            stmt = stmt.where(Document.kb == kb)
        if product_id:
            stmt = stmt.where(Document.product_id == product_id)
        return [{"doc_id": d.doc_id, "kb": d.kb, "doc_name": d.doc_name, "source_url": d.source_url,
                 "missed_crawls": d.missed_crawls, "last_seen_at": iso(d.last_seen_at),
                 "tombstoned_at": iso(d.tombstoned_at), "status": d.status} for d in s.scalars(stmt)]
