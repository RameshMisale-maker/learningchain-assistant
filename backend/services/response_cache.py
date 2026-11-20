"""Answer cache in the database (RESPONSE_CACHE_ENABLED) - free, no Redis needed.

Lookup happens after safety checks, normalisation, the conversation layer and follow-up rewriting, on the
standalone question - so greetings, account questions and follow-ups are never answered from the cache.

* Exact hit   : same product, scope (allowed KBs), user roles, index version, prompt version and normalised question.
* Semantic hit: same everything except the wording, and cosine(question, cached question) >= the product's
                `semantic_cache_threshold` (default 0.95). Skipped while dense search is down.
* Never cached: refusals, out-of-scope, account / escalation / clarification replies, service-unavailable replies,
                answers below CACHE_MIN_CONFIDENCE, answers with warnings (e.g. failed grounding).
* A new index version or prompt version changes the key, so old answers are never served after a change.
* Entries expire after CACHE_TTL_HOURS; a thumbs-down on an answer deletes the entry that produced it.
"""
from __future__ import annotations

import hashlib
import logging
import re
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np

from config.settings import get_settings

logger = logging.getLogger(__name__)

CACHEABLE_TYPES = {"generated", "golden_direct", "golden_semantic"}
SEMANTIC_SCAN_LIMIT = 500


@dataclass
class CacheContext:
    product_id: str
    cache_key: str
    scope_key: str
    normalized_query: str
    embedding: list[float] | None
    index_version: str
    prompt_version: str
    hit_id: int | None = None


_pending: ContextVar[CacheContext | None] = ContextVar("response_cache_ctx", default=None)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _aware(dt: datetime | None) -> datetime | None:
    return dt.replace(tzinfo=timezone.utc) if dt is not None and dt.tzinfo is None else dt


def normalize(query: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", query.lower())).strip()


def _sha(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


def begin(query: str, kbs: list[str], roles: set[str], qvec) -> CacheContext:
    """Build the cache key of this request (stored for `store()` after the answer is ready)."""
    from backend.generation.failover import dense_search_down
    from backend.indexing.versions import index_version_label, prompt_version_label
    from backend.products import current_product

    product_id = current_product().product_id
    index_version, prompt_version = index_version_label(sorted(kbs)), prompt_version_label()
    scope = _sha(product_id, ",".join(sorted(kbs)), ",".join(sorted(roles)), index_version, prompt_version)
    norm = normalize(query)
    embedding = None
    if qvec is not None and not dense_search_down():
        vec = np.asarray(qvec, dtype=np.float32)
        if np.linalg.norm(vec) > 0:
            embedding = [round(float(x), 6) for x in vec]
    ctx = CacheContext(product_id, _sha(scope, norm), scope, norm, embedding, index_version, prompt_version)
    _pending.set(ctx)
    return ctx


def clear_pending() -> None:
    _pending.set(None)


def lookup(ctx: CacheContext) -> dict[str, Any] | None:
    """Cached response dict (exact first, then semantic) or None."""
    from sqlalchemy import select

    from backend.db.models import ResponseCache
    from backend.db.session import db_session
    from backend.products import current_product

    now = _utcnow()
    with db_session() as s:
        row = s.scalars(select(ResponseCache).where(ResponseCache.cache_key == ctx.cache_key)).first()
        kind = "exact"
        if row is not None and _aware(row.expires_at) <= now:
            s.delete(row)
            row = None
        if row is None and ctx.embedding is not None:
            threshold = current_product().semantic_cache_threshold
            q = np.asarray(ctx.embedding, dtype=np.float32)
            q /= np.linalg.norm(q) or 1.0
            best, best_sim = None, threshold
            candidates = s.scalars(select(ResponseCache).where(ResponseCache.scope_key == ctx.scope_key,
                                                               ResponseCache.expires_at > now)
                                   .order_by(ResponseCache.id.desc()).limit(SEMANTIC_SCAN_LIMIT))
            for cand in candidates:
                if not cand.embedding:
                    continue
                v = np.asarray(cand.embedding, dtype=np.float32)
                sim = float(q @ (v / (np.linalg.norm(v) or 1.0)))
                if sim >= best_sim:
                    best, best_sim = cand, sim
            row, kind = best, "semantic"
        if row is None:
            return None
        row.hits += 1
        row.last_hit_at = now
        ctx.hit_id = row.id
        return dict(row.response) | {"_cache": {"kind": kind, "id": row.id, "hits": row.hits}}


def store(resp) -> bool:
    """Cache this response when it is safe to reuse (see module docstring). Returns True when stored."""
    from sqlalchemy import select

    from backend.db.models import ResponseCache
    from backend.db.session import db_session

    ctx = _pending.get()
    settings = get_settings()
    if ctx is None:
        return False
    if ctx.hit_id is not None:                 # served from the cache: remember which message it produced
        with db_session() as s:
            row = s.get(ResponseCache, ctx.hit_id)
            if row is not None and resp.message_id:
                row.last_message_id = resp.message_id
        return False
    if resp.answer_type not in CACHEABLE_TYPES or resp.warnings or resp.confidence < settings.cache_min_confidence \
            or not resp.answer.strip():
        return False
    answer = resp.answer
    prefix = (resp.route or {}).get("greeting_prefix")
    if prefix and answer.startswith(prefix):          # never cache one user's "Hi Ramesh!" for everybody
        answer = answer[len(prefix):].lstrip()
    payload = {"answer": answer, "answer_type": resp.answer_type, "knowledge_bases": list(resp.knowledge_bases),
               "citations": [c.__dict__ for c in resp.citations], "confidence": resp.confidence,
               "golden_match": resp.golden_match, "grounding": resp.grounding, "model": resp.model,
               "contains_sample_data": resp.contains_sample_data,
               "route": {k: v for k, v in (resp.route or {}).items() if k in ("intent", "knowledge_bases", "reason",
                                                                              "method", "rewritten_query")}}
    now = _utcnow()
    with db_session() as s:
        row = s.scalars(select(ResponseCache).where(ResponseCache.cache_key == ctx.cache_key)).first()
        if row is None:
            row = ResponseCache(cache_key=ctx.cache_key)
            s.add(row)
        row.product_id, row.scope_key, row.normalized_query = ctx.product_id, ctx.scope_key, ctx.normalized_query
        row.embedding, row.index_version, row.prompt_version = ctx.embedding, ctx.index_version, ctx.prompt_version
        row.response, row.hits, row.source_message_id = payload, 0, resp.message_id
        row.created_at, row.expires_at = now, now + timedelta(hours=settings.cache_ttl_hours)
    return True


def invalidate_message(message_id: int) -> int:
    """Thumbs-down: drop the cache entry that produced (or served) this message."""
    from sqlalchemy import delete, or_

    from backend.db.models import ResponseCache
    from backend.db.session import db_session

    with db_session() as s:
        result = s.execute(delete(ResponseCache).where(or_(ResponseCache.source_message_id == message_id,
                                                           ResponseCache.last_message_id == message_id)))
        return result.rowcount or 0


def invalidate(product_id: str | None = None) -> int:
    """Drop every entry (of one product). Used after content changes that keep the same index version."""
    from sqlalchemy import delete

    from backend.db.models import ResponseCache
    from backend.db.session import db_session

    with db_session() as s:
        stmt = delete(ResponseCache)
        if product_id:
            stmt = stmt.where(ResponseCache.product_id == product_id)
        return s.execute(stmt).rowcount or 0


def purge_expired() -> int:
    from sqlalchemy import delete

    from backend.db.models import ResponseCache
    from backend.db.session import db_session

    with db_session() as s:
        return s.execute(delete(ResponseCache).where(ResponseCache.expires_at <= _utcnow())).rowcount or 0


def stats() -> dict[str, Any]:
    from sqlalchemy import func, select

    from backend.db.models import ResponseCache
    from backend.db.session import db_session

    with db_session() as s:
        entries, hits = s.execute(select(func.count(ResponseCache.id), func.coalesce(func.sum(ResponseCache.hits), 0))).one()
        return {"enabled": get_settings().response_cache_enabled, "entries": entries, "hits": int(hits)}
