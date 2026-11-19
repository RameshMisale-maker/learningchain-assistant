"""Versioned knowledge-base indexes with an eval gate, promote and instant rollback.

How it works
------------
* Every vector-store call resolves its collection through `ingestion.vector_store.collection_name(kb)`, which asks
  `resolve_collection(kb)` here first:
    - inside `building(kb, name)` (the thread doing a build) -> the collection being built;
    - INDEX_VERSIONING_ENABLED and an `index_aliases` row exists -> the live version's collection;
    - otherwise None -> the classic single collection `<prefix><kb>` (today's behaviour).
* `build_version(kb)` ingests every source file of the KB into a NEW collection `<prefix><kb>_v<id>`, scores it with
  the retrieval eval gate and switches the alias only if  score >= live score - EVAL_TOLERANCE.
  Users keep being served by the live version during the whole build.
* `promote(version_id)` / `rollback(kb)` switch the alias row in one transaction; the KB revision is bumped so every
  worker reloads its BM25 / in-memory index on the next request.
* The newest KEEP_INDEX_VERSIONS versions are kept; older retired collections are dropped. The pre-versioning
  collection is registered as the first version so a rollback can always return to it, and it is never dropped.
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Iterator

from config.settings import get_settings

logger = logging.getLogger(__name__)

_building: ContextVar[dict[str, str] | None] = ContextVar("index_building", default=None)
_alias_cache: dict[str, tuple[float, str | None, int | None]] = {}
_alias_lock = threading.Lock()
ALIAS_TTL_SECONDS = 1.0
EVAL_ROLES = {"admin"}          # the gate scores the whole index, independent of user roles


# ---------------------------------------------------------------- collection resolution
@contextmanager
def building(kb: str, collection: str) -> Iterator[None]:
    """Route every vector-store call for `kb` made by this thread/context to `collection`."""
    current = dict(_building.get() or {})
    current[kb] = collection
    token = _building.set(current)
    try:
        yield
    finally:
        _building.reset(token)


def _alias(kb: str) -> tuple[str | None, int | None]:
    now = time.monotonic()
    cached = _alias_cache.get(kb)
    if cached and now - cached[0] < ALIAS_TTL_SECONDS:
        return cached[1], cached[2]
    from backend.db.models import IndexAlias, IndexVersion
    from backend.db.session import db_session

    name, version_id = None, None
    try:
        with db_session() as s:
            alias = s.get(IndexAlias, kb)
            if alias and alias.live_version_id:
                version = s.get(IndexVersion, alias.live_version_id)
                if version:
                    name, version_id = version.collection, version.version_id
    except Exception:   # the table may not exist yet (first start before migrations) - use the classic collection
        logger.debug("index alias lookup failed for %s", kb, exc_info=True)
    with _alias_lock:
        _alias_cache[kb] = (now, name, version_id)
    return name, version_id


def invalidate_alias_cache() -> None:
    with _alias_lock:
        _alias_cache.clear()


def resolve_collection(kb: str) -> str | None:
    override = _building.get()
    if override and kb in override:
        return override[kb]
    if not get_settings().index_versioning_enabled:
        return None
    return _alias(kb)[0]


def live_version_id(kb: str) -> int | None:
    if not get_settings().index_versioning_enabled:
        return None
    return _alias(kb)[1]


def index_version_label(kbs: list[str]) -> str:
    """Request-log label of the indexes used, e.g. "products:v4,company:v2" ("base" = unversioned collection)."""
    from backend.db.repository import get_kb_revision

    parts = []
    for kb in kbs:
        vid = live_version_id(kb)
        parts.append(f"{kb}:v{vid}" if vid else f"{kb}:base.r{get_kb_revision(kb)}")
    return ",".join(parts)[:200]


def prompt_version_label() -> str:
    """Product prompt_version + a short hash of the prompt files, so any prompt edit is visible in the logs."""
    from backend.products import current_product

    return f"{current_product().prompt_version}+{_prompts_hash()}"[:64]


_prompt_hash: tuple[float, str] | None = None


def _prompts_hash() -> str:
    global _prompt_hash
    now = time.monotonic()
    if _prompt_hash and now - _prompt_hash[0] < 30:
        return _prompt_hash[1]
    digest = hashlib.sha256()
    folder = get_settings().prompts_dir
    for f in sorted(folder.glob("*.txt")) if folder.exists() else []:
        digest.update(f.name.encode())
        digest.update(f.read_bytes())
    _prompt_hash = (now, digest.hexdigest()[:8])
    return _prompt_hash[1]


# ---------------------------------------------------------------- eval gate (retrieval only, no LLM - free)
def _eval_cases(kb: str, store, max_cases: int) -> list[dict[str, Any]]:
    """Cases from evals/<product_id>.jsonl ({"question", "kb", "expected": [substring of chunk id / source]}),
    else the approved Q&A of the collection being scored (alternate phrasing -> its approved-answer chunk)."""
    from backend.products import product_for_kb

    path = get_settings().evals_dir / f"{product_for_kb(kb).product_id}.jsonl"
    cases: list[dict[str, Any]] = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("kb", kb) == kb and row.get("question") and row.get("expected"):
                expected = row["expected"] if isinstance(row["expected"], list) else [row["expected"]]
                cases.append({"question": row["question"], "expected": [str(e) for e in expected]})
    if not cases:
        golden = sorted((c for c in store.get_all(kb) if c.metadata.get("doc_type") == "golden_qa"), key=lambda c: c.id)
        for c in golden:
            alternates = [a for a in str(c.metadata.get("alternates", "")).split(" || ") if a.strip()]
            question = alternates[0] if alternates else str(c.metadata.get("question", ""))
            if question:
                cases.append({"question": question, "expected": [c.id]})
    if len(cases) > max_cases:     # deterministic, evenly spread sample
        step = len(cases) / max_cases
        cases = [cases[int(i * step)] for i in range(max_cases)]
    try:                           # regression cases added from the review queue are always included
        from backend.services.review import list_eval_cases

        cases += [{"question": c["question"], "expected": [str(e) for e in c["expected_sources"]]}
                  for c in list_eval_cases(kb=kb) if c["expected_sources"]]
    except Exception:
        logger.debug("eval_cases unavailable", exc_info=True)
    return cases


def evaluate_collection(kb: str, collection: str | None = None) -> dict[str, Any]:
    """hit@5 and MRR@10 of hybrid retrieval over `collection` (default: what currently serves `kb`)."""
    from backend.retrieval.hybrid import HybridRetriever
    from backend.retrieval.kb_index import KBIndexManager
    from ingestion.embedding import get_embedder
    from ingestion.vector_store import get_vector_store

    settings = get_settings()
    store = get_vector_store()

    def run() -> dict[str, Any]:
        cases = _eval_cases(kb, store, settings.eval_gate_max_cases)
        if not cases:
            return {"score": None, "cases": 0, "hit_at_5": None, "mrr": None}
        retriever = HybridRetriever(settings, store=store, embedder=get_embedder(),
                                    indexes=KBIndexManager(store, refresh_seconds=1e9))
        hits, rr = 0, 0.0
        for case in cases:
            found = retriever.retrieve(case["question"], kb, EVAL_ROLES, top_n=10)
            rank = next((i for i, c in enumerate(found[:10], start=1)
                         if any(e == c.chunk.id or e in c.chunk.id or e in str(c.chunk.metadata.get("source_file", ""))
                                or e in str(c.chunk.metadata.get("doc_name", "")) for e in case["expected"])), None)
            hits += bool(rank and rank <= 5)
            rr += 1.0 / rank if rank else 0.0
        hit5, mrr = hits / len(cases), rr / len(cases)
        return {"score": round(0.5 * hit5 + 0.5 * mrr, 4), "cases": len(cases), "hit_at_5": round(hit5, 4),
                "mrr": round(mrr, 4)}

    if collection:
        with building(kb, collection):
            return run()
    return run()


# ---------------------------------------------------------------- build / promote / rollback
def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _row(v) -> dict[str, Any]:
    return {"version_id": v.version_id, "product_id": v.product_id, "kb": v.kb, "collection": v.collection,
            "created_at": v.created_at.isoformat() if v.created_at else None, "crawl_id": v.crawl_id,
            "embedding_model": v.embedding_model, "chunk_config": v.chunk_config, "doc_count": v.doc_count,
            "chunk_count": v.chunk_count, "eval_score": v.eval_score, "eval_details": v.eval_details,
            "status": v.status, "notes": v.notes}


def _register_baseline(kb: str) -> None:
    """First versioned build: register the existing collection as a live version so it can be rolled back to."""
    from backend.db.models import IndexAlias, IndexVersion
    from backend.db.session import db_session
    from backend.products import product_for_kb
    from ingestion.vector_store import base_collection_name, get_vector_store

    with db_session() as s:
        if s.get(IndexAlias, kb) is not None:
            return
        store = get_vector_store()
        base = base_collection_name(kb)
        with building(kb, base):
            count, model = store.count(kb), store.model_id(kb) or ""
        if not count:
            return
        version = IndexVersion(product_id=product_for_kb(kb).product_id, kb=kb, collection=base, embedding_model=model,
                               chunk_count=count, status="live", notes="pre-versioning collection (baseline)")
        s.add(version)
        s.flush()
        s.add(IndexAlias(kb=kb, product_id=version.product_id, live_version_id=version.version_id))
    invalidate_alias_cache()


def build_version(kb: str, requested_by: str = "system", crawl_id: str = "", promote_if_passed: bool = True,
                  pipeline=None) -> dict[str, Any]:
    """Build a new index version of `kb`, score it and promote it when it passes the eval gate."""
    from backend.db.models import IndexVersion
    from backend.db.session import db_session
    from backend.products import product_for_kb
    from ingestion import registry
    from ingestion.pipeline import IngestionPipeline
    from ingestion.vector_store import base_collection_name, get_vector_store

    settings = get_settings()
    pipeline = pipeline or IngestionPipeline()
    pipeline.kb_cfg(kb)                                   # unknown KB -> ValueError
    try:                                                  # never start a build with a broken embedder
        pipeline.embedder.embed_documents(["index build preflight check"])
    except Exception as exc:
        raise RuntimeError(f"Embedding model '{pipeline.embedder.model_id}' is not usable - no build started: {exc}") from exc
    _register_baseline(kb)
    with db_session() as s:
        version = IndexVersion(product_id=product_for_kb(kb).product_id, kb=kb, collection="", crawl_id=crawl_id,
                               embedding_model=pipeline.embedder.model_id, status="building",
                               chunk_config=dict(pipeline.kb_cfg(kb).get("chunking", {})),
                               notes=f"requested by {requested_by}")
        s.add(version)
        s.flush()
        version.collection = f"{base_collection_name(kb)}_v{version.version_id}"
        version_id, collection = version.version_id, version.collection

    store = get_vector_store()
    started = time.perf_counter()
    try:
        with building(kb, collection):
            store.drop(kb)
            files, golden = pipeline.discover(kb)
            roles = {d["source_path"]: d["access_roles"] for d in registry.list_documents(kb)}
            results = [pipeline.ingest_file(kb, f, uploaded_by=requested_by, force=True,
                                            roles=roles.get(pipeline._rel(f)) or None) for f in files]
            results += [pipeline.ingest_file(kb, g, uploaded_by=requested_by, golden=True, force=True) for g in golden]
            chunk_count = store.count(kb)
        failed = [r.source for r in results if r.status == "failed"]
        evaluation = evaluate_collection(kb, collection)
    except Exception as exc:
        _finish(version_id, status="failed", notes=f"build error: {type(exc).__name__}: {str(exc)[:300]}")
        _drop_collection(kb, collection)
        raise
    live = live_version_id(kb) if settings.index_versioning_enabled else _alias(kb)[1]
    live_score = _live_score(kb, live)
    score = evaluation["score"]
    passed = chunk_count > 0 and (live_score is None or score is None or score >= live_score - settings.eval_tolerance)
    details = evaluation | {"live_version_id": live, "live_score": live_score, "tolerance": settings.eval_tolerance,
                            "failed_files": failed[:20], "build_seconds": round(time.perf_counter() - started, 1)}
    _finish(version_id, status="ready" if passed else "rejected", eval_score=score, eval_details=details,
            doc_count=sum(1 for r in results if r.status != "failed"), chunk_count=chunk_count)
    if passed and promote_if_passed:
        promote(version_id, requested_by)
    elif not passed:
        logger.warning("Index version %s of %s rejected by the eval gate: %.3f < live %.3f - %.2f", version_id, kb,
                       score or 0, live_score or 0, settings.eval_tolerance)
        _drop_collection(kb, collection)
    return get_version(version_id) | {"passed": passed, "ingest_results": results}


def _live_score(kb: str, live_id: int | None) -> float | None:
    """Score of the version serving users now (stored, or measured once for a version that has none)."""
    if live_id is None:
        return None
    current = get_version(live_id)
    if current["eval_score"] is not None:
        return current["eval_score"]
    measured = evaluate_collection(kb, current["collection"])
    _finish(live_id, eval_score=measured["score"], eval_details=measured)
    return measured["score"]


def _finish(version_id: int, **fields: Any) -> None:
    from backend.db.models import IndexVersion
    from backend.db.session import db_session

    with db_session() as s:
        v = s.get(IndexVersion, version_id)
        for k, val in fields.items():
            setattr(v, k, val)


def _drop_collection(kb: str, collection: str) -> None:
    from ingestion.vector_store import base_collection_name, get_vector_store

    if collection == base_collection_name(kb):
        return      # the pre-versioning collection is never dropped (turning the flag off falls back to it)
    with building(kb, collection):
        get_vector_store().drop(kb)


def get_version(version_id: int) -> dict[str, Any]:
    from backend.db.models import IndexVersion
    from backend.db.session import db_session

    with db_session() as s:
        v = s.get(IndexVersion, version_id)
        if v is None:
            raise ValueError(f"Unknown index version {version_id}")
        return _row(v)


def list_versions(kb: str | None = None, product_id: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
    from sqlalchemy import select

    from backend.db.models import IndexAlias, IndexVersion
    from backend.db.session import db_session

    with db_session() as s:
        live = {a.kb: a.live_version_id for a in s.scalars(select(IndexAlias))}
        stmt = select(IndexVersion).order_by(IndexVersion.version_id.desc()).limit(limit)
        if kb:
            stmt = stmt.where(IndexVersion.kb == kb)
        if product_id:
            stmt = stmt.where(IndexVersion.product_id == product_id)
        return [_row(v) | {"is_live": live.get(v.kb) == v.version_id} for v in s.scalars(stmt)]


def promote(version_id: int, requested_by: str = "system") -> dict[str, Any]:
    """Make `version_id` the live index of its KB (atomic alias switch) and prune old versions."""
    from backend.db.models import IndexAlias, IndexVersion
    from backend.db.repository import bump_kb_revision, write_audit
    from backend.db.session import db_session

    with db_session() as s:
        v = s.get(IndexVersion, version_id)
        if v is None:
            raise ValueError(f"Unknown index version {version_id}")
        if v.status in ("failed", "pruned", "building"):
            raise ValueError(f"Index version {version_id} is {v.status} and cannot be promoted")
        alias = s.get(IndexAlias, v.kb)
        if alias is None:
            alias = IndexAlias(kb=v.kb, product_id=v.product_id)
            s.add(alias)
        if alias.live_version_id and alias.live_version_id != version_id:
            old = s.get(IndexVersion, alias.live_version_id)
            if old is not None:
                old.status = "retired"
            alias.previous_version_id = alias.live_version_id
        alias.live_version_id, alias.switched_at = version_id, _utcnow()
        v.status = "live"
        kb, model = v.kb, v.embedding_model
    invalidate_alias_cache()
    bump_kb_revision(kb, model)
    write_audit(requested_by, "index_promote", kb, details={"version_id": version_id})
    _prune(kb)
    logger.info("Index version %s is now live for %s", version_id, kb)
    return get_version(version_id)


def rollback(kb: str, requested_by: str = "system") -> dict[str, Any]:
    """Switch `kb` back to the previously live version."""
    from backend.db.models import IndexAlias
    from backend.db.session import db_session

    with db_session() as s:
        alias = s.get(IndexAlias, kb)
        previous = alias.previous_version_id if alias else None
    if not previous:
        raise ValueError(f"No previous index version to roll back to for '{kb}'")
    return promote(previous, requested_by)


def rollback_product(product_id: str, requested_by: str = "system") -> list[dict[str, Any]]:
    """Roll back every knowledge base of a product that has a previous version."""
    from backend.products import get_product

    out = []
    for kb in get_product(product_id).namespaces:
        try:
            out.append(rollback(kb, requested_by))
        except ValueError as exc:
            out.append({"kb": kb, "error": str(exc)})
    return out


def _prune(kb: str) -> None:
    """Keep the newest KEEP_INDEX_VERSIONS usable versions; drop older retired / rejected collections."""
    from sqlalchemy import select

    from backend.db.models import IndexAlias, IndexVersion
    from backend.db.session import db_session

    keep = max(2, get_settings().keep_index_versions)
    with db_session() as s:
        alias = s.get(IndexAlias, kb)
        protected = {alias.live_version_id, alias.previous_version_id} if alias else set()
        versions = list(s.scalars(select(IndexVersion).where(IndexVersion.kb == kb,
                                                             IndexVersion.status.in_(["retired", "rejected", "ready"]))
                                  .order_by(IndexVersion.version_id.desc())))
        doomed = [(v.version_id, v.collection) for v in versions[keep - 1:] if v.version_id not in protected]
    for version_id, collection in doomed:
        _drop_collection(kb, collection)
        _finish(version_id, status="pruned")
