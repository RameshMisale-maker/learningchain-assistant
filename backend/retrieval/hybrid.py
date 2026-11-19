"""Hybrid retrieval: dense vector search + BM25 keyword search, fused per query with Reciprocal Rank Fusion
(HYBRID_FUSION=rrf) or relative score fusion (HYBRID_FUSION=relative); several phrasings of one question can be
searched together and their result lists fused again with RRF (retrieve_multi, "RAG-Fusion").

Access control is enforced here: chunks whose `access_roles` do not intersect the
user's roles are never returned (document-level ACL).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from backend.common.text_utils import tokenize
from backend.common.timing import stage
from backend.retrieval.kb_index import KBIndex, KBIndexManager, get_index_manager
from config.settings import Settings, get_settings
from ingestion.embedding import Embedder, get_embedder
from ingestion.vector_store import StoredChunk, VectorStore, get_vector_store, roles_from_meta

logger = logging.getLogger(__name__)


@dataclass
class Candidate:
    chunk: StoredChunk
    kb: str
    vector_score: float = 0.0
    bm25_score: float = 0.0
    vector_rank: int | None = None
    bm25_rank: int | None = None
    fused_score: float = 0.0
    rerank_score: float = 0.0
    entity_match: bool = False
    debug: dict = field(default_factory=dict)

    @property
    def id(self) -> str:
        return self.chunk.id

    @property
    def meta(self) -> dict:
        return self.chunk.metadata


def can_access(meta: dict, roles: set[str]) -> bool:
    allowed = roles_from_meta(meta)
    return bool(allowed & roles) or "admin" in roles


class HybridRetriever:
    def __init__(self, settings: Settings | None = None, store: VectorStore | None = None,
                 embedder: Embedder | None = None, indexes: KBIndexManager | None = None) -> None:
        self.settings = settings or get_settings()
        self.store = store or get_vector_store()
        self.embedder = embedder or get_embedder()
        self.indexes = indexes or get_index_manager()

    def retrieve(self, query: str, kb: str, roles: set[str], top_n: int = 30, entities: list[str] | None = None,
                 query_vector: np.ndarray | None = None) -> list[Candidate]:
        index = self.indexes.get(kb)
        if index.size == 0:
            return []
        s = self.settings
        candidates: dict[str, Candidate] = {}

        # ---- dense retrieval (over-fetch to compensate for ACL post-filtering)
        if query_vector is not None:
            qvec = query_vector
        else:
            with stage("query_embedding_ms"):
                qvec = self.embedder.embed_query(query)
        with stage("vector_search_ms"):
            dense = [c for c in self.store.query(kb, qvec, s.top_k_vector * 3) if can_access(c.metadata, roles)]
        for rank, chunk in enumerate(dense[: s.top_k_vector], start=1):
            candidates[chunk.id] = Candidate(chunk=chunk, kb=kb, vector_score=chunk.score, vector_rank=rank)

        # ---- keyword retrieval
        q_tokens = tokenize(query)
        if index.bm25 is not None and q_tokens:
            with stage("bm25_search_ms"):
                scores = index.bm25.get_scores(q_tokens)
            order = np.argsort(-scores)
            rank = 0
            for pos in order:
                if scores[pos] <= 0 or rank >= s.top_k_bm25:
                    break
                chunk = index.chunks[pos]
                if not can_access(chunk.metadata, roles):
                    continue
                rank += 1
                cand = candidates.get(chunk.id)
                if cand is None:
                    cand = Candidate(chunk=StoredChunk(chunk.id, chunk.text, chunk.metadata), kb=kb)
                    candidates[chunk.id] = cand
                cand.bm25_score, cand.bm25_rank = float(scores[pos]), rank

        # ---- fusion (+ entity boost)
        wv, wb, k = s.vector_weight, 1.0 - s.vector_weight, s.rrf_k
        entity_set = {e.lower() for e in (entities or [])}
        relative = s.hybrid_fusion == "relative"
        if relative:   # min-max normalise each list's scores to 0..1 (only chunks that list returned)
            v_scores = [c.vector_score for c in candidates.values() if c.vector_rank]
            b_scores = [c.bm25_score for c in candidates.values() if c.bm25_rank]
            v_lo, v_hi = (min(v_scores), max(v_scores)) if v_scores else (0.0, 1.0)
            b_lo, b_hi = (min(b_scores), max(b_scores)) if b_scores else (0.0, 1.0)
        for cand in candidates.values():
            fused = 0.0
            if relative:
                if cand.vector_rank:
                    fused += wv * ((cand.vector_score - v_lo) / (v_hi - v_lo) if v_hi > v_lo else 1.0)
                if cand.bm25_rank:
                    fused += wb * ((cand.bm25_score - b_lo) / (b_hi - b_lo) if b_hi > b_lo else 1.0)
            else:
                if cand.vector_rank:
                    fused += wv / (k + cand.vector_rank)
                if cand.bm25_rank:
                    fused += wb / (k + cand.bm25_rank)
            if entity_set:
                names = {a.lower() for a in str(cand.meta.get("entity_aliases", "")).split(",") if a}
                names.add(str(cand.meta.get("entity_name", "")).lower())
                text = (cand.chunk.text + " " + str(cand.meta.get("doc_name", ""))).lower()
                if names & entity_set or any(e in text for e in entity_set):
                    cand.entity_match = True
                    fused *= 1.25
            cand.fused_score = fused
        ranked = sorted(candidates.values(), key=lambda c: c.fused_score, reverse=True)
        return ranked[:top_n]

    def retrieve_multi(self, queries: list[str], kb: str, roles: set[str], top_n: int = 30,
                       entities: list[str] | None = None, query_vectors: list | None = None) -> list[Candidate]:
        """Multi-query hybrid retrieval: each phrasing gets its own hybrid search, then the ranked lists are fused
        with RRF, so a chunk found by several phrasings rises and one found by any single phrasing is kept."""
        k = self.settings.rrf_k
        fused: dict[str, float] = {}
        hits: dict[str, int] = {}
        best: dict[str, Candidate] = {}
        vectors = query_vectors or [None] * len(queries)
        for q, qv in zip(queries, vectors):
            for rank, cand in enumerate(self.retrieve(q, kb, roles, top_n=max(top_n, 30), entities=entities,
                                                      query_vector=qv), start=1):
                fused[cand.id] = fused.get(cand.id, 0.0) + 1.0 / (k + rank)
                hits[cand.id] = hits.get(cand.id, 0) + 1
                if cand.id not in best or cand.fused_score > best[cand.id].fused_score:
                    best[cand.id] = cand
        for cid, cand in best.items():
            cand.debug["multi_query_hits"] = hits[cid]
            cand.fused_score = fused[cid]
        return sorted(best.values(), key=lambda c: c.fused_score, reverse=True)[:top_n]


def index_for(kb: str) -> KBIndex:
    return get_index_manager().get(kb)
