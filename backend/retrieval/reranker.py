"""Rerankers.

RERANKER_BACKEND=
  cross_encoder -> sentence-transformers CrossEncoder (default: cross-encoder/ms-marco-MiniLM-L-6-v2);
                   reads query and passage together, much more precise than bi-encoder similarity.
  lexical       -> dependency-free scorer: IDF-weighted query-term coverage + fused rank + dense score.
  none          -> keep hybrid (RRF) order.
If the cross-encoder cannot be loaded, the lexical reranker is used automatically.
All rerankers return scores in [0, 1] so MIN_RERANK_SCORE works across backends.
"""
from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from functools import lru_cache

from backend.common.text_utils import tokenize
from backend.retrieval.hybrid import Candidate
from backend.retrieval.kb_index import get_index_manager
from config.settings import get_settings

logger = logging.getLogger(__name__)


def weighted_coverage(query: str, text: str, kb: str | None = None) -> float:
    """IDF-weighted fraction of the query's content terms present in `text`."""
    q_terms = set(tokenize(query))
    if not q_terms:
        return 0.0
    t_terms = set(tokenize(text))
    idx = get_index_manager().get(kb) if kb else None
    weights = {t: (idx.idf(t) if idx else 1.0) for t in q_terms}
    total = sum(weights.values()) or 1.0
    return sum(w for t, w in weights.items() if t in t_terms) / total


class Reranker(ABC):
    name = "base"

    @abstractmethod
    def rerank(self, query: str, candidates: list[Candidate]) -> list[Candidate]: ...


class NoReranker(Reranker):
    name = "none"

    def rerank(self, query, candidates):
        if not candidates:
            return []
        top = max(c.fused_score for c in candidates) or 1.0
        for c in candidates:
            c.rerank_score = c.fused_score / top
        return sorted(candidates, key=lambda c: c.rerank_score, reverse=True)


class LexicalReranker(Reranker):
    name = "lexical"

    def rerank(self, query, candidates):
        if not candidates:
            return []
        top_fused = max(c.fused_score for c in candidates) or 1.0
        for c in candidates:
            text = c.meta.get("embed_text") or c.chunk.text
            coverage = weighted_coverage(query, text, c.kb)
            dense = max(0.0, min(1.0, c.vector_score))
            rank_part = c.fused_score / top_fused
            score = 0.50 * coverage + 0.25 * rank_part + 0.25 * dense
            if c.entity_match:
                score = min(1.0, score + 0.05)
            c.rerank_score = score
            c.debug.update({"coverage": round(coverage, 3), "rank_part": round(rank_part, 3), "dense": round(dense, 3)})
        return sorted(candidates, key=lambda c: c.rerank_score, reverse=True)


class CrossEncoderReranker(Reranker):
    name = "cross_encoder"

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name, max_length=512)

    def rerank(self, query, candidates):
        if not candidates:
            return []
        # score the contextual text (document + section header + chunk), the same text that was embedded: small
        # (child) chunks often never repeat their section's subject, e.g. "- On the next page, select ..." under
        # "Admin: How do I Revoke the Certificates"
        pairs = [(query, (c.meta.get("embed_text") or c.chunk.text)[:2000]) for c in candidates]
        raw = self.model.predict(pairs, show_progress_bar=False)
        for c, score in zip(candidates, raw):
            s = float(score)
            c.rerank_score = s if 0.0 <= s <= 1.0 else 1.0 / (1.0 + math.exp(-s))
            if c.entity_match:
                c.rerank_score = min(1.0, c.rerank_score + 0.05)
        return sorted(candidates, key=lambda c: c.rerank_score, reverse=True)


@lru_cache(maxsize=1)
def get_reranker() -> Reranker:
    settings = get_settings()
    backend = settings.reranker_backend
    if backend == "cross_encoder":
        try:
            reranker = CrossEncoderReranker(settings.reranker_model)
            logger.info("Reranker: cross-encoder %s", settings.reranker_model)
            return reranker
        except Exception as exc:  # missing package / model download blocked
            logger.warning("Cross-encoder unavailable (%s); falling back to lexical reranker", exc)
            return LexicalReranker()
    if backend == "lexical":
        return LexicalReranker()
    if backend == "none":
        return NoReranker()
    raise ValueError(f"Unknown RERANKER_BACKEND '{backend}'")
