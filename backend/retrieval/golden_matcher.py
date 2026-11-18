"""Golden (approved Q&A) matcher.

Scores the user question against every approved question and its alternate phrasings:
  score = 0.45 * dense cosine + 0.30 * stemmed-term Jaccard + 0.25 * character fuzzy ratio
(max over all phrasings of each entry). Keyword hits add a small bonus.

  score >= GOLDEN_DIRECT_THRESHOLD -> the approved answer is returned verbatim (no LLM rewrite)
  score >= GOLDEN_ASSIST_THRESHOLD -> the approved answer is injected as top-priority context
"""
from __future__ import annotations

import hashlib
import re
import threading
from dataclasses import dataclass
from difflib import SequenceMatcher

import numpy as np

from backend.common.text_utils import content_terms, jaccard, normalize_text
from backend.retrieval.hybrid import can_access
from backend.retrieval.kb_index import KBIndexManager, get_index_manager
from config.settings import get_settings
from ingestion.embedding import Embedder, get_embedder
from ingestion.vector_store import StoredChunk


@dataclass
class GoldenMatch:
    chunk: StoredChunk
    kb: str
    score: float
    matched_question: str
    components: dict

    @property
    def qid(self) -> str:
        return str(self.chunk.metadata.get("qid", ""))

    @property
    def answer(self) -> str:
        return str(self.chunk.metadata.get("answer", ""))

    @property
    def question(self) -> str:
        return str(self.chunk.metadata.get("question", ""))


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", normalize_text(text).lower())).strip()


class GoldenMatcher:
    def __init__(self, embedder: Embedder | None = None, indexes: KBIndexManager | None = None) -> None:
        self.embedder = embedder or get_embedder()
        self.indexes = indexes or get_index_manager()
        self._vec_cache: dict[str, np.ndarray] = {}
        self._lock = threading.Lock()

    def _vectors(self, texts: list[str]) -> np.ndarray:
        keys = [hashlib.sha1(t.encode()).hexdigest() for t in texts]
        missing = [(k, t) for k, t in zip(keys, texts) if k not in self._vec_cache]
        if missing:
            vecs = self.embedder.embed_documents([t for _, t in missing])
            with self._lock:
                for (k, _), v in zip(missing, vecs):
                    self._vec_cache[k] = v
        return np.vstack([self._vec_cache[k] for k in keys])

    def match(self, query: str, kb: str, roles: set[str], query_vector: np.ndarray | None = None,
              top_n: int = 3) -> list[GoldenMatch]:
        entries = [c for c in self.indexes.get(kb).golden if can_access(c.metadata, roles)]
        if not entries:
            return []
        qvec = query_vector if query_vector is not None else self.embedder.embed_query(query)
        q_norm, q_terms = _norm(query), content_terms(query)
        results: list[GoldenMatch] = []
        for chunk in entries:
            meta = chunk.metadata
            variants = [str(meta.get("question", ""))] + [a for a in str(meta.get("alternates", "")).split(" || ") if a.strip()]
            vecs = self._vectors(variants)
            dense = vecs @ np.asarray(qvec, dtype=np.float32)
            best_score, best_variant, best_parts = -1.0, variants[0], {}
            for variant, d in zip(variants, dense):
                jac = jaccard(q_terms, content_terms(variant))
                fuzzy = SequenceMatcher(None, q_norm, _norm(variant)).ratio()
                score = 0.45 * max(0.0, float(d)) + 0.30 * jac + 0.25 * fuzzy
                if score > best_score:
                    best_score, best_variant = score, variant
                    best_parts = {"dense": round(float(d), 3), "jaccard": round(jac, 3), "fuzzy": round(fuzzy, 3)}
            keywords = content_terms(str(meta.get("keywords", "")))
            if keywords and q_terms:
                bonus = 0.05 * (len(q_terms & keywords) / len(q_terms))
                best_score += bonus
                best_parts["keyword_bonus"] = round(bonus, 3)
            results.append(GoldenMatch(chunk=chunk, kb=kb, score=min(1.0, best_score), matched_question=best_variant,
                                       components=best_parts))
        results.sort(key=lambda m: m.score, reverse=True)
        return results[:top_n]

    @staticmethod
    def is_direct(match: GoldenMatch | None) -> bool:
        return bool(match) and match.score >= get_settings().golden_direct_threshold

    @staticmethod
    def is_assist(match: GoldenMatch | None) -> bool:
        return bool(match) and match.score >= get_settings().golden_assist_threshold
