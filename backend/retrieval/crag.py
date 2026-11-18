"""Corrective RAG (CRAG) - an LLM relevance check between retrieval and generation.

    retrieve + rerank -> GRADE each passage against the question (one batched LLM call)
        some relevant  -> keep only the relevant passages ("correct" / "filtered")
        none relevant  -> REWRITE the question as a documentation search query, retrieve again from the
                          knowledge base and grade again ("corrected"), else no evidence ("incorrect")

Differences from the CRAG paper, on purpose:
* No web-search fallback - answers must come only from the approved Sapphirus knowledge base (guardrail).
* Fail open: if the grader errors or returns unusable output, the passages selected by the reranker are used
  unchanged, so CRAG can only remove noise - an LLM outage never turns into a refusal.
Switch off with CRAG_ENABLED=false; CRAG_RETRY=false keeps the filter but skips the rewrite-and-retry step.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from backend.generation.prompt_builder import load_prompt
from backend.retrieval.hybrid import Candidate
from config.settings import Settings

logger = logging.getLogger(__name__)


@dataclass
class CragReport:
    action: str                       # correct | filtered | corrected | incorrect | grader_error | skipped
    graded: int = 0
    relevant: int = 0
    dropped_ids: list[str] = field(default_factory=list)
    rewritten_query: str = ""
    retried: bool = False

    @property
    def verified(self) -> bool:
        """The LLM confirmed the remaining passages are relevant to the question."""
        return self.action in ("correct", "filtered", "corrected")

    def to_dict(self) -> dict:
        return {"action": self.action, "graded": self.graded, "relevant": self.relevant,
                "dropped": self.dropped_ids[:10], "rewritten_query": self.rewritten_query, "retried": self.retried}


class RelevanceGrader:
    def __init__(self, llm, settings: Settings) -> None:
        self.llm = llm
        self.settings = settings

    def _passage(self, n: int, c: Candidate) -> str:
        text = str(c.meta.get("embed_text") or c.chunk.text)
        text = re.sub(r"\s+", " ", text)[: self.settings.crag_chunk_chars]
        title = " > ".join(str(c.meta.get(k)) for k in ("doc_name", "section") if c.meta.get(k))
        return f"[{n}] ({title})\n{text}" if title else f"[{n}]\n{text}"

    def grade(self, question: str, candidates: list[Candidate]) -> list[bool] | None:
        """One verdict per candidate, or None when the grader could not be used (caller keeps the passages)."""
        if not candidates:
            return []
        passages = "\n\n".join(self._passage(i, c) for i, c in enumerate(candidates, 1))
        try:
            data, _ = self.llm.complete_json("You grade passage relevance. Output JSON only.",
                                             load_prompt("crag_grade_prompt.txt").format(question=question,
                                                                                          passages=passages),
                                             max_tokens=400)
        except Exception as exc:
            logger.warning("CRAG grader failed (%s); keeping reranked passages", exc)
            return None
        raw = data.get("relevant") if isinstance(data, dict) else None
        if not isinstance(raw, list):
            logger.warning("CRAG grader returned no usable verdict: %s", str(data)[:200])
            return None
        picked = set()
        for v in raw:
            try:
                picked.add(int(str(v).strip().strip("[]S")))
            except ValueError:
                continue
        return [i in picked for i in range(1, len(candidates) + 1)]

    def rewrite(self, question: str) -> str:
        try:
            result = self.llm.complete("You rewrite search queries. Output only the query.",
                                       load_prompt("crag_rewrite_prompt.txt").format(question=question),
                                       max_tokens=200, temperature=0.0)
        except Exception as exc:
            logger.warning("CRAG query rewrite failed: %s", exc)
            return ""
        text = result.text.strip().strip('"').splitlines()[0].strip() if result.text.strip() else ""
        return text if 3 < len(text) < 400 and text.lower() != question.lower() else ""


def apply_grades(candidates: list[Candidate], verdicts: list[bool]) -> tuple[list[Candidate], list[str]]:
    kept = [c for c, ok in zip(candidates, verdicts) if ok]
    dropped = [c.id for c, ok in zip(candidates, verdicts) if not ok]
    return kept, dropped
