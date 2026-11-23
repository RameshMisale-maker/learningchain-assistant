"""Multi-part question decomposition for retrieval.

A question such as "If I upload a certificate as a PDF, where is it stored and how will a verifier check it?" needs
facts from several documents. Retrieving with the whole question tends to return chunks about only one part, so each
part is also retrieved separately (see ChatService._answer). Detection is a cheap regex check; splitting uses the LLM
when one is configured (it resolves pronouns: "where is it stored" -> "Where is the certificate PDF stored?") and a
rule-based clause split otherwise.
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

MAX_SUB_QUESTIONS = 3
_QUESTION_CUE = re.compile(
    r"\b(what|how|where|who|whom|which|when|why|can|could|does|do|did|is|are|was|were|will|would|should|shall|may)\b",
    re.IGNORECASE)
# clause boundaries: "?", ";", " and " / ", and " / " also ", or a comma directly followed by a question word
_SPLIT = re.compile(
    r"\?|;|,?\s+\b(?:and|also|plus)\b\s+|,\s+(?=(?:what|how|where|who|which|when|why|can|could|does|do|is|are|will|"
    r"would|should)\b)", re.IGNORECASE)
_MIN_WORDS = 3


def clauses(question: str) -> list[str]:
    parts = [p.strip(" ,.;") for p in _SPLIT.split(question or "")]
    return [p for p in parts if len(p.split()) >= _MIN_WORDS]


def _asks(clause: str) -> bool:
    """A clause asks something when a question word opens it (first two words): 'how will a verifier check it',
    'as a student, where ...'. 'I can't remember my password' is a statement, not a question."""
    words = re.sub(r"[^\w\s']", " ", clause).split()[:2]
    return any(_QUESTION_CUE.fullmatch(w) for w in words)


def is_multi_part(question: str) -> bool:
    """True when the question contains two or more clauses that each ask something."""
    return sum(1 for c in clauses(question) if _asks(c)) >= 2


def heuristic_split(question: str) -> list[str]:
    """Rule-based fallback: each asking clause becomes a sub-question; leading non-asking clauses (conditions such
    as 'If I upload a certificate as a PDF') are kept as context. Pronouns are not resolved."""
    parts = clauses(question)
    context = ", ".join(c for c in parts[:next((i for i, c in enumerate(parts) if _asks(c)), 0)])
    asking = [c for c in parts if _asks(c)]
    subs = [(f"{context}, {c}" if context else c) for c in asking]
    return [s[0].upper() + s[1:] + "?" for s in subs][:MAX_SUB_QUESTIONS]


_COMPARISON = [
    re.compile(r"\b(?:difference|differences)\s+between\s+(?P<a>.+?)\s+and\s+(?P<b>.+?)\s*[?.!]*$", re.IGNORECASE),
    re.compile(r"\bhow\s+(?:does|do|is|are)\s+(?P<a>.+?)\s+differ\s+from\s+(?P<b>.+?)\s*[?.!]*$", re.IGNORECASE),
    re.compile(r"\bcompare\s+(?P<a>.+?)\s+(?:and|with|to)\s+(?P<b>.+?)\s*[?.!]*$", re.IGNORECASE),
]


def comparison_split(question: str) -> list[str]:
    """The two sides of a comparison ("difference between revoking and recalling a certificate" -> "revoking a
    certificate", "recalling a certificate"), so each side is retrieved on its own; [] when not a comparison.
    A one-word first side borrows the second side's object ("revoking" + "a certificate ...")."""
    for pattern in _COMPARISON:
        m = pattern.search(question or "")
        if not m:
            continue
        a, b = m.group("a").strip(" ,"), m.group("b").strip(" ,")
        if len(a.split()) == 1 and len(b.split()) > 1:
            a = f"{a} {' '.join(b.split()[1:])}"
        if a and b and a.lower() != b.lower():
            return [a[0].upper() + a[1:], b[0].upper() + b[1:]]
    return []


def decompose(question: str, llm=None) -> list[str]:
    """Sub-questions for a multi-part question or a comparison, or [] for a single question."""
    sides = comparison_split(question)
    if sides:
        return sides
    if not is_multi_part(question):
        return []
    if llm is not None:
        try:
            from backend.generation.prompt_builder import load_prompt

            data, _ = llm.complete_json(
                "You split compound questions into standalone sub-questions. Output JSON only.",
                load_prompt("decompose_prompt.txt").replace("{question}", question), max_tokens=250)
            subs = [str(s).strip() for s in (data.get("sub_questions") or []) if str(s).strip()]
            subs = [s for s in subs if s.lower() != question.strip().lower()]
            if len(subs) >= 2:
                return subs[:MAX_SUB_QUESTIONS]
        except Exception as exc:   # decomposition must never break answering
            logger.warning("Question decomposition failed (%s); using rule-based split", exc)
    subs = heuristic_split(question)
    return subs if len(subs) >= 2 else []
