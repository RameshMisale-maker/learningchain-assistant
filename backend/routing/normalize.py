"""Query normalization + lenient scope check: understand informal, misspelt questions like ChatGPT does.

    "wt is <compny>?"            -> "what is <Company>?"         (glossary variant -> product name)
    "howw <prodct> works?"       -> "how <Product> works?"
    "hw to verfy certificat"     -> "how to verify certificate"    (short forms, truncated domain words)
    "<compny> contact no"        -> "<Company> contact number"

Two passes:
1. fuzzy_normalize()  - free and instant: SMS short forms (wt, hw, r, u, pls, abt ...), product-name variants
   (the product config's glossary) and fuzzy matching (rapidfuzz, score >= 80) of unknown words against the
   domain glossary. Words the knowledge base already knows are never "corrected".
2. llm_understand()   - only when unknown words remain after pass 1: one LLM call rewrites the message into clean
   English (with the same glossary) AND classifies its scope: IN_SCOPE / OUT_OF_SCOPE / GREETING / UNCLEAR.
   It is also used to double-check an out-of-scope decision before the bot refuses (lenient: unsure -> IN_SCOPE).

The normalized query is used for scope, retrieval and answering; the original is kept for logging. The user's
spelling is never mentioned in replies.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable

try:
    from rapidfuzz import fuzz, process

    def _best(word: str, choices: list[str]) -> tuple[str, float]:
        hit = process.extractOne(word, choices, scorer=fuzz.ratio)
        return (hit[0], float(hit[1])) if hit else ("", 0.0)
except ImportError:  # pragma: no cover - difflib fallback keeps the feature working without rapidfuzz
    import difflib

    def _best(word: str, choices: list[str]) -> tuple[str, float]:
        scored = [(c, difflib.SequenceMatcher(None, word, c).ratio() * 100) for c in choices]
        return max(scored, key=lambda x: x[1]) if scored else ("", 0.0)

logger = logging.getLogger(__name__)

FUZZY_THRESHOLD = 80          # rapidfuzz ratio for words of 6+ letters
FUZZY_THRESHOLD_SHORT = 86    # shorter words need a closer match ("admit" must not become "admin")
SCOPES = ("IN_SCOPE", "OUT_OF_SCOPE", "GREETING", "UNCLEAR")

# product / company names: canonical spelling -> known variants (matched as whole words, case-insensitive)
def _glossary() -> dict[str, list[str]]:
    """Canonical product / company names -> the variants users type (config/products/<id>.yaml -> glossary)."""
    from backend.products import current_product
    return {g.term: [v.lower() for v in g.variants] for g in current_product().glossary}


def _domain_terms() -> list[str]:
    from backend.products import current_product
    return list(current_product().domain_terms)


def _canon() -> dict[str, str]:
    return {name.lower(): name for name in _glossary()}


QUESTION_WORDS = ["what", "how", "who", "why", "when", "where", "which", "does", "can", "tell", "explain"]
SHORTFORMS = {
    "wt": "what", "wat": "what", "wht": "what", "whet": "what", "wut": "what", "wats": "what is", "whats": "what is",
    "hw": "how", "hww": "how", "howw": "how", "hwo": "how", "r": "are", "u": "you", "ur": "your", "pls": "please",
    "plz": "please", "abt": "about", "y": "why", "thx": "thanks", "reg": "regarding", "wrk": "work", "wrks": "works",
    "cn": "can", "shud": "should", "wen": "when", "whr": "where", "wch": "which", "nd": "and", "n": "and",
    "dis": "this", "da": "the", "d": "the", "frm": "from", "abt.": "about", "info": "information",
}
_WORD_RE = re.compile(r"[A-Za-z0-9']+")


@dataclass
class Understanding:
    original: str
    normalized: str
    scope: str | None = None             # set only when the LLM was asked
    method: str = "none"                 # none | rules | llm
    corrections: list[tuple[str, str]] = field(default_factory=list)
    search_queries: list[str] = field(default_factory=list)   # one per separate question (LLM only, max 3)
    language: str = "en"                 # en | hi | hinglish | other (LLM only)
    needs_account_tool: bool = False     # needs this user's own account data, not the help documentation
    unclear: bool = False                # an ambiguous word was kept as written

    def to_dict(self) -> dict:
        return {"original_query": self.original, "normalized_query": self.normalized, "scope_check": self.scope,
                "normalization": self.method, "corrections": [f"{a}->{b}" for a, b in self.corrections][:10],
                "search_queries": self.search_queries, "language": self.language,
                "needs_account_tool": self.needs_account_tool, "unclear": self.unclear}


def _default_known(word: str) -> bool:
    from backend.common.text_utils import STOPWORDS
    return word in STOPWORDS


def fuzzy_normalize(text: str, known: Callable[[str], bool] | None = None) -> tuple[str, list[tuple[str, str]]]:
    """Rule + fuzzy pass. Returns (normalized text, [(from, to), ...])."""
    known = known or _default_known
    out = text
    fixes: list[tuple[str, str]] = []

    glossary, domain_terms, canon_map = _glossary(), _domain_terms(), _canon()
    # 1. multi-word product-name variants first (glossary: "<variant>" -> <Product>)
    for canon, variants in glossary.items():
        for v in sorted(variants, key=len, reverse=True):
            pattern = re.compile(rf"(?<![A-Za-z]){re.escape(v)}(?![A-Za-z])", re.IGNORECASE)
            if pattern.search(out):
                fixes += [(m.group(0), canon) for m in pattern.finditer(out)]
                out = pattern.sub(canon, out)

    # 2. word by word: short forms, then fuzzy matching of unknown words against names / domain terms
    def fix_word(m: re.Match) -> str:
        w = m.group(0)
        lw = w.lower()
        if lw in canon_map or w in glossary:
            return w
        # "contact no" / "phone no" -> number
        if lw == "no" and re.search(r"(contact|phone|mobile|helpline|whatsapp|cell)\s*$", out[:m.start()], re.I):
            fixes.append((w, "number"))
            return "number"
        if lw in SHORTFORMS and not (lw in ("n", "d", "da", "info") and known(lw)):
            fixes.append((w, SHORTFORMS[lw]))
            return SHORTFORMS[lw]
        # truncated domain word ("revok", "certificat", "verificatio"): complete it. These look "known" to the
        # knowledge base, whose vocabulary is stemmed, but the cross-encoder reranker scores the typo badly.
        if len(lw) >= 5 and lw not in domain_terms:
            completion = next((t for t in domain_terms if t.startswith(lw) and len(t) - len(lw) <= 2), None)
            if completion:
                fixes.append((w, completion))
                return completion
        if lw.isdigit() or len(lw) < 3 or known(lw):
            return w
        squeezed = re.sub(r"(.)\1{2,}", r"\1", lw)                     # "helppp" -> "help"
        threshold = FUZZY_THRESHOLD if len(squeezed) >= 6 else FUZZY_THRESHOLD_SHORT
        name, score = _best(squeezed, list(canon_map))
        if score >= threshold:
            fixes.append((w, canon_map[name]))
            return canon_map[name]
        pool = domain_terms + (QUESTION_WORDS if len(squeezed) <= 5 else [])
        term, score = _best(squeezed, pool)
        if score >= threshold and term != lw:
            fixes.append((w, term))
            return term
        return w

    out = _WORD_RE.sub(fix_word, out)
    return re.sub(r"\s+", " ", out).strip(), fixes


def unknown_words(text: str, known: Callable[[str], bool]) -> list[str]:
    words = [w.lower() for w in _WORD_RE.findall(text)]
    return [w for w in words if not w.isdigit() and w.lower() not in _canon() and not known(w)]


_NUMBER_RE = re.compile(r"\d[\d,./-]*\d|\d")


def _numbers(text: str) -> set[str]:
    return {re.sub(r"[,]", "", n) for n in _NUMBER_RE.findall(text or "")}


def _as_bool(value) -> bool:
    return value is True or str(value).strip().lower() in ("true", "yes", "1")


@lru_cache(maxsize=512)
def _llm_call(llm, text: str, history: str = "(none)") -> dict | None:
    from backend.generation.prompt_builder import load_prompt

    data, _ = llm.complete_json("You normalize and classify chatbot messages. Output JSON only.",
                                load_prompt("normalize_prompt.txt").format(message=text, history=history),
                                max_tokens=500)
    corrected = str(data.get("corrected") or data.get("normalized") or "").strip()
    scope = str(data.get("scope") or "").strip().upper()
    if not corrected or scope not in SCOPES:
        return None
    queries = data.get("search_queries") or []
    queries = [str(q).strip() for q in (queries if isinstance(queries, list) else [queries]) if str(q).strip()][:3]
    language = str(data.get("language") or "en").strip().lower()
    return {"corrected": corrected, "scope": scope, "search_queries": queries,
            "language": language if language in ("en", "hi", "hinglish", "other") else "other",
            "needs_account_tool": _as_bool(data.get("needs_account_tool")), "unclear": _as_bool(data.get("unclear"))}


def llm_clean(text: str, llm, history: str = "(none)") -> dict | None:
    """Full LLM clean-up result, or None when unavailable / unusable. Never raises.
    Rejected: answers instead of rewrites, injected instructions, and rewrites that drop or change a number
    (IDs, amounts and dates must be kept exactly as the user wrote them)."""
    if llm is None:
        return None
    try:
        result = _llm_call(llm, text, history)
    except Exception as exc:
        logger.warning("LLM query normalization failed (%s); using rule-based normalization", exc)
        return None
    if result is None:
        return None
    corrected = result["corrected"]
    if len(corrected) > max(3 * len(text), len(text) + 80):
        return None
    from ingestion.text_cleaner import detect_injection
    if detect_injection(corrected) or any(detect_injection(q) for q in result["search_queries"]):
        return None
    if not _numbers(text) <= _numbers(corrected):
        logger.warning("LLM normalization changed a number; using rule-based normalization")
        return None
    return result


def llm_understand(text: str, llm, history: str = "(none)") -> tuple[str, str] | None:
    """(normalized text, scope) from the LLM, or None. Used by the lenient scope check."""
    result = llm_clean(text, llm, history)
    return (result["corrected"], result["scope"]) if result else None


def reply_language(original: str, label: str | None) -> str:
    """The assistant always answers in English. Questions in Hindi, Hinglish or other languages are still understood
    (the clean-up step translates them to English for search); only the reply language is fixed."""
    return "English"


def understand(text: str, known: Callable[[str], bool] | None = None, llm=None,
               history: list[dict] | None = None) -> Understanding:
    """Rule/fuzzy normalization, plus the LLM when words remain that neither the rules nor the knowledge base know,
    or the message is written in another script (Hindi etc.)."""
    known = known or _default_known
    normalized, fixes = fuzzy_normalize(text, known)
    result = Understanding(original=text, normalized=normalized, method="rules" if fixes else "none",
                           corrections=fixes)
    other_script = bool(re.search(r"[^\x00-\x7f]", re.sub(r"[\u2018-\u201f\u2026\xa0]", "", normalized)))
    if llm is not None and (unknown_words(normalized, known) or other_script):   # typos left, or Hindi etc.
        history_text = "(none)"
        if history:
            from backend.generation.prompt_builder import format_history
            history_text = format_history(history, 2)                            # last 4 messages
        cleaned = llm_clean(normalized, llm, history_text)
        if cleaned:
            result.normalized, result.scope, result.method = cleaned["corrected"], cleaned["scope"], "llm"
            result.search_queries = cleaned["search_queries"]
            result.language = cleaned["language"]
            result.needs_account_tool = cleaned["needs_account_tool"]
            result.unclear = cleaned["unclear"]
    return result
