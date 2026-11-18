"""Query normalization + lenient scope check: understand informal, misspelt questions like ChatGPT does.

    "wt is sapphirus?"          -> "what is Sapphirus?"
    "howw leaning chain works?" -> "how LearningChain works?"
    "hw to verfy certificat"    -> "how to verify certificate"
    "saphirus contact no"       -> "Sapphirus contact number"

Two passes:
1. fuzzy_normalize()  - free and instant: SMS short forms (wt, hw, r, u, pls, abt ...), product-name variants
   (saphirus, leaning chain, LC ...) and fuzzy matching (rapidfuzz, score >= 80) of unknown words against the
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
GLOSSARY: dict[str, list[str]] = {
    "Sapphirus": ["saphirus", "sapphiras", "sapirus", "sapphires", "saphiras", "sapphirous", "sapphiros", "safirus"],
    "LearningChain": ["leaning chain", "learning chain", "lerning chain", "learnchain", "learn chain",
                      "learningchian", "leaningchain", "lerningchain", "learning-chain", "lc"],
}
DOMAIN_TERMS = ["certificate", "certificates", "issue", "issuing", "verify", "verification", "verifier", "share",
                "sharing", "blockchain", "admin", "portal", "user", "credential", "credentials", "revoke", "student",
                "institution", "university", "login", "password", "account", "contact", "pricing", "services",
                "company", "product", "products", "upload", "download", "template", "features", "transcript"]
QUESTION_WORDS = ["what", "how", "who", "why", "when", "where", "which", "does", "can", "tell", "explain"]
SHORTFORMS = {
    "wt": "what", "wat": "what", "wht": "what", "whet": "what", "wut": "what", "wats": "what is", "whats": "what is",
    "hw": "how", "hww": "how", "howw": "how", "hwo": "how", "r": "are", "u": "you", "ur": "your", "pls": "please",
    "plz": "please", "abt": "about", "y": "why", "thx": "thanks", "reg": "regarding", "wrk": "work", "wrks": "works",
    "cn": "can", "shud": "should", "wen": "when", "whr": "where", "wch": "which", "nd": "and", "n": "and",
    "dis": "this", "da": "the", "d": "the", "frm": "from", "abt.": "about", "info": "information",
}
_CANON = {name.lower(): name for name in GLOSSARY}
_WORD_RE = re.compile(r"[A-Za-z0-9']+")


@dataclass
class Understanding:
    original: str
    normalized: str
    scope: str | None = None             # set only when the LLM was asked
    method: str = "none"                 # none | rules | llm
    corrections: list[tuple[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"original_query": self.original, "normalized_query": self.normalized, "scope_check": self.scope,
                "normalization": self.method, "corrections": [f"{a}->{b}" for a, b in self.corrections][:10]}


def _default_known(word: str) -> bool:
    from backend.common.text_utils import STOPWORDS
    return word in STOPWORDS


def fuzzy_normalize(text: str, known: Callable[[str], bool] | None = None) -> tuple[str, list[tuple[str, str]]]:
    """Rule + fuzzy pass. Returns (normalized text, [(from, to), ...])."""
    known = known or _default_known
    out = text
    fixes: list[tuple[str, str]] = []

    # 1. multi-word product-name variants first ("leaning chain" -> LearningChain)
    for canon, variants in GLOSSARY.items():
        for v in sorted(variants, key=len, reverse=True):
            pattern = re.compile(rf"(?<![A-Za-z]){re.escape(v)}(?![A-Za-z])", re.IGNORECASE)
            if pattern.search(out):
                fixes += [(m.group(0), canon) for m in pattern.finditer(out)]
                out = pattern.sub(canon, out)

    # 2. word by word: short forms, then fuzzy matching of unknown words against names / domain terms
    def fix_word(m: re.Match) -> str:
        w = m.group(0)
        lw = w.lower()
        if lw in _CANON or w in GLOSSARY:
            return w
        # "contact no" / "phone no" -> number
        if lw == "no" and re.search(r"(contact|phone|mobile|helpline|whatsapp|cell)\s*$", out[:m.start()], re.I):
            fixes.append((w, "number"))
            return "number"
        if lw in SHORTFORMS and not (lw in ("n", "d", "da", "info") and known(lw)):
            fixes.append((w, SHORTFORMS[lw]))
            return SHORTFORMS[lw]
        if lw.isdigit() or len(lw) < 3 or known(lw):
            return w
        squeezed = re.sub(r"(.)\1{2,}", r"\1", lw)                     # "helppp" -> "help"
        threshold = FUZZY_THRESHOLD if len(squeezed) >= 6 else FUZZY_THRESHOLD_SHORT
        name, score = _best(squeezed, list(_CANON))
        if score >= threshold:
            fixes.append((w, _CANON[name]))
            return _CANON[name]
        pool = DOMAIN_TERMS + (QUESTION_WORDS if len(squeezed) <= 5 else [])
        term, score = _best(squeezed, pool)
        if score >= threshold and term != lw:
            fixes.append((w, term))
            return term
        return w

    out = _WORD_RE.sub(fix_word, out)
    return re.sub(r"\s+", " ", out).strip(), fixes


def unknown_words(text: str, known: Callable[[str], bool]) -> list[str]:
    words = [w.lower() for w in _WORD_RE.findall(text)]
    return [w for w in words if not w.isdigit() and w.lower() not in _CANON and not known(w)]


@lru_cache(maxsize=512)
def _llm_call(llm, text: str) -> tuple[str, str] | None:
    from backend.generation.prompt_builder import load_prompt

    data, _ = llm.complete_json("You normalize and classify chatbot messages. Output JSON only.",
                                load_prompt("normalize_prompt.txt").format(message=text), max_tokens=400)
    normalized = str(data.get("normalized") or "").strip()
    scope = str(data.get("scope") or "").strip().upper()
    if not normalized or scope not in SCOPES:
        return None
    return normalized, scope


def llm_understand(text: str, llm) -> tuple[str, str] | None:
    """(normalized text, scope) from the LLM, or None when unavailable / unusable. Never raises."""
    if llm is None:
        return None
    try:
        result = _llm_call(llm, text)
    except Exception as exc:
        logger.warning("LLM query normalization failed (%s); using rule-based normalization", exc)
        return None
    if result is None:
        return None
    normalized, scope = result
    # sanity: a rewrite, not an answer - keep it short and free of instructions
    if len(normalized) > max(3 * len(text), len(text) + 80):
        return None
    from ingestion.text_cleaner import detect_injection
    if detect_injection(normalized):
        return None
    return normalized, scope


def understand(text: str, known: Callable[[str], bool] | None = None, llm=None) -> Understanding:
    """Rule/fuzzy normalization, plus the LLM when words remain that neither the rules nor the knowledge base know."""
    known = known or _default_known
    normalized, fixes = fuzzy_normalize(text, known)
    result = Understanding(original=text, normalized=normalized, method="rules" if fixes else "none",
                           corrections=fixes)
    if llm is not None and unknown_words(normalized, known):
        judged = llm_understand(normalized, llm)
        if judged:
            result.normalized, result.scope = judged
            result.method = "llm"
    return result
