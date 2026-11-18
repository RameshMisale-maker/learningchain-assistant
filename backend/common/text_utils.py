"""Shared text utilities: tokenisation, stop-words, light stemming, sentence splitting.

Used by ingestion (chunking, BM25 tokens), retrieval (BM25, reranking), routing
(term statistics), grounding checks and evaluation metrics so every component
tokenises text the same way.
"""
from __future__ import annotations

import re
import unicodedata

STOPWORDS = frozenset(
    """a about above after again against all am an and any are aren't as at be because been before being below
    between both but by can can't cannot could couldn't did didn't do does doesn't doing don't down during each
    few for from further had hadn't has hasn't have haven't having he he'd he'll he's her here here's hers herself
    him himself his how how's i i'd i'll i'm i've if in into is isn't it it's its itself let's me more most mustn't
    my myself no nor not of off on once only or other ought our ours ourselves out over own same shan't she she'd
    she'll she's should shouldn't so some such than that that's the their theirs them themselves then there there's
    these they they'd they'll they're they've this those through to too under until up very was wasn't we we'd we'll
    we're we've were weren't what what's when when's where where's which while who who's whom why why's with won't
    would wouldn't you you'd you'll you're you've your yours yourself yourselves tell please explain describe give
    know want need get also us may might shall will just much many available kind sort exactly actually
    basically currently happen happens happened stand stands""".split()
)

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.'][a-z0-9]+)*")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])|\n{2,}|\n(?=\s*[-*•]\s)|\n(?=\s*\d+[.)]\s)")


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    text = text.replace("–", "-").replace("—", "-")
    return text


def stem(token: str) -> str:
    """Very small suffix stripper (deterministic, dependency free)."""
    if len(token) <= 3 or not token.isalpha():
        return token
    base = token
    if base.endswith(("ies", "ied")) and len(base) > 4:
        base = base[:-3] + "y"
    elif base.endswith(("sses", "shes", "ches", "xes", "zes")):
        base = base[:-2]
    else:
        for suffix in ("ations", "ation", "ments", "ment", "ings", "ing", "ers", "ed", "s"):
            if base.endswith(suffix) and len(base) - len(suffix) >= 3:
                if suffix == "s" and base.endswith(("ss", "us", "is")):
                    break
                base = base[: -len(suffix)]
                break
    if base.endswith("e") and len(base) > 4:
        base = base[:-1]
    return base


def tokenize(text: str, remove_stopwords: bool = True, do_stem: bool = True) -> list[str]:
    text = normalize_text(text).lower()
    tokens = _TOKEN_RE.findall(text)
    out: list[str] = []
    for tok in tokens:
        tok = tok.strip(".-'")
        if not tok:
            continue
        if remove_stopwords and tok in STOPWORDS:
            continue
        out.append(stem(tok) if do_stem else tok)
    return out


def content_terms(text: str) -> set[str]:
    return set(tokenize(text))


def split_sentences(text: str) -> list[str]:
    parts = _SENTENCE_RE.split(text.strip())
    return [p.strip() for p in parts if p and p.strip()]


NUMBER_RE = re.compile(r"(?<![\w.])\d+(?:,\d+)*(?:\.\d+)?(?![\w])")


def extract_numbers(text: str) -> set[str]:
    return {n.replace(",", "") for n in NUMBER_RE.findall(text or "")}


def token_overlap(a: str, b: str) -> float:
    """Fraction of content terms in `a` that also appear in `b`."""
    ta, tb = content_terms(a), content_terms(b)
    if not ta:
        return 0.0
    return len(ta & tb) / len(ta)


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)
