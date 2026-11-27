"""Grounding / hallucination checks and citation handling.

Lexical mode (default, no extra cost):
  * every sentence's content terms must be covered by the sources (>= 0.5 overlap)
  * every number in the answer must appear in the sources (numeric hallucination guard)
  * faithfulness = supported sentences / checked sentences
LLM mode (GROUNDING_MODE=llm): an additional LLM fact-check listing unsupported claims.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from backend.common.text_utils import content_terms, extract_numbers, split_sentences, stem
from backend.generation.prompt_builder import SourceRef, format_sources, load_prompt

logger = logging.getLogger(__name__)

REFUSAL_TEXT = "I'm sorry, but the knowledge base does not contain sufficient information to answer that question."
_REFUSAL_PATTERNS = re.compile(
    r"(don't have confirmed information|do not have confirmed information|does not contain (sufficient|enough)|insufficient information|couldn't find sufficient verified information|could not find sufficient verified information|"
    r"(couldn't|could not|could n't) find sufficient information|not (available|found) in the (knowledge base|sources)"
    r"|i (do not|don't) have (enough )?information|cannot find (any )?information|no information (is )?available)",
    re.IGNORECASE)
_CITATION_RE = re.compile(r"\[(S\d+)\]")
BOILERPLATE = ("Based on the knowledge base:", "based on the product catalogue:", "Here is a comparison of",
               "Values come from the product catalogue", "(sample data)", "Not specified")


@dataclass
class GroundingReport:
    faithfulness: float
    supported: bool
    unsupported_sentences: list[str] = field(default_factory=list)
    unsupported_numbers: list[str] = field(default_factory=list)
    is_refusal: bool = False
    method: str = "lexical"

    def to_dict(self) -> dict:
        return {"faithfulness": round(self.faithfulness, 3), "supported": self.supported,
                "unsupported_sentences": self.unsupported_sentences[:5], "unsupported_numbers": self.unsupported_numbers,
                "is_refusal": self.is_refusal, "method": self.method}


def is_refusal(text: str) -> bool:
    return bool(_REFUSAL_PATTERNS.search(text or ""))


def strip_citations(text: str) -> str:
    return _CITATION_RE.sub("", text)


def cited_ids(text: str) -> list[str]:
    seen: list[str] = []
    for sid in _CITATION_RE.findall(text or ""):
        if sid not in seen:
            seen.append(sid)
    return seen


_ALT_CITATION_RE = re.compile(r"【\s*(S\d+)[^】]*】|\(\s*(S\d+)\s*\)")
_MULTI_CITATION_RE = re.compile(r"\[\s*(S\d+(?:\s*[,;]\s*S\d+)+)\s*\]")


def normalize_citations(text: str) -> str:
    """Different models write citations differently (【S2】, 【S2†L3-L5】, (S2), [S1, S3]); convert all to [S2]."""
    text = _ALT_CITATION_RE.sub(lambda m: f"[{m.group(1) or m.group(2)}]", text)
    return _MULTI_CITATION_RE.sub(lambda m: "".join(f"[{s.strip()}]" for s in re.split(r"[,;]", m.group(1))), text)


def remove_invalid_citations(text: str, valid: set[str]) -> str:
    return _CITATION_RE.sub(lambda m: m.group(0) if m.group(1) in valid else "", text)


COUNT_QUESTION_RE = re.compile(r"\b(how many|how much of|number of|total (number|count) of|count of)\b", re.IGNORECASE)
_COUNT_CLAIM_RE = re.compile(r"(?<![\w.,])(\d+(?:,\d{3})*)\s*\+?\s*\**\s*([A-Za-z]{3,})")


def uncounted_numbers(question: str, answer: str, refs: list[SourceRef]) -> list[str]:
    """For a "how many ..." question: counts in the answer ("**12** universities") that the sources do not state
    in the same way ("12 universities"). The model must copy a stated total, never count list items itself; the
    plain number check is not enough because the same digits often occur elsewhere (a date, a page number)."""
    if not COUNT_QUESTION_RE.search(question or ""):
        return []
    text = _CITATION_RE.sub(" ", answer or "")
    text = re.sub(r"(?im)^\s*(?:\d+[.)]|step\s*\d+\s*:?)\s+", " ", text)      # list / step numbering
    corpus = re.sub(r"\s+", " ", _source_corpus(refs).replace("*", "")).lower()
    bad = []
    for num, word in _COUNT_CLAIM_RE.findall(text.replace("*", "")):
        stem_ = word.lower()[:5]
        if not re.search(rf"(?<![\d,.]){re.escape(num.lower())}\s*\+?\s*{re.escape(stem_)}", corpus):
            bad.append(f"{num} {word}")
    return bad


def _source_corpus(refs: list[SourceRef]) -> str:
    parts = []
    for r in refs:
        parts.append(r.text)
        parts.append(str(r.metadata.get("answer", "")))
        parts.append(str(r.metadata.get("doc_name", "")))
    return "\n".join(parts)


def lexical_grounding(answer: str, refs: list[SourceRef], question: str = "", min_support: float = 0.6) -> GroundingReport:
    if is_refusal(answer):
        return GroundingReport(faithfulness=1.0, supported=True, is_refusal=True)
    corpus = _source_corpus(refs)
    corpus_terms = content_terms(corpus) | content_terms(question)
    corpus_numbers = extract_numbers(corpus) | extract_numbers(question)
    clean = strip_citations(answer)
    for boiler in BOILERPLATE:
        clean = clean.replace(boiler, " ")
    sentences = [s for s in split_sentences(clean) if len(content_terms(s)) >= 3]
    # table rows count as sentences too
    sentences += [ln for ln in clean.split("\n") if ln.strip().startswith("|") and "---" not in ln and ln not in sentences]
    # other word forms of a source word count ("recalling" / "recall", "verifiable" / "verification"), so a correct
    # paraphrase is not thrown away for the raw source text; the per-sentence 0.5 bar itself is unchanged
    corpus_stems = {stem(w) for w in corpus_terms}
    corpus_prefixes = {w[:6] for w in corpus_terms if len(w) >= 7}

    def found(word: str) -> bool:
        return word in corpus_terms or stem(word) in corpus_stems or (len(word) >= 7 and word[:6] in corpus_prefixes)

    unsupported = []
    for sent in sentences:
        terms = content_terms(sent)
        if terms and sum(found(w) for w in terms) / len(terms) < 0.5:
            unsupported.append(sent.strip())
    numbers = {n for n in extract_numbers(clean) if not re.fullmatch(r"[1-9]", n)}  # ignore list numbering 1-9
    bad_numbers = sorted(n for n in numbers if n not in corpus_numbers)
    checked = max(len(sentences), 1)
    faithfulness = 1.0 - len(unsupported) / checked if sentences else 1.0
    supported = faithfulness >= min_support and not bad_numbers
    return GroundingReport(faithfulness=faithfulness, supported=supported, unsupported_sentences=unsupported,
                           unsupported_numbers=bad_numbers)


def trim_unsupported(answer: str, report: GroundingReport) -> str:
    """The answer without the sentences the grounding check could not match to the sources ("" when nothing useful
    is left). One unsupported sentence - typically advice the model added - no longer discards a correct answer."""
    drop = {re.sub(r"\s+", " ", strip_citations(s)).strip(" .") for s in report.unsupported_sentences}
    if not drop:
        return answer
    kept_lines = []
    for line in answer.split("\n"):
        parts = split_sentences(line) if line.strip() and not line.strip().startswith("|") else [line]
        kept = [p for p in parts if re.sub(r"\s+", " ", strip_citations(p)).strip(" .") not in drop]
        if kept or not line.strip():
            kept_lines.append(" ".join(kept) if parts != [line] else line)
    out = re.sub(r"\n{3,}", "\n\n", "\n".join(kept_lines)).strip()
    return out if len(content_terms(strip_citations(out))) >= 4 else ""


def llm_grounding(llm, answer: str, refs: list[SourceRef]) -> GroundingReport:
    prompt = load_prompt("grounding_prompt.txt").format(sources=format_sources(refs), answer=answer)
    data, _ = llm.complete_json("You are a meticulous fact-checker. Output JSON only.", prompt)
    claims = [str(c) for c in data.get("unsupported_claims", [])]
    faith = float(data.get("faithfulness", 1.0 if not claims else 0.5))
    return GroundingReport(faithfulness=faith, supported=bool(data.get("supported", not claims)),
                           unsupported_sentences=claims, method="llm")


_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+)\)")
_BARE_URL_RE = re.compile(r"(?<![(\w/])https?://[^\s)\]>\"']+")


def remove_unknown_urls(answer: str, allowed: set[str]) -> str:
    """Drop links the model wrote that are not the URL of a cited source (fabricated-URL protection).
    Markdown links keep their text; bare unknown URLs are removed."""
    allowed = {u.rstrip("/") for u in allowed if u}

    def ok(url: str) -> bool:
        return url.rstrip("/.,;") in allowed

    answer = _MD_LINK_RE.sub(lambda m: m.group(0) if ok(m.group(2)) else m.group(1), answer)
    return _BARE_URL_RE.sub(lambda m: m.group(0) if ok(m.group(0)) else "", answer)
