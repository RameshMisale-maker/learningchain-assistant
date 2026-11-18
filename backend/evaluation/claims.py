"""Claim-level faithfulness / hallucination evaluation.

An answer is split into atomic claims (sentences, bullet items, table rows). Each claim is judged against the
retrieved context by a "claim judge":

  lexical  - the production grounding heuristic (content-word overlap + every number must appear in the sources)
  nli      - a local natural-language-inference cross-encoder (entailment of claim by any context passage)
  claude   - Claude as an LLM judge (only when an Anthropic key is configured)

Judges are *validated* against a labelled claim set (datasets/evaluation_dataset/claims_gold.csv) with a confusion
matrix before their numbers are trusted. Positive class = "supported":
  TP supported claim judged supported      FN supported claim judged unsupported
  TN unsupported claim judged unsupported  FP unsupported claim judged supported (missed hallucination)

Hallucination rate (system answers) = unsupported claims / total claims x 100, using the validated judge.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache

from backend.common.text_utils import content_terms, extract_numbers, split_sentences
from backend.generation.grounding import BOILERPLATE, is_refusal, strip_citations

logger = logging.getLogger(__name__)

URL_RE = re.compile(r"https?://[^\s)\]>\"']+")
_META_LINE = re.compile(r"^\s*(Full guide:|Video walkthrough:|Sources?\b|---|\*\*Sources\*\*)", re.IGNORECASE)


def split_claims(answer: str) -> list[str]:
    """Atomic claims: sentences / list items / table rows with >= 3 content words. Headings, source lists and
    link-only lines are not claims (links are checked separately)."""
    if not answer or is_refusal(answer):
        return []
    from backend.generation.grounding import normalize_citations
    text = strip_citations(normalize_citations(answer)).replace("**", "").replace("__", "")
    for boiler in BOILERPLATE:
        text = text.replace(boiler, " ")
    claims: list[str] = []
    for line in text.split("\n"):
        line = line.strip()
        if not line or _META_LINE.match(line) or line.startswith("#") or set(line) <= set("|-: "):
            continue
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)]|Step \d+:)\s*", "", line).strip("* ")
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|") if c.strip()]
            line = ": ".join(cells)
        for sent in split_sentences(line) or [line]:
            sent = URL_RE.sub("", sent).strip(" .;:")
            if len(content_terms(sent)) >= 3:
                claims.append(sent)
    return claims


@dataclass
class ClaimVerdict:
    claim: str
    supported: bool
    score: float
    judge: str
    evidence: str = ""


class LexicalJudge:
    name = "lexical"

    def __init__(self, min_overlap: float = 0.5) -> None:
        self.min_overlap = min_overlap

    def judge(self, claim: str, passages: list[str]) -> ClaimVerdict:
        corpus = "\n".join(passages)
        terms, corpus_terms = content_terms(claim), content_terms(corpus)
        overlap = len(terms & corpus_terms) / len(terms) if terms else 1.0
        numbers = {n for n in extract_numbers(claim) if not re.fullmatch(r"[1-9]", n)}
        bad_numbers = numbers - extract_numbers(corpus)
        ok = overlap >= self.min_overlap and not bad_numbers
        return ClaimVerdict(claim, ok, round(overlap, 3), self.name)


class NLIJudge:
    """Entailment by any single passage (passages are split to fit the model's input). Labels are read from the model
    config, so the judge works with any NLI cross-encoder (default: cross-encoder/nli-deberta-v3-small)."""
    name = "nli"

    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-small", threshold: float = 0.5) -> None:
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)
        self.model_name, self.threshold = model_name, threshold
        labels = {int(k): v.lower() for k, v in self.model.model.config.id2label.items()}
        self.entail_idx = next(i for i, v in labels.items() if "entail" in v)
        self.name = f"nli:{model_name.split('/')[-1]}"

    @staticmethod
    def _windows(passages: list[str], max_chars: int = 1200) -> list[str]:
        out = []
        for p in passages:
            sents = split_sentences(p) or [p]
            buf = ""
            for s in sents:
                if len(buf) + len(s) > max_chars and buf:
                    out.append(buf)
                    buf = buf[-300:]      # small overlap so a fact split across windows is still seen
                buf = f"{buf} {s}".strip()
            if buf:
                out.append(buf)
        return out or [""]

    def judge(self, claim: str, passages: list[str]) -> ClaimVerdict:
        import numpy as np

        windows = self._windows(passages)
        logits = np.asarray(self.model.predict([(w, claim) for w in windows], show_progress_bar=False))
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = exp / exp.sum(axis=1, keepdims=True)
        best = int(np.argmax(probs[:, self.entail_idx]))
        score = float(probs[best, self.entail_idx])
        return ClaimVerdict(claim, score >= self.threshold, round(score, 3), self.name, windows[best][:300])


class ClaudeJudge:
    """LLM judge via the configured Anthropic client. Not used as ground truth - validated like the others."""
    name = "claude"

    def __init__(self) -> None:
        from backend.generation.llm_client import get_llm

        self.llm = get_llm()
        if self.llm is None or self.llm.provider != "anthropic":
            raise RuntimeError("Claude judge needs LLM_PROVIDER=anthropic and ANTHROPIC_API_KEY")
        self.name = f"claude:{self.llm.model}"

    def judge(self, claim: str, passages: list[str]) -> ClaimVerdict:
        context = "\n\n".join(f"<passage>{p[:3000]}</passage>" for p in passages[:8])
        data, _ = self.llm.complete_json(
            "You verify claims against evidence. Output JSON only: {\"supported\": true|false, \"reason\": \"...\"}. "
            "A claim is supported only if the passages state it; outside knowledge does not count.",
            f"PASSAGES:\n{context}\n\nCLAIM: {claim}", max_tokens=200)
        return ClaimVerdict(claim, bool(data.get("supported")), 1.0 if data.get("supported") else 0.0, self.name,
                            str(data.get("reason", ""))[:300])


@lru_cache(maxsize=4)
def get_judge(kind: str):
    if kind == "lexical":
        return LexicalJudge()
    if kind == "claude":
        return ClaudeJudge()
    if kind.startswith("nli"):
        _, _, model = kind.partition(":")
        return NLIJudge(model or "cross-encoder/nli-deberta-v3-small")
    raise ValueError(kind)


@dataclass
class AnswerClaimReport:
    claims: list[ClaimVerdict] = field(default_factory=list)
    fabricated_urls: list[str] = field(default_factory=list)
    invalid_citations: list[str] = field(default_factory=list)

    @property
    def unsupported(self) -> list[ClaimVerdict]:
        return [c for c in self.claims if not c.supported]


def _norm(text: str) -> str:
    text = text.translate(str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'", "•": "-"}))
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def evaluate_answer(answer: str, passages: list[str], judge, allowed_urls: set[str] | None = None,
                    valid_citation_ids: set[str] | None = None) -> AnswerClaimReport:
    """Verbatim evidence first: a claim whose normalised text occurs in a retrieved passage is supported by
    definition (quoted source text); only the remaining claims go to the judge. URLs that appear in the retrieved
    passages (e.g. video links inside an article) count as sourced, not fabricated."""
    corpus = " ".join(_norm(p) for p in passages)
    claims = []
    for c in split_claims(answer):
        nc = _norm(c)
        if nc and nc in corpus:
            claims.append(ClaimVerdict(c, True, 1.0, "verbatim"))
        else:
            claims.append(judge.judge(c, passages))
    report = AnswerClaimReport(claims=claims)
    if allowed_urls is not None:
        allowed = {u.rstrip("/") for u in allowed_urls if u}
        allowed |= {u.rstrip("/.,;") for p in passages for u in URL_RE.findall(p)}
        report.fabricated_urls = [u for u in URL_RE.findall(answer) if u.rstrip("/.,;") not in allowed]
    if valid_citation_ids is not None:
        report.invalid_citations = sorted({c for c in re.findall(r"\[(S\d+)\]", answer) if c not in valid_citation_ids})
    return report
