"""Claim-level hallucination evaluation of the chatbot's actual answers against the contexts it actually used.

Hallucination Rate = (Unsupported factual claims / Total factual claims) x 100
"Unsupported" = any claim the evidence does not support; CONTRADICTED claims are a subset and are shown separately.

Claims are extracted with the chatbot's existing claim splitter (backend.evaluation.claims.split_claims, imported
read-only). A claim that appears verbatim in a retrieved passage is supported by definition; every other claim goes
to a judge. Judges can be wrong, so every unsupported / contradicted claim is flagged for human verification in the
dashboard; human verdicts are stored separately (evaluation/annotations/) and never overwrite the judge's output.
"""
from __future__ import annotations

import hashlib
import json
import re
from functools import lru_cache
from typing import Any

from evaluation import config

SUPPORTED, UNSUPPORTED, CONTRADICTED = "SUPPORTED", "UNSUPPORTED", "CONTRADICTED"
NOT_A_CLAIM = "NOT_A_CLAIM"
_LEAD_IN_START = re.compile(r"^(to|for|as an?|here (is|are)|follow these|the (following )?steps?)\b", re.IGNORECASE)
_LEAD_IN_END = re.compile(r"\b(should|must|can|will|need to|needs to|has to|have to)$", re.IGNORECASE)
_NEGATION = re.compile(r"\b(not|no|never|cannot|can't|isn't|aren't|doesn't|don't|won't|without|unless|only)\b|n't\b",
                       re.IGNORECASE)


def is_lead_in(claim: str) -> bool:
    """A lead-in line, not a factual claim: "To reissue a single certificate", "To register new users, an admin
    should", "Follow these steps:". Judges mark these unsupported or contradicted although they state nothing."""
    text = (claim or "").strip().rstrip(".").strip()
    words = re.findall(r"[A-Za-z0-9]+", text)
    if not words or text.endswith(":"):
        return True
    if len(words) > 8:
        return False
    if _LEAD_IN_END.search(text):
        return True
    return bool(_LEAD_IN_START.match(text)) and "," not in text      # "To verify, scan the QR code" is a claim


def overlap_supported(claim: str, passages: list[str], threshold: float | None = None) -> bool:
    """The claim's content words (stemmed, stopwords removed) are at least `threshold` present in the passages:
    a close paraphrase of the source ("Enter your Current Password, then the New Password and re-enter it").
    Never for a claim with a negation (meaning can flip) or with a number the passages do not contain."""
    from backend.common.text_utils import content_terms

    threshold = config.EVAL_CLAIM_OVERLAP_SUPPORT if threshold is None else threshold
    if threshold <= 0 or _NEGATION.search(claim or ""):
        return False
    corpus = " ".join(passages)
    if any(n not in corpus for n in re.findall(r"\d+(?:[.,]\d+)*", claim or "")):
        return False
    terms = content_terms(claim)
    if len(terms) < 3:
        return False
    return len(terms & content_terms(corpus)) / len(terms) >= threshold


def claim_key(claim: str) -> str:
    return hashlib.sha1(claim.strip().lower().encode("utf-8")).hexdigest()[:12]


def _norm(text: str) -> str:
    text = (text or "").translate(str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'", "•": "-"}))
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


class LLMClaimJudge:
    """Separate evaluator model (OpenAI-compatible API) with a three-way verdict."""

    SYSTEM = ("You check one factual claim against evidence passages from a knowledge base. Use ONLY the passages; "
              "outside knowledge does not count. Answer with JSON only: "
              '{"verdict": "SUPPORTED" | "UNSUPPORTED" | "CONTRADICTED", "evidence": "<short quote or empty>"}. '
              "SUPPORTED: the passages state the claim (paraphrase is fine). CONTRADICTED: the passages state "
              "something incompatible with the claim. UNSUPPORTED: the passages do not establish the claim.")

    def __init__(self) -> None:
        from openai import OpenAI

        key, base = config.evaluator_credentials()
        if not key:
            raise RuntimeError("No evaluator API key (EVAL_LLM_API_KEY / OPENAI_API_KEY)")
        self.client = OpenAI(api_key=key, base_url=base or None, timeout=config.EVAL_LLM_TIMEOUT, max_retries=4)
        self.model = config.EVAL_LLM_MODEL
        self.name = f"llm:{self.model}"

    def judge(self, claim: str, passages: list[str]) -> tuple[str, str]:
        context = "\n\n".join(f"<passage id={i + 1}>\n{p[:3500]}\n</passage>" for i, p in enumerate(passages[:8]))
        resp = self.client.chat.completions.create(
            model=self.model, temperature=0, max_tokens=300, response_format={"type": "json_object"},
            extra_body=config.evaluator_extra_body(),
            messages=[{"role": "system", "content": self.SYSTEM},
                      {"role": "user", "content": f"PASSAGES:\n{context}\n\nCLAIM: {claim}"}])
        raw = resp.choices[0].message.content or "{}"
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", raw, re.S)
            data = json.loads(m.group(0)) if m else {}
        verdict = str(data.get("verdict", UNSUPPORTED)).upper()
        if verdict not in (SUPPORTED, UNSUPPORTED, CONTRADICTED):
            verdict = UNSUPPORTED
        return verdict, str(data.get("evidence", ""))[:300]


class NLIClaimJudge:
    """Local NLI cross-encoder (no API): entailment -> SUPPORTED, contradiction -> CONTRADICTED, else UNSUPPORTED."""

    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-small", threshold: float = 0.5) -> None:
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)
        labels = {int(k): v.lower() for k, v in self.model.model.config.id2label.items()}
        self.entail = next(i for i, v in labels.items() if "entail" in v)
        self.contra = next(i for i, v in labels.items() if "contra" in v)
        self.threshold = threshold
        self.name = f"nli:{model_name.split('/')[-1]}"

    def judge(self, claim: str, passages: list[str]) -> tuple[str, str]:
        import numpy as np

        from backend.evaluation.claims import NLIJudge

        windows = NLIJudge._windows(passages)
        logits = np.asarray(self.model.predict([(w, claim) for w in windows], show_progress_bar=False))
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = exp / exp.sum(axis=1, keepdims=True)
        best_e = int(np.argmax(probs[:, self.entail]))
        if probs[best_e, self.entail] >= self.threshold:
            return SUPPORTED, windows[best_e][:300]
        best_c = int(np.argmax(probs[:, self.contra]))
        if probs[best_c, self.contra] >= self.threshold:
            return CONTRADICTED, windows[best_c][:300]
        return UNSUPPORTED, ""


@lru_cache(maxsize=2)
def get_judge(kind: str = config.HALLUCINATION_JUDGE):
    return NLIClaimJudge(config.EVAL_NLI_MODEL) if kind == "nli" else LLMClaimJudge()


def analyse(answer: str, contexts: list[str], behavior: str, judge=None) -> dict[str, Any]:
    """Claim analysis for one answer. Abstentions / rejections contain no factual claims."""
    from backend.evaluation.claims import split_claims

    result: dict[str, Any] = {"claims": [], "total_claims": 0, "supported": 0, "unsupported": 0, "contradicted": 0,
                              "hallucination_rate": None, "judge": "", "error": ""}
    if behavior != "Answer":
        return result
    claims = split_claims(answer)
    if not claims:
        result["hallucination_rate"] = 0.0
        return result
    corpus = " ".join(_norm(c) for c in contexts)
    judge = judge or get_judge()
    result["judge"] = getattr(judge, "name", "")
    for claim in claims:
        if not contexts:
            verdict, evidence, by = UNSUPPORTED, "", "no-context"
        elif is_lead_in(claim):
            verdict, evidence, by = NOT_A_CLAIM, "lead-in line, states no fact", "lead-in"
        elif _norm(claim) and _norm(claim) in corpus:
            verdict, evidence, by = SUPPORTED, "verbatim match in retrieved context", "verbatim"
        elif overlap_supported(claim, contexts):
            verdict, evidence, by = SUPPORTED, "close paraphrase: key words found in retrieved context", "word-overlap"
        else:
            try:
                verdict, evidence = judge.judge(claim, contexts)
                by = result["judge"]
            except Exception as exc:   # judge failure is reported, never counted as a hallucination
                result["error"] = f"{type(exc).__name__}: {exc}"[:300]
                verdict, evidence, by = "JUDGE_ERROR", "", result["judge"]
        result["claims"].append({"claim": claim, "claim_key": claim_key(claim), "verdict": verdict,
                                 "evidence": evidence, "judged_by": by,
                                 "needs_human_review": verdict in (UNSUPPORTED, CONTRADICTED)})
    judged = [c for c in result["claims"] if c["verdict"] not in ("JUDGE_ERROR", NOT_A_CLAIM)]
    result["total_claims"] = len(judged)
    result["supported"] = sum(c["verdict"] == SUPPORTED for c in judged)
    result["contradicted"] = sum(c["verdict"] == CONTRADICTED for c in judged)
    result["unsupported"] = sum(c["verdict"] in (UNSUPPORTED, CONTRADICTED) for c in judged)
    result["hallucination_rate"] = (result["unsupported"] / result["total_claims"]) if judged else None
    return result


def apply_human_reviews(claims: list[dict], reviews: dict[str, str]) -> list[dict]:
    """Return claims with `final_verdict` = human verdict when present, else the judge's verdict."""
    out = []
    for c in claims:
        human = reviews.get(c["claim_key"])
        out.append(c | {"human_verdict": human or "", "final_verdict": human or c["verdict"]})
    return out


def aggregate(per_question: list[dict], reviews_by_q: dict[str, dict[str, str]] | None = None) -> dict[str, Any]:
    reviews_by_q = reviews_by_q or {}
    total = supported = unsupported = contradicted = flagged = 0
    failed: list[str] = []
    for q in per_question:
        claims = apply_human_reviews(q.get("claims", []), reviews_by_q.get(q["question_id"], {}))
        judged = [c for c in claims if c["final_verdict"] != "JUDGE_ERROR"]
        total += len(judged)
        supported += sum(c["final_verdict"] == SUPPORTED for c in judged)
        contradicted += sum(c["final_verdict"] == CONTRADICTED for c in judged)
        bad = sum(c["final_verdict"] in (UNSUPPORTED, CONTRADICTED) for c in judged)
        unsupported += bad
        flagged += sum(1 for c in claims if c["needs_human_review"] and not c["human_verdict"])
        if bad:
            failed.append(q["question_id"])
    return {"total_claims": total, "supported_claims": supported, "unsupported_claims": unsupported,
            "contradicted_claims": contradicted, "hallucination_rate": (unsupported / total) if total else None,
            "awaiting_human_review": flagged, "failed_questions": failed}
