"""Evaluation metrics: retrieval (Recall@K, Precision@K, MRR, NDCG@K), answer quality
(faithfulness, answer relevancy, context precision/recall, correctness) and helpers."""
from __future__ import annotations

import math
import re

import numpy as np

from backend.common.text_utils import content_terms, normalize_text, tokenize

REFUSAL_TYPES = {"refusal", "out_of_scope", "blocked", "clarification", "access_denied", "invalid"}


REFUSE_LABEL = "refuse"


def confusion_matrix(actual: list[str], predicted: list[str], labels: list[str] | None = None) -> dict:
    """Confusion matrix as {"labels": [...], "matrix": [[count]]} with rows = actual, columns = predicted."""
    labels = labels or sorted(set(actual) | set(predicted))
    for lab in sorted(set(actual) | set(predicted)):
        if lab not in labels:
            labels = labels + [lab]
    pos = {lab: i for i, lab in enumerate(labels)}
    matrix = [[0] * len(labels) for _ in labels]
    for a, p in zip(actual, predicted):
        matrix[pos[a]][pos[p]] += 1
    return {"labels": labels, "matrix": matrix}


def classification_report(cm: dict) -> dict:
    """Per-class precision / recall / F1 / support plus accuracy and macro / weighted F1 from a confusion matrix."""
    labels, m = cm["labels"], cm["matrix"]
    total = sum(sum(r) for r in m)
    per_class, f1s, weights = {}, [], []
    for i, lab in enumerate(labels):
        tp = m[i][i]
        support = sum(m[i])
        predicted = sum(row[i] for row in m)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[lab] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
        if support:
            f1s.append(f1)
            weights.append(support)
    accuracy = sum(m[i][i] for i in range(len(labels))) / total if total else 0.0
    return {"per_class": per_class, "accuracy": accuracy,
            "macro_f1": sum(f1s) / len(f1s) if f1s else 0.0,
            "weighted_f1": sum(f * w for f, w in zip(f1s, weights)) / sum(weights) if weights else 0.0}


def binary_counts(should_answer: list[bool], answered: list[bool]) -> dict:
    """Answer-vs-refuse decision. Positive = 'answered'. TP: answered when it should; TN: refused when it should;
    FP: answered when it should have refused; FN: refused / clarified a question it should have answered."""
    tp = sum(1 for s, a in zip(should_answer, answered) if s and a)
    tn = sum(1 for s, a in zip(should_answer, answered) if not s and not a)
    fp = sum(1 for s, a in zip(should_answer, answered) if not s and a)
    fn = sum(1 for s, a in zip(should_answer, answered) if s and not a)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "accuracy": (tp + tn) / max(tp + tn + fp + fn, 1),
            "specificity": tn / (tn + fp) if tn + fp else 0.0}


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", normalize_text(text).lower().replace(",", "")).strip()


def is_relevant(item: dict, matchers: list[str]) -> bool:
    """item: retrieved chunk info (doc_name, source_file, section, qid, product_name, ...)."""
    for m in matchers:
        m = m.strip()
        if not m:
            continue
        if "=" in m:
            field, value = m.split("=", 1)
            if str(item.get(field.strip(), "")).lower() == value.strip().lower():
                return True
            continue
        hay = " ".join(str(item.get(k, "")) for k in ("source_file", "doc_name", "section", "qid")).lower()
        if m.lower() in hay:
            return True
    return False


def matched_matchers(items: list[dict], matchers: list[str]) -> set[str]:
    return {m for m in matchers if m.strip() and any(is_relevant(i, [m]) for i in items)}


def retrieval_metrics(items: list[dict], matchers: list[str], k: int) -> dict[str, float]:
    matchers = [m for m in matchers if m.strip()]
    if not matchers:
        return {}
    top = items[:k]
    flags = [is_relevant(i, matchers) for i in top]
    found = matched_matchers(top, matchers)
    # A question is answerable from any one of its alternative sources -> recall is 1 if any source group is found.
    recall = 1.0 if found else 0.0
    precision = sum(flags) / max(len(top), 1) if top else 0.0
    rr = next((1.0 / (i + 1) for i, f in enumerate(flags) if f), 0.0)
    dcg = sum(1.0 / math.log2(i + 2) for i, f in enumerate(flags) if f)
    ideal_n = max(1, min(k, sum(flags) or 1))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_n))
    hits_ctx = 0
    ap = 0.0
    for i, f in enumerate(flags):
        if f:
            hits_ctx += 1
            ap += hits_ctx / (i + 1)
    context_precision = ap / hits_ctx if hits_ctx else 0.0
    return {"recall_at_k": recall, "precision_at_k": precision, "mrr": rr, "ndcg_at_k": dcg / idcg if idcg else 0.0,
            "context_precision": context_precision}


def fact_found(fact: str, text: str) -> bool:
    fact_n, text_n = _norm(fact), _norm(text)
    if not fact_n:
        return True
    if fact_n in text_n:
        return True
    terms = content_terms(fact)
    if not terms:
        return False
    return len(terms & content_terms(text)) / len(terms) >= 0.8


def fact_coverage(facts: list[str], text: str) -> float:
    facts = [f for f in facts if f.strip()]
    if not facts:
        return float("nan")
    return sum(fact_found(f, text) for f in facts) / len(facts)


def token_f1(prediction: str, reference: str) -> float:
    p, r = tokenize(prediction), tokenize(reference)
    if not p or not r:
        return 0.0
    common = {}
    for t in p:
        if t in r:
            common[t] = min(p.count(t), r.count(t))
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(p), overlap / len(r)
    return 2 * precision * recall / (precision + recall)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
    return float(np.dot(a, b) / denom)


def safe_mean(values) -> float:
    vals = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return float(np.mean(vals)) if vals else float("nan")


def percentile(values, q: float) -> float:
    vals = [v for v in values if v is not None]
    return float(np.percentile(vals, q)) if vals else float("nan")
