"""Retrieval metrics computed on the chatbot's ACTUAL re-ranked child chunks.

Relevance labels come from (highest priority first):
  1. human annotations made in the dashboard (keyed by chunk text, so they survive re-indexing),
  2. `relevant_chunk_ids` in the golden dataset (a child id, or a parent id that covers all its children),
  3. `relevant_document` in the golden dataset (document name / file / section / approved-answer id matchers).
When none exist the question is reported as "needs annotation" - no relevance is invented.
"""
from __future__ import annotations

import math
import re
from typing import Any

from backend.evaluation.metrics import fact_found, is_relevant

METRIC_KEYS = ["precision_at_k", "recall_at_k", "mrr", "ndcg_at_k", "hit_rate_at_k", "context_precision",
               "context_recall"]


def split_list(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    return [p.strip() for p in re.split(r"[|;\n]", str(value)) if p.strip()]


def alternatives(matcher: str) -> list[str]:
    """One relevant_document entry: "A OR B" = the source is A or B (either counts); plain "A" = just A.
    Entries separated by | are all required (recall = share of entries found)."""
    return [a.strip() for a in re.split(r"\s+OR\s+", matcher) if a.strip()]


def label_chunks(retrieved: list[dict], row: dict, human: dict[str, bool] | None = None) -> tuple[list[bool | None], str]:
    """Relevance flag per retrieved chunk + the label source used."""
    human = human or {}
    gt_ids = set(split_list(row.get("relevant_chunk_ids")))
    matchers = [a for m in split_list(row.get("relevant_document")) for a in alternatives(m)]
    flags: list[bool | None] = []
    sources = set()
    for ch in retrieved:
        if ch.get("text_key") in human:
            flags.append(bool(human[ch["text_key"]]))
            sources.add("human")
        elif gt_ids:
            flags.append(ch.get("chunk_id") in gt_ids or (bool(ch.get("parent_id")) and ch.get("parent_id") in gt_ids))
            sources.add("chunk_ids")
        elif matchers:
            flags.append(is_relevant({k: ch.get(k, "") for k in ("doc_name", "source_file", "section", "doc_id")},
                                     matchers))
            sources.add("document")
        else:
            flags.append(None)
    source = "+".join(sorted(sources)) if sources else "none"
    return flags, source


def context_recall_lexical(reference: str, contexts: list[str]) -> float | None:
    """Share of reference statements (reference_context, '|' or sentence separated) found in the contexts that were
    sent to the LLM. Lexical, deterministic; RAGAS context recall is the LLM-judged counterpart."""
    statements = split_list(reference) if "|" in (reference or "") else [
        s.strip() for s in re.split(r"(?<=[.!?])\s+", reference or "") if len(s.strip()) > 3]
    if not statements:
        return None
    text = "\n".join(contexts)
    return sum(1 for s in statements if fact_found(s, text)) / len(statements)


def compute(retrieved: list[dict], row: dict, k: int, contexts: list[str],
            human: dict[str, bool] | None = None) -> dict[str, Any]:
    flags, source = label_chunks(retrieved, row, human)
    reference = str(row.get("reference_context") or "") or str(row.get("expected_answer") or "")
    out: dict[str, Any] = {key: None for key in METRIC_KEYS}
    out["context_recall"] = context_recall_lexical(reference, contexts)
    out["label_source"] = source
    out["relevance_flags"] = flags
    if source == "none" or not retrieved:
        out["needs_annotation"] = bool(retrieved) and source == "none"
        return out
    out["needs_annotation"] = any(f is None for f in flags[:k])
    top = [bool(f) for f in flags[:k]]
    gt_ids = set(split_list(row.get("relevant_chunk_ids")))
    matchers = split_list(row.get("relevant_document"))
    top_items = retrieved[:k]
    if "human" in source or not (gt_ids or matchers):
        # relative recall: relevant chunks in top-k / all relevant chunks among everything the retriever returned
        total_rel = sum(1 for f in flags if f)
        recall = (sum(top) / total_rel) if total_rel else 0.0
    elif gt_ids:
        found = {g for g in gt_ids if any(c.get("chunk_id") == g or c.get("parent_id") == g for c in top_items)}
        recall = len(found) / len(gt_ids)
        total_rel = len(gt_ids)
    else:
        found = {m for m in matchers if any(is_relevant({x: c.get(x, "") for x in ("doc_name", "source_file",
                                                                                  "section", "doc_id")}, alternatives(m))
                                            for c in top_items)}
        recall = len(found) / len(matchers)
        total_rel = len(matchers)
    out["precision_at_k"] = sum(top) / len(top) if top else 0.0
    out["recall_at_k"] = recall
    out["hit_rate_at_k"] = 1.0 if any(top) else 0.0
    out["mrr"] = next((1.0 / (i + 1) for i, f in enumerate(top) if f), 0.0)
    dcg = sum(1.0 / math.log2(i + 2) for i, f in enumerate(top) if f)
    # ideal ranking: every known relevant item first. A relevant document can be split into several relevant child
    # chunks, so count relevant chunks in the full candidate list as well as the ground-truth items (NDCG <= 1).
    n_ideal = max(total_rel or 0, sum(1 for f in flags if f), 1)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(k, n_ideal)))
    out["ndcg_at_k"] = dcg / ideal if ideal else 0.0
    # rank-aware context precision: mean precision@i over the ranks that hold a relevant chunk
    hits, ap = 0, 0.0
    for i, f in enumerate(top):
        if f:
            hits += 1
            ap += hits / (i + 1)
    out["context_precision"] = ap / hits if hits else 0.0
    return out


def aggregate(rows: list[dict]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in METRIC_KEYS:
        vals = [r[key] for r in rows if r.get(key) is not None]
        out[key] = sum(vals) / len(vals) if vals else None
        out[f"{key}_n"] = len(vals)
    out["needs_annotation"] = sum(1 for r in rows if r.get("needs_annotation"))
    return out
