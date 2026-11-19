"""Evaluation reports. Each run gets its own folder evaluation/results/<run_id>/ that is never overwritten:

  run_meta.json            configuration, models, dataset hash, git commit, targets
  summary.json / .csv      aggregate metrics vs targets
  question_report.csv      one row per golden question (flat, for spreadsheets)
  failed_questions.csv     the failing subset with reasons
  questions.json           full traces: retrieved child chunks, parent contexts, answer, all metrics, claims
  hallucination_claims.csv every extracted claim with its verdict
  safety_report.csv        safety test results
  performance.json         latency / tokens / cost / errors
"""
from __future__ import annotations

import json
import secrets
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from evaluation import config

FLAT_COLUMNS = [
    "question_id", "question_category", "expected_behavior", "actual_behavior", "answer_type", "status",
    "failure_reason", "question", "expected_answer", "answer", "relevant_document", "label_source",
    "precision_at_k", "recall_at_k", "mrr", "ndcg_at_k", "hit_rate_at_k", "context_precision", "context_recall",
    "faithfulness", "answer_relevancy", "answer_correctness", "ragas_context_precision", "ragas_context_recall",
    "lexical_f1", "total_claims", "unsupported_claims", "contradicted_claims", "hallucination_rate",
    "latency_ms", "retrieval_ms", "llm_total_ms", "ttft_ms", "prompt_tokens", "completion_tokens", "cost_usd",
    "model", "conversation_intent", "normalized_query", "retrieved_chunk_ids", "parent_ids", "context_documents", "ragas_error",
]


def new_run_id() -> str:
    return f"run_{datetime.now():%Y%m%d_%H%M%S}_{secrets.token_hex(2)}"


def create_run_dir(run_id: str) -> Path:
    config.ensure_dirs()
    path = config.RESULTS_DIR / run_id
    path.mkdir(parents=False, exist_ok=False)     # never overwrite an earlier run
    return path


def _json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def flatten(q: dict[str, Any]) -> dict[str, Any]:
    run = q.get("run", {})
    ret, rag, hal = q.get("retrieval", {}), q.get("ragas", {}), q.get("hallucination", {})
    t = run.get("timings") or {}
    return {
        "question_id": q["question_id"], "question_category": q["question_category"],
        "expected_behavior": q["expected_behavior"], "actual_behavior": run.get("behavior"),
        "answer_type": run.get("answer_type"), "status": q.get("status"), "failure_reason": q.get("failure_reason"),
        "question": q["question"], "expected_answer": q.get("expected_answer"), "answer": run.get("answer"),
        "relevant_document": q.get("relevant_document"), "label_source": ret.get("label_source"),
        **{k: ret.get(k) for k in ("precision_at_k", "recall_at_k", "mrr", "ndcg_at_k", "hit_rate_at_k",
                                   "context_precision", "context_recall")},
        "faithfulness": rag.get("faithfulness"), "answer_relevancy": rag.get("answer_relevancy"),
        "answer_correctness": rag.get("answer_correctness"), "ragas_context_precision": rag.get("context_precision"),
        "ragas_context_recall": rag.get("context_recall"), "lexical_f1": q.get("lexical_f1"),
        "total_claims": hal.get("total_claims"), "unsupported_claims": hal.get("unsupported"),
        "contradicted_claims": hal.get("contradicted"), "hallucination_rate": hal.get("hallucination_rate"),
        "latency_ms": run.get("latency_ms"), "retrieval_ms": t.get("retrieval_ms"), "llm_total_ms": t.get("llm_total_ms"),
        "ttft_ms": run.get("ttft_ms"), "prompt_tokens": run.get("prompt_tokens"),
        "completion_tokens": run.get("completion_tokens"), "cost_usd": run.get("cost_usd"), "model": run.get("model"),
        "conversation_intent": (run.get("route") or {}).get("conversation_intent", ""),
        "normalized_query": (run.get("route") or {}).get("normalized_query", ""),
        "retrieved_chunk_ids": "|".join(c["chunk_id"] for c in run.get("retrieved", [])[:config.RETRIEVAL_K]),
        "parent_ids": "|".join(dict.fromkeys(c["parent_id"] for c in run.get("contexts", []) if c.get("parent_id"))),
        "context_documents": "|".join(dict.fromkeys(f"{c['doc_name']} > {c['section']}" for c in run.get("contexts", []))),
        "ragas_error": rag.get("ragas_error", ""),
    }


def write_run(run_dir: Path, meta: dict, questions: list[dict], safety: list[dict], summary: dict,
              performance: dict) -> None:
    _json(run_dir / "run_meta.json", meta)
    _json(run_dir / "questions.json", questions)
    _json(run_dir / "summary.json", summary)
    _json(run_dir / "performance.json", performance)
    flat = pd.DataFrame([flatten(q) for q in questions], columns=FLAT_COLUMNS)
    flat.to_csv(run_dir / "question_report.csv", index=False, encoding="utf-8")
    flat[flat["status"] == "FAIL"].to_csv(run_dir / "failed_questions.csv", index=False, encoding="utf-8")
    claims = [{"question_id": q["question_id"], **c} for q in questions for c in q.get("hallucination", {}).get("claims", [])]
    pd.DataFrame(claims).to_csv(run_dir / "hallucination_claims.csv", index=False, encoding="utf-8")
    pd.DataFrame(safety).to_csv(run_dir / "safety_report.csv", index=False, encoding="utf-8")
    pd.DataFrame([{"metric": k, "value": v} for k, v in summary.get("metrics", {}).items()]).to_csv(
        run_dir / "summary.csv", index=False, encoding="utf-8")
    # the three-layer view (retrieval / generation / end-to-end production) - the main downloadable report
    if summary.get("layers"):
        layers = pd.DataFrame(summary["layers"])
        layers["sample_size"] = layers["sample_size"].astype("Int64")      # whole numbers, blank when unknown
        layers.to_csv(run_dir / "evaluation_metrics.csv", index=False, encoding="utf-8-sig")
    if performance.get("load_test"):
        pd.DataFrame(performance["load_test"]).to_csv(run_dir / "load_test.csv", index=False, encoding="utf-8-sig")


# ------------------------------------------------------------------------------------------------ reading
def list_runs() -> list[dict[str, Any]]:
    if not config.RESULTS_DIR.exists():
        return []
    runs = []
    for d in sorted(config.RESULTS_DIR.iterdir(), reverse=True):
        if d.is_dir() and (d / "summary.json").exists():
            meta = json.loads((d / "run_meta.json").read_text(encoding="utf-8")) if (d / "run_meta.json").exists() else {}
            runs.append({"run_id": d.name, "path": d, "started_at": meta.get("started_at", ""),
                         "questions": meta.get("question_count"), "label": meta.get("label", "")})
    return runs


def load_run(run_id: str) -> dict[str, Any]:
    d = config.RESULTS_DIR / run_id

    def j(name, default):
        p = d / name
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default

    safety_path = d / "safety_report.csv"
    safety = pd.read_csv(safety_path, dtype=str, keep_default_na=False).to_dict("records") \
        if safety_path.exists() and safety_path.stat().st_size > 2 else []
    return {"run_id": run_id, "path": d, "meta": j("run_meta.json", {}), "summary": j("summary.json", {}),
            "questions": j("questions.json", []), "performance": j("performance.json", {}), "safety": safety}


def compare_runs(run_ids: list[str]) -> pd.DataFrame:
    rows = []
    for rid in run_ids:
        r = load_run(rid)
        m = r["summary"].get("metrics", {})
        rows.append({"run_id": rid, "started_at": r["meta"].get("started_at", ""),
                     "questions": r["meta"].get("question_count"), "llm": r["meta"].get("chatbot", {}).get("llm_model"),
                     "embedding": r["meta"].get("chatbot", {}).get("embedding_model"), **m})
    return pd.DataFrame(rows)
