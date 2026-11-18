"""Run the evaluation dataset through the full chatbot pipeline and write per-KB reports.

  python -m backend.evaluation.run_evaluation
  python -m backend.evaluation.run_evaluation --category hr_policies --limit 10
  python -m backend.evaluation.run_evaluation --ragas      # optional RAGAS scoring (needs `pip install ragas` + LLM key)

Outputs (reports/):  evaluation_report.md, evaluation_results.csv, evaluation_summary.json
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from backend.common.logging_config import setup_logging
from backend.evaluation.metrics import (REFUSAL_TYPES, REFUSE_LABEL, binary_counts, classification_report,
                                        confusion_matrix, cosine, fact_coverage, percentile, retrieval_metrics,
                                        safe_mean, token_f1)
from backend.security.auth import AuthenticatedUser
from backend.services.chat_service import ChatService
from config.settings import get_settings
from ingestion.document_loader import read_tabular

logger = logging.getLogger(__name__)
ROLE_SETS = {"public": {"public"}, "employee": {"employee"}, "hr": {"employee", "hr"}, "admin": {"admin", "employee", "hr"}}


def evaluate_row(service: ChatService, row: dict, k: int) -> dict:
    roles = ROLE_SETS.get(str(row.get("role", "employee")).strip(), {"employee"})
    user = AuthenticatedUser(username=f"eval_{row.get('role', 'employee')}", roles=roles)
    resp = service.answer(str(row["question"]), user, persist=False)
    expected_kbs = [k_.strip() for k_ in str(row.get("expected_kb", "")).split("|") if k_.strip()]
    expect_flag = str(row.get("expect_refusal", "false")).lower()
    expect_either = expect_flag == "either"          # ambiguous: clarifying question or a reasonable answer
    expect_refusal = expect_flag == "true"
    facts = [f for f in str(row.get("key_facts", "") or "").split("|") if f.strip()]
    matchers = [m for m in str(row.get("relevant_sources", "") or "").split("|") if m.strip()]
    refused = resp.answer_type in REFUSAL_TYPES

    routed = set(resp.route.get("knowledge_bases") or resp.knowledge_bases or [])
    if expect_refusal:
        routing_ok = (not routed) or bool(routed & set(expected_kbs)) or refused
    else:
        routing_ok = bool(routed & set(expected_kbs))

    # Single-label "destination" for the confusion matrix: the KB that answered, or 'refuse'.
    answered_from = list(resp.knowledge_bases or []) or sorted(routed)
    predicted_dest = REFUSE_LABEL if refused or not answered_from else answered_from[0]
    valid = [k_ for k_ in expected_kbs if k_ != "none"]
    if expect_either:
        expected_dest = predicted_dest
    elif expect_refusal or not valid:
        expected_dest = REFUSE_LABEL
    else:  # several acceptable KBs: credit the prediction when it is one of them
        expected_dest = predicted_dest if predicted_dest in valid else valid[0]

    out = {
        "eval_id": row.get("eval_id"), "category": row.get("category"), "question": row["question"],
        "expected_kb": "|".join(expected_kbs), "routed_kbs": "|".join(sorted(routed)), "intent": resp.route.get("intent"),
        "answer_type": resp.answer_type, "routing_correct": routing_ok, "answer": resp.answer,
        "expected_destination": expected_dest, "predicted_destination": predicted_dest,
        "should_answer": (not refused) if expect_either else not expect_refusal, "answered": not refused,
        "needs_human_review": str(row.get("needs_human_review", "false")).lower() == "true",
        "latency_ms": resp.latency_ms, "prompt_tokens": resp.prompt_tokens, "completion_tokens": resp.completion_tokens,
        "cost_usd": resp.cost_usd, "golden_qid": (resp.golden_match or {}).get("qid", ""),
        "golden_score": (resp.golden_match or {}).get("score", None), "confidence": resp.confidence,
        "citations": "; ".join(f"{c.doc_name}" + (f" p{c.page}" if c.page else "") + (f" {c.qid}" if c.qid else "")
                               for c in resp.citations),
    }
    ret = retrieval_metrics(resp.retrieved, matchers, k) if matchers else {}
    out.update({m: ret.get(m) for m in ("recall_at_k", "precision_at_k", "mrr", "ndcg_at_k", "context_precision")})
    out["retrieval_failure"] = bool(matchers) and not expect_refusal and ret.get("recall_at_k", 0.0) == 0.0

    context = "\n".join(c.snippet for c in resp.citations) + "\n" + "\n".join(
        str(r.get("doc_name", "")) for r in resp.retrieved)
    full_ctx = _context_text(service, resp)
    out["context_recall"] = fact_coverage(facts, full_ctx) if facts else None
    faith = (resp.grounding or {}).get("faithfulness")
    if resp.answer_type in ("golden_direct", "golden_semantic"):
        faith = 1.0
    out["faithfulness"] = None if refused else faith
    if expect_either:
        out["correctness"] = None
        out["answer_relevancy"] = None
    elif expect_refusal:
        out["correctness"] = 1.0 if refused else 0.0
        out["answer_relevancy"] = None
    else:
        fc = fact_coverage(facts, resp.answer) if facts else float("nan")
        f1 = token_f1(resp.answer, str(row.get("expected_answer", ""))) if row.get("expected_answer") else float("nan")
        if math.isnan(fc) and math.isnan(f1):
            out["correctness"] = None
        elif math.isnan(f1):
            out["correctness"] = fc
        elif math.isnan(fc):
            out["correctness"] = f1
        else:
            out["correctness"] = 0.8 * fc + 0.2 * f1
        out["key_fact_coverage"] = None if math.isnan(fc) else fc
        if refused:
            out["answer_relevancy"] = 0.0
        else:
            emb = service.embedder
            out["answer_relevancy"] = max(0.0, cosine(emb.embed_query(str(row["question"])), emb.embed_documents([resp.answer])[0]))
    out["unanswered"] = (not expect_refusal) and (not expect_either) and refused
    out["hallucination"] = (not refused) and resp.answer_type not in ("golden_direct", "golden_semantic") and bool(
        (resp.grounding or {}).get("unsupported_numbers") or (faith is not None and faith < 0.5))
    out["oos_correct"] = refused if expect_refusal else None
    out["timings"] = resp.timings
    out["_context"] = full_ctx[:4000]
    return out


def _context_text(service: ChatService, resp) -> str:
    """Full text of the retrieved chunks (top 5) for context-recall measurement."""
    from backend.retrieval.kb_index import get_index_manager

    texts = []
    for item in resp.retrieved[:5]:
        idx = get_index_manager().get(item["kb"])
        pos = idx.id_to_pos.get(item["id"])
        if pos is not None:
            chunk = idx.chunks[pos]
            texts.append(chunk.text + " " + str(chunk.metadata.get("answer", "")))
    return "\n".join(texts)


def summarize(df: pd.DataFrame) -> dict:
    def block(d: pd.DataFrame) -> dict:
        in_scope = d[d["category"] != "out_of_scope"]
        return {
            "questions": int(len(d)),
            "routing_accuracy": safe_mean(d["routing_correct"].astype(float)),
            "recall_at_k": safe_mean(in_scope["recall_at_k"]), "precision_at_k": safe_mean(in_scope["precision_at_k"]),
            "mrr": safe_mean(in_scope["mrr"]), "ndcg_at_k": safe_mean(in_scope["ndcg_at_k"]),
            "context_precision": safe_mean(in_scope["context_precision"]),
            "context_recall": safe_mean(in_scope["context_recall"]),
            "faithfulness": safe_mean(d["faithfulness"]), "answer_relevancy": safe_mean(in_scope["answer_relevancy"]),
            "correctness": safe_mean(d["correctness"]),
            "latency_p50_ms": percentile(d["latency_ms"], 50), "latency_p95_ms": percentile(d["latency_ms"], 95),
            "avg_prompt_tokens": safe_mean(d["prompt_tokens"]), "avg_completion_tokens": safe_mean(d["completion_tokens"]),
            "avg_cost_usd": safe_mean(d["cost_usd"]), "total_cost_usd": float(d["cost_usd"].sum()),
            "retrieval_failures": int(d["retrieval_failure"].sum()),
            "hallucination_rate": safe_mean(d["hallucination"].astype(float)),
            "unanswered_rate": safe_mean(in_scope["unanswered"].astype(float)) if len(in_scope) else None,
            "out_of_scope_refusal_rate": safe_mean(d["oos_correct"].dropna().astype(float)) if d["oos_correct"].notna().any() else None,
            "golden_direct_answers": int(d["answer_type"].isin(["golden_direct", "golden_semantic"]).sum()),
            "answer_types": d["answer_type"].value_counts().to_dict(),
        }
    summary = {"overall": block(df)}
    for cat, d in df.groupby("category"):
        summary[str(cat)] = block(d)
    return summary


DEST_LABELS = ["company", "products", REFUSE_LABEL]


def confusion_summary(df: pd.DataFrame) -> dict:
    dest = confusion_matrix(df["expected_destination"].tolist(), df["predicted_destination"].tolist(), DEST_LABELS)
    return {"destination": dest, "destination_report": classification_report(dest),
            "answer_decision": binary_counts(df["should_answer"].astype(bool).tolist(), df["answered"].astype(bool).tolist())}


def confusion_markdown(conf: dict) -> list[str]:
    labels, m = conf["destination"]["labels"], conf["destination"]["matrix"]
    rep, ad = conf["destination_report"], conf["answer_decision"]
    lines = ["## Confusion matrices", "",
             "### 1. Destination: where the question should go vs. where it went", "",
             "Rows = expected, columns = actual. `refuse` = out of scope, access denied, not in knowledge base or "
             "clarification. The diagonal is correct.", "",
             "| Expected \\ Actual | " + " | ".join(labels) + " | Total |", "|---|" + "---|" * (len(labels) + 1)]
    for i, lab in enumerate(labels):
        cells = [f"**{v}**" if i == j else str(v) for j, v in enumerate(m[i])]
        lines.append(f"| **{lab}** | " + " | ".join(cells) + f" | {sum(m[i])} |")
    lines.append("| **Total** | " + " | ".join(str(sum(r[j] for r in m)) for j in range(len(labels)))
                 + f" | {sum(sum(r) for r in m)} |")
    lines += ["", "| Class | Precision | Recall | F1 | Support |", "|---|---|---|---|---|"]
    for lab, s in rep["per_class"].items():
        lines.append(f"| {lab} | {s['precision']:.3f} | {s['recall']:.3f} | {s['f1']:.3f} | {s['support']} |")
    lines += ["", f"Accuracy **{rep['accuracy']:.3f}** · Macro F1 **{rep['macro_f1']:.3f}** · "
                  f"Weighted F1 **{rep['weighted_f1']:.3f}**", "",
              "### 2. Answer vs. refuse decision", "",
              "Positive = the bot answered.", "",
              "| | Bot answered | Bot refused |", "|---|---|---|",
              f"| **Should answer** | TP = {ad['tp']} | FN = {ad['fn']} (missed answer) |",
              f"| **Should refuse** | FP = {ad['fp']} (answered when it shouldn't) | TN = {ad['tn']} |", "",
              f"Precision **{ad['precision']:.3f}** · Recall **{ad['recall']:.3f}** · F1 **{ad['f1']:.3f}** · "
              f"Accuracy **{ad['accuracy']:.3f}** · Specificity (correct refusals) **{ad['specificity']:.3f}**", ""]
    return lines


def write_confusion_csv(conf: dict, out_dir: Path) -> None:
    labels = conf["destination"]["labels"]
    pd.DataFrame(conf["destination"]["matrix"], index=[f"expected_{l}" for l in labels],
                 columns=[f"actual_{l}" for l in labels]).to_csv(out_dir / "confusion_matrix_destination.csv")
    ad = conf["answer_decision"]
    pd.DataFrame([[ad["tp"], ad["fn"]], [ad["fp"], ad["tn"]]], index=["should_answer", "should_refuse"],
                 columns=["answered", "refused"]).to_csv(out_dir / "confusion_matrix_answer_decision.csv")


def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def write_report(df: pd.DataFrame, summary: dict, out_dir: Path, k: int, config: dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = ["questions", "routing_accuracy", "recall_at_k", "precision_at_k", "mrr", "ndcg_at_k", "context_precision",
               "context_recall", "faithfulness", "answer_relevancy", "correctness", "hallucination_rate", "unanswered_rate",
               "out_of_scope_refusal_rate", "retrieval_failures", "golden_direct_answers", "latency_p50_ms",
               "latency_p95_ms", "avg_prompt_tokens", "avg_completion_tokens", "avg_cost_usd", "total_cost_usd"]
    order = ["overall", "company", "products", "faq_exact", "faq_paraphrase", "multi_part", "multi_article",
             "comparison", "exact_detail", "conflict", "ambiguous", "missing_answer", "misleading", "out_of_scope"]
    cols = [c for c in order if c in summary]
    lines = [f"# Sapphirus AI Assistant - Evaluation Report", "",
             f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}  ",
             f"K = {k}  ", "Configuration: " + ", ".join(f"{a}={b}" for a, b in config.items()), "",
             "> Knowledge: LearningChain help centre (products) and sapphirus.in (company). The test set is small,",
             "> so treat these numbers as indicative and grow it with real user questions over time.", ""]
    if config.get("llm") == "extractive-fallback" or str(config.get("embedding", "")).startswith("local_hash"):
        lines += ["> OFFLINE MODE: answers were produced by the extractive (no-LLM) answerer and/or the local hashed",
                  "> embedder. Extractive answers copy source text, so faithfulness is ~1.0 by construction and token",
                  "> cost is 0. Set OPENAI_API_KEY (and EMBEDDING_BACKEND=openai) and re-run for production-representative",
                  "> generation, relevancy and cost metrics.", ""]
    lines += [
             "## Metrics by knowledge base", "", "| Metric | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for m in metrics:
        lines.append(f"| {m} | " + " | ".join(_fmt(summary[c].get(m)) for c in cols) + " |")
    lines += [""] + confusion_markdown(summary["confusion"]) if "confusion" in summary else []
    lines += ["", "## Metric definitions", "",
              "- **Recall@K**: question has at least one relevant source in the top K retrieved chunks.",
              "- **Precision@K**: share of the top K chunks that are relevant.",
              "- **MRR**: 1 / rank of the first relevant chunk. **NDCG@K**: rank-discounted relevance.",
              "- **Context precision**: RAGAS-style average precision of relevant chunks in the ranked context.",
              "- **Context recall**: share of expected key facts present in the retrieved context.",
              "- **Faithfulness**: share of answer sentences supported by the sources (lexical grounding check; numbers must match).",
              "- **Answer relevancy**: embedding similarity between question and answer.",
              "- **Correctness**: 0.8 x key-fact coverage + 0.2 x token F1 vs. expected answer (out-of-scope: 1 if correctly refused).",
              "- **Hallucination rate**: answered questions with unsupported numbers or faithfulness < 0.5.",
              "- **Unanswered rate**: in-scope questions that were refused / clarified.", "",
              "## Failures and issues", ""]
    bad = df[(df["routing_correct"] == False) | (df["correctness"].fillna(1) < 0.5) | (df["hallucination"] == True)]  # noqa: E712
    if bad.empty:
        lines.append("No failing questions.")
    else:
        lines += ["| ID | Question | Expected KB | Routed | Type | Correctness |", "|---|---|---|---|---|---|"]
        for _, r in bad.iterrows():
            lines.append(f"| {r['eval_id']} | {str(r['question'])[:70]} | {r['expected_kb']} | {r['routed_kbs']} | "
                         f"{r['answer_type']} | {_fmt(r['correctness'])} |")
    path = out_dir / "evaluation_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_ragas(df: pd.DataFrame) -> dict:
    """Optional: score answers with RAGAS (requires `pip install ragas datasets` and an LLM API key)."""
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
    except Exception as exc:
        return {"error": f"RAGAS not available: {exc}"}
    data = df[df["category"] != "out_of_scope"]
    ds = Dataset.from_dict({"question": data["question"].tolist(), "answer": data["answer"].tolist(),
                            "contexts": [[c] for c in data["_context"].tolist()],
                            "ground_truth": data["expected_answer"].fillna("").tolist()})
    try:
        result = evaluate(ds, metrics=[faithfulness, answer_relevancy, context_precision, context_recall])
        return {k: float(v) for k, v in result.items()} if hasattr(result, "items") else {"result": str(result)}
    except Exception as exc:
        return {"error": f"RAGAS evaluation failed: {exc}"}


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Evaluate the Sapphirus AI Assistant")
    parser.add_argument("--dataset", default=str(settings.base_dir / "datasets/evaluation_dataset/evaluation_dataset.csv"))
    parser.add_argument("--out", default=str(settings.base_dir / "reports"))
    parser.add_argument("--category", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--ragas", action="store_true")
    parser.add_argument("--label", default="", help="name for this run in the dashboard history")
    args = parser.parse_args(argv)

    rows = read_tabular(Path(args.dataset))["Sheet1"].to_dict(orient="records")
    if args.category:
        rows = [r for r in rows if r["category"] == args.category]
    if args.limit:
        rows = rows[: args.limit]
    service = ChatService()
    results = []
    for i, row in enumerate(rows, 1):
        res = evaluate_row(service, row, args.k)
        res["expected_answer"] = row.get("expected_answer", "")
        results.append(res)
        print(f"[{i:3}/{len(rows)}] {res['eval_id']:6} {res['answer_type']:20} route_ok={res['routing_correct']!s:5} "
              f"correct={_fmt(res['correctness'])}")
    df = pd.DataFrame(results)
    summary = summarize(df)
    summary["confusion"] = confusion_summary(df)
    if args.ragas:
        summary["ragas"] = run_ragas(df)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.drop(columns=["_context"]).to_csv(out_dir / "evaluation_results.csv", index=False)
    write_confusion_csv(summary["confusion"], out_dir)
    config = {"llm": settings.llm_model_name, "embedding": service.embedder.model_id,
              "reranker": service.reranker.name, "vector_store": settings.vector_backend}
    (out_dir / "evaluation_summary.json").write_text(json.dumps({"config": config, "summary": summary}, indent=2,
                                                                default=str), encoding="utf-8")
    report = write_report(df, summary, out_dir, args.k, config)
    try:
        from backend.services.metrics_service import record_evaluation_run
        keep = ("questions", "routing_accuracy", "recall_at_k", "precision_at_k", "mrr", "ndcg_at_k",
                "context_precision", "context_recall", "faithfulness", "answer_relevancy", "correctness",
                "hallucination_rate", "unanswered_rate", "out_of_scope_refusal_rate", "latency_p50_ms",
                "latency_p95_ms", "avg_cost_usd", "total_cost_usd")
        compact = {k: {m: v.get(m) for m in keep} for k, v in summary.items() if isinstance(v, dict) and "questions" in v}
        run_id = record_evaluation_run(args.label or f"evaluation {datetime.now(timezone.utc):%Y-%m-%d %H:%M}", "rag",
                                       config | {"dataset": Path(args.dataset).name, "k": args.k},
                                       {"by_category": compact, "confusion": summary["confusion"]})
        print(f"Recorded evaluation run #{run_id} for the dashboard")
    except Exception as exc:   # reporting must not fail the evaluation
        print(f"Could not record evaluation run: {exc}")
    print(f"\nReport written to {report}")
    print(json.dumps({k: {m: summary[k][m] for m in ("routing_accuracy", "recall_at_k", "correctness")} for k in summary
                      if k not in ("ragas", "confusion")}, indent=2, default=str))
    print("\n".join(confusion_markdown(summary["confusion"])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
