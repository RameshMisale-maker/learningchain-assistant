"""Evaluation workflow: validated golden dataset -> existing chatbot pipeline -> retrieval / RAGAS / hallucination /
safety / performance -> reports in evaluation/results/<run_id>/.

Command line (from the project root, with the evaluation venv):
    evaluation\\.venv\\Scripts\\python -m evaluation.run_evaluation --limit 10
    evaluation\\.venv\\Scripts\\python -m evaluation.run_evaluation --categories "Direct factual" "Out-of-scope" --no-ragas
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

import pandas as pd

from evaluation import config
from evaluation import dataset_manager as dm
from evaluation import generation_metrics as gm
from evaluation import hallucination_metrics as hm
from evaluation import performance_metrics as pm
from evaluation import report_generator as rg
from evaluation import retrieval_metrics as rm
from evaluation import safety_metrics as sm

Progress = Callable[[float, str], None]


@dataclass
class RunOptions:
    categories: list[str] = field(default_factory=list)      # empty = all
    limit: int | None = None
    run_ragas: bool = True
    run_hallucination: bool = True
    run_safety: bool = True
    hallucination_judge: str = config.HALLUCINATION_JUDGE
    k: int = config.RETRIEVAL_K
    label: str = ""
    resume_run_id: str = ""        # continue an interrupted run from its checkpoint (no new chatbot calls)
    ragas_sample: int = 0          # 0 = RAGAS on every answered question; N = stratified sample of N (rate-limited judges)
    load_test_url: str = ""                                   # e.g. http://127.0.0.1:8001 ("" = skip load test)
    load_levels: list[int] = field(default_factory=lambda: [1, 4, 8])
    load_requests: int = 12


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=config.PROJECT_ROOT, capture_output=True,
                              text=True, timeout=10).stdout.strip()
    except Exception:
        return ""


def _mean(values) -> float | None:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def judge_question(q: dict[str, Any], k: int) -> tuple[str, str]:
    """PASS / FAIL with the reasons (only metrics that were actually measured are used)."""
    run, reasons = q["run"], []
    if run.get("behavior") == "Error":
        return "FAIL", f"pipeline error: {run.get('error') or run.get('answer_type')}"
    if not q["behavior_correct"]:
        reasons.append(f"expected {q['expected_behavior']}, got {run.get('behavior')} ({run.get('answer_type')})")
    if q["expected_behavior"] == "Answer" and run.get("behavior") == "Answer":
        th = config.QUESTION_THRESHOLDS
        ret, rag, hal = q["retrieval"], q["ragas"], q["hallucination"]
        if ret.get("recall_at_k") is not None and ret["recall_at_k"] < th["recall_at_k"]:
            reasons.append(f"relevant source missing from top {k} (recall@{k}={ret['recall_at_k']:.2f})")
        for key in ("faithfulness", "answer_correctness", "answer_relevancy"):
            if rag.get(key) is not None and rag[key] < th[key]:
                reasons.append(f"{key}={rag[key]:.2f} < {th[key]}")
        if hal.get("hallucination_rate") is not None and hal["hallucination_rate"] > th["hallucination_rate"]:
            reasons.append(f"{hal['unsupported']} unsupported claim(s)")
    return ("FAIL" if reasons else "PASS"), "; ".join(reasons)


def ragas_sample(answered: list[dict], n: int) -> list[dict]:
    """Deterministic sample stratified by question category (round-robin over categories, question_id order), so a
    small RAGAS budget still covers every kind of question. n <= 0 or n >= len -> all."""
    if n <= 0 or n >= len(answered):
        return answered
    by_cat: dict[str, list[dict]] = {}
    for q in sorted(answered, key=lambda x: x["question_id"]):
        by_cat.setdefault(q["question_category"], []).append(q)
    picked: list[dict] = []
    while len(picked) < n and any(by_cat.values()):
        for cat in sorted(by_cat):
            if by_cat[cat] and len(picked) < n:
                picked.append(by_cat[cat].pop(0))
    return picked


def layer_rows(summary: dict[str, Any], perf: dict[str, Any]) -> list[dict[str, Any]]:
    """Every metric grouped into the three evaluation layers - the rows of evaluation_metrics.csv."""
    m, targets, k = summary["metrics"], config.TARGETS, summary.get("k", config.RETRIEVAL_K)
    rows: list[dict[str, Any]] = []

    def add(layer, metric, value, unit, method, n=None, target_key=None):
        t = targets.get(target_key) if target_key else None
        met = None
        if t and value is not None:
            met = value >= t["target"] if t["direction"] == "min" else value <= t["target"]
        rows.append({"layer": layer, "metric": metric, "value": None if value is None else round(float(value), 4),
                     "unit": unit, "target": "" if not t else (">= " if t["direction"] == "min" else "<= ") +
                     str(t["target"]), "target_met": "" if met is None else ("yes" if met else "no"),
                     "sample_size": n, "method": method})

    R, G, E = "1. Retrieval", "2. Generation", "3. End-to-end production"
    n_ret = m.get("retrieval_labelled_questions")
    add(R, f"Precision@{k}", m.get("precision_at_k"), "fraction", "relevance labels vs re-ranked child chunks", n_ret)
    add(R, f"Recall@{k}", m.get("recall_at_k"), "fraction", "relevance labels vs re-ranked child chunks", n_ret,
        "recall_at_k")
    add(R, f"Hit rate@{k}", m.get("hit_rate_at_k"), "fraction", "any relevant chunk in top k", n_ret)
    add(R, "MRR", m.get("mrr"), "fraction", "reciprocal rank of first relevant chunk", n_ret)
    add(R, f"NDCG@{k}", m.get("ndcg_at_k"), "fraction", "rank-discounted relevance", n_ret)
    add(R, "Context precision (rank-based)", m.get("context_precision"), "fraction", "relevance labels, rank-weighted",
        n_ret)
    add(R, "Context recall (lexical)", m.get("context_recall"), "fraction", "reference statements found in context")
    add(R, "Context precision (RAGAS)", m.get("ragas_context_precision"), "fraction",
        f"RAGAS LLM judge ({config.EVAL_LLM_MODEL})", m.get("ragas_scored"), "context_precision")
    add(R, "Context recall (RAGAS)", m.get("ragas_context_recall"), "fraction",
        f"RAGAS LLM judge ({config.EVAL_LLM_MODEL})", m.get("ragas_scored"), "context_recall")

    add(G, "Faithfulness (RAGAS)", m.get("faithfulness"), "fraction", f"RAGAS LLM judge ({config.EVAL_LLM_MODEL})",
        m.get("ragas_scored"), "faithfulness")
    add(G, "Answer relevancy (RAGAS)", m.get("answer_relevancy"), "fraction",
        "RAGAS: judge-generated questions vs. embeddings", m.get("ragas_scored"), "answer_relevancy")
    add(G, "Answer correctness (RAGAS)", m.get("answer_correctness"), "fraction",
        f"RAGAS LLM judge vs. expected answer ({config.EVAL_LLM_MODEL})", m.get("ragas_scored"), "answer_correctness")
    add(G, "Hallucination rate (claim level)", m.get("hallucination_rate"), "fraction",
        f"unsupported claims / all claims, judged per claim by {summary.get('hallucination_judge', 'judge')}",
        m.get("total_claims"), "hallucination_rate")
    add(G, "Contradicted claims", m.get("contradicted_claims"), "count",
        f"judged per claim by {summary.get('hallucination_judge', 'judge')}")
    add(G, "Behaviour accuracy (answer / abstain / reject)", m.get("behavior_accuracy"), "fraction",
        "expected vs. actual behaviour")
    add(G, "Abstention accuracy", m.get("abstention_accuracy"), "fraction", "declines unanswerable questions", None,
        "abstention_accuracy")
    add(G, "False abstention rate", m.get("false_abstention_rate"), "fraction", "answerable questions declined")

    add(E, "Question pass rate", m.get("pass_rate"), "fraction", "all per-question checks passed",
        m.get("questions_evaluated"))
    add(E, "Safety test pass rate", m.get("safety_pass_rate"), "fraction", "safety_tests.csv via live pipeline",
        m.get("safety_tests"))
    add(E, "Restricted request rejection", m.get("restricted_rejection_rate"), "fraction",
        "restricted / out-of-scope prompts rejected", None, "restricted_rejection_rate")
    add(E, "Average latency", perf.get("avg_latency_ms"), "ms", "end-to-end, sequential", perf.get("requests"))
    add(E, "P50 latency", perf.get("p50_latency_ms"), "ms", "end-to-end, sequential", perf.get("requests"))
    add(E, "P95 latency", perf.get("p95_latency_ms"), "ms", "end-to-end, sequential", perf.get("requests"))
    add(E, "P99 latency", perf.get("p99_latency_ms"), "ms", "end-to-end, sequential", perf.get("requests"))
    add(E, "Retrieval latency (avg)", perf.get("avg_retrieval_ms"), "ms", "vector + BM25 + fusion")
    add(E, "Re-ranking latency (avg)", perf.get("avg_rerank_ms"), "ms", "cross-encoder")
    add(E, "LLM latency (avg)", perf.get("avg_llm_total_ms"), "ms", "LLM call incl. fallback retries")
    add(E, "TTFT (LLM)", perf.get("avg_ttft_ms") if perf.get("ttft_measured") else None, "ms",
        perf.get("ttft_note") or "streaming LLM")
    add(E, "Input tokens (total)", perf.get("total_input_tokens"), "tokens", "reported by the LLM API")
    add(E, "Output tokens (total)", perf.get("total_output_tokens"), "tokens", "reported by the LLM API")
    cost_known = bool(perf.get("total_cost_usd")) or not perf.get("total_input_tokens")
    add(E, "Cost per query", perf.get("cost_per_query_usd") if cost_known else None, "USD",
        "tokens x configured price" if cost_known else "price not configured for this provider (.env)")
    add(E, "Error rate", perf.get("error_rate"), "fraction", "pipeline errors / requests", perf.get("requests"))
    add(E, "Answers by fallback model", perf.get("fallback_model_share"), "fraction",
        "primary model rate-limited -> OPENAI_FALLBACK_MODELS", perf.get("llm_answers"))
    add(E, "Sequential throughput", perf.get("throughput_qps"), "requests/s", "one request at a time")
    for lvl in perf.get("load_test") or []:
        c = lvl["concurrency"]
        add(E, f"Load c={c}: throughput", lvl["throughput_rps"], "requests/s", "concurrent HTTP load test",
            lvl["requests"])
        add(E, f"Load c={c}: P95 latency", lvl["latency_p95_ms"], "ms", "concurrent HTTP load test", lvl["requests"])
        add(E, f"Load c={c}: first streamed token P50", lvl["first_token_p50_ms"], "ms",
            "streaming endpoint (answer verified before streaming)", lvl["requests"])
        add(E, f"Load c={c}: error rate", lvl["error_rate"], "fraction", "concurrent HTTP load test", lvl["requests"])
    return rows


def summarize(questions: list[dict], safety: list[dict], perf: dict, k: int) -> dict[str, Any]:
    answered = [q for q in questions if q["expected_behavior"] == "Answer" and q["run"].get("behavior") == "Answer"]
    ret = rm.aggregate([q["retrieval"] for q in questions if q["expected_behavior"] == "Answer"])
    hal = hm.aggregate([{"question_id": q["question_id"], **q["hallucination"]} for q in questions])
    beh = gm.behavior_summary([{"expected_behavior": q["expected_behavior"], "actual_behavior": q["run"].get("behavior"),
                                "behavior_correct": q["behavior_correct"],
                                "question_category": q["question_category"]} for q in questions])
    saf = sm.summarize(safety)
    restricted = beh["restricted_rejection_rate"]
    if saf.get("strict_rejection_rate") is not None:   # golden 'Reject' rows + safety-suite rejection tests
        n_g, n_s = beh["restricted_n"], sum(1 for s in safety if s["expected_behavior"] == "Reject")
        ok_g = (restricted or 0) * n_g
        ok_s = saf["strict_rejection_rate"] * n_s
        restricted = (ok_g + ok_s) / (n_g + n_s) if (n_g + n_s) else None
    metrics = {
        "questions_evaluated": len(questions),
        "pass_rate": _mean([1.0 if q["status"] == "PASS" else 0.0 for q in questions]),
        "ragas_scored": sum(1 for q in answered if q["ragas"].get("faithfulness") is not None
                            or q["ragas"].get("answer_relevancy") is not None),
        "faithfulness": _mean(q["ragas"].get("faithfulness") for q in answered),
        "answer_correctness": _mean(q["ragas"].get("answer_correctness") for q in answered),
        "answer_relevancy": _mean(q["ragas"].get("answer_relevancy") for q in answered),
        "ragas_context_precision": _mean(q["ragas"].get("context_precision") for q in answered),
        "ragas_context_recall": _mean(q["ragas"].get("context_recall") for q in answered),
        "precision_at_k": ret["precision_at_k"], "recall_at_k": ret["recall_at_k"], "mrr": ret["mrr"],
        "ndcg_at_k": ret["ndcg_at_k"], "hit_rate_at_k": ret["hit_rate_at_k"],
        "context_precision": ret["context_precision"], "context_recall": ret["context_recall"],
        "retrieval_labelled_questions": ret["recall_at_k_n"], "needs_relevance_annotation": ret["needs_annotation"],
        "hallucination_rate": hal["hallucination_rate"], "total_claims": hal["total_claims"],
        "unsupported_claims": hal["unsupported_claims"], "contradicted_claims": hal["contradicted_claims"],
        "claims_awaiting_review": hal["awaiting_human_review"],
        "behavior_accuracy": beh["behavior_accuracy"], "abstention_accuracy": beh["abstention_accuracy"],
        "false_abstention_rate": beh["false_abstention_rate"], "restricted_rejection_rate": restricted,
        "safety_pass_rate": saf.get("pass_rate"), "safety_tests": saf.get("total"),
        "avg_latency_ms": perf.get("avg_latency_ms"), "p95_latency_ms": perf.get("p95_latency_ms"),
        "cost_per_query_usd": perf.get("cost_per_query_usd"), "error_rate": perf.get("error_rate"),
    }
    # RAGAS context metrics are the headline context precision/recall when available
    targets = {}
    for key, t in config.TARGETS.items():
        value = metrics.get(key)
        if key in ("context_precision", "context_recall") and metrics.get(f"ragas_{key}") is not None:
            value = metrics[f"ragas_{key}"]
        met = None if value is None else (value >= t["target"] if t["direction"] == "min" else value <= t["target"])
        targets[key] = {**t, "value": value, "met": met}
    return {"metrics": metrics, "targets": targets, "k": k, "safety": saf,
            "failed_questions": [q["question_id"] for q in questions if q["status"] == "FAIL"],
            "hallucination_failed_questions": hal["failed_questions"]}


def pace_seconds(tokens: int, elapsed_s: float, tokens_per_minute: int | None = None) -> float:
    """Seconds to wait after a chatbot call that used `tokens` AI tokens and took `elapsed_s`, so the run stays
    under EVAL_TOKENS_PER_MINUTE. A test run fires questions far faster than real users; without pacing it hits the
    provider's per-minute limit and answers come from the fallback model instead of the one being evaluated."""
    tpm = config.EVAL_TOKENS_PER_MINUTE if tokens_per_minute is None else tokens_per_minute
    if tpm <= 0 or tokens <= 0:
        return 0.0
    return max(0.0, tokens / tpm * 60.0 - elapsed_s)


def _ask_paced(adapter, question: str, sleep=time.sleep, **kwargs) -> dict:
    started = time.perf_counter()
    r = adapter.ask(question, **kwargs).to_dict()
    wait = pace_seconds(int(r.get("prompt_tokens") or 0) + int(r.get("completion_tokens") or 0),
                        time.perf_counter() - started)
    if wait:
        sleep(wait)
    return r


def _execute_chatbot(adapter, rows, safety_tests, labels, options, report, total_calls):
    """Stage 1: run every golden question and safety prompt through the existing chatbot pipeline."""
    questions, wall0 = [], time.perf_counter()
    for i, row in enumerate(rows):
        report(0.05 + 0.45 * i / total_calls, f"Chatbot: {row['question_id']} - {row['question'][:70]}")
        r = _ask_paced(adapter, row["question"], role=row.get("role") or "public")
        q = {k: row.get(k, "") for k in config.GOLDEN_COLUMNS + ["role", "notes"]}
        q["run"] = r
        q["behavior_correct"] = gm.behavior_correct(row["expected_behavior"], r["behavior"])
        q["lexical_f1"] = gm.lexical_f1(r["answer"], row["expected_answer"])             if row["expected_behavior"] == "Answer" and r["behavior"] == "Answer" else None
        ctx = [c["text"] for c in r["contexts"]]
        q["retrieval"] = rm.compute(r["retrieved"], row, options.k, ctx, labels.get(row["question_id"]))             if row["expected_behavior"] == "Answer" else {}
        questions.append(q)
    safety_runs = []
    for j, test in enumerate(safety_tests):
        report(0.05 + 0.45 * (len(rows) + j) / total_calls, f"Safety: {test['test_id']} - {test['prompt'][:60]}")
        safety_runs.append(_ask_paced(adapter, test["prompt"]))
    return questions, safety_runs, time.perf_counter() - wall0


def run(dataset: pd.DataFrame, options: RunOptions, progress: Progress | None = None) -> str:
    """Execute the full evaluation; returns the run id. Raises ValueError when the dataset is invalid."""
    from evaluation.chatbot_adapter import ChatbotAdapter, index_inventory

    def report(frac: float, msg: str) -> None:
        if progress:
            progress(min(max(frac, 0.0), 1.0), msg)
        print(f"[{frac:5.1%}] {msg}", flush=True)

    dataset = dm.normalize(dataset)
    report(0.0, "Validating golden dataset")
    errors, warnings = dm.validate(dataset, index_inventory())
    if errors:
        raise ValueError("Dataset is invalid: " + " | ".join(errors))
    if options.categories:
        dataset = dataset[dataset["question_category"].isin(options.categories)]
    if options.limit:
        dataset = dataset.head(options.limit)
    if dataset.empty:
        raise ValueError("No questions match the selected categories.")

    if options.resume_run_id:
        run_id = options.resume_run_id
        run_dir = config.RESULTS_DIR / run_id
        if not (run_dir / "checkpoint.json").exists():
            raise ValueError(f"No checkpoint to resume in {run_dir}")
        if (run_dir / "summary.json").exists():
            raise ValueError(f"Run {run_id} is already complete - results are never overwritten")
    else:
        run_id = rg.new_run_id()
        run_dir = rg.create_run_dir(run_id)
    report(0.02, "Loading the existing chatbot pipeline (embedder, vector index, re-ranker, LLM client)")
    adapter = ChatbotAdapter()
    started = datetime.now(timezone.utc).isoformat()
    labels = dm.relevance_labels()
    rows = dataset.to_dict("records")
    safety_tests = sm.load_tests().to_dict("records") if options.run_safety else []
    total_calls = len(rows) + len(safety_tests)
    prompt_lines = sm._system_prompt_lines()
    checkpoint = run_dir / "checkpoint.json"

    if options.resume_run_id:
        # 1) resumed: reuse the chatbot answers captured by the interrupted run (no new chatbot calls)
        cp = json.loads(checkpoint.read_text(encoding="utf-8"))
        questions, safety_tests, safety_runs = cp["questions"], cp["safety_tests"], cp["safety_runs"]
        wall, started = cp["wall_seconds"], cp["started_at"]
        report(0.5, f"Resumed {run_id}: {len(questions)} answers and {len(safety_runs)} safety runs from checkpoint")
    else:
        questions, safety_runs, wall = _execute_chatbot(adapter, rows, safety_tests, labels, options, report,
                                                        total_calls)
        checkpoint.write_text(json.dumps({"questions": questions, "safety_tests": safety_tests,
                                          "safety_runs": safety_runs, "wall_seconds": wall, "started_at": started},
                                         ensure_ascii=False, default=str), encoding="utf-8")
        report(0.5, f"Checkpoint saved ({checkpoint.name}) - resume with --resume {run_id} if scoring is interrupted")
    safety = [sm.grade(t, r, prompt_lines) for t, r in zip(safety_tests, safety_runs)]


    # 2) RAGAS on actual answers + actual contexts (separate evaluator model)
    ragas_note = "skipped"
    for q in questions:
        q["ragas"] = {}
    if options.run_ragas:
        from evaluation import ragas_evaluator as re_

        ok, ragas_note = re_.ragas_status()
        targets = ragas_sample([q for q in questions if q["run"]["behavior"] == "Answer"], options.ragas_sample)
        for q in questions:
            if q["run"]["behavior"] == "Answer" and q not in targets:
                q["ragas"] = {"ragas_error": f"not sampled (RAGAS ran on a {options.ragas_sample}-question sample)"}
        if ok and targets:
            report(0.55, f"RAGAS: scoring {len(targets)} answered question(s) with {config.EVAL_LLM_MODEL}")
            try:
                scores = re_.evaluate_rows(
                    [{"question_id": q["question_id"], "question": q["question"], "answer": q["run"]["answer"],
                      "contexts": [{"sid": c["sid"], "text": c["text"]} for c in q["run"]["contexts"]],
                      "reference": q["expected_answer"]} for q in targets],
                    progress=lambda d, n: report(0.55 + 0.2 * d / max(n, 1), f"RAGAS: {d}/{n}"))
                for q in targets:
                    q["ragas"] = scores.get(q["question_id"], {})
            except Exception as exc:
                ragas_note = f"RAGAS failed: {type(exc).__name__}: {exc}"
                for q in targets:
                    q["ragas"] = {"ragas_error": ragas_note[:300]}
        elif not ok:
            for q in targets:
                q["ragas"] = {"ragas_error": ragas_note}

    # 3) claim-level hallucination evaluation
    judge_name = "skipped"
    for q in questions:
        q["hallucination"] = {}
    if options.run_hallucination:
        try:
            judge = hm.get_judge(options.hallucination_judge)
            judge_name = getattr(judge, "name", options.hallucination_judge)
        except Exception as exc:
            judge, judge_name = None, f"unavailable: {type(exc).__name__}: {exc}"
        answered = [q for q in questions if q["run"]["behavior"] == "Answer"]
        for i, q in enumerate(answered):
            report(0.76 + 0.18 * i / max(len(answered), 1), f"Hallucination check: {q['question_id']}")
            if judge is None:
                q["hallucination"] = {"error": judge_name, "claims": []}
                continue
            q["hallucination"] = hm.analyse(q["run"]["answer"], [c["text"] for c in q["run"]["contexts"]],
                                            q["run"]["behavior"], judge)

    # 4) pass / fail, performance, reports
    for q in questions:
        q["status"], q["failure_reason"] = judge_question(q, options.k)
    all_runs = [q["run"] for q in questions] + safety_runs
    perf = pm.summarize(all_runs, wall)
    primary = adapter.llm_model
    llm_runs = [r for r in all_runs if r.get("prompt_tokens")]
    perf["answers_by_fallback_model"] = sum(1 for r in llm_runs if r.get("model") and r["model"] != primary)
    perf["llm_answers"] = len(llm_runs)
    perf["fallback_model_share"] = (perf["answers_by_fallback_model"] / len(llm_runs)) if llm_runs else None
    load_note = "not run"
    if options.load_test_url:
        from evaluation import load_test as lt

        report(0.95, f"Load test: {options.load_test_url} at concurrency {options.load_levels}")
        try:
            load = lt.run(options.load_test_url, [q["question"] for q in questions], options.load_levels,
                          options.load_requests)
            perf["load_test"] = lt.rows(load)
            load_note = f"{options.load_test_url} levels={options.load_levels} x {options.load_requests} requests"
        except Exception as exc:
            load_note = f"failed: {type(exc).__name__}: {exc}"
            perf["load_test"] = []
    summary = summarize(questions, safety, perf, options.k)
    summary["hallucination_judge"] = judge_name
    summary["layers"] = layer_rows(summary, perf)
    meta = {
        "run_id": run_id, "label": options.label, "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(), "question_count": len(questions),
        "safety_test_count": len(safety), "categories": options.categories or "all", "limit": options.limit,
        "dataset_hash": dm.dataset_hash(dataset), "dataset_path": str(config.GOLDEN_DATASET_PATH),
        "dataset_warnings": warnings, "git_commit": _git_commit(), "chatbot": adapter.describe(),
        "evaluator": {"ragas": ragas_note, "ragas_model": config.EVAL_LLM_MODEL, "hallucination_judge": judge_name},
        "load_test": load_note,
        "k": options.k, "targets": config.TARGETS, "thresholds": config.QUESTION_THRESHOLDS,
    }
    report(0.96, "Writing reports")
    rg.write_run(run_dir, meta, questions, safety, summary, perf)
    report(1.0, f"Done: {run_id}")
    return run_id


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run the RAG evaluation against the existing chatbot")
    p.add_argument("--dataset", default=str(config.GOLDEN_DATASET_PATH))
    p.add_argument("--categories", nargs="*", default=[])
    p.add_argument("--limit", type=int)
    p.add_argument("--no-ragas", action="store_true")
    p.add_argument("--no-hallucination", action="store_true")
    p.add_argument("--no-safety", action="store_true")
    p.add_argument("--judge", default=config.HALLUCINATION_JUDGE, choices=["llm", "nli"])
    p.add_argument("--label", default="")
    p.add_argument("--resume", default="", help="run id of an interrupted run to finish from its checkpoint")
    p.add_argument("--ragas-sample", type=int, default=0, help="RAGAS on a stratified sample of N answered questions")
    p.add_argument("--load-test-url", default="", help="running API for the concurrency test, e.g. http://127.0.0.1:8001")
    p.add_argument("--load-levels", default="1,4,8")
    p.add_argument("--load-requests", type=int, default=12)
    a = p.parse_args(argv)
    from pathlib import Path

    run_id = run(dm.load(Path(a.dataset)), RunOptions(categories=a.categories, limit=a.limit, run_ragas=not a.no_ragas,
                                                       run_hallucination=not a.no_hallucination,
                                                       run_safety=not a.no_safety, hallucination_judge=a.judge,
                                                       label=a.label, load_test_url=a.load_test_url, ragas_sample=a.ragas_sample, resume_run_id=a.resume,
                                                       load_levels=[int(x) for x in a.load_levels.split(",") if x],
                                                       load_requests=a.load_requests))
    print(f"Results: {config.RESULTS_DIR / run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
