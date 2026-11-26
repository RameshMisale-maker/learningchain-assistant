"""Re-check the flagged claims of a finished run, without asking the chatbot again.

    python -m evaluation.recheck_claims --run <run_id>                # local rules only (free)
    python -m evaluation.recheck_claims --run <run_id> --judge llm    # + the AI judge for what is still flagged

Only claims marked UNSUPPORTED / CONTRADICTED are looked at again: first the free local rules (lead-in lines are not
claims; close paraphrases of the passages are supported), then - with --judge llm - the evaluator model
(EVAL_LLM_MODEL) for the rest. When the judge's rate limit is reached, the remaining claims keep their old verdict
and are listed as not re-checked. Answers, retrieval, RAGAS and safety results are unchanged. The result is written to
a NEW folder <run_id>_rechecked (the original run is never modified).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from typing import Any

from evaluation import config
from evaluation import hallucination_metrics as hm


def recheck_question(hal: dict[str, Any], contexts: list[str], judge=None) -> tuple[dict[str, Any], dict[str, int]]:
    """Re-check one question's flagged claims. Returns (updated hallucination dict, counts by outcome)."""
    counts = {"lead_in": 0, "overlap": 0, "judge_supported": 0, "judge_flagged": 0, "not_rechecked": 0}
    claims = []
    stop_judge = False
    for c in hal.get("claims") or []:
        c = dict(c)
        if c.get("verdict") in (hm.UNSUPPORTED, hm.CONTRADICTED):
            if hm.is_lead_in(c["claim"]):
                c.update(verdict=hm.NOT_A_CLAIM, evidence="lead-in line, states no fact", judged_by="lead-in",
                         needs_human_review=False)
                counts["lead_in"] += 1
            elif hm.overlap_supported(c["claim"], contexts):
                c.update(verdict=hm.SUPPORTED, evidence="close paraphrase: key words found in retrieved context",
                         judged_by="word-overlap", needs_human_review=False)
                counts["overlap"] += 1
            elif judge is not None and not stop_judge:
                try:
                    verdict, evidence = judge.judge(c["claim"], contexts)
                    c.update(verdict=verdict, evidence=evidence, judged_by=getattr(judge, "name", "judge"),
                             needs_human_review=verdict in (hm.UNSUPPORTED, hm.CONTRADICTED))
                    counts["judge_supported" if verdict == hm.SUPPORTED else "judge_flagged"] += 1
                except Exception as exc:          # rate limit / outage: keep the old verdict, report it
                    stop_judge = "rate" in str(exc).lower() or "429" in str(exc)
                    counts["not_rechecked"] += 1
            else:
                counts["not_rechecked"] += 1
        claims.append(c)
    judged = [c for c in claims if c["verdict"] not in ("JUDGE_ERROR", hm.NOT_A_CLAIM)]
    out = dict(hal, claims=claims, total_claims=len(judged),
               supported=sum(c["verdict"] == hm.SUPPORTED for c in judged),
               contradicted=sum(c["verdict"] == hm.CONTRADICTED for c in judged),
               unsupported=sum(c["verdict"] in (hm.UNSUPPORTED, hm.CONTRADICTED) for c in judged))
    out["hallucination_rate"] = (out["unsupported"] / out["total_claims"]) if judged else hal.get("hallucination_rate")
    return out, counts


def main(argv: list[str] | None = None) -> int:
    from evaluation import report_generator as rg
    from evaluation import safety_metrics as sm
    from evaluation.run_evaluation import judge_question, layer_rows, summarize

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--judge", choices=("none", "llm"), default="none")
    args = ap.parse_args(argv)

    src = config.RESULTS_DIR / args.run
    dst = config.RESULTS_DIR / f"{args.run}_rechecked"
    questions = json.loads((src / "questions.json").read_text(encoding="utf-8"))
    ck = json.loads((src / "checkpoint.json").read_text(encoding="utf-8"))
    judge = hm.LLMClaimJudge() if args.judge == "llm" else None
    totals: dict[str, int] = {}
    for q in questions:
        hal = q.get("hallucination") or {}
        if not hal.get("claims"):
            continue
        contexts = [c.get("text", "") for c in q["run"].get("contexts") or []]
        q["hallucination"], counts = recheck_question(hal, contexts, judge)
        for k, v in counts.items():
            totals[k] = totals.get(k, 0) + v
        q["status"], q["failure_reason"] = judge_question(q, 5)
    safety = [sm.grade(t, r, sm._system_prompt_lines()) for t, r in zip(ck["safety_tests"], ck["safety_runs"])]
    perf = json.loads((src / "performance.json").read_text(encoding="utf-8"))
    summary = summarize(questions, safety, perf, 5)
    summary["hallucination_judge"] = f"rechecked: local rules{' + ' + judge.name if judge else ''}"
    summary["layers"] = layer_rows(summary, perf)
    meta = json.loads((src / "run_meta.json").read_text(encoding="utf-8"))
    meta.update({"run_id": dst.name, "rechecked_from": args.run, "rechecked_at": datetime.now(timezone.utc).isoformat(),
                 "recheck": totals, "note": "flagged claims re-checked (lead-in lines, close paraphrases"
                                            + (", evaluator model" if judge else "") + "); answers unchanged"})
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    rg.write_run(dst, meta, questions, safety, summary, perf)
    passed = sum(q["status"] == "PASS" for q in questions)
    print(f"{dst.name}: {passed}/{len(questions)} pass | re-check: {totals}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
