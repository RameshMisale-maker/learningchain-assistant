"""Validate the chatbot against a golden reference page (default: the learningchain.in home page).

For every question in datasets/evaluation_dataset/learningchain_home_golden.csv:
  * answer it with the production pipeline (live index, same configuration as the API)
  * accuracy  - share of the reference key facts present in the answer
  * behaviour - answered when it should / declined the negative controls
  * claims    - every factual claim is checked against the approved evidence (the golden page text fetched live +
                the knowledge-base passages the answer was built from). Verbatim quotes are supported by definition;
                other claims go to the validated NLI judge (DeBERTa-v3-base: precision 1.00, catches 100% of
                labelled hallucinations, rejects ~18% of true paraphrased claims - so the rate is an upper bound).
  Hallucination rate = unsupported claims / total claims x 100.

Usage: python scripts/validate_golden.py [--url https://learningchain.in/] [--judge nli:cross-encoder/nli-deberta-v3-base]
Outputs: reports/hallucination/golden_validation.{json,md} and an evaluation run in the dashboard history.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
logging.disable(logging.WARNING)

from backend.evaluation.claims import evaluate_answer, get_judge  # noqa: E402
from backend.evaluation.metrics import REFUSAL_TYPES, fact_coverage  # noqa: E402
from backend.retrieval.kb_index import get_index_manager  # noqa: E402
from ingestion.web import scrape_page  # noqa: E402  (Firecrawl when configured, else fetch + trafilatura)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="https://learningchain.in/")
    ap.add_argument("--dataset", default=str(ROOT / "datasets/evaluation_dataset/learningchain_home_golden.csv"))
    ap.add_argument("--judge", default="nli:cross-encoder/nli-deberta-v3-base")
    args = ap.parse_args()

    from backend.security.auth import AuthenticatedUser
    from backend.services.chat_service import ChatService

    page = scrape_page(args.url).page.markdown
    rows = list(csv.DictReader(open(args.dataset, encoding="utf-8")))
    judge = get_judge(args.judge)
    svc = ChatService()
    user = AuthenticatedUser("golden_validation", {"public"})
    idx = get_index_manager()
    results, tot = [], {"claims": 0, "unsupported": 0, "unsupported_vs_page_only": 0, "fabricated_urls": 0}
    started = time.perf_counter()
    for row in rows:
        resp = svc.answer(row["question"], user, persist=True, allowed_kbs=["products"], channel="eval",
                          assistant="products")
        refused = resp.answer_type in REFUSAL_TYPES
        kb_passages, urls = [], set()
        for item in resp.retrieved[:8]:
            pos = idx.get(item["kb"]).id_to_pos.get(item["id"])
            if pos is not None:
                ch = idx.get(item["kb"]).chunks[pos]
                kb_passages.append(str(ch.metadata.get("answer")) if ch.metadata.get("doc_type") == "golden_qa" else ch.text)
                urls.add(str(ch.metadata.get("source_url") or ""))
        rep = evaluate_answer(resp.answer, [page] + kb_passages, judge, allowed_urls=urls | {c.url for c in resp.citations})
        page_only = evaluate_answer(resp.answer, [page], judge)
        facts = [f for f in row["key_facts"].split("|") if f.strip()]
        expect_answer = row["expect"] == "answer"
        r = {"id": row["id"], "question": row["question"], "answer_type": resp.answer_type,
             "behaviour_ok": (not refused) if expect_answer else refused,
             "fact_coverage": fact_coverage(facts, resp.answer) if (facts and expect_answer) else None,
             "claims": len(rep.claims), "unsupported": [c.claim for c in rep.unsupported],
             "unsupported_vs_page_only": len(page_only.unsupported), "fabricated_urls": rep.fabricated_urls,
             "citations": [c.url or c.doc_name for c in resp.citations][:3], "answer": resp.answer[:600],
             "latency_ms": resp.latency_ms}
        results.append(r)
        tot["claims"] += r["claims"]
        tot["unsupported"] += len(r["unsupported"])
        tot["unsupported_vs_page_only"] += r["unsupported_vs_page_only"]
        tot["fabricated_urls"] += len(r["fabricated_urls"])
        cov = "-" if r["fact_coverage"] is None else f"{r['fact_coverage']:.2f}"
        print(f"{r['id']} {'OK ' if r['behaviour_ok'] else 'BAD'} {resp.answer_type:16} facts={cov:4} "
              f"claims={r['claims']:2} unsupported={len(r['unsupported'])}  {row['question'][:60]}")

    answered = [r for r in results if r["fact_coverage"] is not None]
    summary = {
        "golden_source": args.url, "questions": len(results), "judge": judge.name,
        "answer_model": svc.llm.model if svc.llm else "extractive (no LLM key configured)",
        "behaviour_accuracy": round(sum(r["behaviour_ok"] for r in results) / len(results), 3),
        "answer_accuracy_key_facts": round(sum(r["fact_coverage"] for r in answered) / len(answered), 3),
        "negative_controls_declined": f"{sum(r['behaviour_ok'] for r in results if r['fact_coverage'] is None)}/"
                                      f"{sum(1 for r in results if r['fact_coverage'] is None)}",
        "total_claims": tot["claims"], "unsupported_claims": tot["unsupported"],
        "hallucination_rate_pct": round(100 * tot["unsupported"] / tot["claims"], 2) if tot["claims"] else 0.0,
        "hallucination_rate_vs_golden_page_only_pct": round(100 * tot["unsupported_vs_page_only"] / tot["claims"], 2)
        if tot["claims"] else 0.0,
        "fabricated_urls": tot["fabricated_urls"], "seconds": round(time.perf_counter() - started, 1)}
    out = ROOT / "reports" / "hallucination"
    out.mkdir(parents=True, exist_ok=True)
    (out / "golden_validation.json").write_text(json.dumps({"summary": summary, "results": results}, indent=2),
                                               encoding="utf-8")
    lines = [f"# Golden validation - {args.url}", "", *(f"- **{k}**: {v}" for k, v in summary.items()), "",
             "| ID | Question | Type | Behaviour | Key facts | Claims | Unsupported |", "|---|---|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['id']} | {r['question']} | {r['answer_type']} | {'ok' if r['behaviour_ok'] else 'WRONG'} | "
                     f"{'-' if r['fact_coverage'] is None else round(r['fact_coverage'], 2)} | {r['claims']} | "
                     f"{len(r['unsupported'])} |")
    lines += ["", "## Unsupported claims (review manually - the judge is strict on paraphrases)", ""]
    lines += [f"- {r['id']}: {c}" for r in results for c in r["unsupported"]]
    (out / "golden_validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        from backend.services.metrics_service import record_evaluation_run
        record_evaluation_run(f"golden validation: {args.url}", "hallucination", {"judge": judge.name}, summary)
    except Exception as exc:
        print("could not record run:", exc)
    print("\n" + json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
