"""Claim-level hallucination evaluation.

Step 1 - validate claim judges against labelled claims (datasets/evaluation_dataset/claims_gold.csv): confusion
         matrix, accuracy, precision, recall, F1, specificity per judge. Positive class = "supported".
Step 2 - run the chatbot on the evaluation questions and judge every claim of every answer with the chosen judge:
         hallucination rate = unsupported claims / total claims x 100, plus fabricated URLs and invalid citations,
         broken down by question category, with examples.

Usage:
  python scripts/evaluate_claims.py                       # judges: lexical + nli (local); system judge = best F1
  python scripts/evaluate_claims.py --judges lexical,nli,claude --system-judge claude
Outputs: reports/hallucination/{judge_validation,system_hallucination}.{json,md}
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
logging.disable(logging.WARNING)

from backend.evaluation.claims import evaluate_answer, get_judge  # noqa: E402
from backend.evaluation.metrics import REFUSAL_TYPES, binary_counts  # noqa: E402
from backend.retrieval.kb_index import get_index_manager  # noqa: E402

OUT = ROOT / "reports" / "hallucination"


def _context_for(source_match: str) -> list[str]:
    """Texts of indexed chunks whose section / qid / approved question matches `source_match`."""
    key = source_match.lower().strip()
    found = []
    for kb in get_index_manager().kb_names():
        for c in get_index_manager().get(kb).chunks:
            m = c.metadata
            hay = " ".join(str(m.get(k, "")) for k in ("section", "qid", "question")).lower()
            if key == str(m.get("qid", "")).lower() or key in hay:
                found.append(str(m.get("answer")) if m.get("doc_type") == "golden_qa" and m.get("answer") else c.text)
    return found


def validate_judges(judges: list[str]) -> dict:
    rows = list(csv.DictReader(open(ROOT / "datasets/evaluation_dataset/claims_gold.csv", encoding="utf-8")))
    contexts = {r["claim_id"]: _context_for(r["source_match"]) for r in rows}
    missing = [r["claim_id"] for r in rows if not contexts[r["claim_id"]]]
    results = {"claims": len(rows), "context_missing": missing, "label_origin": rows[0]["origin"], "judges": {}}
    for name in judges:
        try:
            judge = get_judge(name)
        except Exception as exc:
            results["judges"][name] = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        started = time.perf_counter()
        verdicts = [judge.judge(r["claim"], contexts[r["claim_id"]]) for r in rows]
        truth = [r["label"] == "supported" for r in rows]
        pred = [v.supported for v in verdicts]
        cm = binary_counts(truth, pred)
        by_type = defaultdict(lambda: [0, 0])
        for r, p in zip(rows, pred):
            if r["label"] == "unsupported":
                by_type[r["error_type"]][0] += 1
                by_type[r["error_type"]][1] += int(not p)
        errors = [{"claim_id": r["claim_id"], "claim": r["claim"], "label": r["label"], "judged_supported": v.supported,
                   "score": v.score} for r, v in zip(rows, verdicts) if (r["label"] == "supported") != v.supported]
        results["judges"][judge.name] = {
            **cm, "ms_per_claim": round((time.perf_counter() - started) * 1000 / len(rows), 1),
            "hallucination_detection_rate": cm["specificity"],
            "detected_by_error_type": {k: f"{v[1]}/{v[0]}" for k, v in by_type.items()}, "errors": errors}
    return results


def system_hallucination(judge_name: str, limit: int = 0) -> dict:
    from backend.security.auth import AuthenticatedUser
    from backend.services.chat_service import ChatService

    judge = get_judge(judge_name)
    rows = list(csv.DictReader(open(ROOT / "datasets/evaluation_dataset/evaluation_dataset.csv", encoding="utf-8")))
    rows = [r for r in rows if r["expect_refusal"].lower() != "true"][: limit or None]
    svc = ChatService()
    user = AuthenticatedUser("claims_eval", {"public"})
    idx = get_index_manager()
    per_cat = defaultdict(lambda: {"answers": 0, "claims": 0, "unsupported": 0, "fabricated_urls": 0,
                                   "invalid_citations": 0})
    examples, totals = [], {"answers": 0, "claims": 0, "unsupported": 0, "fabricated_urls": 0, "invalid_citations": 0}
    for row in rows:
        resp = svc.answer(row["question"], user, persist=False)
        if resp.answer_type in REFUSAL_TYPES or resp.answer_type == "greeting":
            continue
        passages, urls = [], set()
        for item in resp.retrieved[:8]:
            pos = idx.get(item["kb"]).id_to_pos.get(item["id"])
            if pos is not None:
                ch = idx.get(item["kb"]).chunks[pos]
                passages.append(str(ch.metadata.get("answer")) if ch.metadata.get("doc_type") == "golden_qa" else ch.text)
                urls.add(str(ch.metadata.get("source_url") or ""))
        rep = evaluate_answer(resp.answer, passages, judge, allowed_urls=urls | {c.url for c in resp.citations},
                              valid_citation_ids={c.id for c in resp.citations} | {f"S{i}" for i in range(1, 9)})
        cat = per_cat[row["category"]]
        for bucket in (cat, totals):
            bucket["answers"] += 1
            bucket["claims"] += len(rep.claims)
            bucket["unsupported"] += len(rep.unsupported)
            bucket["fabricated_urls"] += len(rep.fabricated_urls)
            bucket["invalid_citations"] += len(rep.invalid_citations)
        for v in rep.unsupported[:2]:
            examples.append({"eval_id": row["eval_id"], "question": row["question"], "answer_type": resp.answer_type,
                             "claim": v.claim, "score": v.score})
    rate = lambda d: round(100 * d["unsupported"] / d["claims"], 2) if d["claims"] else 0.0  # noqa: E731
    return {"judge": judge.name, "model": svc.llm.model if svc.llm else "extractive (no LLM key)",
            "totals": {**totals, "hallucination_rate_pct": rate(totals)},
            "by_category": {k: {**v, "hallucination_rate_pct": rate(v)} for k, v in sorted(per_cat.items())},
            "examples": examples[:25]}


def to_md(val: dict, sysrep: dict | None) -> str:
    lines = ["# Hallucination evaluation", "", f"Labelled claims: {val['claims']} ({val['label_origin']}).",
             "Positive class = supported. FP = an unsupported (hallucinated) claim the judge accepted.", "",
             "## 1. Judge validation (confusion matrix vs. labelled claims)", "",
             "| Judge | TP | FN | FP | TN | Accuracy | Precision | Recall | F1 | Specificity (hallucinations caught) "
             "| ms/claim |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, j in val["judges"].items():
        if "error" in j:
            lines.append(f"| {name} | not run: {j['error'][:120]} |")
            continue
        lines.append(f"| {name} | {j['tp']} | {j['fn']} | {j['fp']} | {j['tn']} | {j['accuracy']:.3f} | "
                     f"{j['precision']:.3f} | {j['recall']:.3f} | {j['f1']:.3f} | {j['specificity']:.3f} | "
                     f"{j['ms_per_claim']} |")
    for name, j in val["judges"].items():
        if "error" in j:
            continue
        lines += ["", f"**{name}** - hallucinations caught by type: " +
                  ", ".join(f"{k} {v}" for k, v in j["detected_by_error_type"].items())]
        for e in j["errors"][:8]:
            lines.append(f"- {e['claim_id']} ({e['label']}, judged {'supported' if e['judged_supported'] else 'unsupported'},"
                         f" score {e['score']}): {e['claim']}")
    if sysrep:
        t = sysrep["totals"]
        lines += ["", "## 2. System answers", "", f"Judge: {sysrep['judge']}. Answer model: {sysrep['model']}.", "",
                  f"Answers checked: {t['answers']}, claims: {t['claims']}, unsupported: {t['unsupported']} -> "
                  f"**hallucination rate {t['hallucination_rate_pct']}%** (unsupported claims / total claims x 100). "
                  f"Fabricated URLs: {t['fabricated_urls']}. Invalid citations: {t['invalid_citations']}.", "",
                  "| Category | Answers | Claims | Unsupported | Rate % | Fabricated URLs | Invalid citations |",
                  "|---|---|---|---|---|---|---|"]
        for k, v in sysrep["by_category"].items():
            lines.append(f"| {k} | {v['answers']} | {v['claims']} | {v['unsupported']} | {v['hallucination_rate_pct']} | "
                         f"{v['fabricated_urls']} | {v['invalid_citations']} |")
        lines += ["", "### Unsupported-claim examples (judge output - review manually)", ""]
        for e in sysrep["examples"][:15]:
            lines.append(f"- [{e['eval_id']}] {e['question']} -> \"{e['claim'][:160]}\" (score {e['score']})")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judges", default="lexical,nli")
    ap.add_argument("--system-judge", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip-system", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    val = validate_judges([j.strip() for j in args.judges.split(",") if j.strip()])
    (OUT / "judge_validation.json").write_text(json.dumps(val, indent=2), encoding="utf-8")
    ok = {k: v for k, v in val["judges"].items() if "error" not in v}
    for k, v in ok.items():
        print(f"{k}: acc={v['accuracy']:.3f} precision={v['precision']:.3f} recall={v['recall']:.3f} "
              f"F1={v['f1']:.3f} specificity={v['specificity']:.3f}")
    sysrep = None
    if not args.skip_system and ok:
        chosen = args.system_judge or max(ok, key=lambda k: (ok[k]["f1"] + ok[k]["specificity"]) / 2)
        chosen_kind = "lexical" if chosen == "lexical" else ("claude" if chosen.startswith("claude") else
                                                             "nli:cross-encoder/" + chosen.split(":", 1)[1]
                                                             if chosen.startswith("nli:") else chosen)
        sysrep = system_hallucination(chosen_kind, args.limit)
        (OUT / "system_hallucination.json").write_text(json.dumps(sysrep, indent=2), encoding="utf-8")
        print(f"system hallucination rate ({sysrep['judge']}): {sysrep['totals']['hallucination_rate_pct']}% "
              f"of {sysrep['totals']['claims']} claims")
    (OUT / "report.md").write_text(to_md(val, sysrep), encoding="utf-8")
    print(f"Report: {OUT / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
