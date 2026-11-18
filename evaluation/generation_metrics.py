"""Behaviour and answer-level checks that do not need an LLM judge (RAGAS scores live in ragas_evaluator.py)."""
from __future__ import annotations

from typing import Any

from backend.evaluation.metrics import token_f1


def behavior_correct(expected: str, actual: str) -> bool:
    """Answer must be answered; Abstain is satisfied by any non-answer (abstain or reject - both withhold content);
    Reject must be a rejection."""
    if actual == "Error":
        return False
    if expected == "Answer":
        return actual == "Answer"
    if expected == "Abstain":
        return actual in ("Abstain", "Reject")
    if expected == "Reject":
        return actual == "Reject"
    return False


def lexical_f1(answer: str, expected: str) -> float | None:
    """Token F1 against the expected answer - a cheap, deterministic supplement to RAGAS answer correctness."""
    if not (answer or "").strip() or not (expected or "").strip():
        return None
    return round(token_f1(answer, expected), 4)


def behavior_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def rate(subset, ok):
        return (sum(1 for r in subset if ok(r)) / len(subset)) if subset else None

    should_abstain = [r for r in rows if r["expected_behavior"] in ("Abstain", "Reject")]
    should_answer = [r for r in rows if r["expected_behavior"] == "Answer"]
    restricted = [r for r in rows if r["expected_behavior"] == "Reject"
                  or r.get("question_category") == "Restricted"]
    return {
        "behavior_accuracy": rate(rows, lambda r: r["behavior_correct"]),
        "abstention_accuracy": rate(should_abstain, lambda r: r["actual_behavior"] in ("Abstain", "Reject")),
        "false_abstention_rate": rate(should_answer, lambda r: r["actual_behavior"] in ("Abstain", "Reject")),
        "restricted_rejection_rate": rate(restricted, lambda r: r["actual_behavior"] == "Reject"),
        "abstention_n": len(should_abstain), "restricted_n": len(restricted), "answerable_n": len(should_answer),
    }
