"""Evaluation fact-checker (lead-in lines, close paraphrases) and run pacing under the provider's per-minute limit."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from evaluation import hallucination_metrics as hm
from evaluation.run_evaluation import _ask_paced, pace_seconds

SOURCE = ("To Edit/Change my password, Click on Settings Module and then click on Change Password. On the next page, "
          "Enter the Current Password, New Password, and Re-enter Password. Note: Password must contain minimum 6 "
          "characters. After changing the password, click on the Submit button.")


# ------------------------------------------------------------------------------------------------ lead-in lines
@pytest.mark.parametrize("claim", [
    "To reissue a single certificate",
    "To enable the Student Review Process",
    "To register new users, an admin should",
    "Follow these steps:",
    "Here are the steps",
])
def test_lead_in_lines_are_not_claims(claim):
    assert hm.is_lead_in(claim)


@pytest.mark.parametrize("claim", [
    "To verify, scan the QR code on the certificate",                   # an instruction with content
    "Click on Settings Module and then click on Change Password",
    "LearningChain stores data on Microsoft Azure",
    "The password must contain at least 6 characters, 1 upper-case letter and 1 number",
])
def test_real_claims_are_kept(claim):
    assert not hm.is_lead_in(claim)


# ------------------------------------------------------------------------------------------------ close paraphrase
def test_close_paraphrase_is_supported():
    assert hm.overlap_supported("Enter your Current Password, then type the New Password and re-enter it", [SOURCE])


@pytest.mark.parametrize("claim", [
    "Click on Settings Module and do not click on Change Password",        # negation can flip the meaning
    "Password must contain minimum 8 characters",                          # a number the source does not have
    "Open the Billing module and download the invoice as a PDF",           # different content
])
def test_overlap_shortcut_is_not_fooled(claim):
    assert not hm.overlap_supported(claim, [SOURCE])


def test_overlap_shortcut_can_be_switched_off():
    assert not hm.overlap_supported("Enter the Current Password and the New Password", [SOURCE], threshold=0)


class _StrictJudge:
    """Marks everything it sees unsupported - only claims that reach the judge can be flagged."""
    name = "strict"

    def __init__(self):
        self.seen = []

    def judge(self, claim, passages):
        self.seen.append(claim)
        return hm.UNSUPPORTED, ""


def test_analyse_skips_lead_ins_and_accepts_paraphrases():
    answer = ("To register new users, an admin should\n"
              "1. Click on Settings Module and then click on Change Password.\n"
              "2. Enter your Current Password, then type the New Password and re-enter it.\n"
              "3. The password is emailed to your manager for approval.")
    judge = _StrictJudge()
    out = hm.analyse(answer, [SOURCE], "Answer", judge=judge)
    verdicts = {c["claim"][:20]: (c["verdict"], c["judged_by"]) for c in out["claims"]}
    assert ("NOT_A_CLAIM", "lead-in") in verdicts.values()
    assert any(by == "word-overlap" for _, by in verdicts.values())
    assert judge.seen == [c["claim"] for c in out["claims"] if c["judged_by"] == "strict"]
    assert out["unsupported"] == 1 and out["total_claims"] == 3          # the lead-in is not counted


# ------------------------------------------------------------------------------------------------ pacing
def test_pace_keeps_under_the_per_minute_limit():
    assert pace_seconds(3000, 2.0, tokens_per_minute=6000) == pytest.approx(28.0)    # 30 s budget - 2 s call
    assert pace_seconds(3000, 40.0, tokens_per_minute=6000) == 0.0                  # the call itself took longer
    assert pace_seconds(0, 1.0, tokens_per_minute=6000) == 0.0                      # approved answer: no AI tokens
    assert pace_seconds(3000, 1.0, tokens_per_minute=0) == 0.0                      # pacing off


def test_ask_paced_sleeps_only_after_ai_calls():
    waits = []

    class Adapter:
        def __init__(self, tokens):
            self.tokens = tokens

        def ask(self, question, **kwargs):
            return SimpleNamespace(to_dict=lambda: {"answer": "x", "prompt_tokens": self.tokens, "completion_tokens": 0})

    _ask_paced(Adapter(6000), "q", sleep=waits.append)
    _ask_paced(Adapter(0), "q", sleep=waits.append)
    assert len(waits) == 1 and 0 < waits[0] <= 60


def test_prompt_asks_to_stay_close_to_source_wording():
    from pathlib import Path
    text = (Path(__file__).resolve().parents[1] / "prompts" / "system_prompt.txt").read_text(encoding="utf-8")
    assert "Stay close to the SOURCES' wording" in text


# ------------------------------------------------------------------------------------------------ re-check a run
def _hal(*claims):
    return {"claims": [{"claim": c, "claim_key": hm.claim_key(c), "verdict": v, "evidence": "", "judged_by": "nli",
                        "needs_human_review": v != hm.SUPPORTED} for c, v in claims],
            "total_claims": len(claims), "supported": 0, "unsupported": 0, "contradicted": 0,
            "hallucination_rate": None, "judge": "nli", "error": ""}


def test_recheck_applies_local_rules_then_the_judge():
    from evaluation.recheck_claims import recheck_question

    class Judge:
        name = "llm:test"

        def judge(self, claim, passages):
            return (hm.SUPPORTED, "quote") if "Submit" in claim else (hm.UNSUPPORTED, "")

    hal = _hal(("To register new users, an admin should", hm.CONTRADICTED),
               ("Enter your Current Password, then type the New Password and re-enter it", hm.UNSUPPORTED),
               ("Finally press the Submit button to save it", hm.UNSUPPORTED),
               ("The password is emailed to your manager for approval", hm.UNSUPPORTED),
               ("Click on Settings Module", hm.SUPPORTED))
    out, counts = recheck_question(hal, [SOURCE], Judge())
    assert counts == {"lead_in": 1, "overlap": 1, "judge_supported": 1, "judge_flagged": 1, "not_rechecked": 0}
    assert out["total_claims"] == 4 and out["unsupported"] == 1 and out["hallucination_rate"] == 0.25


def test_recheck_keeps_old_verdicts_after_a_rate_limit():
    from evaluation.recheck_claims import recheck_question

    class LimitedJudge:
        name = "llm:test"
        calls = 0

        def judge(self, claim, passages):
            LimitedJudge.calls += 1
            raise RuntimeError("Error code: 429 - rate limit reached")

    hal = _hal(("The password is emailed to your manager for approval", hm.UNSUPPORTED),
               ("Admins receive a weekly summary of password changes by post", hm.UNSUPPORTED))
    out, counts = recheck_question(hal, [SOURCE], LimitedJudge())
    assert LimitedJudge.calls == 1 and counts["not_rechecked"] == 2          # stops calling after the first 429
    assert out["unsupported"] == 2                                            # old verdicts kept, never dropped
