"""Understanding the user: opinions / worries are not account questions, "any alternative" continues the previous
topic, and opinion / comparison questions get a reply to the point instead of the bare "no confirmed information"."""
from __future__ import annotations

import pytest

from backend.routing import intent as convo

HISTORY = [{"role": "user", "content": "any negatives of using this application"},
           {"role": "assistant", "content": "I don't have a published list of drawbacks or limitations ..."}]


# ------------------------------------------------------------------------------------------------ account check
@pytest.mark.parametrize("text", [
    "i assume lot of problems using this application",
    "I think it is hard to use",
    "Can I use it on my phone?",
    "any negatives of using this application",
])
def test_opinions_and_general_questions_are_not_account_questions(text):
    assert not convo.mentions_own_account(text)


@pytest.mark.parametrize("text", [
    "Has my payment gone through?", "our verification status is pending", "I paid twice",
    "money got deducted from my account", "where is my receipt",
])
def test_own_account_questions_still_detected(text):
    assert convo.mentions_own_account(text)


# ------------------------------------------------------------------------------------------------ opinion kinds
@pytest.mark.parametrize("text, kind", [
    ("any alternative", "alternatives"),
    ("are there other tools like this?", "alternatives"),
    ("who are your competitors", "alternatives"),
    ("any negatives of using this application", "limitations"),
    ("what are the disadvantages", "limitations"),
    ("cons?", "limitations"),
    ("i assume lot of problems using this application", "concern"),
    ("I'm worried it will be complicated", "concern"),
    ("How do I verify a certificate?", ""),
    ("What is the fee for verification?", ""),
    ("The verification result shows an error", ""),          # a support question, not a worry
])
def test_opinion_kind(text, kind):
    assert convo.opinion_kind(text) == kind


def test_opinion_replies_name_the_product_and_support_and_invent_nothing(workspace):
    from backend.products import current_product

    product = current_product()
    for kind in convo.OPINION_KINDS:
        reply = convo.opinion_reply(kind)
        assert product.covers_text("and") in reply and product.escalation.email in reply
        assert "{" not in reply
        low = reply.lower()
        assert "no other alternative" not in low and "no problems" not in low and "no limitations" not in low


# ------------------------------------------------------------------------------------------------ follow-ups
@pytest.mark.parametrize("text", ["any alternative", "any alternatives?", "any negatives", "what else",
                                  "any other option", "cons?"])
def test_short_messages_continue_the_previous_topic(text):
    assert convo.classify_message(text, HISTORY).intent == convo.FOLLOW_UP


def test_without_history_they_are_ordinary_questions():
    assert convo.classify_message("any alternative", []).intent != convo.FOLLOW_UP


def test_full_questions_are_not_treated_as_follow_ups():
    assert not convo.is_elliptical_follow_up("How do I verify a certificate on the portal?")


# ------------------------------------------------------------------------------------------------ end to end
@pytest.mark.parametrize("message, kind", [
    ("any negatives of using this application", "limitations"),
    ("i assume lot of problems using this application", "concern"),
    ("any alternative", "alternatives"),
])
def test_chat_answers_the_point_instead_of_the_fixed_sentence(chat, public_user, message, kind):
    from config.settings import get_settings

    resp = chat.answer(message, public_user, persist=False, history=HISTORY)
    assert resp.answer_type != "account"                                  # never the "your account details" reply
    # nothing documented (or outside the documentation) -> the tailored reply, never the generic sentence
    if resp.answer_type in ("refusal", "out_of_scope"):
        assert resp.route.get("refusal_kind") == kind
        assert resp.answer != get_settings().out_of_scope_text
        assert resp.answer == convo.opinion_reply(kind)
        assert resp.answer != get_settings().insufficient_info_text


def test_unrelated_refusals_keep_the_fixed_sentence(chat, public_user):
    from config.settings import get_settings

    resp = chat.answer("What uptime SLA do you guarantee for the quantum module?", public_user, persist=False,
                       history=[])
    if resp.answer_type == "refusal":
        assert resp.answer == get_settings().insufficient_info_text and not resp.route.get("refusal_kind")


def test_prompt_has_understanding_rules():
    from pathlib import Path

    text = (Path(__file__).resolve().parents[1] / "prompts" / "system_prompt.txt").read_text(encoding="utf-8")
    assert "UNDERSTANDING THE USER" in text and "2b. Exception - opinion" in text


def test_off_topic_requests_keep_the_generic_decline(chat, public_user):
    from config.settings import get_settings

    resp = chat.answer("tell me a joke", public_user, persist=False, history=[])
    if resp.answer_type == "out_of_scope":
        assert not resp.route.get("refusal_kind")


# ------------------------------------------------------------------------------------------------ evaluation: Engage
@pytest.mark.parametrize("run, ok", [
    ({"behavior": "Abstain", "answer_type": "refusal", "route": {"refusal_kind": "limitations"}}, True),
    ({"behavior": "Reject", "answer_type": "out_of_scope", "route": {"refusal_kind": "alternatives"}}, True),
    ({"behavior": "Answer", "answer_type": "generated", "route": {}}, True),
    ({"behavior": "Abstain", "answer_type": "account", "route": {}}, False),        # the old wrong reply
    ({"behavior": "Abstain", "answer_type": "refusal", "route": {}}, False),        # the bare fixed sentence
    ({"behavior": "Error", "answer_type": "error", "route": {}}, False),
])
def test_engage_expected_behaviour(run, ok):
    from evaluation import generation_metrics as gm

    assert gm.behavior_correct("Engage", run["behavior"], run) is ok


def test_dataset_has_the_conversational_questions():
    from evaluation import dataset_manager as dm

    df = dm.load()
    rows = df[df["question_category"] == "Conversational"]
    assert set(rows["question_id"]) == {"CV1", "CV2", "CV3"} and set(rows["expected_behavior"]) == {"Engage"}
    assert not dm.validate(df)[0]


def test_model_refusal_of_an_opinion_question_gets_the_tailored_reply(workspace, public_user, monkeypatch):
    """Passages were found and the AI model itself wrote the "no confirmed information" sentence (the deployed bot
    did this for "any negatives of using this application") - the user still gets the reply to the point."""
    import json

    from backend.generation.llm_client import LLMClient, LLMResult
    from backend.services.chat_service import ChatService
    from config.settings import get_settings

    settings = get_settings()

    class RefusingLLM(LLMClient):
        provider, model = "openai", "refusing"

        def complete(self, system, user, max_tokens=None, temperature=None, json_mode=False):
            if "Return ONLY this JSON" in user or "Classify" in user or "clean up user messages" in user:
                return LLMResult(text="{}", model=self.model, provider="openai")
            return LLMResult(text=json.dumps({"reply": settings.insufficient_info_text, "ui_action": "none",
                                              "conversation_state": "active"}),
                             prompt_tokens=10, completion_tokens=5, model=self.model, provider="openai")

    chat = ChatService(llm=RefusingLLM(settings))
    monkeypatch.setattr(chat, "_insufficient", lambda *a, **k: False)       # passages found -> the model is asked
    resp = chat.answer("any negatives of using this application", public_user, persist=False, history=[])
    assert resp.answer_type == "refusal" and resp.route.get("refusal_kind") == "limitations"
    assert resp.answer == convo.opinion_reply("limitations")
