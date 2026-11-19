"""Conversation handling: greet / introduce once, short small talk, calm handling of frustration (support after two
in a row), one-sentence off-topic replies, short clarifications, and never the same reply twice in a row."""
from __future__ import annotations

import pytest

ALL_KBS = ["products", "company"]
EARLIER = [{"role": "user", "content": "How do I revoke a certificate?"},
           {"role": "assistant", "content": "Open the Reports module [S1].", "answer_type": "generated"}]


def ask(chat, user, text, history=None):
    return chat.answer(text, user, persist=False, allowed_kbs=ALL_KBS, history=history or [])


def test_introduction_only_on_the_first_message(chat, public_user):
    first = ask(chat, public_user, "Hi")
    again = ask(chat, public_user, "Hi", EARLIER)
    assert "Sapphirus Assistant" in first.answer
    assert "Sapphirus Assistant" not in again.answer and again.answer.startswith("Hi again")
    assert ask(chat, public_user, "Good morning", EARLIER).answer == "Good morning! How can I help?"


@pytest.mark.parametrize("text", ["thanks", "ok", "who are you", "how are you"])
def test_small_talk_is_one_short_sentence(chat, public_user, text):
    resp = ask(chat, public_user, text, EARLIER)
    assert resp.answer_type == "greeting" and len(resp.answer) <= 110
    assert resp.answer.count(". ") <= 1 and "Welcome to" not in resp.answer


@pytest.mark.parametrize("text", ["go to hell", "this is useless", "worst app", "you are stupid", "wtf"])
def test_frustration_gets_a_calm_short_reply(chat, public_user, text):
    resp = ask(chat, public_user, text)
    assert resp.answer_type == "frustration"
    assert resp.answer.startswith("Sorry you're having a frustrating time") and "?" in resp.answer
    assert not resp.citations


def test_frustration_offers_to_retry_the_last_question(chat, public_user):
    resp = ask(chat, public_user, "this is useless", EARLIER)
    assert "How do I revoke a certificate" in resp.answer and resp.route.get("uses_history")


def test_frustration_twice_in_a_row_offers_the_support_team(chat, public_user):
    first = ask(chat, public_user, "useless bot", EARLIER)
    history = EARLIER + [{"role": "user", "content": "useless bot"},
                         {"role": "assistant", "content": first.answer, "answer_type": first.answer_type}]
    second = ask(chat, public_user, "go to hell", history)
    assert second.answer_type == "frustration" and "info@sapphirus.in" in second.answer
    assert second.answer != first.answer


def test_real_questions_mentioning_frustration_are_answered(chat, public_user):
    resp = ask(chat, public_user, "How do I fix a frustrating login error?")
    assert resp.answer_type != "frustration"


def test_off_topic_is_one_sentence_with_one_suggestion(chat, public_user):
    resp = ask(chat, public_user, "Tell me a joke.")
    assert resp.answer_type == "out_of_scope"
    assert resp.answer.startswith("I can only help with Sapphirus Systems and LearningChain")
    assert resp.answer.count(". ") == 0 and "for example" in resp.answer


def test_unclear_gets_one_short_question_without_introduction(chat, public_user):
    resp = ask(chat, public_user, "What?", EARLIER)
    assert resp.answer_type == "clarification" and resp.answer.endswith("?")
    assert "Sapphirus Assistant" not in resp.answer and len(resp.answer) < 200


@pytest.mark.parametrize("text", ["Tell me a joke.", "What uptime SLA does LearningChain guarantee?", "What?", "Hi"])
def test_never_the_same_reply_twice_in_a_row(chat, public_user, text):
    first = ask(chat, public_user, text, EARLIER)
    history = EARLIER + [{"role": "user", "content": text},
                         {"role": "assistant", "content": first.answer, "answer_type": first.answer_type}]
    second = ask(chat, public_user, text, history)
    assert second.answer.strip() != first.answer.strip(), (first.answer_type, first.answer)
