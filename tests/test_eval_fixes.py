"""Fixes from the 23 Nov 2026 full evaluation: over-strict undocumented-term guard, broad topic requests, comparison
retrieval, revoke vs recall in the FAQ."""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from backend.common.english_vocab import common_words, is_common_word
from backend.routing.decompose import comparison_split, decompose
from backend.routing.intent import broad_topic

ROOT = Path(__file__).resolve().parents[1]
HISTORY = [{"role": "user", "content": "How do I generate a certificate?"},
           {"role": "assistant", "content": "Go to Certificates > Generate Single ..."}]


# ------------------------------------------------------------------ 1. undocumented-term guard: everyday words pass
needs_vocab = pytest.mark.skipif(not common_words(), reason="reranker vocabulary not available offline")


@needs_vocab
@pytest.mark.parametrize("word", ["ordinary", "forgot", "faked", "recalling"])
def test_everyday_words_are_common(word):
    assert is_common_word(word)


@needs_vocab
@pytest.mark.parametrize("word", ["uptime", "aadhaar", "sla"])
def test_specific_terms_are_not_common(word):
    assert not is_common_word(word)


@needs_vocab
def test_guard_ignores_everyday_words_but_catches_names(chat):
    terms = chat._undocumented_terms
    assert terms("Why is a blockchain QR code more secure than an ordinary QR code?",
                 "Unlike an ordinary QR code, it is checked against the blockchain.") == []
    assert terms("What if I forgot my password?", "If you forgot your password, click Forget password.") == []
    assert terms("Does LearningChain have an iOS app?", "Yes, there is an iOS app.") == ["ios"]
    assert terms("does learningchain have an ios app?", "Yes, there is an ios app.") == ["ios"]   # lower case too
    assert "uptime" in terms("What uptime SLA is guaranteed?", "The uptime SLA is 99.9%.")


# ------------------------------------------------------------------ 2. broad topic requests -> clarifying question
@pytest.mark.parametrize("text,topic", [
    ("Tell me about the review.", "review"), ("What about the fees?", "fees"),
    ("Can you tell me more about verification", "verification"), ("info on student review", "student review")])
def test_broad_topic_detected(text, topic):
    assert broad_topic(text, []) == topic


@pytest.mark.parametrize("text", ["How do I enable the student review process?", "What is the verification fee?",
                                  "Tell me about the review process for bulk certificates in detail"])
def test_specific_questions_are_not_broad(text):
    assert broad_topic(text, []) == ""


def test_broad_topic_after_an_answer_is_a_follow_up():
    assert broad_topic("What about the fees?", HISTORY) == ""


def test_broad_topic_gets_clarification_with_options(chat, public_user):
    resp = chat.answer("Tell me about the review.", public_user, persist=False)
    assert resp.answer_type == "clarification"
    assert "review" in resp.answer.lower() and resp.answer.count("\n- ") >= 2


def test_named_topic_still_answered(chat, public_user):
    resp = chat.answer("Tell me about LearningChain", public_user, persist=False)
    assert resp.answer_type not in ("clarification", "refusal", "out_of_scope")


# ------------------------------------------------------------------ 3. comparison: both sides retrieved
@pytest.mark.parametrize("question,sides", [
    ("What is the difference between revoking and recalling a certificate in LearningChain?",
     ["Revoking a certificate in LearningChain", "Recalling a certificate in LearningChain"]),
    ("What is the difference between LearningChain and DigiLocker?", ["LearningChain", "DigiLocker"]),
    ("Compare DigiLocker with LearningChain", ["DigiLocker", "LearningChain"]),
    ("How does DigiLocker differ from LearningChain?", ["DigiLocker", "LearningChain"]),
])
def test_comparison_split(question, sides):
    assert comparison_split(question) == sides
    assert decompose(question) == sides


def test_non_comparison_not_split():
    assert comparison_split("How do I recall a certificate?") == []


def test_revoke_vs_recall_retrieves_both_articles(chat, public_user):
    # the tests run without an LLM, so this checks retrieval: both documented actions reach the top passages
    resp = chat.answer("What is the difference between revoking and recalling a certificate?", public_user,
                       persist=False)
    sections = " ".join(str(r.get("section")) for r in resp.retrieved[:6])
    assert "Revoke the Certificates" in sections and "Recall the Certificates" in sections


# ------------------------------------------------------------------ 4. FAQ: revoke is not an alternate of recall
def test_recall_faq_has_no_revoke_alternate():
    with open(ROOT / "datasets" / "learningchain_faq_master.csv", encoding="utf-8-sig") as f:
        row = next(r for r in csv.DictReader(f) if r["Question ID"] == "LC-090")
    assert "revoke" not in row["Alternate Questions"].lower()
    assert "revoke" not in row["Keywords"].lower()
