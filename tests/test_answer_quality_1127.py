"""27 Nov fixes toward 132/132: paraphrase-tolerant grounding, trimming one unsupported sentence instead of discarding
the answer, wrong-assumption retry, "any alternative" on its own, per-cell table claims, file-type synonyms and a
thread-safe database start-up."""
from __future__ import annotations

import json
import threading

import pytest

from backend.generation.grounding import GroundingReport, lexical_grounding, trim_unsupported
from backend.generation.prompt_builder import SourceRef
from backend.retrieval.hybrid import Candidate
from ingestion.vector_store import StoredChunk

REVOKE_RECALL = (
    "To Revoke the certificate, Click on Reports Module and then click on Generated Certificates tab. Now, select the "
    "student and click on Revoke under Actions. After the certificate is Revoked we cannot do the verification. "
    "To Recall the certificate, Click on Reports Module and then click on Generated Certificates. Select the student "
    "and click on Recall under Actions. After the certificate is Recalled, verification is possible again and a "
    "successful message is displayed.")


def _ref(text: str, sid: str = "S1") -> SourceRef:
    cand = Candidate(chunk=StoredChunk(id=sid, text=text, metadata={"doc_name": "Doc"}), kb="products")
    return SourceRef(sid=sid, candidate=cand, metadata=cand.chunk.metadata)


# ------------------------------------------------------------------------------------------------ grounding
def test_paraphrase_with_other_word_forms_is_grounded():
    answer = ("Revoking a certificate permanently disables its verification - once revoked, it cannot be checked "
              "any more [S1].\n\nRecalling a certificate removes the revocation, so the certificate becomes verifiable "
              "again and a success message is shown [S1].")
    assert lexical_grounding(answer, [_ref(REVOKE_RECALL)], "difference between revoking and recalling").supported


def test_unrelated_sentence_is_still_unsupported():
    answer = "The certificate is mailed to the student's parents by courier within five working days [S1]."
    report = lexical_grounding(answer, [_ref(REVOKE_RECALL)], "how do I recall a certificate")
    assert not report.supported and report.unsupported_sentences


def test_trim_keeps_the_supported_part():
    answer = ("The platform does **not** have access to or control over client data [S1].\n\nIf you need to update "
              "your own data, please log in to your account and use the appropriate data-management tools.")
    report = GroundingReport(faithfulness=0.5, supported=False, unsupported_sentences=[
        "If you need to update your own data, please log in to your account and use the appropriate data-management "
        "tools."])
    assert trim_unsupported(answer, report) == ("The platform does **not** have access to or control over client "
                                                "data [S1].")


def test_trim_gives_up_when_nothing_useful_is_left():
    answer = "Please contact your administrator for help."
    report = GroundingReport(faithfulness=0.0, supported=False, unsupported_sentences=[answer])
    assert trim_unsupported(answer, report) == ""


# ------------------------------------------------------------------------------------------------ wrong assumptions
@pytest.mark.parametrize("text, expected", [
    ("Which of Sapphirus's offices in Bangalore handles support?", True),
    ("Since the data is stored on AWS, which AWS region is used?", True),
    ("As the platform can edit our data, how do we request changes?", True),
    ("Does it support Aadhaar-based identity verification?", False),       # must still be declined
    ("What uptime SLA do you guarantee?", False),
    ("As an admin, how do I revoke a certificate?", False),
    ("Which of these plans is cheaper?", False),
])
def test_premise_detection(text, expected):
    from backend.routing import intent as convo

    assert convo.has_premise(text) is expected


def test_wrong_assumption_is_corrected_on_retry(workspace, public_user, monkeypatch):
    from backend.generation.llm_client import LLMClient, LLMResult
    from backend.services.chat_service import ChatService
    from config.settings import get_settings

    settings = get_settings()
    calls = []

    class FirstDeclinesLLM(LLMClient):
        provider, model = "openai", "first-declines"

        def complete(self, system, user, max_tokens=None, temperature=None, json_mode=False):
            if "Return ONLY this JSON" in user or "Classify" in user or "clean up user messages" in user:
                return LLMResult(text="{}", model=self.model, provider="openai")
            calls.append(user)
            reply = settings.insufficient_info_text if len(calls) == 1 else \
                "The listed offices are in Hyderabad and Pune [S1] - Bangalore is not among them."
            return LLMResult(text=json.dumps({"reply": reply, "ui_action": "none", "conversation_state": "active"}),
                             prompt_tokens=10, completion_tokens=5, model=self.model, provider="openai")

    chat = ChatService(llm=FirstDeclinesLLM(settings))
    monkeypatch.setattr(chat, "_insufficient", lambda *a, **k: False)
    monkeypatch.setattr(chat.settings, "grounding_mode", "off")
    resp = chat.answer("Which of your offices in Bangalore handles support?", public_user, persist=False, history=[])
    assert len(calls) == 2 and "NOTE: the question" in calls[1]
    assert resp.answer_type == "generated" and "Bangalore is not among them" in resp.answer
    assert any(w.startswith("wrong assumption corrected on retry") for w in resp.warnings)


def test_no_retry_for_a_plain_question_the_sources_do_not_answer(workspace, public_user, monkeypatch):
    from backend.generation.llm_client import LLMClient, LLMResult
    from backend.services.chat_service import ChatService
    from config.settings import get_settings

    settings = get_settings()
    calls = []

    class DecliningLLM(LLMClient):
        provider, model = "openai", "declining"

        def complete(self, system, user, max_tokens=None, temperature=None, json_mode=False):
            if "Return ONLY this JSON" in user or "Classify" in user or "clean up user messages" in user:
                return LLMResult(text="{}", model=self.model, provider="openai")
            calls.append(user)
            return LLMResult(text=json.dumps({"reply": settings.insufficient_info_text, "ui_action": "none",
                                              "conversation_state": "active"}), model=self.model, provider="openai")

    chat = ChatService(llm=DecliningLLM(settings))
    monkeypatch.setattr(chat, "_insufficient", lambda *a, **k: False)
    resp = chat.answer("Does it support Aadhaar-based identity verification?", public_user, persist=False, history=[])
    assert len(calls) <= 1 and resp.answer_type == "refusal"


# ------------------------------------------------------------------------------------------------ "any alternative"
def test_opinion_question_alone_is_not_sent_to_clarification():
    from backend.routing import intent as convo

    class SaysUnclear:
        def complete_json(self, *a, **k):
            return {"intent": "CLARIFICATION_REQUIRED"}, None

    for text in ("any alternative", "any negatives", "is it worth it"):
        assert convo.classify_message(text, [], llm=SaysUnclear()).intent == convo.KNOWLEDGE, text


def test_any_alternative_alone_gets_the_reply_to_the_point(chat, public_user):
    from backend.routing import intent as convo

    resp = chat.answer("any alternative", public_user, persist=False, history=[])
    assert resp.answer_type in ("refusal", "out_of_scope") and resp.route.get("refusal_kind") == "alternatives"
    assert resp.answer == convo.opinion_reply("alternatives")


# ------------------------------------------------------------------------------------------------ evaluation claims
def test_table_rows_become_one_claim_per_cell():
    from backend.evaluation.claims import split_claims

    answer = ("| Aspect | Product A | Product B |\n|---|---|---|\n"
              "| Storage | Centralized database, prone to failure | Decentralized database, immutable |\n")
    claims = split_claims(answer)
    assert claims == ["Product A - Storage: Centralized database, prone to failure",
                      "Product B - Storage: Decentralized database, immutable"]


# ------------------------------------------------------------------------------------------------ retrieval
def test_a_named_file_type_also_searches_the_supported_formats(workspace):
    from backend.routing.router import QueryRouter
    from config.settings import get_settings

    expanded = QueryRouter(get_settings(), None, llm=None).expand("How do I upload Word documents?")
    assert "file formats supported" in expanded


# ------------------------------------------------------------------------------------------------ start-up
def test_concurrent_database_start_up_is_safe(workspace):
    from backend.db import session

    session.reset_engine()
    engines, errors = [], []

    def start():
        try:
            engines.append(session.get_engine())
        except Exception as exc:          # the warm-up thread used to crash here (KeyError: 'config')
            errors.append(exc)

    threads = [threading.Thread(target=start) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors and len({id(e) for e in engines}) == 1


@pytest.mark.parametrize("answer, ok", [
    ("The listed offices are in Hyderabad and Pune [S1] - bangalore is not among them.", True),
    ("The data is stored on Microsoft Azure, not bangalore [S1].", True),
    ("There is no bangalore office [S1].", False),            # a bare negative is not a correction
    ("bangalore is not supported [S1].", False),
    ("The offices are in Hyderabad - bangalore is not among them.", False),   # no source tag
])
def test_what_counts_as_correcting_an_assumption(answer, ok):
    from backend.services.chat_service import ChatService

    assert ChatService._only_corrected("bangalore", answer.lower()) is ok
