"""Corrective RAG (backend/retrieval/crag.py): relevance grading, filtering, rewrite-and-retry and fail-open."""
from __future__ import annotations

import re

import pytest

from backend.generation.llm_client import LLMResult
from backend.retrieval.crag import RelevanceGrader, apply_grades

QUESTION = "What file formats can be uploaded when issuing certificates in bulk?"


class FakeLLM:
    """Scripted LLM: `grades` is a list of grader replies used in order (dict, or Exception to raise)."""
    provider = "fake"

    def __init__(self, grades, rewrite="bulk certificate upload Excel template file format"):
        self.grades, self.rewrite_text = list(grades), rewrite
        self.grade_calls, self.rewrite_calls, self.graded_questions = 0, 0, []

    def complete_json(self, system, user, max_tokens=400):
        if "grade passage relevance" in system:
            self.grade_calls += 1
            self.graded_questions.append(re.search(r"Question: (.*)", user).group(1))
            reply = self.grades.pop(0) if self.grades else {"relevant": [1]}
            if isinstance(reply, Exception):
                raise reply
            return reply, None
        raise RuntimeError("not scripted")          # router / decompose / intent fall back to their rules

    def complete(self, system, user, max_tokens=None, temperature=None, json_mode=False):
        if "rewrite search queries" in system:
            self.rewrite_calls += 1
            return LLMResult(text=self.rewrite_text)
        if "rewrite questions" in system:
            return LLMResult(text=re.search(r"Latest message: (.*)", user).group(1))
        sid = (re.findall(r"\[(S\d+)\]", user) or ["S1"])[0]
        return LLMResult(text=f"Certificates can be uploaded in bulk using the template [{sid}].", model="fake")


# ------------------------------------------------------------------------------------------------ unit
class _Chunk:
    def __init__(self, i):
        self.id, self.text, self.metadata = f"c{i}", f"passage {i}", {"doc_name": "Manual", "section": f"S{i}"}


class _Cand:
    def __init__(self, i):
        self.chunk = _Chunk(i)
        self.id, self.meta = self.chunk.id, self.chunk.metadata


@pytest.fixture()
def grader_settings():
    from config.settings import get_settings
    return get_settings()


def test_grader_keeps_only_relevant_passages(grader_settings):
    cands = [_Cand(i) for i in range(1, 5)]
    verdicts = RelevanceGrader(FakeLLM([{"relevant": [2, "4", "S9", "x"]}]), grader_settings).grade(QUESTION, cands)
    assert verdicts == [False, True, False, True]
    kept, dropped = apply_grades(cands, verdicts)
    assert [c.id for c in kept] == ["c2", "c4"] and dropped == ["c1", "c3"]


@pytest.mark.parametrize("reply", [RuntimeError("rate limited"), {"answer": "yes"}, {"relevant": "1"}])
def test_grader_failure_returns_none_so_caller_fails_open(grader_settings, reply):
    assert RelevanceGrader(FakeLLM([reply]), grader_settings).grade(QUESTION, [_Cand(1)]) is None


def test_rewrite_rejects_empty_or_identical_output(grader_settings):
    assert RelevanceGrader(FakeLLM([], rewrite=QUESTION), grader_settings).rewrite(QUESTION) == ""
    assert RelevanceGrader(FakeLLM([], rewrite="bulk upload formats"), grader_settings).rewrite(QUESTION) == \
        "bulk upload formats"


# ------------------------------------------------------------------------------------------------ pipeline
def _service(llm):
    from backend.services.chat_service import ChatService
    return ChatService(llm=llm)


def _ask(service, user, question=QUESTION):
    return service.answer(question, user, persist=False, allowed_kbs=["products"])


def test_crag_filters_irrelevant_passages(workspace, public_user):
    llm = FakeLLM([{"relevant": [1]}])
    resp = _ask(_service(llm), public_user)
    crag = resp.route["crag"]
    assert llm.grade_calls == 1 and crag["action"] in ("filtered", "correct") and crag["relevant"] == 1
    assert crag["graded"] >= 1 and len(crag["dropped"]) == crag["graded"] - 1
    assert resp.answer_type not in ("refusal", "out_of_scope") and resp.citations


def test_crag_rewrites_and_retrieves_again_when_nothing_is_relevant(workspace, public_user):
    llm = FakeLLM([{"relevant": []}, {"relevant": [1, 2]}])
    resp = _ask(_service(llm), public_user)
    crag = resp.route["crag"]
    assert crag["action"] == "corrected" and crag["retried"] and crag["rewritten_query"]
    assert llm.rewrite_calls == 1 and llm.grade_calls == 2
    assert llm.graded_questions[1] == QUESTION        # the retry is graded against the ORIGINAL question
    assert resp.answer_type not in ("refusal", "out_of_scope")


def test_crag_refuses_when_retry_also_finds_nothing(workspace, public_user):
    llm = FakeLLM([{"relevant": []}, {"relevant": []}])
    service = _service(llm)
    resp = _ask(service, public_user)
    assert resp.route["crag"]["action"] == "incorrect"
    assert resp.answer_type == "refusal" and resp.answer == service.settings.insufficient_info_text
    assert not resp.citations


def test_crag_fails_open_when_grader_is_down(workspace, public_user):
    resp = _ask(_service(FakeLLM([RuntimeError("429 rate limit")])), public_user)
    assert resp.route["crag"]["action"] == "grader_error"
    assert resp.answer_type not in ("refusal", "out_of_scope") and resp.citations


def test_crag_can_be_disabled(workspace, public_user):
    service = _service(FakeLLM([{"relevant": []}]))
    service.settings.crag_enabled = False
    try:
        resp = _ask(service, public_user)
        assert "crag" not in resp.route and resp.answer_type not in ("refusal", "out_of_scope")
    finally:
        service.settings.crag_enabled = True


def test_greetings_never_reach_crag(workspace, public_user):
    llm = FakeLLM([])
    resp = _ask(_service(llm), public_user, "Hello")
    assert resp.answer_type == "greeting" and llm.grade_calls == 0 and "crag" not in resp.route
