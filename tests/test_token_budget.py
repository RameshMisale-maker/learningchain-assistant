"""Token budget per user and per product (TOKEN_BUDGET_ENABLED) and the token / budget metrics."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from backend.generation.llm_client import LLMClient, LLMResult


class CountingLLM(LLMClient):
    provider = "openai"

    def __init__(self, settings):
        super().__init__(settings)
        self.calls, self.model = 0, "counting"

    def complete(self, system, user, max_tokens=None, temperature=None, json_mode=False):
        self.calls += 1
        if "Classify" in user or "clean up user messages" in user or "Return ONLY this JSON" in user:
            return LLMResult(text="{}", model=self.model, provider="openai")
        return LLMResult(text=json.dumps({"reply": "To verify a certificate, open the Verification URL [S1].",
                                          "ui_action": "none", "conversation_state": "active"}),
                         prompt_tokens=10, completion_tokens=5, model=self.model, provider="openai")


@pytest.fixture()
def budget(workspace, monkeypatch):
    from config.settings import get_settings
    s = get_settings()
    monkeypatch.setattr(s, "token_budget_enabled", True)
    monkeypatch.setattr(s, "token_budget_user_tokens", 1000)
    monkeypatch.setattr(s, "token_budget_product_tokens", 5000)
    monkeypatch.setattr(s, "token_budget_product_limits", "")
    monkeypatch.setattr(s, "token_budget_window_hours", 24.0)
    monkeypatch.setattr(s, "token_budget_exempt_users", [])
    return s


def _spend(username, product_id, tokens, hours_ago=0.0):
    from backend.db import repository
    from backend.db.models import RequestMetric
    from backend.db.session import db_session

    row_id = repository.write_request_metric(username=username, product_id=product_id, input_tokens=tokens,
                                             output_tokens=0, channel="test")
    if hours_ago:
        with db_session() as s:
            s.get(RequestMetric, row_id).created_at = datetime.now(timezone.utc) - timedelta(hours=hours_ago)


# ------------------------------------------------------------------------------------------------ check()
def test_disabled_budget_always_allows(workspace):
    from backend.services import token_budget
    _spend("budget_off_user", "budget_p0", 10_000_000)
    assert token_budget.check("budget_off_user", "budget_p0").allowed


def test_user_over_budget_is_blocked(budget):
    from backend.services import token_budget
    _spend("budget_user_a", "budget_p1", 600)
    assert token_budget.check("budget_user_a", "budget_p1").allowed            # 600 < 1000
    _spend("budget_user_a", "budget_p1", 500)
    c = token_budget.check("budget_user_a", "budget_p1")
    assert not c.allowed and c.scope == "user" and c.used == 1100 and c.limit == 1000


def test_user_budget_counts_all_products(budget):
    from backend.services import token_budget
    _spend("budget_user_b", "budget_p2", 600)
    _spend("budget_user_b", "budget_p3", 600)
    assert token_budget.check("budget_user_b", "budget_p3").scope == "user"


def test_product_over_budget_blocks_every_user(budget):
    from backend.services import token_budget
    for i in range(6):                                   # six users, each under their own budget
        _spend(f"budget_many_{i}", "budget_p4", 900)
    c = token_budget.check("budget_new_user", "budget_p4")
    assert not c.allowed and c.scope == "product" and c.used == 5400 and c.limit == 5000


def test_per_product_override(budget, monkeypatch):
    from backend.services import token_budget
    monkeypatch.setattr(budget, "token_budget_product_limits", "budget_p5=100, other=9")
    assert token_budget.product_limit("budget_p5") == 100 and token_budget.product_limit("budget_p6") == 5000
    _spend("budget_user_c", "budget_p5", 150)
    assert token_budget.check("budget_user_d", "budget_p5").scope == "product"


def test_tokens_outside_the_window_do_not_count(budget):
    from backend.services import token_budget
    _spend("budget_user_e", "budget_p7", 5000, hours_ago=30)    # yesterday
    _spend("budget_user_e", "budget_p7", 100)
    assert token_budget.tokens_used(username="budget_user_e") == 100
    assert token_budget.check("budget_user_e", "budget_p7").allowed


def test_exempt_user_is_never_blocked(budget, monkeypatch):
    from backend.services import token_budget
    monkeypatch.setattr(budget, "token_budget_exempt_users", ["evaluation_public"])
    _spend("evaluation_public", "budget_p8", 99_999)
    assert token_budget.check("evaluation_public", "budget_p8").allowed


def test_zero_limit_means_no_limit(budget, monkeypatch):
    from backend.services import token_budget
    monkeypatch.setattr(budget, "token_budget_user_tokens", 0)
    monkeypatch.setattr(budget, "token_budget_product_tokens", 0)
    _spend("budget_user_f", "budget_p9", 10_000_000)
    assert token_budget.check("budget_user_f", "budget_p9").allowed


def test_budget_reply_names_the_scope_and_support():
    from backend.services.token_budget import BudgetCheck, budget_reply
    text = budget_reply(BudgetCheck(False, "user", 1100, 1000, 24.0), "email help@example.com")
    assert "your account" in text and "24 hours" in text and "help@example.com" in text
    assert "this assistant" in budget_reply(BudgetCheck(False, "product", 1, 1, 24.0), "x")


# ------------------------------------------------------------------------------------------------ in the chat flow
def test_suspended_request_sees_no_llm(workspace):
    from backend.routing.router import QueryRouter
    from backend.services.token_budget import llm_suspended, suspend_llm
    router = QueryRouter(llm="fake-llm")
    with suspend_llm(True):
        assert llm_suspended() and router.llm is None
    assert not llm_suspended() and router.llm == "fake-llm"


def test_over_budget_question_makes_no_llm_call(budget, public_user):
    from backend.services.chat_service import ChatService
    from config.settings import get_settings

    llm = CountingLLM(get_settings())
    chat = ChatService(llm=llm)
    question = "How do I verify a student certificate as an employer?"
    _spend(public_user.username, "budget_other", 2000)                      # the user's budget is used up
    resp = chat.answer(question, public_user, persist=False)
    assert llm.calls == 0                                                    # the shared quota was not touched
    assert resp.route["token_budget"]["allowed"] is False and resp.route["token_budget"]["scope"] == "user"
    assert resp.answer_type in ("budget_exceeded", "golden_direct", "golden_semantic")
    if resp.answer_type == "budget_exceeded":
        assert "limit" in resp.answer and not resp.citations


def test_within_budget_question_uses_the_llm(budget, monkeypatch, employee):
    from backend.services.chat_service import ChatService
    from config.settings import get_settings

    monkeypatch.setattr(budget, "token_budget_user_tokens", 10_000_000)
    monkeypatch.setattr(budget, "token_budget_product_tokens", 0)
    chat = ChatService(llm=CountingLLM(get_settings()))
    resp = chat.answer("How do I verify a student certificate as an employer?", employee, persist=False)
    assert "token_budget" not in resp.route


# ------------------------------------------------------------------------------------------------ metrics
def test_budget_block_and_token_window_metrics(budget, monkeypatch):
    from backend.observability import metrics
    monkeypatch.setattr(budget, "metrics_enabled", True)
    metrics.reset()
    metrics.record_budget_block("budget_p10", "user")
    _spend("budget_user_g", "budget_p10", 321)
    text = metrics.render()
    assert 'assistant_token_budget_blocks_total{product="budget_p10",scope="user"} 1' in text
    assert 'assistant_tokens_window{product="budget_p10"} 321' in text
    assert 'assistant_token_budget_product{product="budget_p10"} 5000' in text
