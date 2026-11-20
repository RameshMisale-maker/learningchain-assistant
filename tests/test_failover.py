"""Phase 5: provider failover - circuit breaker, provider chain, static fallback reply, BM25-only search."""
from __future__ import annotations

import pytest

from backend.generation.failover import AllProvidersDown, CircuitBreaker, FailoverLLM, ResilientEmbedder
from backend.generation.llm_client import LLMClient, LLMRefusal, LLMResult


class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


class FakeLLM(LLMClient):
    def __init__(self, settings, name, fail=False, refuse=False):
        super().__init__(settings)
        self.name, self.fail, self.refuse, self.calls, self.model = name, fail, refuse, 0, f"{name}-model"

    def complete(self, system, user, max_tokens=None, temperature=None, json_mode=False):
        self.calls += 1
        if self.refuse:
            raise LLMRefusal("refusal")
        if self.fail:
            raise TimeoutError("provider timed out")
        return LLMResult(text=f"answer from {self.name} [S1]", prompt_tokens=10, completion_tokens=5, model=self.model,
                         provider="openai")


@pytest.fixture()
def fast_settings():
    from config.settings import Settings
    s = Settings()
    s.breaker_failures, s.breaker_window_seconds, s.breaker_cooldown_seconds = 3, 60, 120
    return s


# ------------------------------------------------------------------------------------------------ breaker
def test_breaker_opens_after_failures_and_half_opens_after_cooldown():
    clock = FakeClock()
    b = CircuitBreaker("x", failures=3, window_seconds=60, cooldown_seconds=120, clock=clock)
    for _ in range(2):
        b.record_failure()
    assert b.state == "closed" and b.allow()
    b.record_failure()
    assert b.state == "open" and not b.allow()
    clock.t += 121
    assert b.state == "half_open"
    assert b.allow() and not b.allow()              # exactly one trial call
    b.record_failure()                              # trial failed -> open again
    assert b.state == "open"
    clock.t += 121
    assert b.allow()
    b.record_success()
    assert b.state == "closed"


def test_failures_outside_the_window_do_not_open():
    clock = FakeClock()
    b = CircuitBreaker("x", failures=3, window_seconds=60, cooldown_seconds=120, clock=clock)
    for _ in range(5):
        b.record_failure()
        clock.t += 31                               # never 3 failures within 60 s
    assert b.state == "closed"


# ------------------------------------------------------------------------------------------------ provider chain
def test_chain_fails_over_and_skips_an_open_breaker(fast_settings):
    primary, secondary = FakeLLM(fast_settings, "groq", fail=True), FakeLLM(fast_settings, "secondary")
    llm = FailoverLLM(fast_settings, [("openai", primary), ("secondary", secondary)])
    for _ in range(3):
        result = llm.complete("s", "u")
        assert result.text.startswith("answer from secondary") and result.provider_name == "secondary"
    assert llm.breakers["openai"].state == "open"
    llm.complete("s", "u")
    assert primary.calls == 3                       # open breaker: primary is no longer called
    assert {s["name"]: s["state"] for s in llm.status()}["llm:openai"] == "open"


def test_refusal_is_not_an_outage(fast_settings):
    refusing, other = FakeLLM(fast_settings, "a", refuse=True), FakeLLM(fast_settings, "b")
    llm = FailoverLLM(fast_settings, [("a", refusing), ("b", other)])
    with pytest.raises(LLMRefusal):
        llm.complete("s", "u")
    assert other.calls == 0 and llm.breakers["a"].state == "closed"


def test_all_down_raises(fast_settings):
    llm = FailoverLLM(fast_settings, [("a", FakeLLM(fast_settings, "a", fail=True)),
                                      ("b", FakeLLM(fast_settings, "b", fail=True))])
    with pytest.raises(AllProvidersDown):
        llm.complete("s", "u")


def test_stream_fails_over_before_the_first_token(fast_settings):
    llm = FailoverLLM(fast_settings, [("a", FakeLLM(fast_settings, "a", fail=True)), ("b", FakeLLM(fast_settings, "b"))])
    items = list(llm.stream("s", "u"))
    assert items[0].startswith("answer from b") and items[-1].provider_name == "b"


def test_flag_off_keeps_the_single_provider(workspace):
    from backend.generation import llm_client
    from config.settings import get_settings

    assert not get_settings().llm_failover_enabled
    llm_client.get_llm.cache_clear()
    assert not isinstance(llm_client.get_llm(), FailoverLLM)


# ------------------------------------------------------------------------------------------------ integration
def test_all_providers_down_gives_static_reply_with_sources(workspace, public_user, fast_settings):
    from backend.products import get_product
    from backend.services.chat_service import ChatService

    down = FailoverLLM(fast_settings, [("a", FakeLLM(fast_settings, "a", fail=True))])
    service = ChatService(llm=down)
    resp = service.answer("How do I change my password?", public_user, persist=False)
    product = get_product(None)
    assert resp.answer_type == "service_unavailable", resp.answer
    assert product.escalation.email in resp.answer and "llm_unavailable" in resp.warnings
    assert resp.citations                             # the best sources are still linked

    working = FailoverLLM(fast_settings, [("a", FakeLLM(fast_settings, "a", fail=True)),
                                          ("secondary", FakeLLM(fast_settings, "secondary"))])
    resp = ChatService(llm=working).answer("How do I change my password?", public_user, persist=False)
    assert resp.route.get("provider") == "secondary"


def test_embedding_outage_falls_back_to_bm25(workspace, public_user, monkeypatch):
    from backend.generation.failover import dense_search_down
    from backend.services.chat_service import ChatService
    from config.settings import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "llm_failover_enabled", True)
    service = ChatService(llm=None)
    assert isinstance(service.embedder, ResilientEmbedder)

    def broken(text):
        raise ConnectionError("embedding API down")
    monkeypatch.setattr(service.embedder._inner, "embed_query", broken)
    resp = service.answer("How do I change my password?", public_user, persist=False)
    assert dense_search_down()
    assert resp.answer_type not in ("error",) and resp.citations
    assert all(r.get("kb") for r in resp.retrieved)
