"""OpenAI-compatible client falls back to OPENAI_FALLBACK_MODELS when the primary model is rate-limited."""
import httpx
import pytest
from openai import BadRequestError, RateLimitError

from backend.generation import llm_client
from config.settings import get_settings


def _err(cls, status):
    req = httpx.Request("POST", "https://api.example/v1/chat/completions")
    return cls("limit", response=httpx.Response(status, request=req), body=None)


class _FakeCompletions:
    def __init__(self, failing: dict):
        self.failing, self.calls = failing, []

    def create(self, **kw):
        self.calls.append(kw["model"])
        if kw["model"] in self.failing:
            raise self.failing[kw["model"]]

        class Msg: content = "Answer [S1]"
        class Choice: message, finish_reason = Msg(), "stop"
        class Usage: prompt_tokens, completion_tokens = 10, 5
        class Resp: choices, usage = [Choice()], Usage()
        return Resp()


def _client(monkeypatch, failing, fallbacks):
    s = get_settings()
    monkeypatch.setattr(s, "openai_api_key", "k")
    monkeypatch.setattr(s, "openai_model", "big-model")
    monkeypatch.setattr(s, "openai_fallback_models", fallbacks)
    c = llm_client.OpenAIClient(s)
    fake = _FakeCompletions(failing)
    c.client = type("C", (), {"chat": type("Ch", (), {"completions": fake})()})()
    return c, fake


def test_rate_limited_primary_uses_fallback_model(monkeypatch):
    c, fake = _client(monkeypatch, {"big-model": _err(RateLimitError, 429)}, ["small-model"])
    res = c.complete("sys", "user")
    assert res.text == "Answer [S1]" and res.model == "small-model" and fake.calls == ["big-model", "small-model"]


def test_bad_request_is_not_retried_on_other_models(monkeypatch):
    c, fake = _client(monkeypatch, {"big-model": _err(BadRequestError, 400)}, ["small-model"])
    with pytest.raises(BadRequestError):
        c.complete("sys", "user")
    assert fake.calls == ["big-model"]


def test_all_models_limited_raises_last_error(monkeypatch):
    c, _ = _client(monkeypatch, {"big-model": _err(RateLimitError, 429), "small-model": _err(RateLimitError, 429)},
                   ["small-model"])
    with pytest.raises(RateLimitError):
        c.complete("sys", "user")
