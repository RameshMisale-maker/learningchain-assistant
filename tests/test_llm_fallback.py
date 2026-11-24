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


@pytest.fixture(autouse=True)
def _clear_cooldowns():
    llm_client._RATE_LIMITED_UNTIL.clear()
    yield
    llm_client._RATE_LIMITED_UNTIL.clear()


def _groq_daily_limit():
    req = httpx.Request("POST", "https://api.example/v1/chat/completions")
    msg = ("Rate limit reached for model `big-model` on tokens per day (TPD): Limit 200000, Used 199894, "
           "Requested 1252. Please try again in 8m15.072s.")
    return RateLimitError(msg, response=httpx.Response(429, request=req), body=None)


def test_wait_parsed_from_retry_after_header():
    req = httpx.Request("POST", "https://api.example/v1/chat/completions")
    exc = RateLimitError("limit", response=httpx.Response(429, request=req, headers={"retry-after": "12"}), body=None)
    assert llm_client.rate_limit_wait_seconds(exc) == 12


def test_wait_parsed_from_groq_message():
    assert llm_client.rate_limit_wait_seconds(_groq_daily_limit()) == pytest.approx(495.072)


def test_wait_defaults_and_is_capped():
    assert llm_client.rate_limit_wait_seconds(_err(RateLimitError, 429)) == llm_client.RATE_LIMIT_DEFAULT_WAIT
    req = httpx.Request("POST", "https://api.example/v1/chat/completions")
    long = RateLimitError("Please try again in 5h1m0s", response=httpx.Response(429, request=req), body=None)
    assert llm_client.rate_limit_wait_seconds(long) == llm_client.RATE_LIMIT_MAX_WAIT


def test_rate_limited_primary_is_skipped_on_next_calls(monkeypatch):
    """After a daily-limit error the primary is not called again until the cool-down ends: later questions go
    straight to the fallback instead of failing (and retrying) on the primary every time."""
    c, fake = _client(monkeypatch, {"big-model": _groq_daily_limit()}, ["small-model"])
    c.complete("sys", "q1")
    c.complete("sys", "q2")
    assert fake.calls == ["big-model", "small-model", "small-model"]


def test_primary_used_again_after_cooldown(monkeypatch):
    c, fake = _client(monkeypatch, {"big-model": _err(RateLimitError, 429)}, ["small-model"])
    c.complete("sys", "q1")
    for key in list(llm_client._RATE_LIMITED_UNTIL):
        llm_client._RATE_LIMITED_UNTIL[key] = 0.0          # cool-down over
    fake.failing = {}
    res = c.complete("sys", "q2")
    assert res.model == "big-model" and fake.calls == ["big-model", "small-model", "big-model"]


def test_last_model_always_tried_even_if_cooling_down(monkeypatch):
    c, fake = _client(monkeypatch, {"big-model": _err(RateLimitError, 429), "small-model": _err(RateLimitError, 429)},
                      ["small-model"])
    with pytest.raises(RateLimitError):
        c.complete("sys", "q1")
    fake.failing = {}
    assert c.complete("sys", "q2").model == "small-model"


def test_primary_fails_fast_without_sdk_retries(monkeypatch):
    """Only the last model keeps the SDK's retries; earlier ones are called with max_retries=0."""
    c, _ = _client(monkeypatch, {}, ["small-model"])
    seen = {}

    class Real:
        def with_options(self, **kw):
            seen.update(kw)
            return "fast-client"

    c.client = Real()
    models = c._models_to_try()
    assert c._client_for("big-model", models) == "fast-client" and seen == {"max_retries": 0}
    assert c._client_for("small-model", models) is c.client
