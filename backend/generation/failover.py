"""Provider failover (LLM_FAILOVER_ENABLED): circuit breakers, a provider chain and a resilient query embedder.

* Every configured LLM provider gets a circuit breaker: BREAKER_FAILURES failures within BREAKER_WINDOW_SECONDS
  open it for BREAKER_COOLDOWN_SECONDS (calls skip that provider); afterwards one trial call decides (half-open).
* `FailoverLLM` tries the providers in order - the primary (LLM_PROVIDER), the other built-in provider when its key
  is set, then an optional secondary OpenAI-compatible endpoint (LLM_SECONDARY_*; e.g. a free-tier Gemini or
  OpenRouter key). Each call uses LLM_FAILOVER_TIMEOUT_SECONDS and one SDK retry. A model refusal is an answer,
  not an outage, so it never triggers failover.
* When every provider is down `AllProvidersDown` is raised; the chat service then answers with the product's static
  `service_unavailable` reply (support contact + common questions) instead of an error.
* `ResilientEmbedder` keeps search working during an embedding outage: the query vector becomes a zero vector, the
  request is marked "dense search off" and hybrid retrieval runs on BM25 only.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from contextvars import ContextVar
from typing import Any, Iterator

import numpy as np

from backend.generation.llm_client import LLMClient, LLMRefusal, LLMResult
from config.settings import Settings

logger = logging.getLogger(__name__)

_dense_down: ContextVar[bool] = ContextVar("dense_search_down", default=False)


class AllProvidersDown(RuntimeError):
    """No LLM provider could answer (all failed or their circuit breakers are open)."""


class CircuitBreaker:
    CLOSED, OPEN, HALF_OPEN = "closed", "open", "half_open"

    def __init__(self, name: str, failures: int = 5, window_seconds: float = 60, cooldown_seconds: float = 120,
                 clock=time.monotonic) -> None:
        self.name, self.failures, self.window, self.cooldown = name, failures, window_seconds, cooldown_seconds
        self._clock = clock
        self._errors: deque[float] = deque()
        self._opened_at: float | None = None
        self._trial = False
        self._lock = threading.Lock()

    @property
    def state(self) -> str:
        with self._lock:
            return self._state()

    def _state(self) -> str:
        if self._opened_at is None:
            return self.CLOSED
        return self.HALF_OPEN if self._clock() - self._opened_at >= self.cooldown else self.OPEN

    def allow(self) -> bool:
        with self._lock:
            state = self._state()
            if state == self.CLOSED:
                return True
            if state == self.HALF_OPEN and not self._trial:   # exactly one trial call after the cool-down
                self._trial = True
                return True
            return False

    def record_success(self) -> None:
        with self._lock:
            self._errors.clear()
            self._opened_at, self._trial = None, False

    def record_failure(self) -> None:
        with self._lock:
            now = self._clock()
            if self._state() == self.HALF_OPEN:          # trial failed: open again for a full cool-down
                self._opened_at, self._trial = now, False
                return
            self._errors.append(now)
            while self._errors and now - self._errors[0] > self.window:
                self._errors.popleft()
            if len(self._errors) >= self.failures:
                self._opened_at, self._trial = now, False
                logger.error("Circuit breaker '%s' OPEN after %d failures in %ss (cool-down %ss)", self.name,
                             len(self._errors), self.window, self.cooldown)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {"name": self.name, "state": self._state(), "recent_failures": len(self._errors)}


def _breaker(name: str, settings: Settings) -> CircuitBreaker:
    return CircuitBreaker(name, settings.breaker_failures, settings.breaker_window_seconds,
                          settings.breaker_cooldown_seconds)


class FailoverLLM(LLMClient):
    provider = "failover"

    def __init__(self, settings: Settings, providers: list[tuple[str, LLMClient]]) -> None:
        super().__init__(settings)
        if not providers:
            raise ValueError("FailoverLLM needs at least one provider")
        self.providers = providers
        self.breakers = {name: _breaker(f"llm:{name}", settings) for name, _ in providers}
        self.last_provider = ""

    @property
    def model(self) -> str:
        return getattr(self.providers[0][1], "model", "")

    def _available(self) -> list[tuple[str, LLMClient]]:
        return [(name, client) for name, client in self.providers if self.breakers[name].allow()]

    def complete(self, system, user, max_tokens=None, temperature=None, json_mode=False) -> LLMResult:
        errors = []
        for name, client in self._available():
            try:
                result = client.complete(system, user, max_tokens=max_tokens, temperature=temperature,
                                         json_mode=json_mode)
            except LLMRefusal:
                self.breakers[name].record_success()
                raise
            except Exception as exc:
                self.breakers[name].record_failure()
                errors.append(f"{name}: {type(exc).__name__}")
                logger.warning("LLM provider %s failed (%s) - trying the next provider", name, type(exc).__name__)
                continue
            self.breakers[name].record_success()
            result.provider_name = name                    # type: ignore[attr-defined]
            self.last_provider = name
            return result
        raise AllProvidersDown("; ".join(errors) or "all LLM circuit breakers are open")

    def stream(self, system, user, max_tokens=None, json_mode=False) -> Iterator[str | LLMResult]:
        errors = []
        for name, client in self._available():
            started = False
            try:
                for item in client.stream(system, user, max_tokens=max_tokens, json_mode=json_mode):
                    started = True
                    if isinstance(item, LLMResult):
                        item.provider_name = name          # type: ignore[attr-defined]
                    yield item
            except LLMRefusal:
                self.breakers[name].record_success()
                raise
            except Exception as exc:
                self.breakers[name].record_failure()
                if started:                                # part of the answer was already sent: cannot switch
                    raise
                errors.append(f"{name}: {type(exc).__name__}")
                logger.warning("LLM provider %s failed before streaming (%s) - trying the next provider", name,
                               type(exc).__name__)
                continue
            self.breakers[name].record_success()
            self.last_provider = name
            return
        raise AllProvidersDown("; ".join(errors) or "all LLM circuit breakers are open")

    def status(self) -> list[dict[str, Any]]:
        return [self.breakers[name].snapshot() | {"model": getattr(c, "model", "")} for name, c in self.providers]


def build_failover_llm(settings: Settings) -> FailoverLLM | None:
    """Primary provider, the other built-in provider (if its key is set), then the optional secondary endpoint."""
    from dataclasses import replace

    from backend.generation.llm_client import AnthropicClient, OpenAIClient

    fast = replace(settings, llm_timeout_seconds=settings.llm_failover_timeout_seconds, llm_max_retries=1)
    providers: list[tuple[str, LLMClient]] = []
    order = [settings.llm_provider] + [p for p in ("openai", "anthropic") if p != settings.llm_provider]
    keys = {"openai": settings.openai_api_key, "anthropic": settings.anthropic_api_key}
    for name in order:
        if not keys.get(name):
            continue
        try:
            client = OpenAIClient(fast) if name == "openai" else AnthropicClient(fast)
            if name == "openai":
                client.client = client.client.with_options(max_retries=1)
            providers.append((name, client))
        except Exception as exc:
            logger.error("Could not initialise LLM provider %s: %s", name, exc)
    if settings.llm_secondary_api_key and settings.llm_secondary_model:
        try:
            secondary = replace(fast, openai_api_key=settings.llm_secondary_api_key,
                                openai_base_url=settings.llm_secondary_base_url, openai_model=settings.llm_secondary_model,
                                openai_fallback_models=[], openai_reasoning_effort="")
            client = OpenAIClient(secondary)
            client.client = client.client.with_options(max_retries=1)
            providers.append(("secondary", client))
        except Exception as exc:
            logger.error("Could not initialise the secondary LLM provider: %s", exc)
    return FailoverLLM(settings, providers) if providers else None


# ---------------------------------------------------------------- embedding outage -> BM25-only search
def dense_search_down() -> bool:
    return _dense_down.get()


def reset_dense_flag():
    return _dense_down.set(False)


class ResilientEmbedder:
    """Wraps the query embedder: on failure (or open breaker) returns a zero vector and turns dense search off for
    the current request. Every other attribute is the wrapped embedder's."""

    def __init__(self, inner, settings: Settings) -> None:
        self._inner = inner
        self.breaker = _breaker("embedding", settings)
        self._dim: int | None = getattr(inner, "dim", None)

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def embed_query(self, text: str) -> np.ndarray:
        if self.breaker.allow():
            try:
                vec = self._inner.embed_query(text)
                self.breaker.record_success()
                self._dim = int(np.asarray(vec).shape[-1])
                return vec
            except Exception as exc:
                self.breaker.record_failure()
                logger.error("Query embedding failed (%s) - BM25-only search for this request", type(exc).__name__)
        _dense_down.set(True)
        return np.zeros(self._dim or 384, dtype=np.float32)
