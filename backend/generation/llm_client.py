"""Provider-agnostic LLM client (OpenAI default, Anthropic Claude optional).

LLM_PROVIDER=openai | anthropic | extractive
When the selected provider has no API key, `get_llm()` returns None and the
pipeline uses the extractive (no-LLM) answerer, so the system still works offline.
"""
from __future__ import annotations

import json
import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterator

from config.settings import Settings, get_settings

logger = logging.getLogger(__name__)


@dataclass
class LLMResult:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""
    latency_ms: float = 0.0          # request sent -> last token received (total LLM time)
    ttft_ms: float = 0.0             # request sent -> first text token received
    generation_ms: float = 0.0       # first text token -> last token (pure generation)
    stop_reason: str = ""
    request_id: str = ""
    provider: str = ""

    @property
    def tokens_per_second(self) -> float:
        """Output tokens divided by generation duration (first token -> last token)."""
        if self.completion_tokens <= 1 or self.generation_ms <= 0:
            return 0.0
        return round((self.completion_tokens - 1) / (self.generation_ms / 1000), 2)

    def cost_usd(self, settings: Settings) -> float:
        if self.provider == "openai":   # OpenAI-compatible provider (e.g. Groq) has its own configurable price
            pin, pout = settings.openai_price_input_per_1m, settings.openai_price_output_per_1m
        else:
            pin, pout = settings.price_input_per_1m, settings.price_output_per_1m
        return (self.prompt_tokens * pin + self.completion_tokens * pout) / 1_000_000


class LLMRefusal(RuntimeError):
    """The model declined to answer (stop_reason == "refusal")."""


class LLMClient(ABC):
    provider = "base"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @abstractmethod
    def complete(self, system: str, user: str, max_tokens: int | None = None, temperature: float | None = None,
                 json_mode: bool = False) -> LLMResult:
        """Return the model's reply to a system + user prompt."""

    def stream(self, system: str, user: str, max_tokens: int | None = None,
               json_mode: bool = False) -> Iterator[str | LLMResult]:
        """Providers without native streaming: one text chunk, then the result (ttft == total latency)."""
        result = self.complete(system, user, max_tokens=max_tokens, json_mode=json_mode)
        result.ttft_ms = result.ttft_ms or result.latency_ms
        yield result.text
        yield result

    def complete_json(self, system: str, user: str, max_tokens: int = 400) -> tuple[dict, LLMResult]:
        result = self.complete(system, user, max_tokens=max_tokens, temperature=0.0, json_mode=True)
        match = re.search(r"\{.*\}", result.text, flags=re.DOTALL)
        try:
            return (json.loads(match.group(0)) if match else {}), result
        except json.JSONDecodeError:
            logger.warning("LLM returned invalid JSON: %s", result.text[:200])
            return {}, result


class OpenAIClient(LLMClient):
    provider = "openai"

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        from openai import OpenAI

        self.client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url or None,
                             timeout=settings.llm_timeout_seconds, max_retries=2)
        self.model = settings.openai_model
        # Same endpoint, other models: providers such as Groq apply rate limits (incl. tokens-per-day) per model, so a
        # second model keeps answers generated when the primary model's quota is used up.
        self.fallback_models = [m for m in settings.openai_fallback_models if m and m != self.model]

    def complete(self, system, user, max_tokens=None, temperature=None, json_mode=False):
        from openai import APIConnectionError, APIStatusError, RateLimitError

        last_exc: Exception | None = None
        for model in [self.model, *self.fallback_models]:
            try:
                result = self._complete(model, system, user, max_tokens, temperature, json_mode)
                if model != self.model:
                    logger.warning("LLM answered by fallback model %s (primary %s unavailable)", model, self.model)
                return result
            except RateLimitError as exc:
                last_exc = exc
            except APIStatusError as exc:
                if exc.status_code < 500:
                    raise          # bad request / auth: another model would fail the same way
                last_exc = exc
            except APIConnectionError as exc:
                last_exc = exc
            logger.warning("LLM model %s unavailable (%s)%s", model, type(last_exc).__name__,
                           " - trying fallback model" if model != ([self.model, *self.fallback_models][-1]) else "")
        raise last_exc  # type: ignore[misc]

    def _complete(self, model, system, user, max_tokens=None, temperature=None, json_mode=False):
        started = time.perf_counter()
        kwargs = {"model": model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                  "max_completion_tokens": max_tokens or self.settings.llm_max_tokens,
                  "temperature": self.settings.llm_temperature if temperature is None else temperature}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if self.settings.openai_reasoning_effort:          # reasoning models (e.g. gpt-oss on Groq)
            kwargs["reasoning_effort"] = self.settings.openai_reasoning_effort
        try:
            response = self.client.chat.completions.create(**kwargs)
        except Exception as exc:  # some (reasoning) models reject a custom temperature
            if "temperature" not in str(exc).lower():
                raise
            kwargs.pop("temperature")
            response = self.client.chat.completions.create(**kwargs)
        usage = getattr(response, "usage", None)
        total = (time.perf_counter() - started) * 1000
        return LLMResult(text=(response.choices[0].message.content or "").strip(),
                         prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                         completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                         model=model, latency_ms=total, ttft_ms=total, provider="openai",
                         stop_reason=response.choices[0].finish_reason or "")


class AnthropicClient(LLMClient):
    """Claude via the official Anthropic SDK. Always streams, so time-to-first-token and generation speed are
    measured on every call. `temperature` is not sent: current Messages API versions reject sampling parameters,
    and grounding is enforced by the prompt + the post-generation grounding check instead."""
    provider = "anthropic"

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        import anthropic

        self._sdk = anthropic
        # The key comes only from the backend environment (.env / secret store); it is never sent to clients.
        self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key, timeout=settings.llm_timeout_seconds,
                                          max_retries=settings.llm_max_retries)
        self.model = settings.anthropic_model

    def complete(self, system, user, max_tokens=None, temperature=None, json_mode=False):
        result: LLMResult | None = None
        for item in self.stream(system, user, max_tokens=max_tokens, json_mode=json_mode):
            if isinstance(item, LLMResult):
                result = item
        assert result is not None
        return result

    def stream(self, system: str, user: str, max_tokens: int | None = None,
               json_mode: bool = False) -> Iterator[str | LLMResult]:
        """Yield text deltas as they arrive, then a final LLMResult with usage and timings."""
        if json_mode:
            user += "\n\nRespond with a single JSON object only."
        started = time.perf_counter()
        first: float | None = None
        parts: list[str] = []
        try:
            with self.client.messages.stream(
                    model=self.model, max_tokens=max_tokens or self.settings.llm_max_tokens, system=system,
                    messages=[{"role": "user", "content": user}]) as stream:
                for text in stream.text_stream:
                    if first is None:
                        first = time.perf_counter()
                    parts.append(text)
                    yield text
                message = stream.get_final_message()
        except self._sdk.RateLimitError:
            logger.warning("Anthropic rate limit reached (after SDK retries)")
            raise
        except self._sdk.APIStatusError as exc:
            logger.error("Anthropic API error %s (request_id=%s)", exc.status_code,
                         getattr(exc, "request_id", None) or exc.response.headers.get("request-id"))
            raise
        except self._sdk.APIConnectionError:
            logger.error("Could not reach the Anthropic API")
            raise
        ended = time.perf_counter()
        if message.stop_reason == "refusal":
            raise LLMRefusal(getattr(getattr(message, "stop_details", None), "category", None) or "refusal")
        usage = message.usage
        yield LLMResult(text="".join(parts).strip(), prompt_tokens=usage.input_tokens or 0,
                        completion_tokens=usage.output_tokens or 0, model=self.model,
                        latency_ms=(ended - started) * 1000,
                        ttft_ms=((first or ended) - started) * 1000,
                        generation_ms=(ended - (first or ended)) * 1000,
                        stop_reason=message.stop_reason or "", request_id=getattr(message, "_request_id", "") or "",
                        provider="anthropic")


@lru_cache(maxsize=1)
def get_llm() -> LLMClient | None:
    settings = get_settings()
    if settings.llm_failover_enabled:
        from backend.generation.failover import build_failover_llm

        chain = build_failover_llm(settings)
        if chain is not None:
            logger.info("LLM failover chain: %s", " -> ".join(name for name, _ in chain.providers))
            return chain
    provider = settings.effective_llm_provider
    if provider != settings.llm_provider:
        logger.warning("LLM_PROVIDER=%s has no API key configured - using %s", settings.llm_provider,
                       f"fallback provider '{provider}'" if provider != "extractive" else "extractive (no-LLM) answers")
    try:
        if provider == "openai":
            return OpenAIClient(settings)
        if provider == "anthropic":
            return AnthropicClient(settings)
    except Exception as exc:
        logger.error("Could not initialise LLM provider %s: %s - using extractive answers", provider, exc)
    return None
