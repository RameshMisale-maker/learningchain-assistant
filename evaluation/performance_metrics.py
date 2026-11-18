"""Performance metrics from MEASURED values only.

* Latency / stage timings: the chatbot's own per-request timings (backend/common/timing) + wall-clock end-to-end.
* TTFT: only when the production LLM streams (Anthropic client). The OpenAI-compatible client (currently Groq) does
  not stream, so its "ttft" equals total latency - reported as "not measured", never estimated.
* Tokens / cost: the usage returned by the LLM API and the chatbot's configured prices (.env).
* Throughput: questions completed per second of evaluation wall time (sequential, one request at a time).
"""
from __future__ import annotations

from typing import Any

import numpy as np


def _pct(values: list[float], q: float) -> float | None:
    return float(np.percentile(values, q)) if values else None


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def summarize(runs: list[dict[str, Any]], wall_seconds: float) -> dict[str, Any]:
    lat = [r["latency_ms"] for r in runs if r.get("latency_ms") is not None and r.get("behavior") != "Error"]
    t = [r.get("timings") or {} for r in runs]
    retrieval = [x["retrieval_ms"] for x in t if x.get("retrieval_ms") is not None]
    rerank = [x["rerank_ms"] for x in t if x.get("rerank_ms") is not None]
    llm_total = [x["llm_total_ms"] for x in t if x.get("llm_total_ms")]
    llm_gen = [x["llm_generation_ms"] for x in t if x.get("llm_generation_ms")]
    ttft = [r["ttft_ms"] for r in runs if r.get("ttft_measured") and r.get("ttft_ms")]
    llm_calls = [r for r in runs if r.get("prompt_tokens") or r.get("completion_tokens")]
    errors = sum(1 for r in runs if r.get("behavior") == "Error" or r.get("error"))
    providers = sorted({r.get("llm_provider", "") for r in runs if r.get("llm_provider")})
    n = len(runs)
    return {
        "requests": n, "llm_calls": len(llm_calls), "llm_providers": providers,
        "avg_latency_ms": _mean(lat), "p50_latency_ms": _pct(lat, 50), "p95_latency_ms": _pct(lat, 95),
        "p99_latency_ms": _pct(lat, 99), "max_latency_ms": max(lat) if lat else None,
        "avg_retrieval_ms": _mean(retrieval), "avg_rerank_ms": _mean(rerank),
        "avg_llm_total_ms": _mean(llm_total), "avg_llm_generation_ms": _mean(llm_gen),
        "ttft_measured": bool(ttft), "avg_ttft_ms": _mean(ttft), "p95_ttft_ms": _pct(ttft, 95),
        "ttft_note": "" if ttft else ("TTFT not measured: the production LLM client does not stream "
                                      f"(provider: {', '.join(providers) or 'none'})"),
        "total_input_tokens": int(sum(r.get("prompt_tokens") or 0 for r in runs)),
        "total_output_tokens": int(sum(r.get("completion_tokens") or 0 for r in runs)),
        "avg_input_tokens": _mean([r["prompt_tokens"] for r in llm_calls]),
        "avg_output_tokens": _mean([r["completion_tokens"] for r in llm_calls]),
        "total_cost_usd": float(sum(r.get("cost_usd") or 0 for r in runs)),
        "cost_per_query_usd": (float(sum(r.get("cost_usd") or 0 for r in runs)) / n) if n else None,
        "cost_note": ("Token prices for this provider are not configured in the chatbot's .env (OPENAI_PRICE_INPUT_PER_1M"
                      " / OPENAI_PRICE_OUTPUT_PER_1M), so the chatbot reports $0 - real cost is unknown, not zero."
                      if llm_calls and not sum(r.get("cost_usd") or 0 for r in runs) else ""),
        "error_rate": (errors / n) if n else None, "errors": errors,
        "wall_seconds": wall_seconds, "throughput_qps": (n / wall_seconds) if wall_seconds else None,
    }
