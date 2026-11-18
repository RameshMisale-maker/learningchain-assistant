"""Production load test: concurrent requests against a RUNNING chatbot API (the same HTTP path real users take).

Reuses scripts/load_test.py (read-only import). Uses the streaming endpoint (/api/v1/chat/stream) so the time to the
first streamed answer token is measured as the user sees it. Note: the chatbot verifies an answer (grounding,
citations) before streaming it, so first-token time includes generation - it is reported as measured, not estimated.

Run the target API with a high per-user rate limit (RATE_LIMIT_PER_MINUTE) or the load test measures 429s.
"""
from __future__ import annotations

import asyncio
import random
from typing import Any

import httpx

from evaluation import config


def run(url: str, questions: list[str], levels: list[int], requests_per_level: int, seed: int = 7) -> dict[str, Any]:
    from scripts.load_test import level

    random.seed(seed)
    health = httpx.get(f"{url.rstrip('/')}/health/ready", timeout=30)
    if health.status_code != 200:
        raise RuntimeError(f"Load-test target {url} is not ready (HTTP {health.status_code})")
    asyncio.run(level(url, 2, 4, questions, {}, True))          # warm-up (models, caches)
    results = [asyncio.run(level(url, c, requests_per_level, questions, {}, True)) for c in levels]
    return {"url": url, "endpoint": "/api/v1/chat/stream", "requests_per_level": requests_per_level,
            "levels": results}


def rows(load: dict[str, Any]) -> list[dict[str, Any]]:
    """One flat row per concurrency level (for CSV / tables)."""
    out = []
    for r in load.get("levels", []):
        lat, ttfb = r.get("latency_ms") or {}, r.get("ttfb_ms") or {}
        out.append({"concurrency": r["concurrency"], "requests": r["requests"], "successful": r["successful"],
                    "errors": r["errors"], "error_rate": r["error_rate"], "throughput_rps": r["throughput_rps"],
                    "completed_per_minute": r["completed_per_minute"], "latency_mean_ms": lat.get("mean"),
                    "latency_p50_ms": lat.get("p50"), "latency_p95_ms": lat.get("p95"),
                    "latency_p99_ms": lat.get("p99"), "first_token_p50_ms": ttfb.get("p50"),
                    "first_token_p95_ms": ttfb.get("p95"), "status_codes": str(r.get("status_codes"))})
    return out


DEFAULT_LEVELS = [int(x) for x in str(config.os.getenv("EVAL_LOAD_LEVELS", "1,4,8")).split(",") if x.strip()]
DEFAULT_REQUESTS = int(config.os.getenv("EVAL_LOAD_REQUESTS", "12"))
