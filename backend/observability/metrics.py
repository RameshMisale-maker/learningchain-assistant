"""Prometheus metrics (METRICS_ENABLED) in the plain-text exposition format - no extra dependency.

Per-process counters / histograms are updated after every answered request; gauges (circuit breakers, live index
versions, open review items, cache size) are read when /metrics is scraped. Scrape with Prometheus
(ops/prometheus/prometheus.yml) and chart with the Grafana dashboard in ops/grafana/.
"""
from __future__ import annotations

import threading
from collections import defaultdict
from typing import Any

from config.settings import get_settings

LATENCY_BUCKETS_MS = (250, 500, 1000, 2000, 4000, 8000, 15000, 30000)

_lock = threading.Lock()
_counters: dict[tuple[str, tuple], float] = defaultdict(float)
_hist: dict[tuple, dict[str, Any]] = {}

HELP = {
    "assistant_requests_total": ("counter", "Answered chat requests"),
    "assistant_refusals_total": ("counter", "Requests answered with a refusal / out-of-scope / error reply"),
    "assistant_cache_hits_total": ("counter", "Requests answered from the response cache"),
    "assistant_golden_hits_total": ("counter", "Requests answered with an approved (golden) answer"),
    "assistant_errors_total": ("counter", "Requests that raised an internal error"),
    "assistant_llm_tokens_total": ("counter", "LLM tokens used"),
    "assistant_llm_cost_usd_total": ("counter", "Estimated LLM cost in USD"),
    "assistant_feedback_total": ("counter", "User ratings"),
    "assistant_token_budget_blocks_total": ("counter", "Requests answered without the AI model because a token "
                                                       "budget (user or product) was used up"),
    "assistant_request_latency_ms": ("histogram", "End-to-end request latency in milliseconds"),
}


def _inc(name: str, labels: dict[str, str], value: float = 1.0) -> None:
    with _lock:
        _counters[(name, tuple(sorted(labels.items())))] += value


def _observe(name: str, labels: dict[str, str], value: float) -> None:
    key = (name, tuple(sorted(labels.items())))
    with _lock:
        h = _hist.setdefault(key, {"buckets": [0] * len(LATENCY_BUCKETS_MS), "sum": 0.0, "count": 0})
        for i, bound in enumerate(LATENCY_BUCKETS_MS):
            if value <= bound:
                h["buckets"][i] += 1
        h["sum"] += value
        h["count"] += 1


def record_request(resp, channel: str, product_id: str, error: str = "") -> None:
    """Called once per answered request (after metrics are written)."""
    if not get_settings().metrics_enabled:
        return
    from backend.services.chat_service import REFUSED_ANSWER_TYPES

    labels = {"product": product_id, "channel": channel, "answer_type": resp.answer_type}
    _inc("assistant_requests_total", labels)
    base = {"product": product_id, "channel": channel}
    if resp.answer_type in REFUSED_ANSWER_TYPES:
        _inc("assistant_refusals_total", base)
    if (resp.route or {}).get("cache_hit"):
        _inc("assistant_cache_hits_total", base)
    if resp.answer_type.startswith("golden"):
        _inc("assistant_golden_hits_total", base)
    if error:
        _inc("assistant_errors_total", base)
    provider = str((resp.route or {}).get("provider") or "")
    if resp.prompt_tokens or resp.completion_tokens:
        model = str(resp.model or "")       # provider quotas are per model (Groq: tokens per day per model)
        _inc("assistant_llm_tokens_total", base | {"provider": provider, "model": model, "kind": "input"},
             resp.prompt_tokens)
        _inc("assistant_llm_tokens_total", base | {"provider": provider, "model": model, "kind": "output"},
             resp.completion_tokens)
        _inc("assistant_llm_cost_usd_total", base | {"provider": provider}, resp.cost_usd)
    _observe("assistant_request_latency_ms", base, resp.latency_ms)


def record_budget_block(product_id: str, scope: str) -> None:
    if get_settings().metrics_enabled:
        _inc("assistant_token_budget_blocks_total", {"product": product_id, "scope": scope})


def record_feedback(rating: str, product_id: str = "") -> None:
    if get_settings().metrics_enabled:
        _inc("assistant_feedback_total", {"product": product_id, "rating": rating})


def reset() -> None:
    with _lock:
        _counters.clear()
        _hist.clear()


def _fmt_labels(labels: tuple) -> str:
    if not labels:
        return ""
    esc = lambda v: str(v).replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")  # noqa: E731
    return "{" + ",".join(f'{k}="{esc(v)}"' for k, v in labels) + "}"


def _gauges() -> list[tuple[str, str, list[tuple[tuple, float]]]]:
    """(name, help, [(labels, value)]) read at scrape time; each source is optional."""
    out = []
    try:
        from backend.generation.llm_client import get_llm

        llm = get_llm()
        if hasattr(llm, "status"):
            state = {"closed": 0, "half_open": 1, "open": 2}
            out.append(("assistant_circuit_breaker_state", "LLM provider breaker: 0 closed, 1 half-open, 2 open",
                        [((("provider", s["name"]),), state[s["state"]]) for s in llm.status()]))
    except Exception:
        pass
    try:
        from backend.indexing.versions import live_version_id
        from backend.retrieval.kb_index import get_index_manager

        kbs = get_index_manager().kb_names()
        out.append(("assistant_index_live_version", "Live index version per knowledge base (0 = classic index)",
                    [((("kb", kb),), float(live_version_id(kb) or 0)) for kb in kbs]))
    except Exception:
        pass
    try:
        from sqlalchemy import func, select

        from backend.db.models import RequestMetric
        from backend.db.session import db_session
        from backend.services.token_budget import product_limit
        from datetime import datetime, timedelta, timezone

        s_ = get_settings()
        since = datetime.now(timezone.utc) - timedelta(hours=s_.token_budget_window_hours)
        with db_session() as s:
            rows = s.execute(select(RequestMetric.product_id, func.sum(RequestMetric.input_tokens
                                                                         + RequestMetric.output_tokens))
                             .where(RequestMetric.created_at >= since).group_by(RequestMetric.product_id)).all()
        used = {pid or "": float(n or 0) for pid, n in rows}
        out.append(("assistant_tokens_window", "AI tokens used per product in the token-budget window "
                    "(TOKEN_BUDGET_WINDOW_HOURS)", [((("product", pid),), n) for pid, n in sorted(used.items())]))
        out.append(("assistant_token_budget_product", "Token budget per product for the window (0 = no limit)",
                    [((("product", pid),), float(product_limit(pid))) for pid in sorted(used)]))
    except Exception:
        pass
    try:
        from sqlalchemy import func, select

        from backend.db.models import ResponseCache, ReviewItem
        from backend.db.session import db_session

        with db_session() as s:
            open_items = s.execute(select(ReviewItem.source, func.count(ReviewItem.id))
                                   .where(ReviewItem.status == "open").group_by(ReviewItem.source)).all()
            cache_entries = s.scalar(select(func.count(ResponseCache.id))) or 0
        out.append(("assistant_review_items_open", "Open review queue items",
                    [((("source", src),), float(n)) for src, n in open_items]))
        out.append(("assistant_cache_entries", "Response cache entries", [((), float(cache_entries))]))
    except Exception:
        pass
    return out


def render() -> str:
    lines: list[str] = []
    with _lock:
        counters = dict(_counters)
        hists = {k: {"buckets": list(v["buckets"]), "sum": v["sum"], "count": v["count"]} for k, v in _hist.items()}
    for name, (kind, text) in HELP.items():
        lines += [f"# HELP {name} {text}", f"# TYPE {name} {kind}"]
        if kind == "counter":
            lines += [f"{name}{_fmt_labels(labels)} {value:g}" for (n, labels), value in counters.items() if n == name]
        else:
            for (n, labels), h in hists.items():
                if n != name:
                    continue
                for bound, count in zip(LATENCY_BUCKETS_MS, h["buckets"]):
                    lines.append(f"{name}_bucket{_fmt_labels(labels + (('le', str(bound)),))} {count}")
                lines.append(f"{name}_bucket{_fmt_labels(labels + (('le', '+Inf'),))} {h['count']}")
                lines.append(f"{name}_sum{_fmt_labels(labels)} {h['sum']:g}")
                lines.append(f"{name}_count{_fmt_labels(labels)} {h['count']}")
    for name, text, samples in _gauges():
        lines += [f"# HELP {name} {text}", f"# TYPE {name} gauge"]
        lines += [f"{name}{_fmt_labels(labels)} {value:g}" for labels, value in samples]
    return "\n".join(lines) + "\n"
