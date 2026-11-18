"""Aggregations over request_metrics and evaluation_runs for the admin dashboard.

All numbers come from recorded rows (no estimates). Percentiles are computed over the filtered window.
Definitions:
  end_to_end_ms   request received -> complete answer produced
  ttft_ms         request received -> first answer byte available to the client
  llm_ttft_ms     LLM request sent -> first generated token
  tokens/second   output tokens / generation duration (first -> last token)
  throughput      completed requests / window length (per second and per minute)
"""
from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import select

from backend.db.models import EvaluationRun, RequestMetric
from backend.db.session import db_session

LATENCY_FIELDS = ["end_to_end_ms", "ttft_ms", "llm_ttft_ms", "llm_generation_ms", "llm_total_ms", "retrieval_ms",
                  "vector_search_ms", "bm25_search_ms", "rerank_ms", "context_ms"]


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    return round(s[min(len(s) - 1, int(q * (len(s) - 1) + 0.5))], 1)


def _parse(ts: str | None, default: datetime) -> datetime:
    if not ts:
        return default
    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _rows(start: datetime, end: datetime, channel: str = "", assistant: str = "") -> list[RequestMetric]:
    with db_session() as s:
        q = select(RequestMetric).where(RequestMetric.created_at >= start, RequestMetric.created_at <= end)
        if channel:
            q = q.where(RequestMetric.channel == channel)
        if assistant:
            q = q.where(RequestMetric.assistant == assistant)
        return list(s.scalars(q).all())


def _aggregate(rows: list[RequestMetric], seconds: float) -> dict[str, Any]:
    n = len(rows)
    ok = [r for r in rows if r.success]
    out: dict[str, Any] = {"requests": n, "successful": len(ok), "failed": n - len(ok),
                           "success_rate": round(len(ok) / n, 4) if n else None,
                           "throughput_rps": round(n / seconds, 4) if seconds > 0 else None,
                           "throughput_rpm": round(60 * n / seconds, 2) if seconds > 0 else None}
    for f in LATENCY_FIELDS:
        vals = [getattr(r, f) for r in ok if getattr(r, f)]
        out[f] = {"avg": round(statistics.mean(vals), 1) if vals else None, "p50": _pct(vals, 0.5),
                  "p95": _pct(vals, 0.95), "n": len(vals)}
    tps = [r.tokens_per_second for r in ok if r.tokens_per_second]
    out["tokens_per_second"] = {"avg": round(statistics.mean(tps), 1) if tps else None, "p50": _pct(tps, 0.5),
                                "n": len(tps)}
    llm_rows = [r for r in ok if r.output_tokens]
    out["llm_requests"] = len(llm_rows)
    out["input_tokens"] = sum(r.input_tokens for r in rows)
    out["output_tokens"] = sum(r.output_tokens for r in rows)
    total_cost = sum(r.total_cost_usd for r in rows)
    out["total_cost_usd"] = round(total_cost, 6)
    out["avg_cost_per_request_usd"] = round(total_cost / n, 6) if n else None
    out["avg_cost_per_llm_request_usd"] = round(sum(r.total_cost_usd for r in llm_rows) / len(llm_rows), 6) \
        if llm_rows else None
    out["answer_types"] = dict(Counter(r.answer_type for r in rows))
    out["models"] = dict(Counter(r.model or "extractive" for r in rows))
    return out


def summary(start: str | None = None, end: str | None = None, channel: str = "", assistant: str = "") -> dict:
    now = datetime.now(timezone.utc)
    t_end = _parse(end, now)
    t_start = _parse(start, t_end - timedelta(days=7))
    rows = _rows(t_start, t_end, channel, assistant)
    if rows:   # throughput over the active span, not the empty part of the window
        first = min(r.created_at for r in rows)
        last = max(r.created_at for r in rows)
        first = first if first.tzinfo else first.replace(tzinfo=timezone.utc)
        last = last if last.tzinfo else last.replace(tzinfo=timezone.utc)
        span = max((last - first).total_seconds(), 1.0)
    else:
        span = (t_end - t_start).total_seconds()
    result = {"window": {"start": t_start.isoformat(), "end": t_end.isoformat(), "active_seconds": round(span, 1)},
              "filters": {"channel": channel, "assistant": assistant}, "overall": _aggregate(rows, span)}
    by_assistant: dict[str, list] = defaultdict(list)
    by_kb: dict[str, list] = defaultdict(list)
    for r in rows:
        by_assistant[r.assistant or "all"].append(r)
        for kb in (r.knowledge_bases or "none").split(","):
            by_kb[kb or "none"].append(r)
    result["by_assistant"] = {k: _aggregate(v, span) for k, v in by_assistant.items()}
    result["by_knowledge_base"] = {k: _aggregate(v, span) for k, v in by_kb.items()}
    result["channels"] = sorted({r.channel for r in rows})
    return result


def timeseries(start: str | None = None, end: str | None = None, bucket: str = "hour", channel: str = "",
               assistant: str = "") -> list[dict]:
    now = datetime.now(timezone.utc)
    t_end = _parse(end, now)
    t_start = _parse(start, t_end - timedelta(days=7))
    fmt = "%Y-%m-%dT%H:00" if bucket == "hour" else "%Y-%m-%d"
    buckets: dict[str, list] = defaultdict(list)
    for r in _rows(t_start, t_end, channel, assistant):
        buckets[r.created_at.strftime(fmt)].append(r)
    out = []
    for key in sorted(buckets):
        rows = buckets[key]
        ok = [r for r in rows if r.success]
        e2e = [r.end_to_end_ms for r in ok]
        out.append({"bucket": key, "requests": len(rows), "failed": len(rows) - len(ok),
                    "p50_ms": _pct(e2e, 0.5), "p95_ms": _pct(e2e, 0.95),
                    "llm_ttft_p50_ms": _pct([r.llm_ttft_ms for r in ok if r.llm_ttft_ms], 0.5),
                    "retrieval_p50_ms": _pct([r.retrieval_ms for r in ok if r.retrieval_ms], 0.5),
                    "cost_usd": round(sum(r.total_cost_usd for r in rows), 6)})
    return out


def record_evaluation_run(label: str, run_type: str, config: dict, summary_data: dict) -> int:
    with db_session() as s:
        run = EvaluationRun(label=label, run_type=run_type, config=config, summary=summary_data)
        s.add(run)
        s.flush()
        return run.id


def evaluation_runs(run_type: str = "", limit: int = 50) -> list[dict]:
    with db_session() as s:
        q = select(EvaluationRun).order_by(EvaluationRun.id.desc()).limit(limit)
        if run_type:
            q = q.where(EvaluationRun.run_type == run_type)
        return [{"id": r.id, "created_at": r.created_at.isoformat(), "label": r.label, "run_type": r.run_type,
                 "config": r.config, "summary": r.summary} for r in s.scalars(q).all()]
