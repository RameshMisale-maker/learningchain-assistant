"""Load / concurrency test against a running assistant API.

  python scripts/load_test.py --url http://127.0.0.1:8000 --levels 1,4,8,16 --requests 40 [--token ...]
  python scripts/load_test.py --stream        # use /api/v1/chat/stream and measure time to first byte

For each concurrency level it sends real questions (sampled from the evaluation dataset, mixed in-scope and out-of-
scope) and reports: throughput (requests/second, completed requests/minute), latency p50/p95/p99 (ms and s),
time to first byte for streaming, error count/rate and status codes. Results: reports/load_test/load_test.{json,md}.
Rate limiting: run the test with a token whose user has a high RATE_LIMIT_PER_MINUTE, or raise the limit for the run.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import random
import statistics
import time
from collections import Counter
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent


def pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    return round(s[min(len(s) - 1, int(q * (len(s) - 1) + 0.5))], 1)


async def one(client: httpx.AsyncClient, url: str, question: str, headers: dict, stream: bool) -> dict:
    started = time.perf_counter()
    body = {"message": question, "assistant": "all"}
    try:
        if stream:
            first = None
            async with client.stream("POST", f"{url}/api/v1/chat/stream", json=body, headers=headers) as r:
                async for chunk in r.aiter_bytes():
                    if first is None and b"event: delta" in chunk:
                        first = time.perf_counter()
                status = r.status_code
            ttfb = ((first or time.perf_counter()) - started) * 1000
        else:
            r = await client.post(f"{url}/api/v1/chat", json=body, headers=headers)
            status, ttfb = r.status_code, None
        return {"status": status, "ms": (time.perf_counter() - started) * 1000, "ttfb_ms": ttfb}
    except Exception as exc:
        return {"status": f"error:{type(exc).__name__}", "ms": (time.perf_counter() - started) * 1000, "ttfb_ms": None}


async def level(url: str, concurrency: int, n: int, questions: list[str], headers: dict, stream: bool) -> dict:
    sem = asyncio.Semaphore(concurrency)
    results: list[dict] = []
    async with httpx.AsyncClient(timeout=180) as client:
        async def task(q: str):
            async with sem:
                results.append(await one(client, url, q, headers, stream))
        started = time.perf_counter()
        await asyncio.gather(*(task(random.choice(questions)) for _ in range(n)))
        wall = time.perf_counter() - started
    ok = [r for r in results if r["status"] == 200]
    lat = [r["ms"] for r in ok]
    ttfb = [r["ttfb_ms"] for r in ok if r["ttfb_ms"]]
    return {"concurrency": concurrency, "requests": n, "successful": len(ok), "errors": n - len(ok),
            "error_rate": round((n - len(ok)) / n, 4), "status_codes": dict(Counter(str(r["status"]) for r in results)),
            "wall_seconds": round(wall, 2), "throughput_rps": round(len(ok) / wall, 3),
            "completed_per_minute": round(60 * len(ok) / wall, 1),
            "latency_ms": {"mean": round(statistics.mean(lat), 1) if lat else None, "p50": pct(lat, .5),
                           "p95": pct(lat, .95), "p99": pct(lat, .99), "max": round(max(lat), 1) if lat else None},
            "latency_s": {"p50": round(pct(lat, .5) / 1000, 2) if lat else None,
                          "p95": round(pct(lat, .95) / 1000, 2) if lat else None},
            "ttfb_ms": {"p50": pct(ttfb, .5), "p95": pct(ttfb, .95)} if ttfb else None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--levels", default="1,4,8,16")
    ap.add_argument("--requests", type=int, default=40)
    ap.add_argument("--token", default="")
    ap.add_argument("--stream", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    random.seed(args.seed)
    rows = list(csv.DictReader(open(ROOT / "datasets/evaluation_dataset/evaluation_dataset.csv", encoding="utf-8")))
    questions = [r["question"] for r in rows]
    headers = {"Authorization": f"Bearer {args.token}"} if args.token else {}
    health = httpx.get(f"{args.url}/health/ready", timeout=30)
    print(f"Readiness: {health.status_code} {health.json().get('status')}")
    asyncio.run(level(args.url, 2, 4, questions, headers, args.stream))    # warm-up (models, caches)
    out = []
    for c in [int(x) for x in args.levels.split(",")]:
        res = asyncio.run(level(args.url, c, args.requests, questions, headers, args.stream))
        out.append(res)
        print(f"c={c:>3}: {res['throughput_rps']} req/s ({res['completed_per_minute']}/min)  p50={res['latency_ms']['p50']}ms "
              f"p95={res['latency_ms']['p95']}ms  errors={res['errors']} {res['status_codes']}"
              + (f"  TTFB p50={res['ttfb_ms']['p50']}ms" if res.get("ttfb_ms") else ""))
    target = ROOT / "reports" / "load_test"
    target.mkdir(parents=True, exist_ok=True)
    meta = {"url": args.url, "endpoint": "/api/v1/chat/stream" if args.stream else "/api/v1/chat",
            "requests_per_level": args.requests, "readiness": health.json(), "levels": out,
            "generated": time.strftime("%Y-%m-%d %H:%M")}
    (target / "load_test.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    lines = [f"# Load test ({meta['generated']})", "", f"Endpoint `{meta['endpoint']}` at {args.url}, "
             f"{args.requests} requests per level, questions sampled from the evaluation set.", "",
             "| Concurrency | Req/s | Completed/min | p50 ms | p95 ms | p99 ms | p95 s | Errors | Status codes |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in out:
        lines.append(f"| {r['concurrency']} | {r['throughput_rps']} | {r['completed_per_minute']} | {r['latency_ms']['p50']} | "
                     f"{r['latency_ms']['p95']} | {r['latency_ms']['p99']} | {r['latency_s']['p95']} | {r['errors']} | "
                     f"{r['status_codes']} |")
    (target / "load_test.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Report: {target / 'load_test.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
