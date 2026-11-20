"""Content-change webhooks (CMS / help-centre publish events) -> immediate re-crawl of the affected source.

POST /api/v1/webhooks/content-updated
  Body (JSON): {"url": "<changed page URL>"} or {"source_id": 3}; with neither, every enabled help-centre/sitemap
  source on the same host is re-checked.
  Authentication - HMAC-SHA256 with WEBHOOK_SECRET, one of:
    * <WEBHOOK_SIGNATURE_HEADER>: sha256=<hex(hmac(secret, raw_body))>     (generic CMS / custom scripts)
    * X-Zendesk-Webhook-Signature + X-Zendesk-Webhook-Signature-Timestamp  (Zendesk webhooks:
      base64(hmac(secret, timestamp + raw_body)))
  Unsigned or wrongly signed calls are rejected; if WEBHOOK_SECRET is not set the endpoint is disabled (503).
  Replay protection: a signed timestamp more than MAX_SKEW_SECONDS (10 min) from the server clock is rejected, and
  a signature already accepted within that window is rejected (409). Generic callers can sign a timestamp too
  (<WEBHOOK_SIGNATURE_HEADER>-timestamp: <unix seconds>, signature over "<timestamp>.<raw body>");
  WEBHOOK_REQUIRE_TIMESTAMP=true makes that mandatory (recommended once the CMS sends it).
The webhook only *triggers* a crawl of an already-registered source; it cannot add sources or URLs (no SSRF path).
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import threading
import time
from datetime import datetime

from fastapi import APIRouter, HTTPException, Request

from backend.api.deps import client_ip
from backend.db import repository
from config.settings import get_settings
from ingestion import automation

router = APIRouter(prefix="/api/v1/webhooks", tags=["webhooks"])
MAX_SKEW_SECONDS = 600
_seen: dict[str, float] = {}            # accepted signature -> expiry (monotonic): replay cache
_seen_lock = threading.Lock()


def _timestamp_seconds(value: str) -> float | None:
    """Unix seconds or ISO-8601 (Zendesk sends e.g. 2026-11-20T10:15:00Z) -> epoch seconds; None if unparsable."""
    value = (value or "").strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        pass
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _fresh(ts: str, now: float | None = None) -> bool:
    seconds = _timestamp_seconds(ts)
    return seconds is not None and abs((now if now is not None else time.time()) - seconds) <= MAX_SKEW_SECONDS


def _first_use(signature: str) -> bool:
    """True the first time a signature is seen within the skew window (a repeat = replay)."""
    now = time.monotonic()
    with _seen_lock:
        for key in [k for k, exp in _seen.items() if exp <= now]:
            del _seen[key]
        if signature in _seen:
            return False
        _seen[signature] = now + MAX_SKEW_SECONDS
        return True


def verify_signature(headers: dict[str, str], body: bytes, secret: str) -> bool:
    """Valid HMAC signature over the body (and the timestamp, when one is signed). Freshness and replay are checked
    separately in check_request()."""
    h = {k.lower(): v for k, v in headers.items()}
    header = get_settings().webhook_signature_header.lower()
    sig, ts = h.get(header, ""), h.get(f"{header}-timestamp", "")
    if sig:
        signed = (ts.encode() + b"." + body) if ts else body
        expected = "sha256=" + hmac.new(secret.encode(), signed, hashlib.sha256).hexdigest()
        return hmac.compare_digest(sig.strip(), expected)
    zsig, zts = h.get("x-zendesk-webhook-signature", ""), h.get("x-zendesk-webhook-signature-timestamp", "")
    if zsig and zts:
        expected = base64.b64encode(hmac.new(secret.encode(), zts.encode() + body, hashlib.sha256).digest()).decode()
        return hmac.compare_digest(zsig.strip(), expected)
    return False


def check_request(headers: dict[str, str], body: bytes, secret: str) -> tuple[int, str]:
    """(0, "") when the call may proceed, else (HTTP status, reason): bad signature 401, stale or missing timestamp
    401, replayed signature 409."""
    if not verify_signature(headers, body, secret):
        return 401, "bad signature"
    h = {k.lower(): v for k, v in headers.items()}
    header = get_settings().webhook_signature_header.lower()
    ts = h.get("x-zendesk-webhook-signature-timestamp", "") if not h.get(header) else h.get(f"{header}-timestamp", "")
    if ts and not _fresh(ts):
        return 401, "stale timestamp"
    if not ts and get_settings().webhook_require_timestamp:
        return 401, "timestamp required"
    signature = h.get(header) or h.get("x-zendesk-webhook-signature", "")
    if not _first_use(signature.strip()):
        return 409, "replayed request"
    return 0, ""


@router.post("/content-updated", status_code=202)
async def content_updated(request: Request) -> dict:
    secret = get_settings().webhook_secret
    if not secret:
        raise HTTPException(status_code=503, detail="Webhooks are not configured (set WEBHOOK_SECRET)")
    body = await request.body()
    if len(body) > 64 * 1024:
        raise HTTPException(status_code=413, detail="Payload too large")
    status, reason = check_request(dict(request.headers), body, secret)
    if status:
        repository.write_audit("webhook", "webhook_rejected", request.url.path, outcome="denied",
                               details={"reason": reason}, ip_address=client_ip(request))
        raise HTTPException(status_code=status, detail="Replayed request" if status == 409 else "Invalid signature")
    try:
        payload = json.loads(body or b"{}")
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="Body must be JSON") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Body must be a JSON object")
    if payload.get("source_id") is not None:
        source_ids = [int(payload["source_id"])]
    else:
        url = str(payload.get("url") or payload.get("article_url") or "")
        source_ids = automation.find_sources_for_url(url) if url else \
            [s["id"] for s in automation.list_sources() if s["enabled"] and s["source_type"] in ("zendesk", "faq")]
    queued = []
    for sid in source_ids:
        try:
            job, created = automation.enqueue_crawl(sid, "webhook", force=False)
            queued.append({"source_id": sid, "job_id": job["id"], "created": created})
        except KeyError:
            continue
    repository.write_audit("webhook", "webhook_received", request.url.path,
                           details={"sources": source_ids, "ts": int(time.time())}, ip_address=client_ip(request))
    return {"status": "accepted", "queued": queued}
