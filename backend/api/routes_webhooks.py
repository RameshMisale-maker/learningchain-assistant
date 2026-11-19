"""Content-change webhooks (CMS / help-centre publish events) -> immediate re-crawl of the affected source.

POST /api/v1/webhooks/content-updated
  Body (JSON): {"url": "<changed page URL>"} or {"source_id": 3}; with neither, every enabled help-centre/sitemap
  source on the same host is re-checked.
  Authentication - HMAC-SHA256 with WEBHOOK_SECRET, one of:
    * <WEBHOOK_SIGNATURE_HEADER>: sha256=<hex(hmac(secret, raw_body))>     (generic CMS / custom scripts)
    * X-Zendesk-Webhook-Signature + X-Zendesk-Webhook-Signature-Timestamp  (Zendesk webhooks:
      base64(hmac(secret, timestamp + raw_body)))
  Unsigned or wrongly signed calls are rejected; if WEBHOOK_SECRET is not set the endpoint is disabled (503).
The webhook only *triggers* a crawl of an already-registered source; it cannot add sources or URLs (no SSRF path).
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time

from fastapi import APIRouter, HTTPException, Request

from backend.api.deps import client_ip
from backend.db import repository
from config.settings import get_settings
from ingestion import automation

router = APIRouter(prefix="/api/v1/webhooks", tags=["webhooks"])
MAX_SKEW_SECONDS = 600


def verify_signature(headers: dict[str, str], body: bytes, secret: str) -> bool:
    h = {k.lower(): v for k, v in headers.items()}
    from config.settings import get_settings
    sig = h.get(get_settings().webhook_signature_header.lower(), "")
    if sig:
        expected = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
        return hmac.compare_digest(sig.strip(), expected)
    zsig, zts = h.get("x-zendesk-webhook-signature", ""), h.get("x-zendesk-webhook-signature-timestamp", "")
    if zsig and zts:
        expected = base64.b64encode(hmac.new(secret.encode(), zts.encode() + body, hashlib.sha256).digest()).decode()
        return hmac.compare_digest(zsig.strip(), expected)
    return False


@router.post("/content-updated", status_code=202)
async def content_updated(request: Request) -> dict:
    secret = get_settings().webhook_secret
    if not secret:
        raise HTTPException(status_code=503, detail="Webhooks are not configured (set WEBHOOK_SECRET)")
    body = await request.body()
    if len(body) > 64 * 1024:
        raise HTTPException(status_code=413, detail="Payload too large")
    if not verify_signature(dict(request.headers), body, secret):
        repository.write_audit("webhook", "webhook_rejected", request.url.path, outcome="denied",
                               details={"reason": "bad signature"}, ip_address=client_ip(request))
        raise HTTPException(status_code=401, detail="Invalid signature")
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
