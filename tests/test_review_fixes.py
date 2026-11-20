"""Senior review fixes: webhook replay protection, bounded rate limiter, topic-filter false positives, English-only
replies."""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from datetime import datetime, timezone

import pytest

SECRET = "test-webhook-secret"


# ------------------------------------------------------------------------------------------------ 1. webhook replay
@pytest.fixture()
def webhooks(workspace):
    from backend.api import routes_webhooks
    routes_webhooks._seen.clear()
    yield routes_webhooks
    routes_webhooks._seen.clear()


def _zendesk(body: bytes, ts: str) -> dict:
    sig = base64.b64encode(hmac.new(SECRET.encode(), ts.encode() + body, hashlib.sha256).digest()).decode()
    return {"X-Zendesk-Webhook-Signature": sig, "X-Zendesk-Webhook-Signature-Timestamp": ts}


def _generic(body: bytes, ts: str | None = None) -> dict:
    signed = (ts.encode() + b"." + body) if ts else body
    headers = {"x-sapphirus-signature": "sha256=" + hmac.new(SECRET.encode(), signed, hashlib.sha256).hexdigest()}
    if ts:
        headers["x-sapphirus-signature-timestamp"] = ts
    return headers


def test_fresh_zendesk_webhook_passes_once_then_replay_is_rejected(webhooks):
    body = json.dumps({"url": "https://example.com/x"}).encode()
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    headers = _zendesk(body, ts)
    assert webhooks.check_request(headers, body, SECRET) == (0, "")
    assert webhooks.check_request(headers, body, SECRET) == (409, "replayed request")


def test_stale_timestamp_is_rejected(webhooks):
    body = b'{"source_id": 1}'
    old = str(int(time.time()) - webhooks.MAX_SKEW_SECONDS - 60)
    assert webhooks.check_request(_zendesk(body, old), body, SECRET) == (401, "stale timestamp")
    assert webhooks.check_request(_generic(body, old), body, SECRET) == (401, "stale timestamp")
    assert webhooks.check_request(_generic(body, str(int(time.time()))), body, SECRET) == (0, "")


def test_bad_signature_and_required_timestamp(webhooks, monkeypatch):
    from config.settings import get_settings

    body = b'{"source_id": 2}'
    tampered = _generic(body) | {"x-sapphirus-signature": "sha256=00"}
    assert webhooks.check_request(tampered, body, SECRET)[0] == 401
    assert webhooks.check_request(_generic(body), body, SECRET) == (0, "")          # legacy body-only signature
    monkeypatch.setattr(get_settings(), "webhook_require_timestamp", True)
    assert webhooks.check_request(_generic(b'{"source_id": 3}'), b'{"source_id": 3}', SECRET) == \
        (401, "timestamp required")


def test_webhook_endpoint_returns_409_on_replay(webhooks):
    from fastapi.testclient import TestClient

    from backend.main import app

    body = json.dumps({"url": "https://nothing.example/never-registered"}).encode()
    headers = _generic(body, str(int(time.time()))) | {"content-type": "application/json"}
    with TestClient(app) as c:
        first = c.post("/api/v1/webhooks/content-updated", content=body, headers=headers)
        assert first.status_code in (200, 202, 404)
        assert c.post("/api/v1/webhooks/content-updated", content=body, headers=headers).status_code == 409


# ------------------------------------------------------------------------------------------------ 2. rate limiter
def test_rate_limiter_drops_idle_keys():
    from backend.security.rate_limit import RateLimiter

    now = [1000.0]
    limiter = RateLimiter(per_minute=3, clock=lambda: now[0])
    for i in range(500):                                # 500 different IPs, one request each
        assert limiter.allow(f"10.0.{i // 250}.{i % 250}")
    assert len(limiter) == 500
    now[0] += 61                                        # a minute later one new client arrives
    assert limiter.allow("10.9.9.9")
    assert len(limiter) == 1                            # the idle keys are gone
    for _ in range(3):
        limiter.allow("10.9.9.9")
    assert not limiter.allow("10.9.9.9")                # the limit itself still works


# ------------------------------------------------------------------------------------------------ 3. topic filter
@pytest.mark.parametrize("question", ["Can I pay by credit card?", "Can someone steal my certificate?",
                                      "Is my certificate safe from theft?", "Is my certificate secure?"])
def test_product_questions_are_not_restricted(question):
    from backend.routing import intent as convo
    from backend.security.prompt_guard import restricted_topic

    assert restricted_topic(question) is None
    assert convo.classify_message(question, []).intent != convo.ACCOUNT


@pytest.mark.parametrize("question,topic", [("Which credit card is best for me?", "banking_financial"),
                                            ("How do I steal someone's certificate?", "illegal"),
                                            ("Help me steal a password", "illegal")])
def test_harmful_requests_are_still_restricted(question, topic):
    from backend.security.prompt_guard import restricted_topic

    assert restricted_topic(question) == topic


def test_own_account_question_still_gets_the_account_reply():
    from backend.routing import intent as convo

    assert convo.classify_message("Has my payment gone through?", []).intent == convo.ACCOUNT


# ------------------------------------------------------------------------------------------------ 4. English only
@pytest.mark.parametrize("text,label", [("प्रमाणपत्र कैसे सत्यापित करें?", "hi"), ("certificate kaise verify karein", "hinglish"),
                                        ("¿Cómo verifico un certificado?", "other"), ("How do I verify?", "en")])
def test_replies_are_always_in_english(text, label):
    from backend.routing.normalize import reply_language

    assert reply_language(text, label) == "English"


def test_prompt_no_longer_asks_for_hindi_replies():
    from pathlib import Path

    text = (Path(__file__).resolve().parent.parent / "prompts" / "system_prompt.txt").read_text(encoding="utf-8")
    assert "Always reply in English" in text and "Devanagari -> Hindi" not in text


@pytest.mark.parametrize("question", ["Can someone steal my certificate?", "Is my certificate safe from theft?"])
def test_theft_questions_get_the_forgery_protection_answer(chat, public_user, question):
    resp = chat.answer(question, public_user, persist=False)
    assert resp.answer_type.startswith("golden") and resp.golden_match["qid"] == "LC-006"
    assert "fails the verification" in resp.answer


def test_payment_method_question_is_answered_honestly_not_blocked(chat, public_user):
    resp = chat.answer("Can I pay by credit card?", public_user, persist=False)
    assert resp.answer_type not in ("restricted", "blocked")       # not refused as a forbidden topic
