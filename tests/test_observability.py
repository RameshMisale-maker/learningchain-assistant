"""Phase 7: Prometheus /metrics, alert rules (config/alerts.yaml) and the nightly retrieval eval."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

ADMIN = {"Authorization": "Bearer admin-token-for-tests-0001"}
PUBLIC = {"Authorization": "Bearer public-token-for-tests-0001"}


@pytest.fixture()
def metrics_on(workspace):
    from backend.observability import metrics
    from config.settings import get_settings

    settings = get_settings()
    settings.metrics_enabled = True
    metrics.reset()
    yield settings
    settings.metrics_enabled, settings.metrics_token = False, ""
    metrics.reset()


def _clear_alerts():
    from sqlalchemy import delete

    from backend.db.models import AlertEvent
    from backend.db.session import db_session
    with db_session() as s:
        s.execute(delete(AlertEvent))


def test_metrics_endpoint_off_by_default(workspace):
    from fastapi.testclient import TestClient

    from backend.main import app
    with TestClient(app) as c:
        assert c.get("/metrics").status_code == 404


def test_metrics_after_requests_and_token(metrics_on):
    from fastapi.testclient import TestClient

    from backend.main import app
    with TestClient(app) as c:
        c.post("/api/v1/chat", json={"message": "How do I revoke a certificate?", "assistant": "all"}, headers=PUBLIC)
        c.post("/api/v1/chat", json={"message": "What is the capital of France?", "assistant": "all"}, headers=PUBLIC)
        body = c.get("/metrics").text
        assert "# TYPE assistant_requests_total counter" in body
        assert 'assistant_requests_total{answer_type="out_of_scope",channel="api",product="sapphirus"} 1' in body
        assert 'assistant_refusals_total{channel="api",product="sapphirus"} 1' in body
        assert 'assistant_request_latency_ms_bucket{channel="api",product="sapphirus",le="+Inf"} 2' in body
        assert "assistant_index_live_version" in body and "assistant_cache_entries" in body
        metrics_on.metrics_token = "scrape-secret-123"
        assert c.get("/metrics").status_code == 401
        assert c.get("/metrics", headers={"Authorization": "Bearer scrape-secret-123"}).status_code == 200


def test_alert_rules_file_is_valid():
    from backend.observability.alerts import load_rules

    rules = load_rules()
    assert {r["name"] for r in rules} >= {"high_refusal_rate", "high_error_rate", "retrieval_quality_drop"}


def test_refusal_rate_rule_fires_once_per_cooldown(workspace):
    from backend.db.repository import write_request_metric
    from backend.observability.alerts import evaluate_alerts, list_events

    _clear_alerts()
    for i in range(6):
        write_request_metric(product_id="learningchain", channel="api", answer_type="refusal", refused=True,
                             success=True, end_to_end_ms=100.0)
    rule = {"name": "test_refusals", "metric": "refusal_rate", "threshold": 0.5, "window_minutes": 5,
            "min_requests": 5, "severity": "warning", "cooldown_minutes": 60, "below": False}
    fired = evaluate_alerts([rule])
    assert [f["product_id"] for f in fired] == ["learningchain"] and fired[0]["value"] > 0.5
    assert evaluate_alerts([rule]) == []                              # cool-down: not repeated
    assert list_events()[0]["rule"] == "test_refusals"
    too_few = dict(rule, name="test_min", min_requests=1000)
    assert evaluate_alerts([too_few]) == []
    _clear_alerts()


def test_notifications_go_to_slack_when_configured(workspace, monkeypatch):
    import httpx

    from backend.observability import alerts
    from config.settings import get_settings

    _clear_alerts()
    sent = []

    class Resp:
        def raise_for_status(self):
            return None
    monkeypatch.setattr(get_settings(), "alert_slack_webhook_url", "https://hooks.example.invalid/x")
    monkeypatch.setattr(httpx, "post", lambda url, json=None, timeout=None: sent.append(json["text"]) or Resp())
    rule = {"name": "test_slack", "threshold": 1, "severity": "critical", "cooldown_minutes": 5}
    assert alerts.fire(rule, "sapphirus", 3, "llm down")
    assert sent and sent[0].startswith("[CRITICAL] test_slack (sapphirus)")
    assert alerts.list_events()[0]["notified"] is True
    _clear_alerts()


def test_nightly_eval_saves_and_alerts_on_drop(workspace, monkeypatch):
    from backend.indexing import versions
    from backend.observability import alerts

    _clear_alerts()
    folder = alerts.nightly_dir()
    folder.mkdir(parents=True, exist_ok=True)
    before = {f.name for f in folder.glob("nightly_*.json")}
    (folder / "nightly_00000000_000000.json").write_text(json.dumps(
        {"knowledge_bases": {"products": {"score": 0.95}, "company": {"score": 0.5}}}), encoding="utf-8")
    monkeypatch.setattr(versions, "evaluate_collection",
                        lambda kb, collection=None: {"score": 0.7 if kb == "products" else 0.5, "cases": 10})
    rule = {"name": "drop", "metric": "eval_score_drop", "threshold": 0.05, "window_minutes": 1440,
            "min_requests": 1, "severity": "warning", "cooldown_minutes": 60, "below": False}
    out = alerts.nightly_eval([rule])
    assert out["knowledge_bases"]["products"]["score"] == 0.7
    assert [a["kb"] for a in out["alerts"]] == ["products"]
    files = {f.name for f in folder.glob("nightly_*.json")} - before
    assert len(files) == 2                                             # the seeded one + tonight's, nothing replaced
    for name in files:
        (folder / name).unlink()
    _clear_alerts()


def test_scheduler_tick_runs_due_jobs(workspace, monkeypatch):
    from backend.observability import alerts, scheduler
    from config.settings import get_settings

    calls = []
    monkeypatch.setattr(alerts, "evaluate_alerts", lambda: calls.append("alerts") or [])
    monkeypatch.setattr(alerts, "nightly_eval", lambda: calls.append("eval") or {})
    s = get_settings()
    monkeypatch.setattr(s, "alerts_enabled", True)
    monkeypatch.setattr(s, "nightly_eval_enabled", True)
    monkeypatch.setattr(s, "nightly_eval_hour", datetime.now(timezone.utc).hour)
    sched = scheduler.OpsScheduler()
    sched.tick()
    sched.tick()                                                       # same day / within the check interval
    assert calls == ["alerts", "eval"]


def test_admin_alert_endpoints(workspace):
    from fastapi.testclient import TestClient

    from backend.main import app
    with TestClient(app) as c:
        assert c.get("/api/v1/admin/alerts", headers=PUBLIC).status_code == 403
        assert "events" in c.get("/api/v1/admin/alerts", headers=ADMIN).json()
        assert "fired" in c.post("/api/v1/admin/alerts/check", headers=ADMIN).json()


def test_answer_eval_scores_questions_and_refusal_probes(workspace, monkeypatch, tmp_path):
    from types import SimpleNamespace

    from backend.observability import alerts
    from backend.services import chat_service
    from config.settings import get_settings

    (tmp_path / "learningchain.jsonl").write_text(
        '{"question": "good", "kb": "products", "expected": ["Guide"], "must_include": ["blue dial"]}\n'
        '{"question": "wrong source", "kb": "products", "expected": ["Other"]}\n'
        '{"question": "off topic", "kb": "products", "expect_refusal": true}\n', encoding="utf-8")
    monkeypatch.setattr(get_settings(), "evals_dir", tmp_path)
    cite = SimpleNamespace(id="c1", source_file="kb/Guide.md", doc_name="Guide")
    replies = {"good": ("generated", "Turn the blue dial.", [cite]),
               "wrong source": ("generated", "Some answer.", [cite]),
               "off topic": ("out_of_scope", "I can only help with ...", [])}

    class FakeService:
        def answer(self, question, user, **kw):
            t, a, c = replies[question]
            return SimpleNamespace(answer_type=t, answer=a, citations=c)
    monkeypatch.setattr(chat_service, "get_chat_service", lambda: FakeService())
    out = alerts.answer_eval("learningchain")
    assert out["cases"] == 3 and out["refusal_probe_rate"] == 1.0 and out["answered_rate"] == 1.0
    assert out["answer_score"] == round((1 + 0.75 + 1) / 3, 4)
    assert [f["question"] for f in out["failures"]] == ["wrong source"]


def test_nightly_answer_quality_alerts(workspace, monkeypatch):
    import json as _json

    from backend.indexing import versions
    from backend.observability import alerts

    _clear_alerts()
    folder = alerts.nightly_dir()
    folder.mkdir(parents=True, exist_ok=True)
    before = {f.name for f in folder.glob("nightly_*.json")}
    (folder / "nightly_00000000_000001.json").write_text(_json.dumps(
        {"knowledge_bases": {}, "answers": {"learningchain": {"answer_score": 0.95}}}), encoding="utf-8")
    monkeypatch.setattr(versions, "evaluate_collection", lambda kb, collection=None: {"score": 0.9, "cases": 10})
    monkeypatch.setattr(alerts, "answer_eval", lambda pid, max_cases=None: {"answer_score": 0.7 if pid == "learningchain"
                                                                            else 0.95, "cases": 10})
    rules = [{"name": "drop", "metric": "answer_score_drop", "threshold": 0.08, "cooldown_minutes": 60,
              "severity": "warning"},
             {"name": "low", "metric": "answer_score_min", "threshold": 0.75, "cooldown_minutes": 60,
              "severity": "critical"}]
    out = alerts.nightly_eval(rules)
    assert out["answers"]["learningchain"]["answer_score"] == 0.7
    assert sorted(a.get("drop", a.get("score")) for a in out["alerts"]) == [0.25, 0.7]
    for name in {f.name for f in folder.glob("nightly_*.json")} - before:
        (folder / name).unlink()
    _clear_alerts()
