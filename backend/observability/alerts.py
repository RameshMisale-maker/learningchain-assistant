"""Alert rules (config/alerts.yaml) evaluated over request_metrics, plus the nightly retrieval eval.

* `evaluate_alerts()` - run every ALERT_CHECK_MINUTES by the ops scheduler (ALERTS_ENABLED). Each rule is computed per
  product over its window; a firing is stored in `alert_events` and sent to Slack (ALERT_SLACK_WEBHOOK_URL) and / or
  e-mail (ALERT_EMAIL_TO via SMTP_*) when configured. A rule does not fire again within its cool-down.
* `nightly_eval()` - retrieval-only eval (hit@5 / MRR, no LLM, free) of every knowledge base's live index; results go
  to `<DATA_DIR>/evals/nightly_<date>.json` (one file per night, never overwritten) and a drop larger than the
  `eval_score_drop` threshold versus the previous night fires an alert.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import yaml

from config.settings import BASE_DIR, get_settings

logger = logging.getLogger(__name__)

METRICS = {"refusal_rate", "error_rate", "p95_latency_ms", "thumbs_down_rate", "llm_unavailable", "cache_hit_rate",
           "eval_score_drop", "answer_score_drop", "answer_score_min"}
NIGHTLY_METRICS = ("eval_score_drop", "answer_score_drop", "answer_score_min")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def load_rules(path: Path | None = None) -> list[dict[str, Any]]:
    path = path or (BASE_DIR / "config" / "alerts.yaml")
    if not path.exists():
        return []
    rules = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("rules", [])
    for r in rules:
        if r.get("metric") not in METRICS:
            raise ValueError(f"alert rule {r.get('name')}: unknown metric {r.get('metric')} (use one of {sorted(METRICS)})")
        r.setdefault("window_minutes", 60)
        r.setdefault("min_requests", 1)
        r.setdefault("severity", "warning")
        r.setdefault("cooldown_minutes", 60)
        r.setdefault("below", False)
    return rules


def _window_stats(product_id: str, since: datetime) -> dict[str, Any]:
    from sqlalchemy import func, select

    from backend.db.models import ChatMessage, Feedback, RequestMetric
    from backend.db.session import db_session

    with db_session() as s:
        rows = s.execute(select(RequestMetric.refused, RequestMetric.success, RequestMetric.end_to_end_ms,
                                RequestMetric.answer_type, RequestMetric.cache_hit)
                         .where(RequestMetric.product_id == product_id, RequestMetric.created_at >= since,
                                RequestMetric.channel != "eval")).all()
        ratings = s.execute(select(Feedback.rating, func.count(Feedback.id))
                            .join(ChatMessage, Feedback.message_id == ChatMessage.id)
                            .where(ChatMessage.product_id == product_id, Feedback.created_at >= since)
                            .group_by(Feedback.rating)).all()
    n = len(rows)
    latencies = sorted(r.end_to_end_ms for r in rows)
    rated = sum(c for rating, c in ratings if rating in ("helpful", "not_helpful"))
    down = sum(c for rating, c in ratings if rating == "not_helpful")
    return {
        "requests": n,
        "refusal_rate": sum(r.refused for r in rows) / n if n else 0.0,
        "error_rate": sum(not r.success for r in rows) / n if n else 0.0,
        "p95_latency_ms": latencies[min(n - 1, int(0.95 * n))] if n else 0.0,
        "cache_hit_rate": sum(bool(r.cache_hit) for r in rows) / n if n else 0.0,
        "llm_unavailable": float(sum(r.answer_type == "service_unavailable" for r in rows)),
        "rated": rated, "thumbs_down_rate": down / rated if rated else 0.0,
    }


def _recently_fired(rule: str, product_id: str, cooldown_minutes: float) -> bool:
    from sqlalchemy import select

    from backend.db.models import AlertEvent
    from backend.db.session import db_session

    since = _now() - timedelta(minutes=cooldown_minutes)
    with db_session() as s:
        return s.scalar(select(AlertEvent.id).where(AlertEvent.rule == rule, AlertEvent.product_id == product_id,
                                                    AlertEvent.created_at >= since)) is not None


def fire(rule: dict[str, Any], product_id: str, value: float, message: str) -> int | None:
    """Store + notify, unless the same rule fired for this product within its cool-down."""
    from backend.db.models import AlertEvent
    from backend.db.session import db_session

    if _recently_fired(rule["name"], product_id, rule["cooldown_minutes"]):
        return None
    notified = _notify(f"[{rule['severity'].upper()}] {rule['name']} ({product_id}): {message}")
    with db_session() as s:
        event = AlertEvent(rule=rule["name"], severity=rule["severity"], product_id=product_id, value=float(value),
                           threshold=float(rule["threshold"]), message=message[:2000], notified=notified)
        s.add(event)
        s.flush()
        logger.warning("ALERT %s (%s): %s", rule["name"], product_id, message)
        return event.id


def _notify(text: str) -> bool:
    settings = get_settings()
    sent = False
    if settings.alert_slack_webhook_url:
        try:
            import httpx

            httpx.post(settings.alert_slack_webhook_url, json={"text": text}, timeout=10).raise_for_status()
            sent = True
        except Exception:
            logger.warning("Alert Slack webhook failed", exc_info=True)
    if settings.alert_email_to and settings.smtp_host:
        try:
            import smtplib
            from email.message import EmailMessage

            msg = EmailMessage()
            msg["Subject"], msg["From"], msg["To"] = text[:120], settings.smtp_from, settings.alert_email_to
            msg.set_content(text)
            with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=15) as smtp:
                if settings.smtp_starttls:
                    smtp.starttls()
                if settings.smtp_user:
                    smtp.login(settings.smtp_user, settings.smtp_password)
                smtp.send_message(msg)
            sent = True
        except Exception:
            logger.warning("Alert e-mail failed", exc_info=True)
    return sent


def evaluate_alerts(rules: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Check every request-metric rule for every product; returns the alerts that fired now."""
    from backend.products import load_products

    rules = [r for r in (rules if rules is not None else load_rules()) if r["metric"] not in NIGHTLY_METRICS]
    fired = []
    for product_id in sorted(load_products()):
        cache: dict[int, dict[str, Any]] = {}
        for rule in rules:
            window = int(rule["window_minutes"])
            stats = cache.setdefault(window, _window_stats(product_id, _now() - timedelta(minutes=window)))
            sample = stats["rated"] if rule["metric"] == "thumbs_down_rate" else stats["requests"]
            if sample < rule["min_requests"]:
                continue
            value = stats[rule["metric"]]
            crossed = value < rule["threshold"] if rule["below"] else value > rule["threshold"]
            if not crossed:
                continue
            message = (f"{rule['metric']} = {value:.3g} ({'below' if rule['below'] else 'above'} {rule['threshold']}) "
                       f"over the last {window} min, {sample} samples")
            event_id = fire(rule, product_id, value, message)
            if event_id:
                fired.append({"id": event_id, "rule": rule["name"], "product_id": product_id, "value": value})
    return fired


# ---------------------------------------------------------------- nightly retrieval eval
def nightly_dir() -> Path:
    return get_settings().data_dir / "evals"


def nightly_eval(rules: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Retrieval eval of every knowledge base; saved per night; alerts on a drop vs the previous night."""
    from backend.indexing.versions import evaluate_collection, live_version_id
    from backend.products import product_for_kb
    from backend.retrieval.kb_index import get_index_manager

    folder = nightly_dir()
    folder.mkdir(parents=True, exist_ok=True)
    previous_files = sorted(folder.glob("nightly_*.json"))
    previous = json.loads(previous_files[-1].read_text(encoding="utf-8")) if previous_files else {}
    results: dict[str, Any] = {"run_at": _now().isoformat(), "knowledge_bases": {}}
    for kb in get_index_manager().kb_names():
        try:
            results["knowledge_bases"][kb] = evaluate_collection(kb) | {"index_version": live_version_id(kb)}
        except Exception as exc:
            logger.exception("Nightly eval of %s failed", kb)
            results["knowledge_bases"][kb] = {"error": f"{type(exc).__name__}: {exc}"}
    if get_settings().nightly_answer_eval:      # end-to-end answer quality, not only retrieval
        from backend.products import load_products

        results["answers"] = {}
        for product_id in sorted(load_products()):
            try:
                results["answers"][product_id] = answer_eval(product_id)
            except Exception as exc:
                logger.exception("Nightly answer eval of %s failed", product_id)
                results["answers"][product_id] = {"error": f"{type(exc).__name__}: {exc}"}
    stamp = _now().strftime("%Y%m%d_%H%M%S")
    path = folder / f"nightly_{stamp}.json"
    n = 1
    while path.exists():                     # never overwrite an earlier result
        path, n = folder / f"nightly_{stamp}_{n}.json", n + 1
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    results["file"] = str(path)
    rule = next((r for r in (rules if rules is not None else load_rules()) if r["metric"] == "eval_score_drop"), None)
    results["alerts"] = []
    if rule:
        for kb, cur in results["knowledge_bases"].items():
            before = (previous.get("knowledge_bases") or {}).get(kb, {})
            if cur.get("score") is None or before.get("score") is None:
                continue
            drop = before["score"] - cur["score"]
            if drop > rule["threshold"]:
                event = fire(rule | {"name": f"{rule['name']}:{kb}"}, product_for_kb(kb).product_id, drop,
                             f"retrieval eval score of '{kb}' fell from {before['score']:.3f} to {cur['score']:.3f}")
                if event:
                    results["alerts"].append({"kb": kb, "drop": round(drop, 4), "event_id": event})
    all_rules = rules if rules is not None else load_rules()
    drop_rule = next((r for r in all_rules if r["metric"] == "answer_score_drop"), None)
    min_rule = next((r for r in all_rules if r["metric"] == "answer_score_min"), None)
    for product_id, cur in (results.get("answers") or {}).items():
        score = cur.get("answer_score")
        if score is None:
            continue
        before = ((previous.get("answers") or {}).get(product_id) or {}).get("answer_score")
        if drop_rule and before is not None and before - score > drop_rule["threshold"]:
            event = fire(drop_rule | {"name": f"{drop_rule['name']}:{product_id}"}, product_id, before - score,
                         f"answer quality of '{product_id}' fell from {before:.3f} to {score:.3f}")
            if event:
                results["alerts"].append({"product": product_id, "drop": round(before - score, 4), "event_id": event})
        if min_rule and score < min_rule["threshold"]:
            event = fire(min_rule | {"name": f"{min_rule['name']}:{product_id}"}, product_id, score,
                         f"answer quality of '{product_id}' is {score:.3f}, below {min_rule['threshold']}")
            if event:
                results["alerts"].append({"product": product_id, "score": score, "event_id": event})
    return results


REFUSED_TYPES = ("refusal", "out_of_scope", "blocked", "restricted", "access_denied")


def answer_eval(product_id: str, max_cases: int | None = None) -> dict[str, Any]:
    """End-to-end answer quality for one product, from the hand-written cases in evals/<product_id>.jsonl.

    Each case runs through the real chat pipeline (no history, nothing stored) and is scored 0-1:
    * a question ("expected" sources): answered (not refused) 0.25, cites a source 0.25, cites an expected source
      0.25, contains every "must_include" phrase 0.25 (counted as met when the case lists none);
    * a refusal probe ("expect_refusal": true): 1 when the assistant declines, else 0.
    No LLM judge - the checks are objective and free; only the answers themselves use the LLM."""
    from backend.products import get_product
    from backend.security.auth import AuthenticatedUser
    from backend.services.chat_service import get_chat_service

    settings = get_settings()
    path = settings.evals_dir / f"{product_id}.jsonl"
    if not path.exists():
        return {"answer_score": None, "cases": 0, "note": f"no {path.name}"}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = rows[: max_cases or settings.nightly_answer_cases]
    product = get_product(product_id)
    service, user = get_chat_service(), AuthenticatedUser("nightly-eval", {"public"})
    results = []
    for row in rows:
        try:
            resp = service.answer(row["question"], user, persist=False, record_metrics=False, channel="eval",
                                  allowed_kbs=product.allowed_kbs(None), product=product)
        except Exception as exc:
            results.append({"question": row["question"], "score": 0.0, "error": type(exc).__name__})
            continue
        refused = resp.answer_type in REFUSED_TYPES
        if row.get("expect_refusal"):
            results.append({"question": row["question"], "kind": "refusal_probe", "score": float(refused),
                            "answer_type": resp.answer_type})
            continue
        sources = " ".join(f"{c.id} {c.source_file} {c.doc_name}" for c in resp.citations)
        expected = [str(e) for e in row.get("expected", [])]
        must = [str(m) for m in row.get("must_include", [])]
        checks = {"answered": not refused and bool(resp.answer.strip()),
                  "cited": bool(resp.citations),
                  "expected_source": bool(expected) and any(e in sources for e in expected),
                  "key_facts": all(m.lower() in resp.answer.lower() for m in must)}
        results.append({"question": row["question"], "kind": "question", "score": sum(checks.values()) / 4,
                        "checks": checks, "answer_type": resp.answer_type})
    scores = [r["score"] for r in results]
    questions = [r for r in results if r.get("kind") == "question"]
    probes = [r for r in results if r.get("kind") == "refusal_probe"]
    return {"answer_score": round(sum(scores) / len(scores), 4) if scores else None, "cases": len(results),
            "answered_rate": round(sum(r["checks"]["answered"] for r in questions) / len(questions), 4) if questions else None,
            "refusal_probe_rate": round(sum(r["score"] for r in probes) / len(probes), 4) if probes else None,
            "failures": [r for r in results if r["score"] < 1][:20]}


def list_events(limit: int = 100) -> list[dict[str, Any]]:
    from sqlalchemy import select

    from backend.db.models import AlertEvent
    from backend.db.session import db_session

    with db_session() as s:
        return [{"id": e.id, "created_at": e.created_at.isoformat() if e.created_at else None, "rule": e.rule,
                 "severity": e.severity, "product_id": e.product_id, "value": e.value, "threshold": e.threshold,
                 "message": e.message, "notified": e.notified}
                for e in s.scalars(select(AlertEvent).order_by(AlertEvent.id.desc()).limit(limit))]
