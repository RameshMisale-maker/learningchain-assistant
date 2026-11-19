"""Phase 4 (response cache) and Phase 6 (review queue, approved answers, eval cases)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

GOLDEN_Q = "How can I get a LearningChain demo?"


@pytest.fixture()
def cache_on(workspace):
    from backend.services import response_cache
    from config.settings import get_settings

    settings = get_settings()
    settings.response_cache_enabled = True
    response_cache.invalidate()
    yield settings
    settings.response_cache_enabled = False
    response_cache.invalidate()


@pytest.fixture()
def review_on(workspace):
    from config.settings import get_settings

    settings = get_settings()
    settings.review_queue_enabled = True
    yield settings
    settings.review_queue_enabled = False


def _cache_rows():
    from sqlalchemy import select

    from backend.db.models import ResponseCache
    from backend.db.session import db_session
    with db_session() as s:
        return [(r.normalized_query, r.response["answer_type"], r.source_message_id) for r in s.scalars(select(ResponseCache))]


# ------------------------------------------------------------------------------------------------ Phase 4: cache
def test_cache_off_stores_nothing(chat, public_user):
    from backend.services import response_cache
    response_cache.invalidate()
    chat.answer(GOLDEN_Q, public_user)
    assert _cache_rows() == []


def test_exact_hit_serves_the_same_answer_and_is_logged(cache_on, chat, public_user):
    from sqlalchemy import select

    from backend.db.models import RequestMetric
    from backend.db.session import db_session

    first = chat.answer(GOLDEN_Q, public_user, record_metrics=True)
    assert first.answer_type.startswith("golden") and not first.route.get("cache_hit")
    assert len(_cache_rows()) == 1
    second = chat.answer(GOLDEN_Q, public_user, record_metrics=True)
    assert second.route.get("cache_hit") and second.route["cache"]["kind"] == "exact"
    assert second.answer == first.answer and [c.id for c in second.citations] == [c.id for c in first.citations]
    assert second.message_id and second.message_id != first.message_id          # still saved in the history
    with db_session() as s:
        row = s.scalars(select(RequestMetric).where(RequestMetric.message_id == second.message_id)).first()
        assert row.cache_hit and row.golden_hit


def test_semantic_hit_and_scope_isolation(cache_on, chat, public_user, employee, monkeypatch):
    from backend.products import get_product

    chat.answer(GOLDEN_Q, public_user)
    monkeypatch.setattr(get_product(None), "semantic_cache_threshold", 0.9)
    similar = chat.answer("How can I get a demo of LearningChain?", public_user)
    assert similar.route.get("cache_hit") and similar.route["cache"]["kind"] == "semantic"
    other_roles = chat.answer(GOLDEN_Q, employee)                                 # different roles -> own entry
    assert not other_roles.route.get("cache_hit")


def test_refusals_and_out_of_scope_are_never_cached(cache_on, chat, public_user):
    chat.answer("What is the capital of France?", public_user)
    chat.answer("hi", public_user)
    assert _cache_rows() == []
    fake = SimpleNamespace(answer_type="generated", warnings=[], confidence=0.1, answer="x", message_id=1)
    from backend.services import response_cache
    response_cache._pending.set(response_cache.CacheContext("p", "k", "s", "q", None, "i", "v"))
    assert not response_cache.store(fake)                                         # low confidence
    fake.confidence, fake.warnings = 0.9, ["Generated answer failed grounding check"]
    assert not response_cache.store(fake)                                         # failed grounding
    response_cache.clear_pending()


def test_new_index_revision_or_thumbs_down_invalidates(cache_on, chat, public_user):
    from backend.db.repository import bump_kb_revision
    from backend.services.feedback_hooks import on_feedback

    first = chat.answer(GOLDEN_Q, public_user)
    bump_kb_revision("products")                                                  # new index -> new key
    after_reindex = chat.answer(GOLDEN_Q, public_user)
    assert not after_reindex.route.get("cache_hit")
    hit = chat.answer(GOLDEN_Q, public_user)
    assert hit.route.get("cache_hit")
    on_feedback(hit.message_id, "not_helpful")                                    # 👎 on a cached answer
    assert not chat.answer(GOLDEN_Q, public_user).route.get("cache_hit")
    assert first.message_id


# ------------------------------------------------------------------------------------------------ Phase 6: review
def test_clustering_counts_occurrences(review_on):
    from backend.services import review

    a = review.add_item("refusal", "How do I export certificates to Excel?")
    b = review.add_item("refusal", "how do i EXPORT certificates to excel")
    c = review.add_item("refusal", "How do I delete my account?")
    assert a == b != c
    item = next(i for i in review.list_items(source="refusal") if i["id"] == a)
    assert item["occurrences"] == 2 and item["details"]["variants"]
    for i in (a, c):
        review.resolve(i, "test", "dismiss", note="test")


def test_auto_review_refusal_and_low_confidence(review_on):
    from backend.services import review

    refusal = SimpleNamespace(answer_type="refusal", confidence=0.0, answer="no", message_id=None, knowledge_bases=[])
    low = SimpleNamespace(answer_type="generated", confidence=0.1, answer="maybe", message_id=None, knowledge_bases=[])
    good = SimpleNamespace(answer_type="generated", confidence=0.9, answer="yes", message_id=None, knowledge_bases=[])
    ids = [review.auto_review(refusal, "Zq refusal question one"), review.auto_review(low, "Zq low question two")]
    assert all(ids) and review.auto_review(good, "fine") is None
    for i in ids:
        review.resolve(i, "test", "dismiss")


def test_thumbs_down_via_api_creates_item(review_on):
    from fastapi.testclient import TestClient

    from backend.main import app
    from backend.services import review

    public = {"Authorization": "Bearer public-token-for-tests-0001"}
    with TestClient(app) as c:
        r = c.post("/api/v1/chat", json={"message": "How do I revoke a certificate?", "assistant": "all"}, headers=public)
        mid = r.json()["message_id"]
        assert c.post("/api/v1/feedback", json={"message_id": mid, "rating": "not_helpful", "comment": "wrong steps"},
                      headers=public).status_code == 200
        item = next(i for i in review.list_items(source="thumbs_down") if i["message_id"] == mid)
        assert item["question"] == "How do I revoke a certificate?" and item["details"]["comment"] == "wrong steps"
        admin = {"Authorization": "Bearer admin-token-for-tests-0001"}
        assert c.get("/api/v1/admin/review-items", headers=public).status_code == 403
        listed = c.get("/api/v1/admin/review-items?source=thumbs_down", headers=admin).json()
        assert any(i["id"] == item["id"] for i in listed["items"])
        assert c.post(f"/api/v1/admin/review-items/{item['id']}/resolve", json={"action": "dismiss"},
                      headers=admin).json()["status"] == "dismissed"


def test_approved_answer_is_served_and_becomes_an_eval_case(review_on, chat, public_user):
    from backend.services import review
    from ingestion import registry
    from ingestion.pipeline import IngestionPipeline

    question = "What is the Zorblat refund window for LearningChain?"
    item = review.add_item("refusal", question, "I don't have confirmed information.")
    out = review.resolve(item, "test", "approve", answer="The Zorblat refund window is 45 days from purchase.",
                         kb="products")
    try:
        assert out["status"] == "resolved" and out["ingest"]["status"] in ("added", "updated")
        assert any(g["qid"] == out["qid"] for g in review.list_golden("learningchain"))
        assert any(c["question"] == question and out["qid"] in c["expected_sources"]
                   for c in review.list_eval_cases("learningchain", "products"))
        resp = chat.answer(question, public_user, persist=False)
        assert "45 days" in resp.answer and resp.answer_type not in ("refusal", "out_of_scope")
        assert any(c.qid == out["qid"] for c in resp.citations)       # served from the approved answer
        pipeline = IngestionPipeline()
        assert review.reviewed_golden_path("products") in pipeline.discover("products")[1]   # survives a sync
    finally:   # leave the shared test knowledge base as it was
        path = review.reviewed_golden_path("products")
        doc = next(d for d in registry.list_documents("products") if d["source_path"].endswith(path.name))
        IngestionPipeline().delete_document(doc["doc_id"], archive_file=False)
        path.unlink(missing_ok=True)
        from sqlalchemy import delete

        from backend.db.models import EvalCase, GoldenQA
        from backend.db.session import db_session
        with db_session() as s:
            s.execute(delete(GoldenQA).where(GoldenQA.qid == out["qid"]))
            s.execute(delete(EvalCase).where(EvalCase.question == question))


def test_golden_csv_import_validates_rows(workspace, monkeypatch):
    from backend.services import review

    published = []
    monkeypatch.setattr(review, "publish_golden", lambda p, kb: published.append((p, kb)) or {"status": "added"})
    csv_text = ("Question ID,Question,Approved Answer,Alternate Questions\n"
                "IMP1,What is X?,X is a test.,What's X? | Explain X\n"
                ",Missing answer,,\n")
    out = review.import_golden_csv("learningchain", "products", csv_text)
    assert out["imported"] == 1 and len(out["skipped"]) == 1 and published == [("learningchain", "products")]
    row = next(g for g in review.list_golden("learningchain") if g["qid"] == "IMP1")
    assert row["alternates"] == ["What's X?", "Explain X"]
    from sqlalchemy import delete

    from backend.db.models import GoldenQA
    from backend.db.session import db_session
    with db_session() as s:
        s.execute(delete(GoldenQA).where(GoldenQA.qid == "IMP1"))


def test_migrations_0004_to_0007_round_trip(tmp_path):
    from alembic import command
    from alembic.config import Config
    from sqlalchemy import create_engine, inspect

    from config.settings import BASE_DIR

    engine = create_engine(f"sqlite:///{tmp_path / 'm.db'}")
    cfg = Config(str(BASE_DIR / "alembic.ini"))
    cfg.set_main_option("script_location", str(BASE_DIR / "migrations"))
    new_tables = {"response_cache", "review_items", "golden_qa", "eval_cases", "alert_events"}
    with engine.begin() as conn:
        cfg.attributes["connection"] = conn
        command.upgrade(cfg, "head")
        assert new_tables <= set(inspect(conn).get_table_names())
        assert {"missed_crawls", "tombstoned_at"} <= {c["name"] for c in inspect(conn).get_columns("documents")}
        command.downgrade(cfg, "0003")
        assert not new_tables & set(inspect(conn).get_table_names())
        assert "missed_crawls" not in {c["name"] for c in inspect(conn).get_columns("documents")}
