"""Phase 2: versioned indexes (build -> eval gate -> promote / rollback) and full request logging."""
from __future__ import annotations

import pytest


@pytest.fixture()
def versioning(workspace):
    """Turn INDEX_VERSIONING_ENABLED on for one test; afterwards serve the classic collection again."""
    from backend.indexing import versions
    from config.settings import get_settings

    settings = get_settings()
    settings.index_versioning_enabled = True
    versions.invalidate_alias_cache()
    yield settings
    settings.index_versioning_enabled = False
    versions.invalidate_alias_cache()


# ------------------------------------------------------------------------------------------------ unit
def test_flag_off_uses_the_classic_collection(workspace):
    from backend.indexing.versions import building, resolve_collection
    from ingestion.vector_store import base_collection_name, collection_name

    assert resolve_collection("products") is None
    assert collection_name("products") == base_collection_name("products")
    with building("products", "x_build"):          # a build is redirected even with the flag off
        assert collection_name("products") == "x_build"
        assert collection_name("company") == base_collection_name("company")
    assert collection_name("products") == base_collection_name("products")


def test_prompt_and_index_version_labels(workspace):
    from backend.indexing.versions import index_version_label, prompt_version_label

    label = prompt_version_label()
    assert "+" in label and len(label.split("+")[1]) == 8
    assert index_version_label(["products"]).startswith("products:base.r")


# ------------------------------------------------------------------------------------------------ integration
def test_build_gate_promote_and_rollback(versioning, chat, public_user):
    from backend.indexing import versions
    from ingestion.vector_store import base_collection_name, get_vector_store

    built = versions.build_version("products", "test")
    assert built["passed"] and built["status"] == "live", built
    assert built["collection"] == f"{base_collection_name('products')}_v{built['version_id']}"
    assert built["eval_details"]["cases"] > 0 and built["chunk_count"] > 0
    rows = versions.list_versions("products")
    baseline = next(v for v in rows if v["collection"] == base_collection_name("products"))
    assert baseline["status"] == "retired" and baseline["eval_score"] is not None   # live score measured once
    assert next(v for v in rows if v["version_id"] == built["version_id"])["is_live"]
    assert versions.live_version_id("products") == built["version_id"]
    assert get_vector_store().count("products") == built["chunk_count"]            # reads go to the new version

    resp = chat.answer("How do I verify a student certificate?", public_user, record_metrics=True)
    assert resp.answer_type not in ("refusal", "error")

    back = versions.rollback("products", "test")
    assert back["version_id"] == baseline["version_id"] and back["status"] == "live"
    assert versions.live_version_id("products") == baseline["version_id"]
    # promote the newer version again (the old one is still kept for an instant switch)
    assert versions.promote(built["version_id"], "test")["status"] == "live"
    versions.rollback("products", "test")


def test_gate_rejects_a_worse_index(versioning, monkeypatch):
    from backend.indexing import versions
    from ingestion.vector_store import get_vector_store

    live_before = versions.live_version_id("products")
    real = versions.evaluate_collection
    monkeypatch.setattr(versions, "evaluate_collection",
                        lambda kb, collection=None: real(kb, collection) if collection and collection.endswith("products")
                        else {"score": 0.0, "cases": 5, "hit_at_5": 0.0, "mrr": 0.0})
    monkeypatch.setattr(versions, "_live_score", lambda kb, live: 0.9)
    out = versions.build_version("products", "test")
    assert not out["passed"] and out["status"] == "rejected"
    assert versions.live_version_id("products") == live_before                 # users still on the old index
    with versions.building("products", out["collection"]):
        assert get_vector_store().count("products") == 0                       # rejected collection was dropped


def test_request_log_has_versions_intent_and_message_id(chat, public_user):
    from sqlalchemy import select

    from backend.db.models import RequestMetric
    from backend.db.session import db_session

    resp = chat.answer("How do I revoke a certificate?", public_user, record_metrics=True)
    with db_session() as s:
        row = s.scalars(select(RequestMetric).where(RequestMetric.session_id == resp.session_id)).first()
        assert row.message_id == resp.message_id and row.message_id
        assert row.index_version.startswith("products:") and "+" in row.prompt_version
        assert row.intent and row.refused == (resp.answer_type in ("refusal", "out_of_scope"))
        assert row.retrieved and {"id", "kb", "score"} <= set(row.retrieved[0])
        assert row.golden_hit == resp.answer_type.startswith("golden")


def test_admin_index_version_endpoints(workspace):
    from fastapi.testclient import TestClient

    from backend.main import app

    admin = {"Authorization": "Bearer admin-token-for-tests-0001"}
    with TestClient(app) as c:
        assert c.get("/api/v1/admin/index-versions", headers={"Authorization": "Bearer public-token-for-tests-0001"}
                     ).status_code == 403
        r = c.get("/api/v1/admin/index-versions?kb=products", headers=admin)
        assert r.status_code == 200 and "versions" in r.json()
        assert c.post("/api/v1/admin/index-versions/999999/promote", headers=admin).status_code == 400
        assert c.post("/api/v1/admin/index-versions/rollback", headers=admin).status_code == 400


def test_migration_0003_round_trip(tmp_path):
    from alembic import command
    from alembic.config import Config
    from sqlalchemy import create_engine, inspect

    from config.settings import BASE_DIR

    engine = create_engine(f"sqlite:///{tmp_path / 'm.db'}")
    cfg = Config(str(BASE_DIR / "alembic.ini"))
    cfg.set_main_option("script_location", str(BASE_DIR / "migrations"))
    with engine.begin() as conn:
        cfg.attributes["connection"] = conn
        command.upgrade(cfg, "0003")
        assert {"index_versions", "index_aliases"} <= set(inspect(conn).get_table_names())
        assert "index_version" in {c["name"] for c in inspect(conn).get_columns("request_metrics")}
        command.downgrade(cfg, "0002")
        assert "index_versions" not in set(inspect(conn).get_table_names())
        assert "index_version" not in {c["name"] for c in inspect(conn).get_columns("request_metrics")}
