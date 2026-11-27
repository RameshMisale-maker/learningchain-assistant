"""Shared fixtures: an isolated copy of the knowledge bases + datasets, offline backends, fresh database."""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _reset_singletons() -> None:
    from backend.db.session import reset_engine
    from backend.generation import llm_client, prompt_builder
    from backend.retrieval import kb_index, reranker
    from backend.services import chat_service
    from backend.api import deps, routes_admin
    from backend.products import reset_products
    from config.settings import reset_settings
    from ingestion import embedding, vector_store

    reset_settings()
    reset_products()
    reset_engine()
    embedding.get_embedder.cache_clear()
    vector_store.get_vector_store.cache_clear()
    reranker.get_reranker.cache_clear()
    llm_client.get_llm.cache_clear()
    prompt_builder.load_prompt.cache_clear()
    kb_index._manager = None
    chat_service._service = None
    deps._limiter = None
    routes_admin._admin = None


@pytest.fixture(scope="session")
def workspace(tmp_path_factory):
    base = tmp_path_factory.mktemp("sapphirus")
    shutil.copytree(ROOT / "knowledge_bases", base / "knowledge_bases")
    shutil.copytree(ROOT / "datasets", base / "datasets")
    (base / "config").mkdir()
    shutil.copy(ROOT / "config" / "knowledge_bases.yaml", base / "config" / "knowledge_bases.yaml")
    env = {
        "DATA_DIR": str(base / "data"), "LOGS_DIR": str(base / "logs"),
        "KB_CONFIG_PATH": str(base / "config" / "knowledge_bases.yaml"),
        # Production retrieval models by default (routing thresholds are calibrated for them);
        # TEST_FAST=1 switches to the dependency-free local_hash embedder + lexical reranker.
        "EMBEDDING_BACKEND": "local_hash" if os.getenv("TEST_FAST") else "sentence_transformers",
        "EMBEDDING_MODEL": "local_hash" if os.getenv("TEST_FAST") else "BAAI/bge-small-en-v1.5",
        "VECTOR_BACKEND": os.getenv("TEST_VECTOR_BACKEND", "chroma"),
        "RERANKER_BACKEND": "lexical" if os.getenv("TEST_FAST") else "cross_encoder",
        "RERANKER_MODEL": "cross-encoder/ms-marco-MiniLM-L-6-v2", "LLM_PROVIDER": "extractive", "OPENAI_API_KEY": "", "ANTHROPIC_API_KEY": "",
        "AUTH_MODE": "token", "TOKEN_HASH_SECRET": "test-secret", "ALLOW_ANONYMOUS_PUBLIC": "true",
        "BOOTSTRAP_ADMIN_TOKEN": "admin-token-for-tests-0001", "BOOTSTRAP_EMPLOYEE_TOKEN": "employee-token-for-tests-01",
        "BOOTSTRAP_HR_TOKEN": "hr-token-for-tests-00000001", "BOOTSTRAP_PUBLIC_TOKEN": "public-token-for-tests-0001",
        "RATE_LIMIT_PER_MINUTE": "1000", "DATABASE_URL": "", "LOG_LEVEL": "WARNING",
        "INGEST_WORKER_ENABLED": "false", "WEB_MONITOR_ENABLED": "false", "INGEST_RETRY_BASE_SECONDS": "0",
        "WEBHOOK_SECRET": "test-webhook-secret",
        # tests assume the shipped defaults; a developer's .env may switch these on (tests turn them on themselves)
        "METRICS_ENABLED": "false", "TOKEN_BUDGET_ENABLED": "false", "GOLDEN_WARMUP": "false",
    }
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    _reset_singletons()
    from config.settings import get_settings

    settings = get_settings()
    settings.base_dir = base   # resolve KB folders relative to the isolated copy
    from ingestion.pipeline import IngestionPipeline

    pipeline = IngestionPipeline()
    for kb in pipeline.kb_names():
        results = pipeline.sync_kb(kb)
        assert all(r.status in ("added", "unchanged") for r in results), [r.to_dict() for r in results if r.status == "failed"]
    yield base
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    _reset_singletons()


@pytest.fixture(scope="session")
def pipeline(workspace):
    from ingestion.pipeline import IngestionPipeline
    return IngestionPipeline()


@pytest.fixture(scope="session")
def chat(workspace):
    from backend.services.chat_service import ChatService
    return ChatService(llm=None)


@pytest.fixture()
def employee():
    from backend.security.auth import AuthenticatedUser
    return AuthenticatedUser("test_employee", {"employee"})


@pytest.fixture()
def public_user():
    from backend.security.auth import AuthenticatedUser
    return AuthenticatedUser("test_public", {"public"})


@pytest.fixture()
def hr_user():
    from backend.security.auth import AuthenticatedUser
    return AuthenticatedUser("test_hr", {"employee", "hr"})
