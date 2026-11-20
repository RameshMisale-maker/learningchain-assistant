"""Phase 1 - product configuration: schema validation (fail fast), per-request resolution, prompts / replies filled
from config, no product names in code, product_id on stored rows, and a dummy product working end to end."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent


def _valid_config(**overrides) -> dict:
    cfg = yaml.safe_load((ROOT / "config/products/learningchain.yaml").read_text(encoding="utf-8"))
    cfg.update(overrides)
    return cfg


# ------------------------------------------------------------------------------------------------ validation
def test_shipped_product_files_are_valid():
    from backend.products.config import _parse
    for path in sorted((ROOT / "config/products").glob("*.yaml")):
        cfg = _parse(path)
        assert cfg.product_id == path.stem and cfg.kb in cfg.namespaces


@pytest.mark.parametrize("change,message", [
    ({"kb": "nope"}, "must be one of namespaces"),
    ({"namespaces": []}, "at least one knowledge base"),
    ({"scopes": {"x": ["company"]}}, "outside namespaces"),
    ({"product_id": "Bad Id"}, "product_id"),
    ({"unknown_field": 1}, "unknown_field"),
])
def test_invalid_product_config_fails_with_a_clear_error(tmp_path, change, message):
    from backend.products.config import ProductConfigError, _parse
    cfg = _valid_config(**change)
    pid = cfg["product_id"] if re.fullmatch(r"[a-z][a-z0-9_]+", str(cfg["product_id"])) else "bad"
    path = tmp_path / f"{pid}.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    with pytest.raises(ProductConfigError, match=re.escape(message)):
        _parse(path)


def test_missing_required_reply_fails(tmp_path):
    from backend.products.config import ProductConfigError, _parse
    cfg = _valid_config()
    del cfg["replies"]["out_of_scope"]
    (tmp_path / "learningchain.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    with pytest.raises(ProductConfigError, match="replies.out_of_scope"):
        _parse(tmp_path / "learningchain.yaml")


def test_unknown_knowledge_base_fails_at_load(tmp_path, monkeypatch):
    from backend.products import config as pc
    cfg = _valid_config(product_id="other", namespaces=["no_such_kb"], kb="no_such_kb", aliases=[])
    cfg["ui"]["api_model_id"] = ""
    (tmp_path / "other.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    monkeypatch.setenv("PRODUCT_CONFIG_DIR", str(tmp_path))
    pc.reset_products()
    try:
        with pytest.raises(pc.ProductConfigError, match="not defined in knowledge_bases.yaml"):
            pc.load_products()
    finally:
        monkeypatch.delenv("PRODUCT_CONFIG_DIR")
        pc.reset_products()


# ------------------------------------------------------------------------------------------------ resolution
def test_flag_off_always_uses_the_default_product(monkeypatch):
    from backend.products import get_product, resolve_product
    from config.settings import get_settings
    monkeypatch.setattr(get_settings(), "product_config_enabled", False)
    default = get_product()
    assert resolve_product(product="learningchain", assistant="products").product_id == default.product_id
    # the legacy assistant scopes still narrow the knowledge bases exactly as before
    assert default.allowed_kbs("all") == ["products", "company"]
    assert default.allowed_kbs("company") == ["company"] and default.allowed_kbs("products") == ["products"]


def test_flag_on_resolution_order(monkeypatch):
    from backend.products import resolve_product
    from backend.products.config import ProductConfigError
    from config.settings import get_settings
    monkeypatch.setattr(get_settings(), "product_config_enabled", True)
    assert resolve_product(product="learningchain").product_id == "learningchain"            # explicit id
    assert resolve_product(assistant="products").product_id == "learningchain"               # legacy alias
    assert resolve_product(assistant="company").product_id == "sapphirus"
    assert resolve_product(api_key_product="learningchain").product_id == "learningchain"     # API key
    assert resolve_product().product_id == "sapphirus"                                        # default
    with pytest.raises(ProductConfigError):
        resolve_product(product="nope")


def test_replies_and_prompts_follow_the_product():
    from backend.generation.prompt_builder import load_prompt
    from backend.products import use_product
    from backend.services.chat_service import small_talk_reply
    from config.settings import get_settings
    with use_product("learningchain"):
        assert "LearningChain Assistant" in small_talk_reply("hi", set())
        assert get_settings().out_of_scope_text.startswith("I can only help with LearningChain")
        system = load_prompt("system_prompt.txt")
        assert "questions about LearningChain:" in system and "[[" not in system
    with use_product("sapphirus"):
        assert "questions about Sapphirus and LearningChain:" in load_prompt("system_prompt.txt")
        assert "Sapphirus Systems or LearningChain" in small_talk_reply("hi", set())


# ------------------------------------------------------------------------------------------------ acceptance: no names in code
def test_no_product_names_in_code():
    """grep -ri "learningchain|sapphirus" over the code returns nothing outside config/ and tests/."""
    hits = []
    for folder in ("backend", "ingestion", "prompts", "streamlit_cloud", "evaluation", "migrations"):
        for path in (ROOT / folder).rglob("*"):
            if path.suffix not in (".py", ".html", ".js", ".txt", ".mako") or ".venv" in path.parts \
                    or "results" in path.parts or "__pycache__" in path.parts:
                continue
            for i, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
                if re.search(r"learningchain|sapphirus", line, re.I):
                    hits.append(f"{path.relative_to(ROOT)}:{i}: {line.strip()[:100]}")
    assert not hits, "\n".join(hits)


# ------------------------------------------------------------------------------------------------ API + storage
def test_ui_endpoint_and_openai_models(workspace):
    from fastapi.testclient import TestClient

    from backend.main import app
    client = TestClient(app)
    ui = client.get("/api/v1/ui").json()
    assert ui["product_id"] == "sapphirus" and ui["suggestions"] and "api_model_id" not in ui
    models = client.get("/v1/models", headers={"Authorization": "Bearer public-token-for-tests-0001"}).json()["data"]
    assert [m["id"] for m in models] == ["sapphirus-assistant"]          # unchanged Open WebUI model id


def test_rows_are_keyed_by_product(chat, public_user):
    from sqlalchemy import select

    from backend.db.models import ChatMessage, ChatSession, Document, RequestMetric
    from backend.db.session import db_session
    resp = chat.answer("hi", public_user, product="learningchain", record_metrics=True)
    with db_session() as s:
        assert s.get(ChatSession, resp.session_id).product_id == "learningchain"
        msgs = s.scalars(select(ChatMessage).where(ChatMessage.session_id == resp.session_id)).all()
        assert msgs and {m.product_id for m in msgs} == {"learningchain"}
        assert s.scalars(select(RequestMetric).where(RequestMetric.session_id == resp.session_id)).first().product_id \
            == "learningchain"
        owners = {(d.kb, d.product_id) for d in s.scalars(select(Document)).all()}
    assert ("products", "learningchain") in owners and ("company", "sapphirus") in owners


def test_database_is_at_the_latest_migration(workspace):
    from sqlalchemy import text

    from backend.db.session import get_engine
    with get_engine().connect() as conn:
        current = conn.execute(text("SELECT version_num FROM alembic_version")).scalar()
    heads = sorted(f.name.split("_")[0] for f in (ROOT / "migrations" / "versions").glob("0*.py"))
    assert current == heads[-1]


# ------------------------------------------------------------------------------------------------ dummy product end to end
DUMMY_PRODUCT = """
product_id: testproduct
display_name: Zorblat
company: Example Widgets Ltd
company_short: Example
bot_name: Zorblat Helper
description: Help for Zorblat widgets.
covers: [Zorblat]
aliases: [zorblat]
kb: testkb
namespaces: [testkb]
glossary: [{term: Zorblat, variants: [zorblet, zorbalt]}]
domain_terms: [widget, dial, calibrate]
topics: [widgets, calibration]
examples: ["How do I calibrate a Zorblat widget?"]
retrieval: {namespace: testkb}
escalation: {email: help@example.test}
models: {primary: {provider: groq, model: test-model}}
prompt_version: test-1
welcome_message: Hi, ask me about Zorblat widgets.
ui: {title: Zorblat Helper, api_model_id: "", api_model_name: ""}
replies:
  out_of_scope: I can only help with Zorblat widgets.
  no_confirmed_information: I don't have confirmed information on that - email help@example.test.
  blocked: I can only help with Zorblat questions.
  restricted_topic: I can't help with that topic.
  greeting_first: "{hello} I'm the Zorblat Helper."
  who_are_you: I'm the Zorblat Helper.
  what_can_you_do: I explain Zorblat widgets.
  bye: Bye from Zorblat.
  closing: Thanks for using Zorblat Helper.
  account: Please log in to see your account.
  escalation: Please email help@example.test.
  frustration_again: Please email help@example.test.
  rephrase: Could you rephrase?
  access_denied_guest: "{names} is internal."
"""
DUMMY_KB = """
  testkb:
    display_name: "Zorblat widgets"
    description: "Zorblat widget calibration guide."
    enabled: true
    allowed_roles: [public, employee, hr, admin]
    default_document_roles: [public, employee, hr, admin]
    default_category: "Product"
    source_dirs: [knowledge_bases/testkb]
    extra_files: []
    golden_files: []
    chunking: {chunk_size: 900, chunk_overlap: 150}
    entity_fields: []
    routing:
      keywords: [zorblat, widget, widgets, dial, calibrate, calibration]
      examples: ["How do I calibrate a Zorblat widget?", "What is the blue dial on a Zorblat?"]
"""
DUMMY_DOC = """# Zorblat Widget Guide

## Calibrating a Zorblat widget

To calibrate a Zorblat widget, turn the blue dial clockwise until the green light blinks three times, then press
the reset button for five seconds. Calibration takes about two minutes.

## Cleaning

Wipe the Zorblat casing with a dry cloth. Never use water on the dial.
"""
SCRIPT = """
import json, os, sys
sys.path.insert(0, os.environ["APP_ROOT"])
from ingestion.pipeline import IngestionPipeline
from backend.products import get_product, load_products
from backend.security.auth import AuthenticatedUser
from backend.services.chat_service import ChatService
from config.settings import get_settings
get_settings().base_dir = __import__("pathlib").Path(os.environ["WORK"])
pipeline = IngestionPipeline()
for kb in pipeline.kb_names():
    pipeline.sync_kb(kb)
product = get_product("testproduct")
svc = ChatService(llm=None)
user = AuthenticatedUser("t", {"public"})
out = {"products": sorted(load_products())}
for q in ("How do I calibrate a Zorblat widget?", "hi", "who won ipl 2024", "How do I verify a student certificate?"):
    r = svc.answer(q, user, persist=False, allowed_kbs=product.allowed_kbs(), product=product)
    out[q] = {"type": r.answer_type, "answer": r.answer, "kbs": r.knowledge_bases,
              "docs": [c.doc_name for c in r.citations]}
print("RESULT " + json.dumps(out))
"""


def test_dummy_product_works_end_to_end_with_config_only(tmp_path):
    """Adding a product = a product YAML + its knowledge base; no pipeline code changes."""
    work = tmp_path / "work"
    (work / "knowledge_bases" / "testkb").mkdir(parents=True)
    (work / "knowledge_bases" / "testkb" / "zorblat_guide.md").write_text(DUMMY_DOC, encoding="utf-8")
    (work / "config" / "products").mkdir(parents=True)
    kb_yaml = (ROOT / "config/knowledge_bases.yaml").read_text(encoding="utf-8")
    kb_yaml = kb_yaml.replace("knowledge_bases:\n", "knowledge_bases:\n" + DUMMY_KB, 1)
    (work / "config" / "knowledge_bases.yaml").write_text(kb_yaml, encoding="utf-8")
    for p in (ROOT / "config/products").glob("*.yaml"):
        (work / "config" / "products" / p.name).write_text(p.read_text(encoding="utf-8"), encoding="utf-8")
    (work / "config" / "products" / "testproduct.yaml").write_text(textwrap.dedent(DUMMY_PRODUCT), encoding="utf-8")
    env = {**os.environ, "APP_ROOT": str(ROOT), "WORK": str(work), "DATA_DIR": str(work / "data"),
           "LOGS_DIR": str(work / "logs"), "KB_CONFIG_PATH": str(work / "config" / "knowledge_bases.yaml"),
           "PRODUCT_CONFIG_DIR": str(work / "config" / "products"), "DEFAULT_PRODUCT": "sapphirus",
           "DATABASE_URL": "", "LLM_PROVIDER": "extractive", "OPENAI_API_KEY": "", "ANTHROPIC_API_KEY": "",
           "INGEST_WORKER_ENABLED": "false", "WEB_MONITOR_ENABLED": "false", "PRODUCT_CONFIG_ENABLED": "true",
           "EMBEDDING_BACKEND": "local_hash" if os.getenv("TEST_FAST") else "sentence_transformers",
           "EMBEDDING_MODEL": "local_hash" if os.getenv("TEST_FAST") else "BAAI/bge-small-en-v1.5",
           "RERANKER_BACKEND": "lexical" if os.getenv("TEST_FAST") else "cross_encoder",
           "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.run([sys.executable, "-c", SCRIPT], cwd=str(work), env=env, capture_output=True, text=True,
                          timeout=1500)
    line = next((l for l in proc.stdout.splitlines() if l.startswith("RESULT ")), None)
    assert line, proc.stdout[-2000:] + proc.stderr[-3000:]
    out = json.loads(line[len("RESULT "):])
    assert "testproduct" in out["products"]
    calib = out["How do I calibrate a Zorblat widget?"]
    assert calib["type"] not in ("refusal", "out_of_scope") and "blue dial" in calib["answer"]
    assert calib["kbs"] == ["testkb"]
    assert out["hi"]["answer"].startswith("Hello! I'm the Zorblat Helper")              # its own replies
    assert out["who won ipl 2024"]["answer"] == "I can only help with Zorblat widgets."
    other = out["How do I verify a student certificate?"]                               # another product's KB
    assert other["type"] in ("refusal", "out_of_scope") and not other["docs"]


# ------------------------------------------------------------------------------------------------ approved Q&A hygiene
def test_approved_answers_have_no_repeated_questions_and_no_placeholders():
    """Every served question has exactly one approved answer; unfinished FAQ rows are never served."""
    import re

    import pandas as pd

    def norm(t):
        return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", str(t).lower())).strip()

    served = []
    for f in ("datasets/products_golden_dataset.csv", "datasets/learningchain_faq_master.csv"):
        d = pd.read_csv(ROOT / f)
        d = d[~d["Status"].astype(str).str.lower().isin({"retired", "inactive", "draft", "rejected", "archived"})]
        for _, r in d.iterrows():
            served.append((norm(r["Question"]), r["Question ID"], str(r["Approved Answer"])))
    questions = [q for q, _, _ in served]
    assert len(questions) == len(set(questions)), "a question has two approved answers"
    for _, qid, answer in served:
        assert "please add" not in answer.lower() and "please confirm" not in answer.lower(), qid
    faq = pd.read_csv(ROOT / "datasets/learningchain_faq_master.csv").set_index("Question ID")
    assert faq.loc["LC-011", "Status"] == "active" and faq.loc["LC-019", "Status"] == "draft"
