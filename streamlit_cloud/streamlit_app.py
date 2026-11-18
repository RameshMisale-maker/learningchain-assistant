"""LearningChain Assistant - Streamlit UI (Streamlit Community Cloud demo).

Runs the SAME chatbot pipeline as the FastAPI service (ChatService: conversational layer, guardrails, hybrid
retrieval, reranking, CRAG, grounding, citations) in-process - no separate API server is needed.

Configuration
* Non-secret settings: streamlit_cloud/defaults.env (same values as the production .env, no keys).
* Secrets: Streamlit Cloud -> App settings -> Secrets, e.g.   OPENAI_API_KEY = "gsk_..."
  (locally: .streamlit/secrets.toml or the project's .env).
* The knowledge-base index is built on first start (the Streamlit Cloud disk is not persistent).
Run locally:  streamlit run streamlit_cloud/streamlit_app.py
"""
from __future__ import annotations

import os
import sys
import time
import uuid
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_config() -> None:
    """Secrets first (Streamlit secrets -> environment), then non-secret defaults; existing values always win."""
    try:
        for key, value in st.secrets.items():
            if isinstance(value, (str, int, float, bool)):
                os.environ.setdefault(key, str(value))
    except Exception:      # no secrets file locally - the project's .env is used instead
        pass
    defaults = Path(__file__).with_name("defaults.env")
    if defaults.exists():
        for line in defaults.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.split(" #")[0].strip())


_load_config()

st.set_page_config(page_title="LearningChain Assistant", page_icon="🎓", layout="centered")

ASSISTANT_KBS = ["products"]            # same scope as the website's product assistant
MAX_QUESTIONS_PER_MINUTE = 10           # per browser session - protects the shared LLM quota
EXAMPLES = ["What is LearningChain?", "How do I verify a student certificate?",
            "What are the key features of LearningChain?", "How do I revoke a certificate?"]


@st.cache_resource(show_spinner=False)
def load_assistant():
    """Build the knowledge-base index if it is missing, then create the chat service (once per server)."""
    from backend.retrieval.kb_index import get_index_manager
    from backend.services.chat_service import ChatService
    from ingestion.pipeline import IngestionPipeline

    pipeline = IngestionPipeline()
    for kb in pipeline.kb_names():
        pipeline.sync_kb(kb)               # unchanged files are skipped, so a warm restart is quick
    get_index_manager()
    return ChatService()


def _user():
    from backend.security.auth import AuthenticatedUser
    return AuthenticatedUser("streamlit_guest", {"public"})


def _rate_limited() -> bool:
    now = time.time()
    recent = [t for t in st.session_state.ask_times if now - t < 60]
    st.session_state.ask_times = recent
    if len(recent) >= MAX_QUESTIONS_PER_MINUTE:
        return True
    recent.append(now)
    return False


def _render_sources(citations: list) -> None:
    if not citations:
        return
    with st.expander(f"Sources ({len(citations)})"):
        for c in citations:
            title = c.question if c.is_golden and c.question else c.doc_name
            line = f"**[{c.id}] {title}**"
            if c.section and not c.is_golden:
                line += f" - {c.section}"
            if c.page:
                line += f" (page {c.page})"
            st.markdown(line)
            if c.url:
                st.markdown(f"[{c.url}]({c.url})")
            if c.snippet:
                st.caption(c.snippet[:300])


def _ask(question: str, service) -> None:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        if _rate_limited():
            text = "You're sending questions quickly - please wait a minute and try again."
            st.warning(text)
            st.session_state.messages.append({"role": "assistant", "content": text, "answer_type": "rate_limited"})
            return
        history = [{"role": m["role"], "content": m["content"], "answer_type": m.get("answer_type", "generated")}
                   for m in st.session_state.messages[:-1]][-8:]
        with st.spinner("Searching the LearningChain documentation..."):
            resp = service.answer(question, _user(), session_id=st.session_state.session_id, persist=False,
                                  history=history, allowed_kbs=ASSISTANT_KBS, channel="streamlit",
                                  assistant="products", record_metrics=False)
        st.markdown(resp.answer)
        _render_sources(resp.citations)
        st.caption(f"{resp.route.get('conversation_intent', resp.answer_type)} · {resp.latency_ms / 1000:.1f}s")
    st.session_state.messages.append({"role": "assistant", "content": resp.answer, "answer_type": resp.answer_type,
                                      "citations": resp.citations, "meta": f"{resp.latency_ms / 1000:.1f}s"})


def main() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("session_id", uuid.uuid4().hex)
    st.session_state.setdefault("ask_times", [])

    with st.sidebar:
        st.markdown("### 🎓 LearningChain Assistant")
        st.caption("Answers come only from the approved LearningChain knowledge base, with sources.")
        st.markdown("**Try asking**")
        for q in EXAMPLES:
            if st.button(q, use_container_width=True):
                st.session_state.pending = q
        st.divider()
        if st.button("🧹 New conversation", use_container_width=True):
            st.session_state.messages, st.session_state.session_id = [], uuid.uuid4().hex
            st.rerun()
        st.caption("Demo build · hybrid search + reranking + Corrective RAG · Sapphirus Systems")

    st.title("LearningChain Assistant")
    if not os.getenv("OPENAI_API_KEY"):
        st.info("No LLM key configured - answers are extractive (quoted from the sources). "
                "Add OPENAI_API_KEY in the app's Secrets for generated answers.")

    try:
        with st.spinner("Preparing the knowledge base (first start takes 1-2 minutes)..."):
            service = load_assistant()
    except Exception as exc:     # show a clear message instead of a stack trace
        st.error(f"The assistant could not start: {type(exc).__name__}. Please try again shortly.")
        raise

    if not st.session_state.messages:
        with st.chat_message("assistant"):
            st.markdown("Hello! I'm the LearningChain Assistant. Ask me about issuing, sharing or verifying "
                        "blockchain certificates, the admin and user portals, or LearningChain features.")
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m["role"] == "assistant":
                _render_sources(m.get("citations") or [])

    question = st.chat_input("Ask a question about LearningChain...")
    question = question or st.session_state.pop("pending", None)
    if question:
        _ask(question.strip()[:2000], service)


main()
