"""Adapter that runs the EXISTING chatbot pipeline and records everything needed for evaluation.

Integration (no chatbot file is modified):
* The chatbot's production entry point is FastAPI (`backend/main.py`); its /api/v1/chat route calls
  `ChatService.answer(...)` with the assistant's knowledge-base scope from `backend.api.routes_chat.ASSISTANT_KBS`.
  This adapter imports the same `ChatService` and calls `answer()` the same way, with `persist=False` and
  `record_metrics=False`, so evaluation questions never enter the production chat history or metrics tables.
* `ChatResponse` only exposes chunk ids, not the text that was retrieved or sent to the LLM. To capture it, this
  adapter wraps - inside the evaluation process only - the service's retriever and re-ranker objects and the
  `select_context` function the service calls. The wrappers call the original code unchanged and only record
  inputs/outputs; the source files are untouched and the running chatbot server is unaffected.
* The existing vector database (data/chroma) is read through the chatbot's own index manager. Nothing is written.
"""
from __future__ import annotations

import hashlib
import threading
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from evaluation import config


def text_key(text: str) -> str:
    """Stable id for a chunk's text (chunk ids change when the KB is re-indexed; the text usually does not)."""
    return hashlib.sha1((text or "").strip().encode("utf-8")).hexdigest()[:16]


@dataclass
class ChunkRecord:
    rank: int                         # 1-based rank after re-ranking (the order the chatbot used)
    chunk_id: str                     # child chunk id in the vector store
    text_key: str
    parent_id: str
    doc_id: str
    doc_name: str
    source_file: str
    section: str
    kb: str
    url: str
    text: str                         # child chunk text (what was searched)
    rerank_score: float
    fused_score: float
    vector_rank: int | None
    bm25_rank: int | None
    sent_to_llm: bool = False         # this chunk (or its parent) was in the final context


@dataclass
class ContextSource:
    sid: str                          # S1, S2 ... as cited by the answer
    chunk_id: str                     # the child that was matched
    parent_id: str
    doc_name: str
    section: str
    url: str
    is_golden: bool
    text: str                         # exact text given to the LLM (parent section when parent-child is used)


@dataclass
class ChatbotRun:
    question: str
    answer: str
    answer_type: str
    behavior: str                     # Answer | Abstain | Reject | Error (mapped from answer_type)
    model: str
    llm_provider: str
    embedding_model: str
    latency_ms: float
    timings: dict[str, float]
    ttft_ms: float | None             # None when the provider does not stream (TTFT not measurable)
    ttft_measured: bool
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    knowledge_bases: list[str]
    route: dict[str, Any]
    golden_match: dict[str, Any] | None
    grounding: dict[str, Any] | None
    citations: list[dict[str, Any]]
    retrieved: list[ChunkRecord]      # re-ranked child chunks (full candidate list, in rank order)
    contexts: list[ContextSource]     # what was actually sent to the LLM
    warnings: list[str]
    error: str = ""
    started_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def context_texts(self) -> list[str]:
        return [c.text for c in self.contexts]


class _Recorder(threading.local):
    def reset(self) -> None:
        self.retrieved: list = []
        self.reranked: list = []
        self.refs: list = []
        self.active = False


class _RecordingRetriever:
    """Pass-through proxy around the chatbot's HybridRetriever."""

    def __init__(self, inner, recorder: _Recorder) -> None:
        self._inner, self._rec = inner, recorder

    def retrieve(self, *args, **kwargs):
        out = self._inner.retrieve(*args, **kwargs)
        if self._rec.active:
            self._rec.retrieved.extend(out)
        return out

    def __getattr__(self, name):
        return getattr(self._inner, name)


class _RecordingReranker:
    """Pass-through proxy around the chatbot's re-ranker."""

    def __init__(self, inner, recorder: _Recorder) -> None:
        self._inner, self._rec = inner, recorder

    def rerank(self, query, candidates):
        out = self._inner.rerank(query, candidates)
        # record the main re-ranking only (the first call); later calls re-rank sub-questions of multi-part questions
        if self._rec.active and not self._rec.reranked:
            self._rec.reranked = list(out)
        return out

    def __getattr__(self, name):
        return getattr(self._inner, name)


class ChatbotAdapter:
    """Calls the real ChatService. One instance per evaluation run (models are loaded once)."""

    def __init__(self, assistant: str = config.ASSISTANT) -> None:
        from backend.api.routes_chat import ASSISTANT_KBS
        from backend.services import chat_service as cs

        self._cs = cs
        self.assistant = assistant
        self.allowed_kbs = ASSISTANT_KBS.get(assistant, ASSISTANT_KBS["all"])
        self.service = cs.ChatService()
        self.settings = self.service.settings
        self._rec = _Recorder()
        self._rec.reset()
        # instance-level wrappers (the service object belongs to this evaluation process only)
        self.service.retriever = _RecordingRetriever(self.service.retriever, self._rec)
        self.service.reranker = _RecordingReranker(self.service.reranker, self._rec)
        self._original_select_context = cs.select_context
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ info
    @property
    def llm_provider(self) -> str:
        return getattr(self.service.llm, "provider", "none") if self.service.llm else "none"

    @property
    def llm_model(self) -> str:
        return getattr(self.service.llm, "model", "") if self.service.llm else "extractive (no LLM)"

    @property
    def embedding_model(self) -> str:
        return getattr(self.service.embedder, "model_id", "") or self.settings.embedding_model

    def describe(self) -> dict[str, Any]:
        s = self.settings
        return {"assistant": self.assistant, "allowed_kbs": self.allowed_kbs, "llm_provider": self.llm_provider,
                "llm_model": self.llm_model, "embedding_backend": s.embedding_backend,
                "embedding_model": self.embedding_model, "vector_backend": s.vector_backend,
                "reranker": f"{s.reranker_backend}:{s.reranker_model}", "router_mode": s.router_mode,
                "grounding_mode": s.grounding_mode, "top_k_rerank": s.top_k_rerank,
                "max_context_chars": s.max_context_chars, "crag_enabled": s.crag_enabled,
                "conversation_layer": s.conversation_layer}

    # ------------------------------------------------------------------ call
    def ask(self, question: str, role: str = "public") -> ChatbotRun:
        from backend.security.auth import AuthenticatedUser

        roles = config.ROLE_SETS.get((role or "public").strip().lower(), {"public"})
        user = AuthenticatedUser(username=f"evaluation_{role or 'public'}", roles=set(roles))
        started_at = datetime.now(timezone.utc).isoformat()
        with self._lock:   # select_context is a module attribute: record one question at a time
            self._rec.reset()
            self._rec.active = True
            original = self._original_select_context

            def recording_select_context(*args, **kwargs):
                refs = original(*args, **kwargs)
                self._rec.refs = list(refs)
                return refs

            self._cs.select_context = recording_select_context
            error = ""
            try:
                t0 = time.perf_counter()
                resp = self.service.answer(question, user, persist=False, allowed_kbs=self.allowed_kbs,
                                           channel="evaluation", assistant=self.assistant, record_metrics=False)
                wall_ms = (time.perf_counter() - t0) * 1000
            except Exception as exc:   # answer() already catches pipeline errors; this is a last resort
                error = f"{type(exc).__name__}: {exc}"
                traceback.print_exc()
                resp, wall_ms = None, 0.0
            finally:
                self._cs.select_context = original
                self._rec.active = False
            reranked, refs = list(self._rec.reranked), list(self._rec.refs)
        if resp is None:
            return ChatbotRun(question=question, answer="", answer_type="error", behavior="Error",
                              model=self.llm_model, llm_provider=self.llm_provider,
                              embedding_model=self.embedding_model, latency_ms=round(wall_ms, 1), timings={},
                              ttft_ms=None, ttft_measured=False, prompt_tokens=0, completion_tokens=0, cost_usd=0.0,
                              knowledge_bases=[], route={}, golden_match=None, grounding=None, citations=[],
                              retrieved=[], contexts=[], warnings=[], error=error, started_at=started_at)
        return self._build_run(question, resp, reranked, refs, started_at)

    # ------------------------------------------------------------------ helpers
    def _build_run(self, question: str, resp, reranked: list, refs: list, started_at: str) -> ChatbotRun:
        contexts = [self._context_from_ref(r) for r in refs]
        if not contexts and resp.answer_type in ("golden_direct", "golden_semantic"):
            contexts = self._golden_context(resp)
        sent = {c.chunk_id for c in contexts} | {c.parent_id for c in contexts if c.parent_id}
        retrieved = []
        for rank, c in enumerate(reranked, start=1):
            m = c.meta
            parent_id = str(m.get("parent_id") or "")
            retrieved.append(ChunkRecord(
                rank=rank, chunk_id=c.id, text_key=text_key(c.chunk.text), parent_id=parent_id,
                doc_id=str(m.get("doc_id", "")), doc_name=str(m.get("doc_name", "")),
                source_file=str(m.get("source_file", "")), section=str(m.get("section", "")), kb=c.kb,
                url=str(m.get("source_url") or ""), text=c.chunk.text, rerank_score=round(float(c.rerank_score), 4),
                fused_score=round(float(c.fused_score), 5), vector_rank=c.vector_rank, bm25_rank=c.bm25_rank,
                sent_to_llm=c.id in sent or (bool(parent_id) and parent_id in sent)))
        t = dict(resp.timings or {})
        ttft_measured = self.llm_provider == "anthropic" and bool(t.get("llm_ttft_ms"))
        return ChatbotRun(
            question=question, answer=resp.answer, answer_type=resp.answer_type,
            behavior=config.BEHAVIOR_BY_ANSWER_TYPE.get(resp.answer_type, "Answer"),
            model=resp.model or self.llm_model, llm_provider=self.llm_provider, embedding_model=self.embedding_model,
            latency_ms=resp.latency_ms, timings=t, ttft_ms=t.get("llm_ttft_ms") if ttft_measured else None,
            ttft_measured=ttft_measured, prompt_tokens=resp.prompt_tokens, completion_tokens=resp.completion_tokens,
            cost_usd=resp.cost_usd, knowledge_bases=list(resp.knowledge_bases), route=dict(resp.route or {}),
            golden_match=resp.golden_match, grounding=resp.grounding,
            citations=[asdict(c) for c in resp.citations], retrieved=retrieved, contexts=contexts,
            warnings=list(resp.warnings), started_at=started_at)

    @staticmethod
    def _context_from_ref(ref) -> ContextSource:
        m = ref.metadata or {}
        text = str(m.get("answer")) if ref.is_golden and m.get("answer") else ref.text
        if ref.is_golden and m.get("question"):
            text = f"Q: {m.get('question')}\nA: {text}"
        return ContextSource(sid=ref.sid, chunk_id=ref.candidate.id, parent_id=str(m.get("parent_id") or ""),
                             doc_name=str(m.get("doc_name", "")), section=str(m.get("section", "")),
                             url=str(m.get("source_url") or ""), is_golden=bool(ref.is_golden), text=text)

    @staticmethod
    def _golden_context(resp) -> list[ContextSource]:
        """Approved-answer fast path: the approved Q&A row is the only source the chatbot used."""
        from backend.retrieval.kb_index import get_index_manager

        out = []
        for item in resp.retrieved[:1]:
            idx = get_index_manager().get(item["kb"])
            pos = idx.id_to_pos.get(item["id"])
            if pos is None:
                continue
            ch = idx.chunks[pos]
            m = ch.metadata
            text = f"Q: {m.get('question', '')}\nA: {m.get('answer') or ch.text}"
            out.append(ContextSource(sid="S1", chunk_id=ch.id, parent_id="", doc_name=str(m.get("doc_name", "")),
                                     section=str(m.get("section", "")), url=str(m.get("source_url") or ""),
                                     is_golden=True, text=text))
        return out


def index_inventory(kb: str | None = None) -> dict[str, Any]:
    """Read-only view of the current vector store: chunk ids, parent ids and document names (for validation)."""
    from backend.retrieval.kb_index import get_index_manager

    mgr = get_index_manager()
    kbs = [kb] if kb else mgr.kb_names()
    chunk_ids, parent_ids, docs = set(), set(), set()
    for name in kbs:
        idx = mgr.get(name)
        for c in idx.chunks:
            chunk_ids.add(c.id)
            if c.metadata.get("parent_id"):
                parent_ids.add(str(c.metadata["parent_id"]))
            for k in ("doc_name", "source_file", "doc_id", "section", "qid"):
                if c.metadata.get(k):
                    docs.add(str(c.metadata[k]))
    return {"chunk_ids": chunk_ids, "parent_ids": parent_ids, "documents": docs, "kbs": kbs}
