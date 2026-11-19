"""Relational schema (PostgreSQL in production, SQLite for local development).

Tables
------
users          : portal users, roles and hashed API tokens
documents      : document registry (versions, hashes, ACLs) - one row per source document
kb_state       : per-knowledge-base revision counter (used to refresh in-memory indexes)
chat_sessions  : conversation containers
chat_messages  : every user question and assistant answer (with routing, citations, latency, tokens, cost)
feedback       : Helpful / Not Helpful ratings on assistant messages
audit_logs     : security-relevant events (logins, uploads, deletions, access denials, injection attempts)
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(120), unique=True, index=True)
    display_name: Mapped[str] = mapped_column(String(200), default="")
    roles: Mapped[str] = mapped_column(String(200), default="employee")  # comma separated
    token_hash: Mapped[str | None] = mapped_column(String(128), unique=True, index=True, nullable=True)
    product_id: Mapped[str] = mapped_column(String(64), default="", index=True)   # API key -> product ("" = any)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    @property
    def role_list(self) -> list[str]:
        return [r.strip() for r in self.roles.split(",") if r.strip()]


class Document(Base):
    __tablename__ = "documents"
    __table_args__ = (UniqueConstraint("kb", "source_path", name="uq_kb_source"),)
    doc_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    kb: Mapped[str] = mapped_column(String(64), index=True)
    product_id: Mapped[str] = mapped_column(String(64), default="", index=True)   # config/products/<id>.yaml
    doc_name: Mapped[str] = mapped_column(String(300))
    source_path: Mapped[str] = mapped_column(String(600))
    file_type: Mapped[str] = mapped_column(String(16))
    doc_type: Mapped[str] = mapped_column(String(32), default="document")  # document | golden_qa | catalog
    category: Mapped[str] = mapped_column(String(120), default="")
    access_roles: Mapped[str] = mapped_column(String(200), default="")
    confidentiality: Mapped[str] = mapped_column(String(32), default="internal")
    content_hash: Mapped[str] = mapped_column(String(64), index=True)
    version: Mapped[int] = mapped_column(Integer, default=1)
    num_chunks: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(16), default="active")  # active | deleted
    is_sample: Mapped[bool] = mapped_column(Boolean, default=False)
    uploaded_by: Mapped[str] = mapped_column(String(120), default="system")
    uploaded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)
    notes: Mapped[str] = mapped_column(Text, default="")
    # freshness tracking (Phase 3, DOC_TRACKING_ENABLED): crawled pages missing from N crawls are tombstoned
    source_url: Mapped[str] = mapped_column(String(600), default="", index=True)
    last_seen_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_changed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    missed_crawls: Mapped[int] = mapped_column(Integer, default=0)
    tombstoned_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_crawl_id: Mapped[str] = mapped_column(String(64), default="")


class KBState(Base):
    __tablename__ = "kb_state"
    kb: Mapped[str] = mapped_column(String(64), primary_key=True)
    revision: Mapped[int] = mapped_column(Integer, default=0)
    embedding_model: Mapped[str] = mapped_column(String(200), default="")
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class ChatSession(Base):
    __tablename__ = "chat_sessions"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    username: Mapped[str] = mapped_column(String(120), index=True)
    product_id: Mapped[str] = mapped_column(String(64), default="", index=True)   # config/products/<id>.yaml
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    cleared: Mapped[bool] = mapped_column(Boolean, default=False)
    messages: Mapped[list["ChatMessage"]] = relationship(back_populates="session", cascade="all, delete-orphan")


class ChatMessage(Base):
    __tablename__ = "chat_messages"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("chat_sessions.id", ondelete="CASCADE"), index=True)
    role: Mapped[str] = mapped_column(String(16))  # user | assistant
    product_id: Mapped[str] = mapped_column(String(64), default="", index=True)   # config/products/<id>.yaml
    content: Mapped[str] = mapped_column(Text)
    knowledge_bases: Mapped[list] = mapped_column(JSON, default=list)
    citations: Mapped[list] = mapped_column(JSON, default=list)
    route_info: Mapped[dict] = mapped_column(JSON, default=dict)
    answer_type: Mapped[str] = mapped_column(String(32), default="")
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    session: Mapped[ChatSession] = relationship(back_populates="messages")


class Feedback(Base):
    __tablename__ = "feedback"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    message_id: Mapped[int] = mapped_column(ForeignKey("chat_messages.id", ondelete="CASCADE"), index=True)
    username: Mapped[str] = mapped_column(String(120))
    rating: Mapped[str] = mapped_column(String(16))  # helpful | not_helpful
    comment: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class AuditLog(Base):
    __tablename__ = "audit_logs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    username: Mapped[str] = mapped_column(String(120), default="anonymous")
    action: Mapped[str] = mapped_column(String(64), index=True)
    resource: Mapped[str] = mapped_column(String(300), default="")
    outcome: Mapped[str] = mapped_column(String(32), default="success")
    details: Mapped[dict] = mapped_column(JSON, default=dict)
    ip_address: Mapped[str] = mapped_column(String(64), default="")


class RequestMetric(Base):
    """One row per chatbot request: latency breakdown, tokens and cost (feeds the performance dashboard)."""
    __tablename__ = "request_metrics"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    channel: Mapped[str] = mapped_column(String(32), default="api", index=True)        # api | openai | widget | eval
    assistant: Mapped[str] = mapped_column(String(32), default="all", index=True)      # company | products | all
    product_id: Mapped[str] = mapped_column(String(64), default="", index=True)   # config/products/<id>.yaml
    username: Mapped[str] = mapped_column(String(120), default="")
    session_id: Mapped[str] = mapped_column(String(64), default="")
    answer_type: Mapped[str] = mapped_column(String(32), default="", index=True)
    knowledge_bases: Mapped[str] = mapped_column(String(200), default="")
    success: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    error: Mapped[str] = mapped_column(String(500), default="")
    model: Mapped[str] = mapped_column(String(80), default="")
    end_to_end_ms: Mapped[float] = mapped_column(Float, default=0.0)
    ttft_ms: Mapped[float] = mapped_column(Float, default=0.0)          # request received -> first answer byte sent
    llm_ttft_ms: Mapped[float] = mapped_column(Float, default=0.0)      # LLM call start -> first model token
    llm_generation_ms: Mapped[float] = mapped_column(Float, default=0.0)
    llm_total_ms: Mapped[float] = mapped_column(Float, default=0.0)
    retrieval_ms: Mapped[float] = mapped_column(Float, default=0.0)
    vector_search_ms: Mapped[float] = mapped_column(Float, default=0.0)
    bm25_search_ms: Mapped[float] = mapped_column(Float, default=0.0)
    rerank_ms: Mapped[float] = mapped_column(Float, default=0.0)
    context_ms: Mapped[float] = mapped_column(Float, default=0.0)
    tokens_per_second: Mapped[float] = mapped_column(Float, default=0.0)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    llm_cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    embedding_cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    rerank_cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    total_cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    timings: Mapped[dict] = mapped_column(JSON, default=dict)
    # full request logging (Phase 2): enough to replay / debug any answer
    index_version: Mapped[str] = mapped_column(String(200), default="")     # "kb:v<id>,..." of the indexes used
    prompt_version: Mapped[str] = mapped_column(String(64), default="")     # product prompt_version + prompt files hash
    provider: Mapped[str] = mapped_column(String(64), default="")           # LLM provider that produced the answer
    intent: Mapped[str] = mapped_column(String(32), default="", index=True)
    cache_hit: Mapped[bool] = mapped_column(Boolean, default=False)
    golden_hit: Mapped[bool] = mapped_column(Boolean, default=False)
    refused: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    retrieved: Mapped[list] = mapped_column(JSON, default=list)             # [{"id", "kb", "score"}] top chunks
    message_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)


class IndexVersion(Base):
    """One build of a knowledge base's vector collection. Exactly one version per KB is `live` (index_aliases)."""
    __tablename__ = "index_versions"
    version_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    product_id: Mapped[str] = mapped_column(String(64), default="", index=True)
    kb: Mapped[str] = mapped_column(String(64), index=True)
    collection: Mapped[str] = mapped_column(String(200))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    crawl_id: Mapped[str] = mapped_column(String(64), default="")
    embedding_model: Mapped[str] = mapped_column(String(200), default="")
    chunk_config: Mapped[dict] = mapped_column(JSON, default=dict)
    doc_count: Mapped[int] = mapped_column(Integer, default=0)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    eval_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    eval_details: Mapped[dict] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(16), default="building", index=True)  # building|live|retired|failed|rejected
    notes: Mapped[str] = mapped_column(Text, default="")


class IndexAlias(Base):
    """Which index version serves a knowledge base; switching the row is the atomic promote / rollback."""
    __tablename__ = "index_aliases"
    kb: Mapped[str] = mapped_column(String(64), primary_key=True)
    product_id: Mapped[str] = mapped_column(String(64), default="")
    live_version_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    previous_version_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    switched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class WebSource(Base):
    """A monitored website page / sitemap / help-centre feed that is crawled into a knowledge base."""
    __tablename__ = "web_sources"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    url: Mapped[str] = mapped_column(String(1000), unique=True)
    kb: Mapped[str] = mapped_column(String(64), index=True)
    product_id: Mapped[str] = mapped_column(String(64), default="", index=True)   # config/products/<id>.yaml
    source_type: Mapped[str] = mapped_column(String(32), default="page")   # page | sitemap | zendesk | faq
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    interval_minutes: Mapped[int] = mapped_column(Integer, default=360)
    content_hash: Mapped[str] = mapped_column(String(64), default="")
    etag: Mapped[str] = mapped_column(String(200), default="")
    last_modified: Mapped[str] = mapped_column(String(100), default="")
    last_checked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_changed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_status: Mapped[str] = mapped_column(String(32), default="new")   # new | unchanged | changed | failed
    last_error: Mapped[str] = mapped_column(String(500), default="")
    created_by: Mapped[str] = mapped_column(String(120), default="system")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class IngestionJob(Base):
    """Queue entry for background ingestion (uploads, replacements, deletions, web crawls, syncs).

    Idempotency: `dedupe_key` (operation + target + content hash) is unique among non-finished jobs, so the same
    work enqueued twice - or retried after a crash - never produces duplicate records."""
    __tablename__ = "ingestion_jobs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    job_type: Mapped[str] = mapped_column(String(32), index=True)    # ingest_file | delete_document | crawl_source | sync_kb
    kb: Mapped[str] = mapped_column(String(64), default="", index=True)
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    dedupe_key: Mapped[str] = mapped_column(String(300), default="", index=True)
    status: Mapped[str] = mapped_column(String(16), default="queued", index=True)  # queued|running|succeeded|failed|dead
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, default=3)
    next_run_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    locked_by: Mapped[str] = mapped_column(String(100), default="")
    locked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    result: Mapped[dict] = mapped_column(JSON, default=dict)
    error: Mapped[str] = mapped_column(Text, default="")
    requested_by: Mapped[str] = mapped_column(String(120), default="system")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class EvaluationRun(Base):
    """One evaluation run (dataset + configuration + summary) - feeds historical trends and before/after comparisons."""
    __tablename__ = "evaluation_runs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    label: Mapped[str] = mapped_column(String(200), default="")
    run_type: Mapped[str] = mapped_column(String(32), default="rag", index=True)   # rag | hallucination | benchmark
    config: Mapped[dict] = mapped_column(JSON, default=dict)
    summary: Mapped[dict] = mapped_column(JSON, default=dict)


class ResponseCache(Base):
    """Answer cache (Phase 4, RESPONSE_CACHE_ENABLED). Keyed by product + scope + roles + index version + prompt
    version + normalised question, so a new index or prompt version can never serve an old answer."""
    __tablename__ = "response_cache"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    product_id: Mapped[str] = mapped_column(String(64), default="", index=True)
    cache_key: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    scope_key: Mapped[str] = mapped_column(String(64), index=True)      # everything but the question (semantic lookup)
    normalized_query: Mapped[str] = mapped_column(Text, default="")
    embedding: Mapped[list | None] = mapped_column(JSON, nullable=True)
    index_version: Mapped[str] = mapped_column(String(200), default="")
    prompt_version: Mapped[str] = mapped_column(String(64), default="")
    response: Mapped[dict] = mapped_column(JSON, default=dict)
    hits: Mapped[int] = mapped_column(Integer, default=0)
    source_message_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    last_message_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    last_hit_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class ReviewItem(Base):
    """Answers a person should look at (Phase 6, REVIEW_QUEUE_ENABLED): thumbs-down, refusals, low confidence,
    changed / removed source pages. Resolving one can add an approved answer (golden_qa) and an eval case."""
    __tablename__ = "review_items"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    product_id: Mapped[str] = mapped_column(String(64), default="", index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    source: Mapped[str] = mapped_column(String(32), index=True)     # thumbs_down | refusal | low_confidence | stale_doc | manual
    status: Mapped[str] = mapped_column(String(16), default="open", index=True)   # open | resolved | dismissed
    question: Mapped[str] = mapped_column(Text, default="")
    answer: Mapped[str] = mapped_column(Text, default="")
    message_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    cluster_key: Mapped[str] = mapped_column(String(64), default="", index=True)
    occurrences: Mapped[int] = mapped_column(Integer, default=1)
    details: Mapped[dict] = mapped_column(JSON, default=dict)
    resolved_by: Mapped[str] = mapped_column(String(120), default="")
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    resolution: Mapped[str] = mapped_column(Text, default="")


class GoldenQA(Base):
    """Approved answers managed in the database (CSV import, review queue). Exported to the KB's golden file."""
    __tablename__ = "golden_qa"
    __table_args__ = (UniqueConstraint("product_id", "qid", name="uq_golden_product_qid"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    product_id: Mapped[str] = mapped_column(String(64), default="", index=True)
    kb: Mapped[str] = mapped_column(String(64), default="")
    qid: Mapped[str] = mapped_column(String(64))
    category: Mapped[str] = mapped_column(String(120), default="")
    question: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text)
    alternates: Mapped[list] = mapped_column(JSON, default=list)
    keywords: Mapped[str] = mapped_column(String(500), default="")
    source_url: Mapped[str] = mapped_column(String(600), default="")
    status: Mapped[str] = mapped_column(String(16), default="active", index=True)   # active | inactive
    origin: Mapped[str] = mapped_column(String(32), default="csv")                  # csv | review | api
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class EvalCase(Base):
    """Regression cases for the retrieval eval (index gate + nightly eval); added from resolved review items."""
    __tablename__ = "eval_cases"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    product_id: Mapped[str] = mapped_column(String(64), default="", index=True)
    kb: Mapped[str] = mapped_column(String(64), default="")
    question: Mapped[str] = mapped_column(Text)
    expected_answer: Mapped[str] = mapped_column(Text, default="")
    expected_sources: Mapped[list] = mapped_column(JSON, default=list)
    origin: Mapped[str] = mapped_column(String(32), default="review")
    active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class AlertEvent(Base):
    """Fired alerts (Phase 7, ALERTS_ENABLED): rule, value vs threshold, whether a notification went out."""
    __tablename__ = "alert_events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    rule: Mapped[str] = mapped_column(String(64), index=True)
    severity: Mapped[str] = mapped_column(String(16), default="warning")
    product_id: Mapped[str] = mapped_column(String(64), default="")
    value: Mapped[float] = mapped_column(Float, default=0.0)
    threshold: Mapped[float] = mapped_column(Float, default=0.0)
    message: Mapped[str] = mapped_column(Text, default="")
    notified: Mapped[bool] = mapped_column(Boolean, default=False)
