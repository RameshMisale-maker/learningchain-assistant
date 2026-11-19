"""Pydantic request/response models for the REST API (OpenAPI docs at /docs)."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000, description="The user's question")
    session_id: str | None = Field(None, max_length=64, description="Conversation id returned by a previous call")
    assistant: Literal["all", "company", "products"] = Field(
        "all", description="Assistant experience: company information, product information, or both")
    product: str | None = Field(None, max_length=64, pattern=r"^[a-z][a-z0-9_]*$",
                                description="Product id (config/products/<id>.yaml); used when PRODUCT_CONFIG_ENABLED")


class CitationOut(BaseModel):
    id: str
    doc_name: str
    kb: str
    source_file: str
    page: int | None = None
    section: str = ""
    doc_id: str = ""
    version: int | None = None
    qid: str = ""
    snippet: str = ""
    score: float = 0.0
    is_sample: bool = False
    is_golden: bool = False
    url: str = ""
    question: str = ""


class ChatResponseOut(BaseModel):
    answer: str
    session_id: str
    message_id: int | None
    answer_type: str
    knowledge_bases: list[str]
    citations: list[CitationOut]
    confidence: float
    latency_ms: float
    contains_sample_data: bool
    route: dict[str, Any] = {}
    golden_match: dict[str, Any] | None = None
    grounding: dict[str, Any] | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    model: str = ""
    warnings: list[str] = []
    timings: dict[str, float] = {}
    tokens_per_second: float = 0.0


class FeedbackRequest(BaseModel):
    message_id: int
    rating: Literal["helpful", "not_helpful", "conversation"]
    stars: int | None = Field(None, ge=1, le=5, description="1-5 conversation rating (rating='conversation')")
    comment: str = Field("", max_length=2000)


class SyncRequest(BaseModel):
    full_reindex: bool = False
