"""Central, environment-driven configuration for the Sapphirus AI Assistant.

Every tunable (LLM provider, embedding backend, vector store, thresholds, security
options) is read from environment variables (optionally loaded from a `.env` file).
Knowledge-base definitions live in `config/knowledge_bases.yaml` so datasets can be
replaced without touching application code.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env", override=False)


def _env(name: str, default: str = "") -> str:
    value = os.getenv(name)
    return default if value is None or value.strip() == "" else value.strip()


def _env_bool(name: str, default: bool) -> bool:
    return _env(name, str(default)).lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    return int(_env(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(_env(name, str(default)))


def _resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else (BASE_DIR / path).resolve()


@dataclass
class Settings:
    # ---- General ----
    app_name: str = "Sapphirus AI Assistant"
    app_env: str = field(default_factory=lambda: _env("APP_ENV", "development"))
    log_level: str = field(default_factory=lambda: _env("LOG_LEVEL", "INFO"))
    base_dir: Path = BASE_DIR
    data_dir: Path = field(default_factory=lambda: _resolve(_env("DATA_DIR", "data")))
    logs_dir: Path = field(default_factory=lambda: _resolve(_env("LOGS_DIR", "logs")))
    kb_config_path: Path = field(default_factory=lambda: _resolve(_env("KB_CONFIG_PATH", "config/knowledge_bases.yaml")))
    prompts_dir: Path = field(default_factory=lambda: _resolve(_env("PROMPTS_DIR", "prompts")))
    database_url: str = field(default_factory=lambda: _env("DATABASE_URL", ""))

    # ---- LLM ----
    llm_provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "openai").lower())
    openai_api_key: str = field(default_factory=lambda: _env("OPENAI_API_KEY", ""))
    openai_base_url: str = field(default_factory=lambda: _env("OPENAI_BASE_URL", ""))
    openai_model: str = field(default_factory=lambda: _env("OPENAI_MODEL", "gpt-4o-mini"))
    # Tried in order when OPENAI_MODEL is rate-limited or erroring (same endpoint and key), e.g. openai/gpt-oss-20b
    openai_fallback_models: list[str] = field(default_factory=lambda: [m.strip() for m in _env(
        "OPENAI_FALLBACK_MODELS", "").split(",") if m.strip()])
    openai_reasoning_effort: str = field(default_factory=lambda: _env("OPENAI_REASONING_EFFORT", ""))  # reasoning models
    openai_price_input_per_1m: float = field(default_factory=lambda: _env_float("OPENAI_PRICE_INPUT_PER_1M", 0.0))
    openai_price_output_per_1m: float = field(default_factory=lambda: _env_float("OPENAI_PRICE_OUTPUT_PER_1M", 0.0))
    # Used only while the primary provider has no API key (e.g. Claude primary, Groq/OpenAI-compatible fallback)
    llm_fallback_provider: str = field(default_factory=lambda: _env("LLM_FALLBACK_PROVIDER", "").lower())
    anthropic_api_key: str = field(default_factory=lambda: _env("ANTHROPIC_API_KEY", ""))
    anthropic_model: str = field(default_factory=lambda: _env("ANTHROPIC_MODEL", "claude-sonnet-4-5"))
    llm_temperature: float = field(default_factory=lambda: _env_float("LLM_TEMPERATURE", 0.0))
    llm_max_tokens: int = field(default_factory=lambda: _env_int("LLM_MAX_TOKENS", 900))
    llm_timeout_seconds: int = field(default_factory=lambda: _env_int("LLM_TIMEOUT_SECONDS", 60))
    llm_max_retries: int = field(default_factory=lambda: _env_int("LLM_MAX_RETRIES", 2))
    # Pricing is configuration, not code: set it to your provider's current published price (USD per 1M tokens).
    price_input_per_1m: float = field(default_factory=lambda: _env_float("LLM_PRICE_INPUT_PER_1M", 0.0))
    price_output_per_1m: float = field(default_factory=lambda: _env_float("LLM_PRICE_OUTPUT_PER_1M", 0.0))
    embedding_price_per_1m: float = field(default_factory=lambda: _env_float("EMBEDDING_PRICE_PER_1M", 0.0))
    rerank_price_per_1k: float = field(default_factory=lambda: _env_float("RERANK_PRICE_PER_1K", 0.0))
    insufficient_info_text: str = field(default_factory=lambda: _env(
        "INSUFFICIENT_INFO_TEXT",
        "I couldn't find sufficient information in the available Sapphirus documentation to answer this question."))
    out_of_scope_text: str = field(default_factory=lambda: _env(
        "OUT_OF_SCOPE_TEXT",
        "I'm sorry, but that's outside what I can help with. I can answer questions about Sapphirus and its "
        "LearningChain platform - for example issuing, sharing or verifying blockchain certificates, the admin and "
        "user portals, or Sapphirus company information."))

    # ---- Automated ingestion (queue worker, website monitor, webhooks) ----
    ingest_worker_enabled: bool = field(default_factory=lambda: _env_bool("INGEST_WORKER_ENABLED", True))
    ingest_poll_seconds: float = field(default_factory=lambda: _env_float("INGEST_POLL_SECONDS", 2.0))
    ingest_max_attempts: int = field(default_factory=lambda: _env_int("INGEST_MAX_ATTEMPTS", 3))
    ingest_retry_base_seconds: float = field(default_factory=lambda: _env_float("INGEST_RETRY_BASE_SECONDS", 30.0))
    web_monitor_enabled: bool = field(default_factory=lambda: _env_bool("WEB_MONITOR_ENABLED", True))
    web_monitor_interval_minutes: int = field(default_factory=lambda: _env_int("WEB_MONITOR_INTERVAL_MINUTES", 360))
    web_fetch_timeout_seconds: int = field(default_factory=lambda: _env_int("WEB_FETCH_TIMEOUT_SECONDS", 20))
    web_max_page_mb: int = field(default_factory=lambda: _env_int("WEB_MAX_PAGE_MB", 5))
    web_allowed_domains: list[str] = field(default_factory=lambda: [d.strip().lower() for d in _env(
        "WEB_ALLOWED_DOMAINS", "sapphirus.in,learningchain.in,learningchainai.zendesk.com").split(",") if d.strip()])
    # Web page scraping for monitored sources: "firecrawl" (Firecrawl API, renders JavaScript pages and returns clean
    # Markdown) or "builtin" (direct HTTP + trafilatura). Without FIRECRAWL_API_KEY the builtin scraper is used.
    web_scraper: str = field(default_factory=lambda: _env("WEB_SCRAPER", "firecrawl").lower())
    firecrawl_api_key: str = field(default_factory=lambda: _env("FIRECRAWL_API_KEY", ""))
    firecrawl_api_url: str = field(default_factory=lambda: _env("FIRECRAWL_API_URL", "https://api.firecrawl.dev/v2"))
    firecrawl_timeout_seconds: int = field(default_factory=lambda: _env_int("FIRECRAWL_TIMEOUT_SECONDS", 60))
    # Firecrawl may serve a cached copy younger than this (ms). 0 = always scrape fresh (needed for change detection).
    firecrawl_max_age_ms: int = field(default_factory=lambda: _env_int("FIRECRAWL_MAX_AGE_MS", 0))
    webhook_secret: str = field(default_factory=lambda: _env("WEBHOOK_SECRET", ""))
    admin_notify_webhook_url: str = field(default_factory=lambda: _env("ADMIN_NOTIFY_WEBHOOK_URL", ""))
    # Sites allowed to embed the chat widget iframe (Content-Security-Policy frame-ancestors)
    widget_allowed_origins: list[str] = field(default_factory=lambda: [o.strip() for o in _env(
        "WIDGET_ALLOWED_ORIGINS", "https://sapphirus.in,https://www.sapphirus.in,https://learningchain.in,"
        "https://www.learningchain.in").split(",") if o.strip()])

    # ---- Embeddings / vector store / reranker ----
    embedding_backend: str = field(default_factory=lambda: _env("EMBEDDING_BACKEND", "openai").lower())
    embedding_model: str = field(default_factory=lambda: _env("EMBEDDING_MODEL", "text-embedding-3-small"))
    embedding_batch_size: int = field(default_factory=lambda: _env_int("EMBEDDING_BATCH_SIZE", 64))
    # Separate credentials for EMBEDDING_BACKEND=openai, so embeddings can use api.openai.com while
    # OPENAI_API_KEY / OPENAI_BASE_URL point at another OpenAI-compatible LLM host (e.g. Groq).
    # Empty -> fall back to OPENAI_API_KEY / OPENAI_BASE_URL.
    embedding_api_key: str = field(default_factory=lambda: _env("EMBEDDING_API_KEY", ""))
    embedding_base_url: str = field(default_factory=lambda: _env("EMBEDDING_BASE_URL", ""))
    vector_backend: str = field(default_factory=lambda: _env("VECTOR_BACKEND", "chroma").lower())
    reranker_backend: str = field(default_factory=lambda: _env("RERANKER_BACKEND", "cross_encoder").lower())
    reranker_model: str = field(default_factory=lambda: _env("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"))

    # ---- Retrieval ----
    top_k_vector: int = field(default_factory=lambda: _env_int("TOP_K_VECTOR", 20))
    top_k_bm25: int = field(default_factory=lambda: _env_int("TOP_K_BM25", 20))
    top_k_rerank: int = field(default_factory=lambda: _env_int("TOP_K_RERANK", 5))
    rrf_k: int = field(default_factory=lambda: _env_int("RRF_K", 60))
    vector_weight: float = field(default_factory=lambda: _env_float("HYBRID_VECTOR_WEIGHT", 0.5))
    min_rerank_score: float = field(default_factory=lambda: _env_float("MIN_RERANK_SCORE", 0.15))
    min_query_coverage_llm: float = field(default_factory=lambda: _env_float("MIN_QUERY_COVERAGE_LLM", 0.25))
    min_query_coverage: float = field(default_factory=lambda: _env_float("MIN_QUERY_COVERAGE", 0.40))
    max_context_chars: int = field(default_factory=lambda: _env_int("MAX_CONTEXT_CHARS", 9000))

    # ---- Golden dataset ----
    golden_direct_threshold: float = field(default_factory=lambda: _env_float("GOLDEN_DIRECT_THRESHOLD", 0.78))
    golden_assist_threshold: float = field(default_factory=lambda: _env_float("GOLDEN_ASSIST_THRESHOLD", 0.50))

    # ---- Routing / grounding ----
    router_mode: str = field(default_factory=lambda: _env("ROUTER_MODE", "hybrid").lower())
    grounding_mode: str = field(default_factory=lambda: _env("GROUNDING_MODE", "lexical").lower())
    grounding_min_support: float = field(default_factory=lambda: _env_float("GROUNDING_MIN_SUPPORT", 0.6))

    # ---- Corrective RAG (backend/retrieval/crag.py) ----
    # An LLM grades each retrieved chunk for relevance before generation; irrelevant chunks are dropped, and when
    # none is relevant the query is rewritten and the knowledge base searched once more (never the web).
    crag_enabled: bool = field(default_factory=lambda: _env_bool("CRAG_ENABLED", True))
    crag_retry: bool = field(default_factory=lambda: _env_bool("CRAG_RETRY", True))
    crag_chunk_chars: int = field(default_factory=lambda: _env_int("CRAG_CHUNK_CHARS", 700))

    # ---- Conversational intelligence layer (backend/routing/intent.py) ----
    conversation_layer: bool = field(default_factory=lambda: _env_bool("CONVERSATION_LAYER", True))
    conversation_llm_intent: bool = field(default_factory=lambda: _env_bool("CONVERSATION_LLM_INTENT", True))

    # ---- Security ----
    auth_mode: str = field(default_factory=lambda: _env("AUTH_MODE", "token").lower())
    token_hash_secret: str = field(default_factory=lambda: _env("TOKEN_HASH_SECRET", ""))
    bootstrap_admin_token: str = field(default_factory=lambda: _env("BOOTSTRAP_ADMIN_TOKEN", ""))
    bootstrap_employee_token: str = field(default_factory=lambda: _env("BOOTSTRAP_EMPLOYEE_TOKEN", ""))
    bootstrap_hr_token: str = field(default_factory=lambda: _env("BOOTSTRAP_HR_TOKEN", ""))
    bootstrap_public_token: str = field(default_factory=lambda: _env("BOOTSTRAP_PUBLIC_TOKEN", ""))
    allow_anonymous_public: bool = field(default_factory=lambda: _env_bool("ALLOW_ANONYMOUS_PUBLIC", True))
    dev_roles: list[str] = field(default_factory=lambda: [r.strip() for r in _env("DEV_ROLES", "employee").split(",") if r.strip()])
    proxy_user_header: str = field(default_factory=lambda: _env("PROXY_USER_HEADER", "X-Portal-User"))
    proxy_roles_header: str = field(default_factory=lambda: _env("PROXY_ROLES_HEADER", "X-Portal-Roles"))
    proxy_shared_secret: str = field(default_factory=lambda: _env("PROXY_SHARED_SECRET", ""))
    cors_origins: list[str] = field(default_factory=lambda: [o.strip() for o in _env("CORS_ORIGINS", "http://localhost:3000").split(",") if o.strip()])
    max_upload_mb: int = field(default_factory=lambda: _env_int("MAX_UPLOAD_MB", 25))
    max_query_chars: int = field(default_factory=lambda: _env_int("MAX_QUERY_CHARS", 1000))
    chat_retention_days: int = field(default_factory=lambda: _env_int("CHAT_RETENTION_DAYS", 90))
    block_prompt_injection: bool = field(default_factory=lambda: _env_bool("BLOCK_PROMPT_INJECTION", True))
    rate_limit_per_minute: int = field(default_factory=lambda: _env_int("RATE_LIMIT_PER_MINUTE", 30))

    # ---- OCR ----
    ocr_enabled: bool = field(default_factory=lambda: _env_bool("OCR_ENABLED", False))
    tesseract_cmd: str = field(default_factory=lambda: _env("TESSERACT_CMD", ""))
    ocr_language: str = field(default_factory=lambda: _env("OCR_LANGUAGE", "eng"))

    # ---- Frontend ----
    api_base_url: str = field(default_factory=lambda: _env("API_BASE_URL", "http://localhost:8000"))
    ui_mode: str = field(default_factory=lambda: _env("UI_MODE", "api").lower())

    def __post_init__(self) -> None:
        if not self.database_url:
            self.database_url = f"sqlite:///{(self.data_dir / 'sapphirus.db').as_posix()}"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    # ---- Derived helpers ----
    @property
    def effective_llm_provider(self) -> str:
        """Provider actually used: the primary if its key is set, else LLM_FALLBACK_PROVIDER if its key is set,
        else extractive (no LLM)."""
        has_key = {"openai": bool(self.openai_api_key), "anthropic": bool(self.anthropic_api_key)}
        for provider in (self.llm_provider, self.llm_fallback_provider):
            if provider in has_key and has_key[provider]:
                return provider
        return "extractive"

    @property
    def llm_model_name(self) -> str:
        provider = self.effective_llm_provider
        if provider == "openai":
            return self.openai_model
        if provider == "anthropic":
            return self.anthropic_model
        return "extractive-fallback"

    def kb_config(self) -> dict[str, Any]:
        return load_kb_config(str(self.kb_config_path))


@lru_cache(maxsize=4)
def load_kb_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if "knowledge_bases" not in config:
        raise ValueError(f"{path} must define a top-level 'knowledge_bases' mapping")
    return config


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings() -> None:
    """Used by tests to pick up new environment variables."""
    global _settings
    _settings = None
    load_kb_config.cache_clear()
