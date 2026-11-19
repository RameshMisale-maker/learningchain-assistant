"""FastAPI application entry point.

Run:  uvicorn backend.main:app --host 0.0.0.0 --port 8000
Docs: http://localhost:8000/docs
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

from fastapi.responses import FileResponse, JSONResponse

from backend.api import routes_admin, routes_chat, routes_openai, routes_webhooks
from backend.common.logging_config import setup_logging
from backend.db import repository
from backend.db.session import get_engine
from backend.security.auth import bootstrap_users
from config.settings import get_settings

logger = logging.getLogger(__name__)


def validate_production_config(settings) -> list[str]:
    """Refuse to start an insecure production configuration; return warnings for weaker-but-allowed choices."""
    if settings.app_env != "production":
        return []
    errors, warnings = [], []
    if len(settings.token_hash_secret) < 32:
        errors.append("TOKEN_HASH_SECRET must be set (32+ random characters)")
    if settings.auth_mode == "disabled":
        errors.append("AUTH_MODE=disabled is not allowed in production")
    if "*" in settings.cors_origins:
        errors.append("CORS_ORIGINS must list explicit origins, not '*'")
    if settings.database_url.startswith("sqlite"):
        warnings.append("SQLite database in production - use PostgreSQL (DATABASE_URL)")
    if settings.webhook_secret and len(settings.webhook_secret) < 24:
        warnings.append("WEBHOOK_SECRET is short - use 24+ random characters")
    if errors:
        raise RuntimeError("Unsafe production configuration: " + "; ".join(errors))
    return warnings


@asynccontextmanager
async def lifespan(_: FastAPI):
    setup_logging()
    settings = get_settings()
    for warning in validate_production_config(settings):
        logger.warning("Configuration: %s", warning)
    from backend.products import default_product_id, load_products
    products = load_products()      # fail fast: an invalid config/products/*.yaml stops startup with a clear error
    logger.info("Products: %s (default %s, per-request resolution %s)", ", ".join(sorted(products)),
                default_product_id(), "on" if settings.product_config_enabled else "off")
    get_engine()                    # also applies pending Alembic migrations
    created = bootstrap_users()
    if created:
        logger.info("Bootstrap users ready: %s", ", ".join(created))
    purged = repository.purge_old_history(settings.chat_retention_days)
    if purged:
        logger.info("Retention: purged %d old chat messages", purged)
    logger.info("Assistant API started (env=%s, llm=%s, embeddings=%s, vector=%s, auth=%s)",
                settings.app_env, settings.llm_model_name, settings.embedding_backend, settings.vector_backend,
                settings.auth_mode)
    if settings.app_env == "production" and settings.llm_provider != "extractive" and             settings.effective_llm_provider == "extractive":
        logger.error("LLM_PROVIDER=%s but its API key is missing - serving degraded extractive answers",
                     settings.llm_provider)
    from backend.observability import scheduler
    if scheduler.start():
        logger.info("Ops scheduler running (alerts %s, nightly eval %s)", "on" if settings.alerts_enabled else "off",
                    "on" if settings.nightly_eval_enabled else "off")
    from ingestion.automation import start_worker, stop_worker
    worker = start_worker()
    if worker:
        logger.info("Background ingestion worker running (website monitor %s)",
                    "on" if settings.web_monitor_enabled else "off")
    yield
    stop_worker()
    scheduler.stop()


app = FastAPI(title="RAG Assistant API", version="1.0.0", lifespan=lifespan,
              description="Multi-product, multi-knowledge-base RAG assistant (products: config/products/*.yaml).")
app.add_middleware(CORSMiddleware, allow_origins=get_settings().cors_origins, allow_credentials=True,
                   allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
                   allow_headers=["Authorization", "Content-Type", "X-Requested-With"])


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers.setdefault("Cache-Control", "no-store")
    path = request.url.path
    page_csp = ("default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data:; connect-src 'self'; base-uri 'none'; form-action 'self'; object-src 'none'")
    if path == "/chat":       # embeddable on the approved company sites only
        ancestors = " ".join(["'self'", *get_settings().widget_allowed_origins])
        response.headers["Content-Security-Policy"] = f"{page_csp}; frame-ancestors {ancestors}"
    elif path == "/admin":
        response.headers["Content-Security-Policy"] = f"{page_csp}; frame-ancestors 'none'"
        response.headers["X-Frame-Options"] = "DENY"
    else:
        response.headers["X-Frame-Options"] = "SAMEORIGIN"
    if request.url.scheme == "https":
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.get("/health", tags=["system"])
def health() -> dict:
    """Summary health (kept for backward compatibility). Reveals no secrets."""
    settings = get_settings()
    return {"status": "ok", "llm": settings.llm_model_name, "llm_configured": settings.effective_llm_provider != "extractive",
            "embedding_backend": settings.embedding_backend, "vector_backend": settings.vector_backend}


@app.get("/metrics", include_in_schema=False)
def prometheus_metrics(request: Request):
    """Prometheus scrape endpoint (METRICS_ENABLED). Protect it with METRICS_TOKEN or keep it on a private network."""
    from fastapi.responses import PlainTextResponse

    settings = get_settings()
    if not settings.metrics_enabled:
        return JSONResponse(status_code=404, content={"detail": "Not Found"})
    if settings.metrics_token:
        import hmac

        given = request.headers.get("authorization", "").removeprefix("Bearer ").strip()
        if not hmac.compare_digest(given, settings.metrics_token):
            return JSONResponse(status_code=401, content={"detail": "metrics token required"})
    from backend.observability.metrics import render

    return PlainTextResponse(render(), media_type="text/plain; version=0.0.4")


@app.get("/health/live", tags=["system"])
def liveness() -> dict:
    """Liveness: the process is running and serving HTTP (no dependency checks - restart only if this fails)."""
    return {"status": "alive"}


@app.get("/health/ready", tags=["system"])
def readiness() -> JSONResponse:
    """Readiness: database reachable and every enabled knowledge base has an index built with the current embedding
    model. Returns 503 until the service can answer correctly (load balancers should not route traffic before)."""
    from sqlalchemy import text

    from backend.retrieval.kb_index import get_index_manager
    from ingestion.embedding import get_embedder

    checks: dict[str, dict] = {}
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        checks["database"] = {"ok": True}
    except Exception as exc:
        checks["database"] = {"ok": False, "error": type(exc).__name__}
    try:
        model_id = get_embedder().model_id
        manager = get_index_manager()
        for kb in manager.kb_names():
            idx = manager.get(kb)
            models = {str(c.metadata.get("embedding_model", "")) for c in idx.chunks[:50]} - {""}
            checks[f"kb:{kb}"] = {"ok": idx.size > 0 and (not models or models == {model_id}), "chunks": idx.size,
                                  "embedding_model": sorted(models) or None}
    except Exception as exc:
        checks["knowledge_bases"] = {"ok": False, "error": type(exc).__name__}
    settings = get_settings()
    checks["llm"] = {"ok": True, "provider": settings.effective_llm_provider,
                     "note": "" if settings.effective_llm_provider != "extractive" else
                     "No LLM key configured - answers are extractive (degraded, still grounded)"}
    from backend.generation.llm_client import get_llm

    llm = get_llm()
    if hasattr(llm, "status"):           # LLM_FAILOVER_ENABLED: per-provider circuit breaker state
        checks["llm"]["providers"] = llm.status()
    ready = all(c.get("ok") for c in checks.values())
    return JSONResponse(status_code=200 if ready else 503, content={"status": "ready" if ready else "not_ready",
                                                                   "checks": checks})


app.include_router(routes_chat.router)
app.include_router(routes_admin.router)
app.include_router(routes_openai.router)  # OpenAI-compatible API for Open WebUI
app.include_router(routes_webhooks.router)  # signed CMS / help-centre change notifications

STATIC_DIR = Path(__file__).resolve().parent / "static"


@app.get("/", include_in_schema=False)
def root():
    """Hosting platforms (e.g. Hugging Face Spaces) open the site root: send visitors to the chat page."""
    from fastapi.responses import RedirectResponse

    return RedirectResponse("/chat")


@app.get("/chat", include_in_schema=False)
def chat_page() -> FileResponse:
    """Website chat (also the iframe target of widget.js). ?assistant=products|company locks the experience."""
    return FileResponse(STATIC_DIR / "chat.html", media_type="text/html")


@app.get("/widget.js", include_in_schema=False)
def widget_script() -> FileResponse:
    return FileResponse(STATIC_DIR / "widget.js", media_type="application/javascript",
                        headers={"Cache-Control": "public, max-age=3600"})


@app.get("/admin", include_in_schema=False)
def admin_page() -> FileResponse:
    """Administrator console (documents, uploads, websites, jobs, dashboard). Requires an admin API token."""
    return FileResponse(STATIC_DIR / "admin.html", media_type="text/html")
