# Sapphirus Assistant - Production Guide

This guide covers the production build of the Sapphirus website assistant: architecture, configuration, API,
deployment, automated knowledge updates, evaluation and operations.

## 1. Architecture

```
Website visitors ──► sapphirus.in page with widget.js ──► iframe /chat ──┐
Staff (optional) ──► Open WebUI (internal) ──► /v1 OpenAI-compatible ────┤
Admins ──► /admin console ──► /api/v1/admin/* ───────────────────────────┤
CMS / Zendesk ──► /api/v1/webhooks/content-updated (HMAC) ───────────────┤
                                                                         ▼
                         nginx (TLS, rate limits, admin IP allow-list)
                                                                         ▼
     FastAPI API (1 process) ─ security: token auth, RBAC, rate limit, prompt guard, audit
       ├─ Chat pipeline: rewrite → route + scope guard → approved answers (exact + semantic)
       │                 → hybrid retrieval (vector + BM25, ACL-filtered) → cross-encoder rerank
       │                 → Claude Sonnet 4.5 (strict grounding prompt) or extractive fallback
       │                 → grounding check → citation + URL validation → answer + sources
       ├─ Metrics: per-request latency breakdown, tokens, cost → request_metrics
       └─ Ingestion worker thread: DB job queue (retries, idempotent) + website monitor
                                                                         ▼
     PostgreSQL (users, documents registry, chats, metrics, jobs, evaluation runs)
     ChromaDB (vectors, embedded) · BM25 (in memory) · data/uploads + data/archive (originals)
```

Key decisions and why:

| Decision | Reason |
|---|---|
| Keep FastAPI + plain-Python pipeline | Already working and auditable; no framework needed for a fixed pipeline. |
| Keep ChromaDB (embedded) | ~300 chunks today; benchmarked retrieval quality is limited by content, not the store. Single API process = single writer. Move to Qdrant server / pgvector only when running several API replicas. |
| DB-backed job queue (not Celery/Redis) | Volume is documents, not events; reuses PostgreSQL; durable, retryable, visible in the admin console; one less service to secure. |
| Answer verified before streaming | Users never see text that is later withdrawn by the grounding check. Trade-off: user time-to-first-byte ≈ full answer time. |
| Separate assistants enforced server-side | `allowed_kbs` is applied in the pipeline (routing, approved answers, retrieval), so the product assistant cannot read company content and vice versa, whatever the client sends. |

## 2. Configuration

All settings are environment variables - see `.env.example` (fully commented). Essentials for production:

| Variable | Value |
|---|---|
| `ANTHROPIC_API_KEY` | **Required** for generated answers (backend only). Without it answers are extractive (quoted sources). |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-5` as requested. **Deprecated - retires 2026-11-30 on the Claude API.** Switch to `claude-sonnet-5-5` and re-run the evaluation before launch. |
| `LLM_PRICE_INPUT_PER_1M` / `LLM_PRICE_OUTPUT_PER_1M` | Your model's current published price (cost tracking). |
| `TOKEN_HASH_SECRET`, `POSTGRES_PASSWORD`, `WEBHOOK_SECRET` | Long random secrets. |
| `CORS_ORIGINS`, `WIDGET_ALLOWED_ORIGINS` | The chat host and the sites that may embed the widget. |
| `WEB_ALLOWED_DOMAINS` | SSRF allow-list for the website monitor. |

Changing `EMBEDDING_MODEL` requires `python -m ingestion.cli reindex` (vectors from different models are never mixed:
the store records the model id per collection).

## 3. API reference

Interactive OpenAPI docs: `GET /docs` (keep internal - nginx blocks it publicly).

### Public (website)
| Method | Path | Purpose |
|---|---|---|
| POST | `/api/v1/chat/stream` | Streaming answer (SSE events `meta`, `delta`, `sources`, `done`, `error`). Body `{"message", "session_id"?, "assistant": "products" \| "company" \| "all"}` |
| POST | `/api/v1/chat` | Same, JSON response (answer, citations with URLs, timings, tokens, cost) |
| POST | `/api/v1/feedback` | `{"message_id", "rating": "helpful" \| "not_helpful", "comment"?}` |
| GET | `/api/v1/suggestions`, `/api/v1/me` | Suggested questions; caller identity/roles |
| GET | `/chat`, `/widget.js` | Chat page / embed script |
| GET | `/health`, `/health/live`, `/health/ready` | Health summary, liveness, readiness (503 until DB + indexes are ready) |

### OpenAI-compatible (Open WebUI, internal)
`GET /v1/models` (models: `sapphirus-assistant`, `sapphirus-company`, `sapphirus-learningchain`),
`POST /v1/chat/completions` (streaming supported). Requires an API token.

### Admin (`Authorization: Bearer <admin token>`)
| Method | Path | Purpose |
|---|---|---|
| GET | `/api/v1/admin/documents` | Documents with versions, chunk counts, roles |
| POST | `/api/v1/admin/documents` | Upload (multipart `file`, `kb`, `roles`?, `category`?, `is_golden`?) → **202 + job**; `wait=true` = synchronous |
| PUT | `/api/v1/admin/documents/{doc_id}` | Replace with a new version (same id, no duplicates) → 202 |
| DELETE | `/api/v1/admin/documents/{doc_id}` | Remove document and its vectors → 202 |
| POST | `/api/v1/admin/knowledge-bases/{kb}/sync` | Re-sync folder (`{"full_reindex": true}` to rebuild) → 202 |
| GET | `/api/v1/admin/jobs`, `/jobs/{id}` | Ingestion queue, worker status |
| POST | `/api/v1/admin/jobs/{id}/retry` | Retry a failed job (idempotent) |
| GET/POST | `/api/v1/admin/web-sources` | Monitored URLs (types `page`, `sitemap`, `zendesk`, `faq`) |
| PATCH/DELETE | `/api/v1/admin/web-sources/{id}` | Pause/resume, interval; remove (and its content) |
| POST | `/api/v1/admin/web-sources/{id}/crawl` | Check now |
| GET | `/api/v1/admin/metrics/summary`, `/metrics/timeseries` | Dashboard data (`start`, `end`, `assistant`, `channel` filters) |
| GET | `/api/v1/admin/evaluations` | Evaluation history + latest reports |
| GET | `/api/v1/admin/feedback`, `/audit-logs`, `/users` | Feedback, audit trail, users |

### Webhook
`POST /api/v1/webhooks/content-updated` - body `{"url": "..."}` or `{"source_id": N}`; header
`X-Sapphirus-Signature: sha256=<hex HMAC-SHA256(WEBHOOK_SECRET, body)>` or Zendesk's
`X-Zendesk-Webhook-Signature` + `-Timestamp`. Only re-crawls registered sources (cannot add URLs).

## 4. Deployment (Docker, single server)

1. Server: 4 vCPU / 8 GB RAM / 40 GB disk (Ubuntu 22.04+), Docker + compose plugin, DNS `chat.sapphirus.in`.
2. `git clone` the project to `/opt/sapphirus`; `cp .env.example .env`; fill in secrets (section 2).
3. TLS: put `fullchain.pem` and `privkey.pem` in `deploy/certs/` (e.g. certbot); edit `server_name` and the admin
   allow-list in `deploy/nginx.conf`.
4. `docker compose up -d --build` (first build downloads the embedding + reranker models into the image).
5. Create users/tokens: `docker compose exec api python -m backend.security.manage_users create --username web-admin --roles admin`
   (store the token in your password manager).
6. Build the index: `docker compose exec api python -m ingestion.cli sync`.
7. Register automatic sources (admin console → Websites, or API):
   - `https://learningchainai.zendesk.com/hc/en-gb` type `zendesk`, KB `products`
   - `https://learningchain.in/faqs/` type `faq`, KB `products`
   - company pages / sitemap (`https://sapphirus.in/sitemap.xml`, type `sitemap`, KB `company`) once the curated
     company files are retired.
8. Check `https://chat.sapphirus.in/health/ready` → `ready`; open `/admin`; test `/chat`.
9. Embed on the website: `<script src="https://chat.sapphirus.in/widget.js" data-assistant="products" defer></script>`.
10. Backups: `deploy/backup.sh` from cron (nightly) + copy off-site. Test a restore once.
11. Optional staff UI: `docker compose --profile staff up -d` (Open WebUI on the internal network).

**Zendesk webhook** (instant updates, optional): in Zendesk Admin Center create a webhook to
`https://chat.sapphirus.in/api/v1/webhooks/content-updated` (POST, JSON) subscribed to Help Center article
events, and use `WEBHOOK_SECRET` as its signing secret (Zendesk signature headers are verified). Any signed payload
without a `url` field re-checks all registered help-centre sources; only changed category files are re-indexed.
Check the exact event names and payload in your Zendesk plan's webhook documentation - this integration has not been
connected to the live Zendesk account yet. Without a webhook, the scheduled monitor (default every 6 h) picks up
changes.

## 5. Evaluation & monitoring

| Command | Output |
|---|---|
| `python scripts/products/learningchain/build_evaluation_dataset.py` | 117-question set (10 categories; 7 flagged for human review) |
| `python -m backend.evaluation.run_evaluation --label "..."` | Relevance, faithfulness, correctness, confusion matrices → `reports/`, dashboard history |
| `python scripts/evaluate_claims.py --judges lexical,nli[,claude]` | Judge validation (confusion matrix vs labelled claims) + hallucination rate |
| `python scripts/benchmark_rag.py --suite chunking\|embedding\|reranker\|retrieval` | Configuration benchmarks → `reports/benchmarks/` |
| `python scripts/load_test.py --url ... --levels 1,4,8,16 [--stream]` | Throughput, latency percentiles, errors |
| `python scripts/audit_knowledge_base.py` | Extraction coverage, duplicates, metadata, chunk sizes |

Run the evaluation after every content or configuration change; compare runs in the dashboard. Monitoring: the
dashboard (`/admin`), `/health/ready` for uptime checks, structured logs in `logs/`, `request_metrics` for latency/
cost alerting, ingestion failures in the audit log (and `ADMIN_NOTIFY_WEBHOOK_URL`). An error-tracking service
(e.g. Sentry) is not wired in - add its SDK in `backend/main.py` if required.
