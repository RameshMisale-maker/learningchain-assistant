# Architecture

How the assistant indexes content and answers a question, after the seven-phase upgrade (19 Nov 2026).
The diagram is in [`docs/diagrams/08_rag_architecture.html`](diagrams/08_rag_architecture.html)
(also `.svg` and `.png`).

Every upgrade is behind a feature flag. **All flags are off by default**, so an existing deployment behaves
exactly as before until a flag is turned on in `.env`. The flags are listed in [section 8](#8-feature-flags).

---

## 1. Overview

```mermaid
flowchart TB
    subgraph Config["Product config - config/products/&lt;id&gt;.yaml"]
      PC[names · glossary · replies · prompts vars<br/>KB namespaces · sources · UI text · models]
    end
    subgraph Indexing
      SRC[Sources: help centre, FAQ, web pages,<br/>product sitemaps, PDFs, approved Q&A] --> Q[Ingestion queue]
      Q --> VL[Validate & load] --> CH[Parent-child chunking] --> EMB[Embed bge-small]
      EMB --> BUILD[Build index version v_n<br/>new collection]
      BUILD --> GATE{Eval gate<br/>hit@5 + MRR >= live - 0.02}
      GATE -- pass --> ALIAS[Alias switch: v_n live<br/>previous kept for rollback]
      GATE -- fail --> DROP[Version rejected, dropped]
      FR[Freshness: seen / missed / tombstone after 2 misses] -.-> Q
    end
    subgraph Answering
      REQ[Widget · chat page · Streamlit · OpenAI API] --> PR[Resolve product]
      PR --> SG[Safety guard] --> NZ[Normalize] --> CL[Conversation layer] --> FU[Follow-up rewrite]
      FU --> QE[Query embedding<br/>outage: BM25 only]
      QE --> CA{Response cache<br/>exact / semantic}
      CA -- hit --> OUT[Reply + sources]
      CA -- miss --> RT[Router + scope] --> GM{Golden match}
      GM -- direct --> OUT
      GM -- no --> HS[Multi-query hybrid search<br/>dense + BM25, relative fusion] --> RR[Cross-encoder rerank]
      RR --> CTX[Context] --> LLM[LLM failover chain<br/>Groq -> 2nd provider -> static reply]
      LLM --> GR[Grounding + citations] --> OUT
    end
    subgraph AfterAnswer["After the answer"]
      OUT --> LOG[History + full request log]
      LOG --> CS[Cache store]
      LOG --> RV[Review queue<br/>thumbs-down · refusal · low confidence]
      RV -- approve --> GQ[golden_qa + eval_cases] --> SRC
    end
    subgraph Ops
      MET[/metrics/] --> PROM[Prometheus] --> GRAF[Grafana]
      AL[Alert rules config/alerts.yaml] --> AE[alert_events -> Slack / e-mail]
      NE[Nightly retrieval eval] --> AE
    end
    Config --> Indexing
    Config --> Answering
    ALIAS --> HS
```

---

## 2. Products (Phase 1)

Everything product-specific lives in `config/products/<product_id>.yaml`, validated at startup by a pydantic schema
(`backend/products/config.py`; unknown keys are rejected). A product has:

- names (display name, company, bot name, aliases, domains), a glossary and domain terms for typo fixing;
- knowledge-base `namespaces` and legacy `scopes` (`all` / `company` / `products`);
- crawl `sources` (sitemaps, include / exclude paths, allowed domains, help-centre layout);
- fixed `replies`, `escalation` contact, `ui` texts, `models` and `prompt_version`.

`resolve_product()` (`backend/products/context.py`) picks the product for a request: explicit `product` id, then alias,
then request host, then API key, then the default product. The product is held in a context variable
(`current_product()`) for the whole request. Prompts in `prompts/*.txt` use `[[placeholders]]` filled from the
product (`load_prompt`). Database rows (sessions, messages, metrics, documents, web sources) carry `product_id`.

Shipped products: `sapphirus` (combined assistant; namespaces `company` + `products`; the default) and
`learningchain` (namespace `products`). Adding a product is described in
[`adding-a-product.md`](adding-a-product.md).

## 3. Indexing

| Stage | What it does | Code |
|---|---|---|
| Sources | Help-centre articles, FAQ, website pages, training PDFs, approved Q&A CSVs, admin uploads; product sitemaps from config (Phase 3) | `knowledge_bases/`, `datasets/`, `ingestion/sources.py`, `ingestion/web.py` |
| Queue | Uploads, CLI sync, the web monitor and signed webhooks enqueue jobs; 3 tries with backoff | `ingestion/jobs.py`, `ingestion/automation.py` |
| Validate & load | Type / size / macro checks, SHA-256 dedupe, text cleanup (OCR off) | `backend/security/upload_validator.py`, `ingestion/document_loader.py`, `ingestion/text_cleaner.py` |
| Chunking | Parent sections up to 1,800 chars split on the heading tree; child chunks of 400 chars with 80 overlap | `ingestion/chunking.py` |
| Embedding | `BAAI/bge-small-en-v1.5` (local, free) | `ingestion/embedding.py` |
| Store | Chroma, one collection per KB; BM25 built in memory per KB revision | `ingestion/vector_store.py`, `backend/retrieval/kb_index.py` |

### Index versions and the eval gate (Phase 2, `INDEX_VERSIONING_ENABLED`)

Every vector-store call resolves its collection through `collection_name(kb)`: inside a build it is the collection
being built; otherwise the live version from `index_aliases`; otherwise the classic `<prefix><kb>` collection.

A rebuild (`reindex`, `build-version`, or an admin full sync) fills a new collection `<prefix><kb>_v<id>` while users
keep the live one. The **eval gate** then scores it with retrieval only (no LLM, free): hit@5 and MRR@10 over up to
40 cases from `evals/<product_id>.jsonl`, or the KB's approved Q&A (alternate phrasing -> its approved answer), plus
the eval cases created in the review queue. Score = 0.5 x hit@5 + 0.5 x MRR. The gate **fails closed** - no score or fewer than
`EVAL_MIN_CASES` (5) cases is a rejection. The version is promoted only when all hold: `score >= EVAL_MIN_SCORE`
(0.85, absolute floor), `score >= live score - EVAL_TOLERANCE` (0.02) and `score >= best score ever - EVAL_MAX_DRIFT`
(0.05, so many small drops cannot add up). Hand-written cases live in `evals/<product_id>.jsonl` and are always
scored together with the approved Q&A sample; promotion is one alias-row switch, then the KB revision is bumped so
every worker reloads BM25. The previous version stays for an instant rollback; `KEEP_INDEX_VERSIONS` (5) are kept.
The pre-versioning collection is registered as the first version and is never deleted.

CLI: `python -m ingestion.cli build-version --kb products`, `list-versions`, `promote --version N`,
`rollback --kb products` / `--product learningchain`. Admin API: `/api/v1/admin/index-versions` (+ `promote`,
`rollback`); admin page tab "Index versions".

### Freshness tracking (Phase 3, `DOC_TRACKING_ENABLED`)

Each crawl updates the page's document row (`source_url`, `last_seen_at`, `last_changed_at`, `missed_crawls`).
A page that returns 404/410 or leaves its sitemap is not deleted at once: after `TOMBSTONE_AFTER_MISSES` (2) misses in
a row it is tombstoned (chunks removed, `tombstoned_at` set, a `stale_doc` review item created). A changed page drops
its product's cached answers and queues every approved answer that cites it for re-checking. Product sitemaps are
registered from config at worker start and crawled every `CRAWL_INTERVAL_MINUTES` (daily), filtered by the
product's `include_paths` / `exclude_paths` / `allowed_domains`. Code: `ingestion/freshness.py`.

## 4. Answering a question

| # | Stage | What it does | Code |
|---|---|---|---|
| 1 | Resolve product | id / alias / host / API key / default | `backend/products/context.py` |
| 2 | Safety guard | prompt-injection screening, sensitive data (card numbers via Luhn, OTP / PIN / password) redacted, restricted topics | `backend/security/prompt_guard.py` |
| 3 | Normalize | typo and short-form fixes from the product glossary; reply language (English / Hindi / Hinglish) | `backend/routing/normalize.py` |
| 4 | Conversation layer | greetings, small talk, frustration, account, escalation, incomplete input - answered without retrieval | `backend/routing/intent.py` |
| 5 | Follow-up rewrite | standalone question from the conversation; multi-part questions split | `backend/routing/router.py`, `decompose.py` |
| 6 | Query embedding | dense vector; with `LLM_FAILOVER_ENABLED` an embedding outage switches this request to BM25-only search | `backend/generation/failover.py` (`ResilientEmbedder`) |
| 7 | **Response cache** (Phase 4) | exact key, then semantic match (cosine >= product `semantic_cache_threshold`, 0.95) | `backend/services/response_cache.py` |
| 8 | Router + scope | intent, KB routing and access control; lenient scope check (LLM / retrieval decides borderline cases) | `backend/routing/router.py`, `chat_service._lenient_scope` |
| 9 | Golden match | approved answer returned verbatim at score >= 0.78, or added as context | `backend/retrieval/golden_matcher.py` |
| 10 | Hybrid search | dense 40 + BM25 40 per phrasing, min-max ("relative") score fusion, multi-query RAG-Fusion | `backend/retrieval/hybrid.py` |
| 11 | Rerank | cross-encoder `ms-marco-MiniLM-L-6-v2`, top 5 | `backend/retrieval/reranker.py` |
| 12 | Context | parent sections of the best children, deduplicated, within `MAX_CONTEXT_CHARS` | `backend/generation/prompt_builder.py` |
| 13 | **LLM failover** (Phase 5) | provider chain with circuit breakers; static `service_unavailable` reply when all are down | `backend/generation/failover.py` |
| 14 | Grounding + citations | lexical grounding check (extractive fallback when it fails), citation and URL validation | `backend/generation/grounding.py` |
| 15 | Persist + log | history, full request log, cache store, review queue | `chat_service.answer` |

### Response cache (Phase 4, `RESPONSE_CACHE_ENABLED`)

A database table (`response_cache`), no Redis. The key is product + allowed KBs + user roles + index version +
prompt version + normalised standalone question, so a new index or prompt version never serves an old answer.
Because the key includes the user's roles and the allowed knowledge bases, a cached answer can never be served
across an access boundary, even though the lookup runs before the router. Only `generated` and `golden_*` answers with confidence >= `CACHE_MIN_CONFIDENCE` and no warnings are stored -
never refusals, out-of-scope, account / escalation / clarification replies or service-unavailable replies.
Entries expire after `CACHE_TTL_HOURS` (24); a thumbs-down deletes the entry that produced or served the answer.

### Provider failover (Phase 5, `LLM_FAILOVER_ENABLED`)

Providers in order: `LLM_PROVIDER` (Groq through the OpenAI-compatible client, with its own fallback model), the other
built-in provider when its key is set, then an optional secondary OpenAI-compatible endpoint (`LLM_SECONDARY_*`, for
example a free-tier Gemini key). Each call: `LLM_FAILOVER_TIMEOUT_SECONDS` (15 s) and one retry. Circuit breaker per
provider: 5 failures within 60 s open it for 120 s, then one trial call. A model refusal is an answer, not an outage.
When every provider is down the reply is the product's `service_unavailable` text (support contact + common
questions) with the best sources linked. Breaker states are shown in `/health/ready` and `/metrics`.

## 5. After the answer

### Full request logging (Phase 2)

`request_metrics` gains `index_version` (e.g. `products:v4`), `prompt_version` (product version + prompt-file hash),
`provider`, `intent`, `cache_hit`, `golden_hit`, `refused`, `retrieved` (top chunk ids, KBs, scores) and
`message_id`. Metrics are written after the message is saved, so every log row points at its message.

### Review queue, approved answers, eval cases (Phase 6, `REVIEW_QUEUE_ENABLED`)

Items are queued automatically for thumbs-down answers, refusals, answers below `REVIEW_LOW_CONFIDENCE` (0.35) and
changed / removed source pages. Similar questions are clustered (same content words), so one item counts every
occurrence. In the admin page (tab "Review queue") a reviewer approves an answer or dismisses the item. Approving
adds the answer to `golden_qa` and an `eval_cases` row, writes the KB's `<DATA_DIR>/golden/<kb>_reviewed.csv` and
re-ingests it, so the next user gets the approved answer and the eval gate checks it from then on. Approved
answers can also be imported from CSV (`POST /api/v1/admin/golden-qa/import`).

## 6. Observability (Phase 7)

| Piece | Flag | What |
|---|---|---|
| `/metrics` | `METRICS_ENABLED` (+ optional `METRICS_TOKEN`) | Prometheus text format, no extra dependency: requests by product / channel / answer type, refusals, cache and golden hits, errors, tokens, cost, feedback, latency histogram; gauges for breakers, live index versions, open review items, cache size |
| Alerts | `ALERTS_ENABLED` | rules in `config/alerts.yaml` (refusal rate, error rate, p95 latency, thumbs-down rate, LLM unavailable, eval score drop) checked every `ALERT_CHECK_MINUTES`; stored in `alert_events`; sent to `ALERT_SLACK_WEBHOOK_URL` and / or e-mail (`ALERT_EMAIL_TO`, `SMTP_*`); cool-down per rule |
| Nightly eval | `NIGHTLY_EVAL_ENABLED` | retrieval eval of every live index **and an end-to-end answer check** per product (`NIGHTLY_ANSWER_EVAL`: each case in `evals/<id>.jsonl` must be answered, cited, from the expected source and contain its key facts; off-topic probes must be refused) at `NIGHTLY_EVAL_HOUR` (UTC); saved as `<DATA_DIR>/evals/nightly_<timestamp>.json` (never overwritten); a drop above the threshold fires an alert |
| Dashboards | - | `ops/grafana/assistant-dashboard.json`, `ops/prometheus/prometheus.yml`, `ops/prometheus/alerts.yml` (self-hosted, free) |

Langfuse tracing is deliberately not enabled (extra paid service); the request log and `/metrics` cover tracing.

## 7. Data and deployment

- **Database:** SQLite by default, PostgreSQL 16 in Docker Compose, through SQLAlchemy. The schema is managed by
  **Alembic** (`migrations/versions/0001`-`0007`); the API applies pending migrations at startup (a pre-Alembic
  database is stamped at `0001` first, data untouched).

  | Migration | Adds |
  |---|---|
  | 0001 | baseline: the 11 existing tables |
  | 0002 | `product_id` on sessions, messages, metrics, documents, web sources, users (backfilled) |
  | 0003 | `index_versions`, `index_aliases`; request-log columns on `request_metrics` |
  | 0004 | `response_cache` |
  | 0005 | `review_items`, `golden_qa`, `eval_cases` |
  | 0006 | document freshness columns (`source_url`, `last_seen_at`, `last_changed_at`, `missed_crawls`, `tombstoned_at`, `last_crawl_id`) |
  | 0007 | `alert_events` |

- **Vector store:** embedded Chroma (one writer process); versioned collections `<prefix><kb>_v<id>`.
- **Runtime:** FastAPI + uvicorn (one worker), background ingestion worker and ops scheduler threads.
- **Front ends:** chat page + embeddable widget (`/chat`, `/widget.js`; texts from `/api/v1/ui`), admin console
  (`/admin`), OpenAI-compatible API (`/v1/chat/completions`, Open WebUI), Streamlit Community Cloud demo.
- **Hosting:** Docker Compose (api + PostgreSQL + nginx); demo on Streamlit Community Cloud from GitHub.

## 8. Feature flags

| Flag | Phase | Default | Off means |
|---|---|---|---|
| `PRODUCT_CONFIG_ENABLED` | 1 | false | every request uses the default product (today's combined assistant) |
| `INDEX_VERSIONING_ENABLED` | 2 | false | one collection per KB, rebuilt in place |
| `DOC_TRACKING_ENABLED` | 3 | false | a 404 page is deleted at once; sitemaps are not seeded from config |
| `RESPONSE_CACHE_ENABLED` | 4 | false | no cache |
| `LLM_FAILOVER_ENABLED` | 5 | false | one provider (+ its fallback model); LLM errors give the extractive answer |
| `REVIEW_QUEUE_ENABLED` | 6 | false | nothing is queued (approved-answer import and admin APIs still work) |
| `METRICS_ENABLED` / `ALERTS_ENABLED` / `NIGHTLY_EVAL_ENABLED` | 7 | false | `/metrics` returns 404; no alert checks; no nightly eval |

Full request logging (Phase 2 columns) is always on: it only adds columns to the existing metrics row.

## 9. Cost

Everything above runs on free components: local embeddings and reranker, Groq free tier, SQLite / PostgreSQL,
self-hosted Prometheus and Grafana, an optional free-tier second LLM provider. Nothing new needs a paid account.
