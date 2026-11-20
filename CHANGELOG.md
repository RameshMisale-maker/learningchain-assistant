# Changelog

## 2026-11-20 - Architecture review fixes

- **Eval gate fails closed:** no score or fewer than `EVAL_MIN_CASES` (5) cases -> rejected (before: passed).
- **Eval gate floor + drift:** promote only if `>= EVAL_MIN_SCORE` (0.85), `>= live - 0.02` and
  `>= best ever - EVAL_MAX_DRIFT` (0.05), so quality cannot slide down a little at a time with the gate green.
- **Eval cases:** `evals/learningchain.jsonl` (16 questions + 3 off-topic probes) and `evals/sapphirus.jsonl`
  (15 + 3), always scored together with the approved Q&A sample. Current scores: LearningChain 0.957 (37 cases),
  Sapphirus 0.924 (34 cases).
- **Nightly answer check:** end-to-end answers per product - answered, cited, expected source, key facts; off-topic
  probes refused. Alerts `answer_quality_drop` (> 0.08 vs the previous night) and `answer_quality_low` (< 0.75).
- **Diagram:** refusal / fallback paths drawn (red); cache key shown (product + KBs + roles + index & prompt
  version); TLS, 20 req/min rate limit, `backup.sh` and "no CI pipeline yet" restored; stricter gate and answer check.
- **Env vars:** `EVAL_MIN_SCORE`, `EVAL_MAX_DRIFT`, `EVAL_MIN_CASES`, `NIGHTLY_ANSWER_EVAL`, `NIGHTLY_ANSWER_CASES`.

## 2026-11-19 - Conversation ending and rating

- **Closing intent:** "no thanks", "that's all", "bye" -> intent `CLOSING` (rules + LLM intent prompt): a goodbye with
  `ui_action = show_rating`. The goodbye text no longer says "please rate below" - the app shows the rating itself.
- **Conversation state per session** (`active | closing | closed`, `chat_sessions.conversation_state`). After the
  goodbye, a short acknowledgement (ok, okay, thanks, thank you, bye, done, a thumbs-up; case-insensitive, after
  normalisation) makes no LLM call and stores no message: state `closed`, the rating stays visible. A real question
  sets the state back to `active` and is answered normally.
- **Structured LLM answer** `{reply, ui_action: none|show_rating|show_escalation|show_login, conversation_state}`,
  validated; invalid output falls back to the text with `ui_action = none` (`STRUCTURED_REPLY`, default on). Only the
  goodbye can show the rating; escalation replies show the support contact, guest access-denied shows sign-in.
- **Greetings with an introduction** ("okay hello chatbot, Ramesh here", "hi, this is Priya") are `GREETING`, never
  frustration; the LLM intent fallback can no longer assign frustration / account / escalation.
- **Rating:** the chat page shows 1-5 stars + optional comment on `show_rating`, kept until submitted or a new
  question; it posts to `POST /api/feedback` (alias of `/api/v1/feedback`) with `session_id`, `rating`, `comment`,
  `product_id`, `versions`. Ratings of 1-2 create a `review_items` entry (`source = low_rating`).
- **Migration:** `0008` - `chat_sessions.conversation_state`; `feedback.session_id`, `product_id`, `stars`,
  `details` (backfilled stars from existing `stars_N` ratings).
- **Data:** `knowledge_bases/products/LearningChain_08_Certificate_Generation_Modes.md` - the 10 certificate / QR
  code generation modes from `LearningChain_Chatbot_FAQ.csv` (33 chunks).
- **Tests:** `tests/test_conversation_end.py`.
- **Manual test:** chat page -> ask a question -> "no thanks" (goodbye + stars) -> "ok" (nothing new, stars stay) ->
  rate 2 stars -> admin "Review queue" shows a `low_rating` item -> ask a new question (stars disappear).

## 2026-11-19 - Pipeline upgrade (phases 1-7)

Implements `docs/upgrade-plan.md` using only free components. Every phase is behind a feature flag that is **off by
default**, so behaviour is unchanged until a flag is turned on. The API applies the database migrations at startup.
Architecture: `docs/architecture.md`; adding a product: `docs/adding-a-product.md`.

### Phase 1 - Product configuration

Everything product-specific moved from code into `config/products/<product_id>.yaml`.

- **Files:** new `backend/products/` (`config.py` schema + loader, `context.py` resolution / context variable),
  `config/products/sapphirus.yaml`, `config/products/learningchain.yaml`, `alembic.ini`, `migrations/`.
  Product names removed from `backend/`, `ingestion/`, `prompts/` (now `[[placeholders]]`), `streamlit_cloud/`,
  `evaluation/`, static pages (texts come from `GET /api/v1/ui`). Product-specific scripts moved to
  `scripts/products/learningchain/`.
- **Migrations:** `0001` baseline (existing 11 tables; an existing database is stamped, not changed), `0002`
  `product_id` on sessions, messages, metrics, documents, web sources, users (backfilled from the KB owner).
- **Env vars:** `PRODUCT_CONFIG_ENABLED`, `PRODUCT_CONFIG_DIR`, `DEFAULT_PRODUCT`, `PRODUCT_ID` (Streamlit),
  `VECTOR_COLLECTION_PREFIX`, `CRAWLER_USER_AGENT`, `WEBHOOK_SIGNATURE_HEADER`, `LOG_FILE_NAME`.
- **Tests:** `tests/test_products.py` (schema validation, resolution with the flag on / off, no product names in
  code, UI endpoint, rows keyed by product, migration head, a made-up product working end to end from config only).
- **Manual test:** start the API; `GET /api/v1/ui` returns the default product's texts; set
  `PRODUCT_CONFIG_ENABLED=true` and open `/chat?product=learningchain` - title and suggestions change; break a YAML
  key and restart - startup stops with a clear error.

### Phase 2 - Index versioning and full request logging

- **Files:** new `backend/indexing/versions.py`; `ingestion/vector_store.py` (`collection_name` resolves build /
  live version; `base_collection_name`), `backend/retrieval/kb_index.py` (reload when the serving collection
  changes), `ingestion/pipeline.py` (`reindex_kb` builds a version when the flag is on), `ingestion/cli.py`
  (`build-version`, `list-versions`, `promote`, `rollback`), `backend/api/routes_admin.py` (`/index-versions`),
  `backend/services/chat_service.py` (metrics written after the message, new log fields), `backend/static/admin.html`
  ("Index versions" tab).
- **Migration:** `0003` - `index_versions`, `index_aliases`; `request_metrics` + `index_version`,
  `prompt_version`, `provider`, `intent`, `cache_hit`, `golden_hit`, `refused`, `retrieved`, `message_id`.
- **Env vars:** `INDEX_VERSIONING_ENABLED`, `EVAL_TOLERANCE` (0.02), `KEEP_INDEX_VERSIONS` (5),
  `EVAL_GATE_MAX_CASES` (40), `EVALS_DIR`.
- **Tests:** `tests/test_index_versions.py` (flag off = classic collection, labels, build -> gate -> promote ->
  rollback, a worse index rejected and dropped, request-log fields, admin endpoints, migration round trip).
- **Manual test:** `INDEX_VERSIONING_ENABLED=true`; `python -m ingestion.cli build-version --kb products` prints
  `version N live (eval 0.9x ...)`; `list-versions` shows the baseline as retired; ask a question; `rollback --kb
  products`; the admin page "Index versions" tab shows the same.

### Phase 3 - Document freshness tracking

- **Files:** new `ingestion/freshness.py`; `ingestion/automation.py` (404/410 counted as misses, tombstone after N,
  sitemap filtered by product rules and missing pages counted, product sitemaps seeded at worker start),
  `backend/api/routes_admin.py` (`/documents/stale`, `/web-sources/seed-from-config`).
- **Migration:** `0006` - `documents.source_url`, `last_seen_at`, `last_changed_at`, `missed_crawls`,
  `tombstoned_at`, `last_crawl_id` (backfilled from `updated_at`).
- **Env vars:** `DOC_TRACKING_ENABLED`, `TOMBSTONE_AFTER_MISSES` (2), `CRAWL_INTERVAL_MINUTES` (1440).
- **Tests:** `tests/test_freshness.py` (URL rules from config, seen -> miss -> recovered -> tombstone with a review
  item, flag off keeps immediate delete, changed page drops cache and flags approved answers, sitemap filtering and
  dropped pages, admin endpoint).
- **Manual test:** `DOC_TRACKING_ENABLED=true`, restart; `GET /api/v1/admin/web-sources` lists the sapphirus.in
  sitemaps (daily); `GET /api/v1/admin/documents/stale` lists pages with misses.

### Phase 4 - Response cache (database, no Redis)

- **Files:** new `backend/services/response_cache.py`, `backend/services/feedback_hooks.py`;
  `backend/services/chat_service.py` (lookup after the query embedding, store after persisting),
  `backend/api/routes_chat.py` (feedback hook), `backend/api/routes_admin.py` (`/cache`, `/cache/clear`).
- **Migration:** `0004` - `response_cache`.
- **Env vars:** `RESPONSE_CACHE_ENABLED`, `CACHE_TTL_HOURS` (24), `CACHE_MIN_CONFIDENCE` (0.5); semantic threshold per
  product (`semantic_cache_threshold`, 0.95).
- **Tests:** in `tests/test_cache_and_review.py` (off stores nothing, exact hit logged as `cache_hit`, semantic hit,
  roles isolate entries, refusals / low confidence / failed grounding never cached, new index revision and
  thumbs-down invalidate).
- **Manual test:** `RESPONSE_CACHE_ENABLED=true`; ask the same question twice - the second answer's `route` has
  `cache_hit: true` and returns in a few ms; thumbs-down it - the next ask is generated again.

### Phase 5 - Provider failover (free version)

- **Files:** new `backend/generation/failover.py` (circuit breaker, `FailoverLLM`, `ResilientEmbedder`);
  `backend/generation/llm_client.py` (`get_llm` builds the chain when the flag is on),
  `backend/retrieval/hybrid.py` (dense search skipped during an embedding outage), `backend/services/chat_service.py`
  (static `service_unavailable` reply, provider recorded), `backend/products/config.py` (`replies.service_unavailable`),
  `backend/main.py` (breaker state in `/health/ready`).
- **Migration:** none.
- **Env vars:** `LLM_FAILOVER_ENABLED`, `LLM_FAILOVER_TIMEOUT_SECONDS` (15), `BREAKER_FAILURES` (5),
  `BREAKER_WINDOW_SECONDS` (60), `BREAKER_COOLDOWN_SECONDS` (120), `LLM_SECONDARY_BASE_URL`, `LLM_SECONDARY_API_KEY`,
  `LLM_SECONDARY_MODEL`.
- **Tests:** `tests/test_failover.py` (breaker open / half-open / window, chain failover and skipping an open
  breaker, refusal is not an outage, streaming failover, flag off, static reply with sources, BM25-only search when
  the embedder fails).
- **Manual test:** `LLM_FAILOVER_ENABLED=true` with a wrong `OPENAI_API_KEY` and no secondary key: an in-scope
  question gets the `service_unavailable` reply with links; `/health/ready` shows the breaker open after 5 failures.

### Phase 6 - Review queue, approved answers, eval cases

- **Files:** new `backend/services/review.py`; `backend/services/chat_service.py` (refusals / low-confidence
  answers queued), `backend/services/feedback_hooks.py` (thumbs-down queued), `ingestion/pipeline.py`
  (`<DATA_DIR>/golden/<kb>_reviewed.csv` discovered as a golden file), `backend/indexing/versions.py` (eval cases join
  the gate), `backend/api/routes_admin.py` (`/review-items`, `/review-items/{id}/resolve`, `/golden-qa`,
  `/golden-qa/import`, `/eval-cases`), `backend/static/admin.html` ("Review queue" tab).
- **Migration:** `0005` - `review_items`, `golden_qa`, `eval_cases`.
- **Env vars:** `REVIEW_QUEUE_ENABLED`, `REVIEW_LOW_CONFIDENCE` (0.35).
- **Tests:** in `tests/test_cache_and_review.py` (clustering, auto items, thumbs-down through the API, approving
  serves the answer and creates an eval case, CSV import validation, migrations 0004-0007 round trip).
- **Manual test:** `REVIEW_QUEUE_ENABLED=true`; thumbs-down an answer in the chat page; the admin page "Review
  queue" shows it; type an approved answer and click Approve; ask the question again - the approved answer is
  returned.

### Phase 7 - Observability

- **Files:** new `backend/observability/` (`metrics.py`, `alerts.py`, `scheduler.py`), `config/alerts.yaml`,
  `ops/prometheus/prometheus.yml`, `ops/prometheus/alerts.yml`, `ops/grafana/assistant-dashboard.json`;
  `backend/main.py` (`/metrics`, scheduler start / stop), `backend/services/chat_service.py` and
  `feedback_hooks.py` (counters), `backend/api/routes_admin.py` (`/alerts`, `/alerts/check`, `/evals/nightly`).
- **Migration:** `0007` - `alert_events`.
- **Env vars:** `METRICS_ENABLED`, `METRICS_TOKEN`, `ALERTS_ENABLED`, `ALERT_CHECK_MINUTES` (5),
  `NIGHTLY_EVAL_ENABLED`, `NIGHTLY_EVAL_HOUR` (2 UTC), `ALERT_SLACK_WEBHOOK_URL`, `ALERT_EMAIL_TO`, `SMTP_HOST`,
  `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD`, `SMTP_FROM`, `SMTP_STARTTLS`. Langfuse is not enabled.
- **Tests:** `tests/test_observability.py` (`/metrics` off by default, counters / histogram / gauges after requests,
  token protection, rules file valid, a rule fires once per cool-down and respects `min_requests`, Slack
  notification, nightly eval saved without overwriting and alerting on a drop, scheduler tick, admin endpoints).
- **Manual test:** `METRICS_ENABLED=true`; ask two questions; `curl localhost:8000/metrics`; run Prometheus with
  `ops/prometheus/` and import `ops/grafana/assistant-dashboard.json` into Grafana; `POST
  /api/v1/admin/evals/nightly` writes `data/evals/nightly_<timestamp>.json`.
