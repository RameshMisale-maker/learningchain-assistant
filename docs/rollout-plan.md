# Rollout plan: switching on the upgrade features

All seven upgrade phases are behind feature flags that are **off by default**, so production still runs the
pre-upgrade behaviour for those features. This plan switches them on **one at a time**, re-running the same
evaluation after each step and comparing it with the previous step. A step that lowers quality is rolled back by
setting its flag back to `false` - no code change, no data migration.

## Already live (no flag)

These changes are active today regardless of the flags:

- conversation ending + 1-5 star rating, per-session conversation state
- replies always in English (other languages are understood)
- one combined LearningChain FAQ file (107 approved answers)
- review fixes: webhook replay protection, rate-limiter pruning, topic filter only refuses harmful intent
- evaluation fixes: account reply only for the user's own data, vague "it" asks which feature, undocumented-term
  guard (e.g. "iOS app" is declined instead of guessed), greetings read as a whole message
- full request logging (index / prompt version, provider, intent, retrieved chunks)

Baseline (20 Nov, 50 questions, all flags off): **90% pass (46/50 after the iOS fix), recall@5 0.95, faithfulness
1.00, 0 unsupported claims, behaviour accuracy 0.96, safety tests 24/24.**

## Order

Safest first (no effect on answers), then the features that change what users see.

| Step | Flag(s) | What users notice | Pass criteria after the eval re-run |
|---|---|---|---|
| 1 | `METRICS_ENABLED` | nothing - monitoring only | scores equal to the baseline; `/metrics` returns data - **on since 26 Nov 2026** (local `.env`; set it in Streamlit Secrets too) |
| 1b | `TOKEN_BUDGET_ENABLED` (+ `TOKEN_BUDGET_USER_TOKENS`, `TOKEN_BUDGET_PRODUCT_TOKENS`, `TOKEN_BUDGET_PRODUCT_LIMITS`, `TOKEN_BUDGET_EXEMPT_USERS`) | only a user or product over its daily token budget: approved answers still work, other questions get a "limit reached" reply instead of using the shared Groq quota | scores equal (evaluation accounts are exempt); `assistant_tokens_window` on `/metrics` stays under the budget; size the budgets from a week of that gauge - **on since 26 Nov 2026** with 40k per user / 150k per product |
| 2 | `REVIEW_QUEUE_ENABLED` | nothing - 👎, refusals and low ratings are queued for admins | scores equal; items appear in the admin "Review queue" tab |
| 3 | `LLM_FAILOVER_ENABLED` | only when Groq fails: 15 s timeout, backup provider (if `LLM_SECONDARY_*` is set) or a support reply | scores equal; error rate 0; p95 latency not higher |
| 4 | `RESPONSE_CACHE_ENABLED` | repeated questions answered instantly | scores equal; run the eval twice - the second run's cached answers must match the first; latency drops |
| 5 | `PRODUCT_CONFIG_ENABLED` | only with separate LearningChain / Sapphirus sites or widgets (`?product=`, domains) | scores equal for both products; correct titles and replies per product |
| 6 | `INDEX_VERSIONING_ENABLED` | nothing until the next content rebuild, which must then pass the eval gate | `python -m ingestion.cli build-version --kb products` passes the gate; score equal or better; one test rollback works |
| 7 | `DOC_TRACKING_ENABLED` | website content refreshed daily; removed pages dropped after 2 missed crawls | after the first crawl, scores equal; `GET /api/v1/admin/documents/stale` reviewed |
| 8 | `ALERTS_ENABLED`, `NIGHTLY_EVAL_ENABLED` | nothing - alerts to Slack / e-mail, automatic nightly check | a test alert arrives; `data/evals/nightly_<date>.json` appears the next morning |

Why this order: steps 1-2 only observe, so they give clean before / after numbers for everything else. Failover (3)
protects users from the Groq free-tier limits before more traffic is served from new paths. The cache (4) changes
latency and must be checked for stale answers. Product resolution (5) only matters once a second site or widget
exists. Versioning (6) must be on before the next re-index; freshness tracking (7) changes content automatically, so
it comes after the gate that protects index quality. Alerts and the nightly eval (8) need Slack / e-mail settings.

## Procedure for each step

1. Set the flag in `.env` (and in Streamlit **Secrets** for the demo), e.g. `METRICS_ENABLED=true`, then restart
   the API (and reboot the Streamlit app).
2. Smoke test in the chat page: one LearningChain question, one company question, one off-topic question.
3. Re-run the evaluation on the same question set, labelled with the step:

   ```bat
   set EVAL_ASSISTANT=all
   evaluation\.venv\Scripts\python -m evaluation.run_evaluation --dataset evaluation/golden_subset_final.csv --ragas-sample 15 --label step1-metrics
   ```

4. Compare with the previous step in the dashboard (`evaluation\run_dashboard.bat`, http://localhost:8502) and fill
   in the table below.
5. **Go / no-go:** continue only if question pass rate, recall@5 and behaviour accuracy are not lower than the
   previous step (allow ±0.02 for run-to-run noise) and there are no unsupported claims or failed safety tests.
6. **Rollback:** set the flag back to `false` and restart. Nothing else is needed.

**Budget:** one evaluation run uses about 75k Groq tokens; the free tier allows about 200k per model per day, and the
AI judge has its own limit. Plan **one step per day**. If the primary model is rate-limited the run falls back to
`gpt-oss-20b` - repeat that step the next day rather than compare runs made with different models.

## Results

| Step | Date | Run id | Pass rate | Recall@5 | Behaviour acc. | Unsupported claims | p95 latency | Go? |
|---|---|---|---|---|---|---|---|---|
| Baseline | 20 Nov | run_20261120_140708_591e | 0.90 (0.92 after fix) | 0.95 | 0.96 | 0 | 22 s | - |
| 1 Metrics | | | | | | | | |
| 2 Review queue | | | | | | | | |
| 3 Failover | | | | | | | | |
| 4 Cache | | | | | | | | |
| 5 Product config | | | | | | | | |
| 6 Index versioning | | | | | | | | |
| 7 Doc tracking | | | | | | | | |
| 8 Alerts + nightly eval | | | | | | | | |

Flag details and defaults: [`architecture.md`](architecture.md#8-feature-flags) and `.env.example`.
