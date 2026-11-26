# RAG Evaluation Module

An independent evaluation dashboard for the Sapphirus / LearningChain chatbot. It **reads** the chatbot's code,
vector index and configuration and calls its real answering pipeline. It does **not** modify any chatbot file,
prompt, setting, index or dependency.

## How it connects to the chatbot

| Need | How it is done (no chatbot file changed) |
|---|---|
| Ask a question | `evaluation/chatbot_adapter.py` imports `backend.services.chat_service.ChatService` and calls `answer()` exactly like the website's `/api/v1/chat` route, using the same assistant scope (`ASSISTANT_KBS["products"]`). `persist=False, record_metrics=False`, so evaluation questions never enter the chat history or production metrics. |
| Retrieved child chunks, parent ids, parent context | `ChatResponse` only returns chunk ids. Inside the evaluation process, the adapter wraps the service's retriever, re-ranker and `select_context` with pass-through recorders. The original code runs unchanged; the running chatbot server is not affected. |
| Vector database | The existing Chroma index in `data/chroma`, read through the chatbot's own index manager. No new database, no re-ingestion. |
| Production LLM | Whatever the chatbot is configured to use. Each run records the model in `run_meta.json`. |
| Evaluator (judge) | A **separate** model: `EVAL_LLM_MODEL` (default `qwen/qwen3.8-27b` via the same OpenAI-compatible endpoint). It never grades its own answers. |

> **The configuration at the time this module was built differed from the original brief:**
> * LLM: `openai/gpt-oss-120b` on Groq, not Claude. No `ANTHROPIC_API_KEY` is set, so the chatbot falls back to its OpenAI-compatible provider. Set the key and the next run evaluates Claude automatically.
> * Embeddings: `BAAI/bge-small-en-v1.5`, not `text-embedding-3-small`. No `EMBEDDING_API_KEY` is set.
> * TTFT is only measurable with a streaming provider (the Anthropic client). With Groq it is reported as *unavailable*, never estimated.

## Installation (one time)

RAGAS needs `openai<2`, and installing it in the chatbot's Python would downgrade `openai` 3.x, which the chatbot uses.
So the module uses its own virtual environment, which sees the chatbot's packages read-only:

```bat
evaluation\setup_eval_env.bat
```

## Start the dashboard

```bat
evaluation\run_dashboard.bat
```
This opens http://localhost:8502. The chatbot keeps running on its own port (8000) as before.
The equivalent command is `evaluation\.venv\Scripts\python -m streamlit run evaluation/dashboard.py --server.port 8502`.
A plain `streamlit run evaluation/dashboard.py` also works, but with the main Python RAGAS is unavailable and the
other metrics still run.

## Workflow

1. **2. Golden Dataset**: upload `golden_dataset.csv`, or edit the included one in the table (category, expected
   behaviour and role are dropdowns). Use *Add a new question*, then **Validate dataset** and **Save changes**. Every
   save keeps the previous file in `evaluation/dataset_history/`.
2. **1. Evaluation Overview → Run a new evaluation**: choose categories and options (RAGAS, hallucination, safety,
   optional limit), then click **Run Evaluation**. A progress bar shows each step. Nothing runs when the page opens.
3. Review pages 1–8. Use the sidebar to pick which run to view.
4. **3. Retrieval**: label chunk relevance where the dataset has none. **5. Hallucination**: verify flagged claims.
   Human labels are saved in `evaluation/annotations/` and never overwrite run results.

Command line (same workflow, no UI):
```bat
evaluation\.venv\Scripts\python -m evaluation.run_evaluation --limit 10 --label quick-check
evaluation\.venv\Scripts\python -m evaluation.run_evaluation --categories "Direct factual" "Out-of-scope" --no-ragas
```

## Golden dataset columns

`question_id, question, expected_answer, reference_context, relevant_document, relevant_chunk_ids,
question_category, expected_behavior`, plus optional `role` (public | employee | hr | admin; default public) and `notes`.

* `question_category` is one of: Direct factual, Product-related, Company-related, Multi-document, Complex,
  Ambiguous, Missing-information, Out-of-scope, Restricted, Conversational.
* `expected_behavior` is one of: Answer, Abstain, Reject. Mapping from the chatbot's `answer_type`: generated, golden or
  extractive answers count as **Answer**; `refusal` (insufficient information) and `clarification` count as **Abstain**;
  `out_of_scope`, `blocked` and `access_denied` count as **Reject**. **Engage** (opinion / comparison / worry
  messages such as "any negatives?" or "any alternative") passes with a grounded answer or the tailored reply to
  the point (route `refusal_kind`), and fails with the account reply, the bare fixed refusal sentence or an error;
  Engage questions are not counted in the abstention rates.
* `relevant_document`: document name, file, section title or approved-answer id. Separate several with `|`.
* `relevant_chunk_ids`: child or parent chunk ids, separated by `|`. **Chunk ids change whenever the knowledge base is
  re-indexed**, so prefer `relevant_document` or the dashboard's annotation. Annotations are keyed by chunk text and
  survive re-indexing.
* `reference_context`: key statements, separated by `|`, used for lexical context recall. RAGAS uses
  `expected_answer` as its reference.

The included `golden_dataset.csv` was converted from the project's `datasets/evaluation_dataset/evaluation_dataset.csv`
(99 LearningChain questions). Company questions were left out because the Company knowledge base is disabled, and
there are no *Restricted* questions yet. Add some before relying on that metric.

## Metrics

| Page | Metrics | Source |
|---|---|---|
| Retrieval | Precision@K, Recall@K, MRR, NDCG@K, Hit Rate@K, rank-based context precision, lexical context recall | Actual re-ranked child chunks. K=5 (`EVAL_RETRIEVAL_K`). |
| Generation | Faithfulness, Answer Relevancy, Answer Correctness, Context Precision, Context Recall | RAGAS 0.4.3 on the actual answer and the actual parent contexts sent to the LLM |
| Hallucination | Claims, supported / unsupported / contradicted, rate = unsupported ÷ total × 100 | The chatbot's claim splitter; lead-in lines ("To reissue a single certificate", lines ending in ":") are not claims; a verbatim match or a close paraphrase (≥ `EVAL_CLAIM_OVERLAP_SUPPORT` of the content words in the passages, no negation, no new number) is supported; the rest goes to a judge (`llm` or local `nli`), with human verification |
| Safety | Pass or fail per test, by category, strict rejection rate | `safety_tests.csv`, run through the unchanged pipeline, with a system-prompt leak check |
| Performance | TTFT (streaming only), average / P50 / P95 / P99 latency, retrieval and LLM latency, tokens, cost, error rate, throughput | The chatbot's own per-stage timings and the token usage the LLM API returns |

Targets are in `config.py` (`TARGETS`). Per-question pass/fail thresholds are in `QUESTION_THRESHOLDS`.
A metric that could not be measured is shown as **Not measured**, never as 0 or an estimate.

## Results

Each run creates a new folder, `evaluation/results/run_<timestamp>_<id>/`, and never overwrites an earlier one. It contains:
`run_meta.json` (models, dataset hash, git commit, targets), `summary.json/.csv`, `question_report.csv`,
`failed_questions.csv`, `questions.json` (full traces: retrieved chunks, parent contexts, answers, claims),
`hallucination_claims.csv`, `safety_report.csv` and `performance.json`. Run comparison is on the Overview page.

## Configuration (environment variables)

| Variable | Default | Purpose |
|---|---|---|
| `EVAL_LLM_MODEL` | `qwen/qwen3.8-27b` | Evaluator / judge model |
| `EVAL_LLM_API_KEY`, `EVAL_LLM_BASE_URL` | the chatbot's `OPENAI_API_KEY` / `OPENAI_BASE_URL` | Evaluator endpoint (any OpenAI-compatible API) |
| `EVAL_RAGAS_WORKERS` | `1` | Parallel judge calls. Raise this on paid tiers. Groq's free tier rate-limits parallel calls. |
| `EVAL_HALLUCINATION_JUDGE` | `llm` | `llm` or `nli` (local cross-encoder, no API) |
| `EVAL_NLI_MODEL` | `cross-encoder/nli-deberta-v3-small` | Local judge model (`...-base` is slower and more accurate) |
| `EVAL_CLAIM_OVERLAP_SUPPORT` | `0.8` | Share of a claim's content words found in the passages to count it as a close paraphrase; `0` = off |
| `EVAL_TOKENS_PER_MINUTE` | `6000` | Pace the chatbot run under the provider's per-minute token limit (Groq free tier: 8,000 for gpt-oss-120b), so answers are not pushed to the fallback model by the test's burst; `0` = no pacing |
| `EVAL_RETRIEVAL_K` | `5` | K for the retrieval metrics |
| `EVAL_ASSISTANT` | `products` | Which website assistant's knowledge-base scope to evaluate |

## Re-checking flagged claims

`python -m evaluation.recheck_claims --run <run_id> [--judge llm]` re-checks only the UNSUPPORTED / CONTRADICTED claims
of a finished run: the local rules first (free), then - with `--judge llm` - the evaluator model for the rest, stopping
cleanly at the judge's rate limit. Answers are not regenerated; the result goes to a new folder `<run_id>_rechecked`.

## Cost note

A full run calls the production LLM once per question and safety prompt, which is billed as normal chatbot usage.
It also makes several evaluator calls per answered question for RAGAS and claim checks. Use **Limit questions** for
quick checks.
