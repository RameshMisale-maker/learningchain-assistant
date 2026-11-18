# Deliverable 9 — Testing and Evaluation Framework

## 1. Automated tests (pytest)

```
python -m pytest            # 71 tests, ~10 s, fully offline
TEST_VECTOR_BACKEND=numpy python -m pytest   # same suite on the NumPy vector store
```

| File | Covers |
|---|---|
| `tests/test_text_and_loaders.py` | tokenizer/stemmer, number extraction, all 7 loaders, PDF tables, DOCX headings, CSV records, cleaning, chunking, injection detection |
| `tests/test_ingestion.py` | initial sync, idempotent re-sync, golden rows, `_access.yaml` ACL, duplicate detection, update → version 2, delete + archive |
| `tests/test_retrieval_and_routing.py` | hybrid retrieval, document ACL, golden matcher (exact, paraphrase, row ACL), **all 7 routing examples from the requirements**, comparison entities, OOS, greeting, access denial, clarification, follow-up rewrite |
| `tests/test_chat_service.py` | approved answer returned verbatim, citations, HR access control, restricted document, product comparison, no invented prices, refusals, injection blocking, persisted history + follow-ups, **LLM path with a fake LLM** (grounded answer used, hallucinated number → fallback, invalid citations removed) |
| `tests/test_security_and_grounding.py` | upload validation (bad types, fake PDFs, macros), safe file names, token auth, query guard, rate limiter, grounding, citations, metric maths |
| `tests/test_zz_dataset_replacement.py` | a new knowledge base and a catalogue with different column names added purely through YAML + files (no code change), routed, answered and access-controlled |
| `tests/test_api.py` | REST API end to end: health, profile, chat, feedback ownership, history, admin-only endpoints, upload → query → ACL → delete → audit |

## 2. Evaluation pipeline

```
python -m backend.evaluation.run_evaluation                 # all 100 questions
python -m backend.evaluation.run_evaluation --category hr_policies
python -m backend.evaluation.run_evaluation --ragas         # + RAGAS (pip install ragas datasets; needs an LLM key)
```

Each question in `datasets/evaluation_dataset/evaluation_dataset.csv` is sent through the **full pipeline**
(router → golden → retrieval → rerank → generation → grounding) as the given role. Outputs in `reports/`:

* `evaluation_report.md` — metrics per knowledge base + failing questions
* `evaluation_results.csv` — per-question answer, route, citations and every metric
* `evaluation_summary.json` — machine-readable summary (use it in CI to fail a build on regressions)

| Group | Metrics |
|---|---|
| Retrieval | Recall@K, Precision@K, MRR, NDCG@K (relevance = chunk from an expected source; `field=value` matchers for catalogue rows) |
| Answer quality | Faithfulness (grounding check), answer relevancy (question/answer embedding similarity), context precision (rank-aware), context recall (expected key facts present in retrieved context), correctness (0.8 × key-fact coverage + 0.2 × token-F1) |
| System | Latency p50/p95, prompt/completion tokens, cost per query, retrieval failures, hallucination rate, unanswered rate, out-of-scope refusal rate, routing accuracy |

## 3. Baseline results (offline mode)

The committed report (`reports/evaluation_report.md`) was produced **without an API key** — local hashed embeddings,
lexical reranker and the extractive answerer — to prove the pipeline end to end:

| Metric | Company | Products | HR | Out-of-scope |
|---|---|---|---|---|
| Routing accuracy | 0.97 | 1.00 | 1.00 | 0.90 |
| Recall@5 | 0.97 | 1.00 | 1.00 | — |
| MRR | 0.97 | 0.98 | 1.00 | — |
| Correctness | 0.95 | 0.98 | 0.87 | 0.80 (refusal rate) |
| Hallucination rate | 0 | 0 | 0 | 0 |

**Read these numbers with care:** thresholds were tuned on this same synthetic set (in-sample), and extractive
answers are faithful by construction. The remaining failures are instructive: two out-of-scope questions whose words
overlap the corpus ("share **price**", "**salary** of the HR head") get an extract instead of a refusal — exactly
the case where the LLM's judgement (with the system prompt's refusal rule) is needed. Re-run with
`OPENAI_API_KEY` set to get production-representative generation, relevancy, token and cost figures.

## 4. Recommended evaluation process with real data

1. Build a **held-out** test set from real portal questions (≥ 50 per KB) — do not tune on it.
2. Run the evaluation after every dataset replacement and before every release; store `evaluation_summary.json`.
3. Gate releases on: routing accuracy ≥ 0.95, Recall@5 ≥ 0.90, hallucination rate ≤ 2 %, OOS refusal ≥ 0.95.
4. Review "Not helpful" feedback weekly; convert recurring questions into golden Q&As.
