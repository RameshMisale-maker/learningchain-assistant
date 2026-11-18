# Deliverables 5 & 7 — Folder Structure and Module Reference

## Folder structure

```
sapphirus_chatbot/
├── app.py                         # Streamlit UI (chat + admin)
├── backend/
│   ├── main.py                    # FastAPI app, startup (DB, bootstrap users, retention purge)
│   ├── api/                       # REST layer: routes_chat.py, routes_admin.py, schemas.py, deps.py
│   ├── services/                  # chat_service.py (RAG orchestration), admin_service.py
│   ├── routing/router.py          # intent classification + KB routing
│   ├── retrieval/                 # kb_index.py, hybrid.py, reranker.py, golden_matcher.py
│   ├── generation/                # llm_client.py, prompt_builder.py, grounding.py, extractive.py
│   ├── security/                  # auth.py, rbac.py, prompt_guard.py, upload_validator.py, rate_limit.py, manage_users.py
│   ├── evaluation/                # metrics.py, run_evaluation.py
│   ├── db/                        # models.py, session.py, repository.py
│   └── common/                    # text_utils.py, logging_config.py
├── ingestion/                     # document_loader.py, text_cleaner.py, chunking.py, embedding.py,
│                                  # vector_store.py, registry.py, pipeline.py, cli.py
├── frontend/api_client.py         # HTTP / embedded client used by Streamlit
├── config/
│   ├── settings.py                # environment-driven settings
│   └── knowledge_bases.yaml       # KB definitions, routing hints, suggested questions
├── knowledge_bases/               # SOURCE DOCUMENTS (drop real files here)
│   ├── company/   (+ _access.yaml)
│   ├── products/  (+ _access.yaml)
│   └── hr_policies/ (+ _access.yaml)
├── datasets/                      # golden Q&A, product catalogue, HR test Q&A, evaluation set
├── prompts/                       # system / answer / router / rewrite / grounding prompts
├── scripts/                       # generate_datasets.py + sample_data/*, windows/*.bat
├── tests/                         # pytest suite (71 tests)
├── reports/                       # evaluation outputs
├── docs/                          # these documents + diagrams/
├── data/                          # runtime: Chroma index, SQLite DB, archive (git-ignored)
├── logs/                          # rotating application logs (git-ignored)
├── Dockerfile, docker-compose.yml # production packaging
├── requirements.txt, requirements-optional.txt, .env.example, pytest.ini, README.md
```

## Module reference (responsibility · dependencies · connections)

### Configuration
| File | Responsibility | Depends on | Used by |
|---|---|---|---|
| `config/settings.py` | Loads `.env`/environment into a `Settings` dataclass; resolves paths; effective LLM provider (falls back to extractive without a key); loads the KB YAML | python-dotenv, PyYAML | everything |
| `config/knowledge_bases.yaml` | Declares KBs (folders, golden files, extra files, allowed roles, chunking, entity fields, routing keywords/examples), router tuning, abbreviation expansions, suggested questions | — | ingestion, router, RBAC, UI |

### Ingestion (`ingestion/`)
| File | Responsibility | Depends on | Used by |
|---|---|---|---|
| `document_loader.py` | Type detection and extraction: PDF (PyMuPDF text, tables → Markdown, OCR for scanned pages), DOCX (headings, lists, tables in order), TXT/MD, CSV/XLSX (one record per row), JSON (records or flattened) | PyMuPDF, python-docx, pandas, openpyxl, pytesseract (optional) | pipeline, tests |
| `text_cleaner.py` | Unicode normalisation, hyphenation/soft-wrap repair, repeated header/footer removal, bullet normalisation; prompt-injection detector and prompt neutraliser | text_utils | pipeline, prompt_guard, prompt_builder |
| `chunking.py` | Structure-aware chunking: heading split → paragraph packing with sentence overlap; records/tables kept intact; contextual header per chunk; stable chunk ids | text_utils | pipeline |
| `embedding.py` | Embedding backends (OpenAI, sentence-transformers, local hashed n-grams), normalised vectors, model id | openai / sentence-transformers / scikit-learn | pipeline, retriever, router, golden matcher |
| `vector_store.py` | Vector store interface + ChromaDB and NumPy implementations; one collection per KB; metadata flattening; embedding-model guard | chromadb, numpy | pipeline, retrieval |
| `registry.py` | Document registry in SQL: versions, hashes, ACL, status | db.session | pipeline, admin API |
| `pipeline.py` | Orchestrates ingestion: hash → unchanged/duplicate check → load → clean → chunk → injection scan → embed → replace chunks → registry → KB revision. Sync, re-index, delete (archive), purge, stats. `_access.yaml` resolution | all of the above | CLI, admin service, tests |
| `cli.py` | `list-kbs`, `sync`, `reindex`, `add`, `delete`, `purge`, `list-docs` | pipeline | operators |

### Retrieval (`backend/retrieval/`)
| File | Responsibility | Depends on | Used by |
|---|---|---|---|
| `kb_index.py` | Per-KB in-memory index (chunks, BM25, document frequencies, entity lexicon, golden entries) rebuilt automatically when the KB revision changes | rank-bm25, vector_store, repository | hybrid, router, golden matcher, reranker |
| `hybrid.py` | Dense + BM25 retrieval, ACL filtering, Reciprocal Rank Fusion, entity boost | kb_index, embedding, vector_store | chat_service |
| `reranker.py` | Cross-encoder / lexical / none rerankers (scores in [0,1]); IDF-weighted coverage helper | sentence-transformers (optional) | chat_service, extractive |
| `golden_matcher.py` | Scores the question against approved questions + alternates (dense, term Jaccard, fuzzy, keyword bonus); row-level ACL | kb_index, embedding | chat_service |

### Routing, generation, services
| File | Responsibility | Depends on | Used by |
|---|---|---|---|
| `routing/router.py` | Greeting/OOS/clarify/comparison/access-denied intents, KB scoring (keywords, examples, corpus association, entities), multi-KB, optional LLM router, follow-up rewrite, abbreviation expansion | kb_index, embedding, llm_client (optional) | chat_service |
| `generation/llm_client.py` | OpenAI / Anthropic clients with token usage and cost; JSON helper; returns `None` without keys | openai, anthropic | chat_service, router |
| `generation/prompt_builder.py` | Loads prompt templates; selects context within a character budget; wraps sources as `<source id="S1" …>` with citation metadata; neutralises injection | prompts/, text_cleaner | chat_service, grounding |
| `generation/grounding.py` | Lexical grounding (sentence support + numeric check), optional LLM fact-check, refusal detection, citation parsing/cleanup | text_utils | chat_service, evaluation |
| `generation/extractive.py` | No-LLM answerer: approved answers, catalogue field answers, comparison tables, FAQ pairs, best-sentence extraction with citations | kb_index | chat_service |
| `services/chat_service.py` | **The RAG pipeline** (validate → rewrite → route → golden → retrieve → rerank → generate → ground → cite → persist) | all retrieval/generation/security modules, repository | API, Streamlit (embedded), evaluation |
| `services/admin_service.py` | Upload (validated, archived), replace, delete, golden dataset replacement, sync/re-index, audit | pipeline, upload_validator, repository | admin API, Streamlit |

### API, security, data, UI, evaluation
| File | Responsibility | Depends on | Used by |
|---|---|---|---|
| `backend/main.py` | FastAPI app, CORS, security headers, startup tasks | api routers, db, auth | uvicorn |
| `api/routes_chat.py` | `/me`, `/suggestions`, `/chat`, session history/clear, `/feedback` | chat_service, repository, rbac | portal, Streamlit |
| `api/routes_admin.py` | KB stats, sync/re-index, documents CRUD, feedback, audit logs, users, retention purge | admin_service, registry, repository | admin UI |
| `api/schemas.py`, `api/deps.py` | Pydantic models; auth, admin guard, rate limiting dependencies | security | routes |
| `security/auth.py` | Token (HMAC-hashed), trusted-proxy SSO headers, dev mode; bootstrap users; anonymous → `public` | repository | deps, embedded client |
| `security/rbac.py` | KB-level access, admin check, role-filtered suggested questions | settings | routes, UI |
| `security/prompt_guard.py` | Query sanitising, length limit, injection blocking | text_cleaner | chat_service |
| `security/upload_validator.py` | Extension allow-list, size, magic bytes, macro/zip-bomb checks, safe file names | — | admin_service |
| `security/rate_limit.py`, `manage_users.py` | Per-user rate limit; CLI to create/list/deactivate users | repository | API, operators |
| `db/models.py`, `session.py`, `repository.py` | SQLAlchemy schema, engine/session, data-access helpers (chat, feedback, audit, KB revision, users, retention) | SQLAlchemy | everything stateful |
| `frontend/api_client.py` | HTTP client for the API, or in-process client (`UI_MODE=embedded`) | httpx | `app.py` |
| `app.py` | Branded chat UI (welcome, suggestions, history, sources, feedback, spinner, errors) and admin console (upload, replace, delete, sync, feedback, audit) | Streamlit, api_client | users |
| `evaluation/metrics.py`, `run_evaluation.py` | Metric implementations; runs the evaluation set through the full pipeline and writes per-KB reports (optional RAGAS) | chat_service | operators, CI |
| `scripts/generate_datasets.py` (+ `sample_data/`) | Generates all fictional sample data in DOCX/PDF/MD/TXT/JSON/XLSX/CSV | python-docx, PyMuPDF, pandas | setup |
