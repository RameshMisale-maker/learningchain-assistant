# Deliverable 4 — Technology Selection and Justification

| Layer | Choice (default) | Alternatives supported | Why |
|---|---|---|---|
| Language | Python 3.11–3.13 | — | Best ecosystem for document processing, RAG and evaluation; same language for API, ingestion and tests. |
| Backend API | **FastAPI** + Uvicorn | — | Typed request/response models, automatic OpenAPI docs (`/docs`) for the portal team, async-friendly, easy dependency-injected auth. |
| Pilot UI | **Streamlit** | React portal (via REST API) | Fast to build a branded chat + admin UI; it talks to the same REST API the React portal will use, so nothing is throw-away. |
| LLM | **OpenAI** (`gpt-4o-mini` default, configurable) | Anthropic Claude (`LLM_PROVIDER=anthropic`), Azure/compatible gateways (`OPENAI_BASE_URL`), offline extractive mode | Provider-agnostic client; temperature 0; token/cost tracking per query. Model names are config, not code. |
| Embeddings | **OpenAI `text-embedding-3-small`** | `sentence_transformers` (e.g. `BAAI/bge-small-en-v1.5`, on-prem), `local_hash` (offline tests) | Strong quality/cost ratio; on-prem option for data-residency; embedding model id is stored per collection and mixing is refused. |
| Vector DB | **ChromaDB** (persistent, cosine HNSW) | Flat NumPy index (`VECTOR_BACKEND=numpy`) | Zero-ops, embedded, one collection per KB, metadata stored alongside vectors. A store interface makes pgvector / Qdrant / Azure AI Search a single class to add later. |
| Keyword search | **BM25** (`rank-bm25`) | — | Captures exact terms (policy IDs, product codes, numbers) that embeddings miss. |
| Hybrid fusion | **Reciprocal Rank Fusion** | weight via `HYBRID_VECTOR_WEIGHT` | Robust to the different score scales of BM25 and cosine; no tuning of raw scores needed. |
| Reranking | **Cross-encoder** `ms-marco-MiniLM-L-6-v2` | `lexical` (dependency-free), `none` | Reads query and passage together — the single biggest precision gain in RAG. Auto-fallback to lexical if the model can't load. |
| Routing | **Hybrid router** (keywords + example similarity + corpus statistics + entity detection), LLM for ambiguous cases | `ROUTER_MODE=rules` / `llm` | Cheap and deterministic for the 90% case; adapts automatically to new documents; LLM only where it adds value. See trade-off table in `02_architecture.md`. |
| Orchestration | **Plain Python service layer** | LangChain/LangGraph can wrap it | The pipeline is a fixed, auditable sequence. A framework would add abstraction without adding capability here; the prompt asked to use them "only where useful". The service is easy to wrap as a LangGraph node later (e.g. for agentic tools). |
| Relational DB | **PostgreSQL** (prod) via SQLAlchemy 2 | SQLite (dev default) | Users, document registry/versions, chat history, feedback and audit logs. Same code for both; switch with `DATABASE_URL`. |
| Documents | PyMuPDF (PDF text + tables), python-docx, pandas/openpyxl, json | Tesseract OCR via pytesseract | Fast, accurate, pure-Python installs on Windows; PyMuPDF table finder keeps tables as Markdown. |
| Evaluation | Custom metrics (Recall@K, Precision@K, MRR, NDCG@K, context precision/recall, faithfulness, relevancy, correctness, latency, tokens, cost, hallucination, unanswered) | **RAGAS** (`--ragas`) | Custom metrics run offline and in CI; RAGAS adds LLM-judged metrics when a key is available. |
| Security | HMAC-hashed bearer tokens, trusted-gateway SSO headers, RBAC + document ACL, upload validation, audit log | OIDC/JWT at the gateway | Authentication-ready without locking into one IdP. |

## Configuration over code

Every row above is selected in `.env` (see `.env.example`) — e.g. to run fully on-premises:

```
LLM_PROVIDER=openai
OPENAI_BASE_URL=https://<your-private-endpoint>/v1
EMBEDDING_BACKEND=sentence_transformers
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
RERANKER_BACKEND=cross_encoder
```

After changing the embedding backend run `python -m ingestion.cli reindex` (the system refuses to mix embedding models in one collection).
