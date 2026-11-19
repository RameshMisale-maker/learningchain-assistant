# Sapphirus AI Assistant

Retrieval-augmented assistant for the Sapphirus website. It answers questions about **Sapphirus** (company
information from sapphirus.in) and **LearningChain** (help centre + official FAQ) **only from approved content**,
cites and links its sources, and declines everything else. Production guide: [`docs/11_production_guide.md`](docs/11_production_guide.md).

## Features

* **Website chat widget** (`/chat`, embeddable with `widget.js`): streaming answers, sources with links, feedback, copy,
  history, suggested questions; separate **Company** and **LearningChain** assistants enforced server-side
* **Admin console** (`/admin`): upload / replace / delete documents, monitored websites, ingestion jobs (retry),
  feedback, and a performance & evaluation dashboard
* **Claude Sonnet 4.5 generation** with a strict grounding prompt, citation and URL validation, grounding check and a
  fixed "insufficient information" reply; extractive fallback when no key is configured
* **Retrieval**: approved-answer matching (exact + semantic), hybrid vector + BM25 with RRF, cross-encoder reranking,
  scope guard, document-level access control - configuration chosen by benchmark (`reports/benchmarks/`)
* **Automated knowledge updates**: background job queue with retries and idempotency, async uploads, version
  replacement without gaps, website / help-centre / FAQ monitoring with change detection, signed webhooks, SSRF guard
* **Evaluation**: 117-question dataset (10 categories), retrieval and answer metrics, confusion matrices, claim-level
  hallucination evaluation with validated judges, configuration benchmarks, load testing, per-request latency/cost
* **Security**: hashed API tokens, RBAC, rate limiting, prompt-injection screening (questions *and* documents), secure
  uploads (incl. HTML), CSP / frame-ancestors, log redaction, fail-fast production config checks
* **Deployment**: Dockerfile, docker-compose (API + PostgreSQL + nginx + optional Open WebUI), backups
* **95+ automated tests** (`python -m pytest`)

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate        # Windows: scripts\windows\setup.bat does all of this
pip install -r requirements.txt
cp .env.example .env                                     # add ANTHROPIC_API_KEY + tokens
python -m ingestion.cli sync                             # index all knowledge bases
uvicorn backend.main:app --port 8000                     # API  -> http://localhost:8000/docs
python scripts/run_open_webui.py                         # UI   -> http://localhost:3000 (Open WebUI via uvx)
```

**Offline / no API key:** set `LLM_PROVIDER=extractive`, `EMBEDDING_BACKEND=local_hash`, `RERANKER_BACKEND=lexical`.

```bash
python -m pytest                                         # tests
python -m backend.evaluation.run_evaluation              # evaluation report -> reports/
python -m backend.security.manage_users create --username priya --roles employee,hr
```

## Deliverables map

| # | Deliverable | Where |
|---|---|---|
| 1 | Requirement analysis | [`docs/01_requirements_analysis.md`](docs/01_requirements_analysis.md) |
| 2 | End-to-end architecture diagram | [`docs/02_architecture.md`](docs/02_architecture.md), [`docs/diagrams/08_rag_architecture.html`](docs/diagrams/08_rag_architecture.html) |
| 3 | Workflow diagrams | [`docs/02_architecture.md`](docs/02_architecture.md) (Mermaid), [`docs/diagrams/08_rag_architecture.png`](docs/diagrams/08_rag_architecture.png) |
| 4 | Technology selection & justification | [`docs/03_technology_selection.md`](docs/03_technology_selection.md) |
| 5 | Folder structure + per-module responsibilities/dependencies | [`docs/04_folder_structure_and_modules.md`](docs/04_folder_structure_and_modules.md) |
| 6 | Synthetic datasets | `datasets/`, `knowledge_bases/`, [`docs/05_datasets.md`](docs/05_datasets.md) |
| 7 | Source code | `backend/`, `ingestion/`, `frontend/`, `config/`, `prompts/`, `scripts/` |
| 8 | Open WebUI front end | `backend/api/routes_openai.py`, `scripts/run_open_webui.py` |
| 9 | Testing & evaluation | `tests/`, `backend/evaluation/`, `reports/`, [`docs/06_evaluation.md`](docs/06_evaluation.md) |
| 10 | Windows installation guide | [`docs/07_windows_setup.md`](docs/07_windows_setup.md), `scripts/windows/` |
| 11 | Portal integration | [`docs/08_portal_integration.md`](docs/08_portal_integration.md) |
| 12 | Production deployment | [`docs/09_production_deployment.md`](docs/09_production_deployment.md), `Dockerfile`, `docker-compose.yml` |
| — | Dataset replacement strategy | [`docs/10_dataset_replacement.md`](docs/10_dataset_replacement.md) |

## How a question is answered

1. Validate and screen the question (length, prompt injection).
2. Rewrite follow-ups into standalone questions; expand abbreviations (WFH, PTO, SSO…).
3. Route: score each KB (keywords, example similarity, corpus statistics, product entities); enforce KB-level roles.
4. Golden fast path: strong approved match → answer verbatim (no LLM).
5. Hybrid retrieval per KB (ACL-filtered) → rerank → select context.
6. Generate with the LLM (sources wrapped as untrusted data) → validate citations → grounding check (sentences and numbers must be supported) → fall back to an extract if not.
7. Store the exchange (route, citations, latency, tokens, cost) and collect feedback.

## Configuration

All settings are environment variables — see [`.env.example`](.env.example). Knowledge bases, routing hints,
abbreviations and suggested questions are in [`config/knowledge_bases.yaml`](config/knowledge_bases.yaml);
per-folder access rules in `_access.yaml` files.
