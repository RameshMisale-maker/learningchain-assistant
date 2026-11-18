# Deliverable 1 — Requirement Analysis

> **Source of requirements.** No separate requirement document was attached, so the *Master Prompt* supplied by
> Sapphirus was used as the requirement specification. Nothing company-specific has been assumed: every company,
> product and HR fact in this repository is **fictional sample data**, clearly labelled, and is designed to be
> replaced (see `10_dataset_replacement.md`). When the formal requirement document is available, re-check the
> items marked **[Confirm]** below.

## 1. Business objective

A single, portal-embedded AI assistant that answers employee, customer and business questions **only from
company-approved knowledge**, with source traceability, role-based access and easy replacement of datasets.

## 2. Users and roles

| Role | Who | Knowledge access (default config) |
|---|---|---|
| `public` | Customers / guests on the portal | Company (public docs + public golden Q&A), Products |
| `employee` | Signed-in employees | + internal company procedures/projects, HR policies |
| `hr` | HR team | + restricted HR documents (e.g. investigation procedure) |
| `admin` | Platform administrators | Everything + document management, feedback, audit logs |

**[Confirm]** final role names and whether customers ever use the assistant (the design supports both).

## 3. Business use cases

| # | Use case | KB | Key requirement |
|---|---|---|---|
| UC1 | Company profile, services, contacts, support process, FAQs | Company (golden) | Return the **approved answer verbatim** when a golden question matches |
| UC2 | Internal procedures (IT tickets, expenses, travel) and projects | Company (internal docs) | Only employees see internal docs |
| UC3 | Product features, specifications, manuals, pricing | Products | Never invent features/prices; cite documents |
| UC4 | Product comparisons (2+ products) | Products | Side-by-side from catalogue data, "Not specified" for gaps |
| UC5 | HR policy questions (leave, WFH, benefits…) | HR | HR not available to public users; document-level ACL |
| UC6 | Cross-KB questions | Several | Search several KBs only when warranted |
| UC7 | Document lifecycle by admins | All | Upload, update (versioned), delete, re-index without code changes |
| UC8 | Quality monitoring | All | Evaluation per KB, feedback, audit logs, latency/token/cost metrics |

## 4. Functional requirements → implementation map

| Requirement (prompt §) | Implemented in |
|---|---|
| Three independent KBs, separate collections (§1, §2) | `config/knowledge_bases.yaml`, one vector collection per KB (`ingestion/vector_store.py`) |
| Golden dataset: closest-question search, approved answer unchanged (§1, §7) | `backend/retrieval/golden_matcher.py`, golden fast path in `backend/services/chat_service.py` |
| Semantic + keyword search, reranking (§3, §6) | `backend/retrieval/hybrid.py` (dense + BM25 + RRF), `backend/retrieval/reranker.py` |
| Automatic routing, multi-KB, clarification, no irrelevant KBs (§5) | `backend/routing/router.py` |
| Ingestion of PDF/DOCX/TXT/CSV/XLSX/JSON, tables, OCR, metadata, versions, dedupe, update/delete (§4) | `ingestion/*` |
| Grounding, citations, refusal when information is missing, injection resistance (§6) | `backend/generation/*`, `backend/security/prompt_guard.py`, `prompts/` |
| Synthetic HR dataset (≥10 policies, ≥30 Q&A), sample golden & product data (§1, §7, §8) | `scripts/generate_datasets.py`, `scripts/sample_data/*` |
| Streamlit UI with all listed elements + admin (§9) | `app.py`, `frontend/api_client.py` |
| Evaluation: retrieval, answer and system metrics; 30/30/30/10 test set (§10) | `backend/evaluation/*`, `datasets/evaluation_dataset/` |
| Security: auth-ready, RBAC, doc ACL, env secrets, upload validation, audit, retention (§11) | `backend/security/*`, `backend/db/*` |
| Config-driven dataset replacement (§14) | YAML config + `_access.yaml` sidecars + `ingestion/cli.py` |

## 5. Non-functional requirements

* **Accuracy first** — approved answers are returned verbatim; generated answers must cite sources and pass a grounding check (numbers must appear in sources) or are replaced by an extract.
* **Traceability** — every chunk carries document id, name, KB, category, page/section, source file, version and upload time.
* **Security** — no hard-coded credentials; HMAC-hashed API tokens; KB-level and document-level ACL enforced *inside retrieval* (restricted text never reaches the LLM).
* **Maintainability** — swappable LLM / embedding / vector store / reranker via `.env`; all thresholds configurable.
* **Operability** — works offline for testing (extractive answers + local embeddings), runs on Windows, PostgreSQL-ready.

## 6. Assumptions and open questions

1. **[Confirm]** LLM: OpenAI is the default (as selected); Claude is supported by switching `LLM_PROVIDER`.
2. **[Confirm]** Data residency: if HR documents must not leave the network, use `EMBEDDING_BACKEND=sentence_transformers` and an Azure-OpenAI / private endpoint (`OPENAI_BASE_URL`).
3. **[Confirm]** Portal SSO technology (OIDC/SAML) — the API supports a trusted-gateway header mode and bearer tokens today.
4. **[Confirm]** Chat retention period (default 90 days) and whether chat logs may contain personal data.
5. **[Confirm]** Ownership and approval workflow for the golden dataset (see `05_datasets.md`).
6. Languages: English only in this version.

## 7. Risks

| Risk | Mitigation |
|---|---|
| Sample thresholds don't transfer to real documents | Re-run evaluation with real data; thresholds are in `.env` |
| Poor-quality scans | OCR path with Tesseract; ingestion warnings surfaced to admins |
| Prompt injection inside documents | Documents wrapped as untrusted data, tag neutralisation, ingestion-time flagging, strict system prompt |
| Stale answers after policy changes | Versioned re-ingestion, revision-based index refresh, `sync` command |
| Over-permissive access | Deny-by-default ACL; HR KB not available to `public`; tests cover ACL |
