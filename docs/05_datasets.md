# Deliverable 6 — Synthetic Testing Datasets (and golden dataset maintenance)

> **Everything below is FICTIONAL test data.** Company answers are prefixed `[Sample data]`, products are invented,
> and every HR document carries the banner *"FICTIONAL TEST DATA … NOT an actual Sapphirus policy"*.
> Regenerate at any time with `python -m scripts.generate_datasets --force`.

| Dataset | Location | Contents |
|---|---|---|
| Company golden dataset | `datasets/company_golden_dataset.csv` / `.json` | 32 approved Q&As (company, services, support, contact, sales, careers, internal IT/finance/travel, projects). Columns: `Question ID, Category, Question, Approved Answer, Keywords, Alternate Questions, Access Roles, Status` |
| Company documents | `knowledge_bases/company/` | Company profile (MD), operational procedures (DOCX, employees only), project portfolio (JSON, employees only) |
| Product catalogue | `datasets/product_sample_dataset.csv` | 6 fictional products (Product A–F: Orion Analytics, Vega CRM, Lyra Helpdesk, Atlas ERP, Nova IoT Gateway, Polaris Secure Access) with features, specifications, deployment, integrations, pricing (some "on request"), versions |
| Product documents | `knowledge_bases/products/` | User guide (DOCX), manual (MD), datasheet & overview (PDF with tables), installation guide (TXT), FAQ (JSON), general FAQ (XLSX) |
| HR policies | `knowledge_bases/hr_policies/` | 12 policies + 1 restricted HR-internal procedure, in DOCX/PDF/MD. Each has Policy ID, effective date, description, eligibility, rules, exceptions, approval authority, FAQs (and tables where useful) |
| HR test questions | `datasets/hr_sample_dataset/hr_test_questions.csv` | 30 questions with expected answers, key facts and source policy |
| Evaluation set | `datasets/evaluation_dataset/evaluation_dataset.csv` | 100 questions: 30 company, 30 product, 30 HR, 10 out-of-scope, each with expected KB, expected answer/criteria (`key_facts`), relevant sources and role |

HR policy list: Annual Leave (001), Sick Leave (002), Casual Leave (003), Work From Home (004), Attendance & Working
Hours (005), Public Holidays (006), Maternity/Paternity/Adoption (007), Onboarding (008), Probation (009), Leave
Approval Process (010), Employee Benefits (011), Code of Conduct (012), Disciplinary Investigation Procedure
(HR-INT-013, **HR/admin only** — demonstrates document-level access control).

## Golden dataset: how it is used

1. Each row becomes one chunk in the company collection (`doc_type=golden_qa`), carrying the question, alternates, keywords, approved answer and row-level roles.
2. For every question the matcher scores all phrasings (`0.45·dense + 0.30·term-Jaccard + 0.25·fuzzy + keyword bonus`).
3. **Direct** (`≥ GOLDEN_DIRECT_THRESHOLD`, default 0.78, and the golden entry covers the question's specific terms): the approved answer is returned **verbatim**, with citation `Qxxx`. No LLM call, no cost, no paraphrasing.
4. **Assist** (`≥ GOLDEN_ASSIST_THRESHOLD`, default 0.50): the approved answer is injected as the top source labelled `APPROVED ANSWER`; the LLM is told to keep its wording.
5. Otherwise the normal hybrid RAG pipeline answers.

## Golden dataset: maintenance process

| Step | Owner | How |
|---|---|---|
| Propose new/changed Q&A | Any team (from "Not helpful" feedback, new launches) | Admin → Feedback tab shows low-rated answers |
| Approve wording | Content owner (Corporate Comms / Product / HR) | Edit the CSV/XLSX; set `Status` (`Approved`, `Draft`, `Retired`) — draft/retired rows are skipped |
| Add paraphrases | Content owner | `Alternate Questions` separated by `|` — the cheapest way to raise match rates |
| Restrict internal answers | Content owner | `Access Roles` e.g. `employee;hr;admin` |
| Publish | Admin | Admin UI upload with "golden dataset" ticked, or replace the file and run `python -m ingestion.cli sync --kb company` |
| Verify | Admin | Ask the question in the UI (badge *Approved answer*), run `python -m backend.evaluation.run_evaluation --category company` |
| Version control | Admin | Keep the CSV in Git; every upload is versioned in the registry and the previous file is archived in `data/archive/` |

Rules of thumb: one question per row; keep IDs stable (`Q001`…); answers self-contained (no "see above"); include
contact details/URLs verbatim; review quarterly.
