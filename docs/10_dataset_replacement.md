# Dataset Replacement Strategy (prompt §14 / Phase 8)

Everything that is dataset-specific lives in **files and configuration**, never in code:

| What | Where |
|---|---|
| Which KBs exist, their folders, golden/extra files, allowed roles, chunk sizes | `config/knowledge_bases.yaml` |
| Per-document roles, category, confidentiality, sample flag | `_access.yaml` in each folder (or upload form) |
| Golden Q&A column names | `golden_columns` in the YAML |
| Product-name recognition | `entity_fields` (catalogue column names) |
| Routing hints & suggested questions | `routing.keywords/examples`, `suggested_questions` |

## A. Remove the sample data

```powershell
python -m ingestion.cli purge --kb hr_policies --yes
python -m ingestion.cli purge --kb products --yes
python -m ingestion.cli purge --kb company --yes
# then delete the sample files
Remove-Item knowledge_bases\hr_policies\* , knowledge_bases\products\* , knowledge_bases\company\*
Remove-Item datasets\company_golden_dataset.* , datasets\product_sample_dataset.csv , datasets\_access.yaml , datasets\.samples_generated
```

(`purge` removes indexed content only; deleting the files prevents re-ingestion.)

## B. Load the real Sapphirus content

1. **HR policies** — copy the approved policy files (PDF/DOCX/…) into `knowledge_bases/hr_policies/`. Create
   `knowledge_bases/hr_policies/_access.yaml`:
   ```yaml
   defaults: {roles: [employee, hr, admin], category: "HR Policy", confidentiality: internal, sample: false}
   files:
     - {pattern: "*Investigation*", roles: [hr, admin], confidentiality: restricted}
     - {pattern: "*Salary*",        roles: [hr, admin], confidentiality: restricted}
   ```
2. **Golden dataset** — save the approved Q&A file as `datasets/company_golden_dataset.csv` (or `.xlsx`/`.json` and
   update `golden_files`). If its columns are named differently, map them under `golden_columns`. Minimum columns:
   question id, question, approved answer. Optional: category, keywords, alternate questions (`|`-separated),
   access roles, status.
3. **Products** — copy manuals/datasheets into `knowledge_bases/products/`; put the catalogue export at
   `datasets/product_sample_dataset.csv` or point `extra_files` to the new file (e.g. `datasets/product_catalogue.xlsx`).
   Set `entity_fields` to the catalogue's product name / code columns.
4. **Company documents** — copy into `knowledge_bases/company/` with an `_access.yaml` marking internal docs `employee`.
5. Update `routing.examples` and `suggested_questions` with real examples (optional but improves routing).
6. Index: `python -m ingestion.cli sync` → check `python -m ingestion.cli list-docs`.
7. Write a real evaluation set (same CSV columns as `datasets/evaluation_dataset/evaluation_dataset.csv`) and run
   `python -m backend.evaluation.run_evaluation --dataset <file>`; adjust thresholds in `.env` if needed.

## C. Day-to-day updates (no rebuild)

| Task | How |
|---|---|
| Add a document | Admin UI → Upload, or drop the file in the KB folder and `sync --kb <kb>` |
| Update a document | Admin UI → Documents → *Upload new version* (same file type), or overwrite the file and `sync` — only that document is re-embedded (version + 1, old file archived) |
| Delete a document | Admin UI → Delete (file archived to `data/archive/`), or delete the file and `sync` |
| Update approved answers | Edit the golden file and `sync --kb company`, or upload it with *golden dataset* ticked |
| Change who can see a document | Edit `_access.yaml` and `python -m ingestion.cli add --kb <kb> --file <path> --roles ...` or re-upload with roles |
| Add a new knowledge base (e.g. Finance) | Add a block to `knowledge_bases.yaml` + folder + `sync --kb finance` — routing, ACL, UI and evaluation pick it up automatically |
| Change embedding model | Edit `.env` then `python -m ingestion.cli reindex` |

Unchanged files are detected by SHA-256 and skipped; identical content uploaded under a different name is rejected
as a duplicate; knowledge-base separation and permissions are preserved because they come from configuration,
not from the files themselves.
