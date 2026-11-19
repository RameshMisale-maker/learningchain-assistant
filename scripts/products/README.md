# Product-specific data tools

One-off import / dataset-building scripts that are specific to one product live in `scripts/products/<product_id>/`
(for example scraping that product's help centre or FAQ page). Everything the running chatbot needs about a product
is in `config/products/<product_id>.yaml`; these scripts only produce knowledge-base files and datasets.

| Product | Script | Purpose |
|---|---|---|
| learningchain | `fetch_zendesk.py`, `zendesk_to_md.py` | Download the help-centre articles and convert them to Markdown |
| learningchain | `import_learningchain_faq.py` | Import the website FAQ as approved Q&A |
| learningchain | `build_evaluation_dataset.py` | Build the legacy evaluation question set |

Run from the project root, e.g. `python scripts/products/learningchain/import_learningchain_faq.py`.
