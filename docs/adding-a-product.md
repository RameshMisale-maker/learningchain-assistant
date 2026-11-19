# Adding a product

A new product needs **a YAML file and a knowledge base - no pipeline code changes**. The test
`tests/test_products.py::test_dummy_product_works_end_to_end_with_config_only` proves this with a made-up product.

Steps:

1. Create its knowledge base.
2. Write `config/products/<product_id>.yaml`.
3. Turn on per-request product resolution.
4. Index, test and (optionally) add approved answers, eval cases and crawl sources.

---

## 1. Knowledge base

Add an entry to `config/knowledge_bases.yaml` (copy an existing one) and put the documents in its folder:

```yaml
knowledge_bases:
  acme:
    display_name: Acme Widgets
    enabled: true
    allowed_roles: [public, employee, hr, admin]
    default_document_roles: [public, employee, hr, admin]
    source_dirs: [knowledge_bases/acme]
    golden_files: [datasets/acme_golden_dataset.csv]     # optional approved Q&A
    chunking: {strategy: parent_child, chunk_size: 1800, chunk_overlap: 0, child_size: 400, child_overlap: 80}
    routing: {keywords: [acme, widget, calibrate], examples: ["How do I calibrate an Acme widget?"]}
```

Supported files: PDF, DOCX, Markdown, TXT, HTML, CSV / XLSX. The approved Q&A CSV columns are `Question ID`,
`Category`, `Question`, `Approved Answer`, `Keywords`, `Alternate Questions` (`|`-separated), `Access Roles`,
`Status`, `Source URL`.

## 2. Product file

`config/products/<product_id>.yaml`. The file name must equal `product_id` (lower case, letters / digits / `_`).
Unknown keys are rejected at startup, so typos fail fast. Start from `config/products/learningchain.yaml`.

```yaml
product_id: acme
display_name: Acme Widgets
company: Acme Industries
company_short: Acme
bot_name: Acme Helper
description: Help with Acme widgets - setup, calibration and support.
covers: [Acme widgets]                 # what the assistant answers about (used in prompts and replies)
help_topics: setup, calibration, warranty and orders
aliases: [acme_help]                   # legacy `assistant` values that select this product
domains: [help.acme.example]           # request host names that select this product
kb: acme                               # the product's own knowledge base
namespaces: [acme]                     # every KB this product may search
scopes: {all: [acme]}                  # optional named subsets (legacy `assistant` scopes)
glossary:                              # typo / short-form fixes before search
  - term: Acme
    variants: [acmee, akme]
domain_terms: [widget, calibrate, dial]
topics: [widgets, calibration, warranty, orders]
examples: ["How do I calibrate an Acme widget?", "What does the warranty cover?"]
escalation: {email: support@acme.example, phone: "", hours: "Mon-Fri 9-6"}
semantic_cache_threshold: 0.95         # response cache: how similar a question must be to reuse an answer
prompt_version: "1"                    # bump when you change this product's wording; logged with every answer
welcome_message: Hi! I'm the Acme Helper. Ask me about setting up and using Acme widgets.
retrieval: {namespace: acme, top_k: 5, rerank_top: 5, min_score: 0.15}
models:                                # informational today; the LLM chain itself comes from .env
  primary: {provider: groq, model: openai/gpt-oss-120b}
  fallback: [{provider: groq, model: openai/gpt-oss-20b}]
sources:                               # website crawl (DOC_TRACKING_ENABLED seeds these sitemaps, crawled daily)
  sitemaps: [https://help.acme.example/sitemap.xml]
  include_paths: ["^/docs/"]           # regular expressions on the URL path; empty = everything
  exclude_paths: ["/tag/", "/author/"]
  allowed_domains: [help.acme.example]
ui:
  title: Acme Helper
  subtitle: Answers from the Acme documentation
  welcome: Hi! Ask me anything about Acme widgets.
  suggestions: ["How do I calibrate an Acme widget?"]
  launcher_label: Ask Acme
  api_model_id: acme-assistant         # model name on the OpenAI-compatible API ("" = not offered there)
  api_model_name: Acme Helper
replies:                               # required fixed replies (see backend/products/config.py `Replies`)
  out_of_scope: I can only help with Acme widgets.
  no_confirmed_information: I don't have confirmed information about that in the Acme documentation.
  blocked: I can only help with Acme widget questions.
  restricted_topic: I can't help with that topic.
  greeting_first: "{hello} I'm the Acme Helper. How can I help?"
  who_are_you: I'm the Acme Helper, an assistant for Acme widgets.
  what_can_you_do: I answer questions about setting up and using Acme widgets.
  bye: Goodbye!
  closing: Glad I could help!
  account: For account questions please contact support@acme.example.
  escalation: I'll pass this to our support team - please email support@acme.example.
  frustration_again: I'm sorry this is still not working. Please contact support@acme.example.
  rephrase: Could you rephrase your question?
  access_denied_guest: "Please sign in to ask about {names}."
```

Optional replies have defaults (greetings, thanks, `service_unavailable` for when every LLM provider is down, ...).

## 3. Turn it on

```env
PRODUCT_CONFIG_ENABLED=true     # resolve the product per request (id / alias / host / API key / default)
DEFAULT_PRODUCT=sapphirus       # used when nothing in the request names a product
```

Callers select the product with:
- the chat API: `{"message": "...", "product": "acme"}`;
- the chat page / widget: `/chat?product=acme` or `<script src=".../widget.js" data-product="acme">`;
- the OpenAI-compatible API: set `ui.api_model_id` and use that model name;
- the request host (`domains`), or the Streamlit app with `PRODUCT_ID=acme`.

Restart the API: an invalid file stops startup with a clear error.

## 4. Index, test, improve

```bash
python -m ingestion.cli sync --kb acme           # or build-version --kb acme with INDEX_VERSIONING_ENABLED=true
python -m ingestion.cli list-kbs
curl -s localhost:8000/api/v1/ui?product=acme    # the product's UI texts
curl -s -X POST localhost:8000/api/v1/chat -H "Content-Type: application/json" \
     -d '{"message": "How do I calibrate an Acme widget?", "product": "acme"}'
```

Then, optionally:

- **Eval cases for the index gate:** `evals/acme.jsonl`, one case per line:
  `{"question": "How do I calibrate it?", "kb": "acme", "expected": ["acme_calibration.md"]}`
  (`expected` matches a chunk id, source file or document name). Without this file the gate uses the KB's approved
  Q&A.
- **Approved answers:** upload a CSV in the admin API (`POST /api/v1/admin/golden-qa/import`, fields `product`,
  `kb`, `file`), or approve answers in the admin page's "Review queue" tab.
- **Website crawl:** with `DOC_TRACKING_ENABLED=true` the product's sitemaps are registered at startup and
  crawled daily; or call `POST /api/v1/admin/web-sources/seed-from-config`.
- **Monitoring:** every metric and alert is labelled with `product`; the Grafana dashboard has a product filter.

## Checklist

- [ ] KB entry in `config/knowledge_bases.yaml` and documents in its folder
- [ ] `config/products/<id>.yaml` (file name = `product_id`), namespaces exist, aliases unique
- [ ] `PRODUCT_CONFIG_ENABLED=true`; API restarted without config errors
- [ ] KB indexed (`list-kbs` shows chunks)
- [ ] A test question answered with sources; an off-topic question gets the product's `out_of_scope` reply
- [ ] Optional: `evals/<id>.jsonl`, approved answers, sitemaps
