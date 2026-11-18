# Deliverables 2 & 3 — Architecture and Workflow Diagrams

Diagrams are written in Mermaid (render in GitHub, VS Code, or the pre-rendered SVGs in `docs/diagrams/`).

## 2.1 End-to-end architecture

```mermaid
flowchart LR
    U[User] --> P[Sapphirus Company Portal<br/>React / existing portal]
    P -->|SSO identity| GW[API gateway / reverse proxy<br/>TLS, SSO, rate limits]
    U2[Streamlit UI<br/>pilot & admin] --> API
    GW --> API[FastAPI backend<br/>/api/v1/chat, /admin]
    subgraph Core[Assistant core - backend/services/chat_service.py]
      direction TB
      Q[Query validation &<br/>injection screening] --> RW[Follow-up rewrite +<br/>abbreviation expansion]
      RW --> R[Intent classification &<br/>KB routing + RBAC]
      R --> G{Golden match?}
      G -- strong --> GA[Approved answer verbatim]
      G -- no/weak --> H[Hybrid retrieval per KB<br/>dense + BM25 + RRF]
      H --> RR[Cross-encoder rerank]
      RR --> CS[Context selection]
      CS --> LLM[LLM generation<br/>OpenAI / Claude]
      LLM --> V[Grounding check<br/>+ citation validation]
      V -->|fails| EX[Extractive fallback]
    end
    API --> Core
    subgraph Stores[Knowledge & data stores]
      VC1[(sapphirus_company)]
      VC2[(sapphirus_products)]
      VC3[(sapphirus_hr_policies)]
      PG[(PostgreSQL<br/>users, documents, chat,<br/>feedback, audit)]
    end
    H --> VC1 & VC2 & VC3
    API --> PG
    ADM[Admin upload / CLI sync] --> ING[Ingestion pipeline] --> Stores
```

**Key design choices**

* **One collection per knowledge base** keeps HR, product and company content physically separate; the router decides which collections are searched.
* **ACL is enforced in retrieval** (KB-level in the router, document-level and golden-row-level in the retriever), so restricted text never reaches the prompt.
* **Golden answers bypass the LLM** when the match is strong and covers the question's specific terms.
* **Everything is pluggable through `.env`**: LLM provider, embedding model, vector store, reranker, router mode, thresholds.

## 3.1 Company golden dataset workflow

```mermaid
flowchart TD
    A[Approved Q&A file<br/>CSV / XLSX / JSON] --> B[Column mapping from<br/>golden_columns in YAML]
    B --> C{Status retired/draft?}
    C -- yes --> X[Skip]
    C -- no --> D[One chunk per Q&A row<br/>question + alternates + keywords + answer]
    D --> E[Row-level roles<br/>Access Roles column]
    E --> F[(company collection<br/>doc_type = golden_qa)]
    Q[User question] --> M[Golden matcher<br/>0.45 dense + 0.30 term Jaccard + 0.25 fuzzy<br/>max over all phrasings]
    F --> M
    M --> T{score >= GOLDEN_DIRECT_THRESHOLD<br/>and specific terms covered?}
    T -- yes --> V[Return approved answer verbatim<br/>citation: Qxxx]
    T -- no --> S{score >= GOLDEN_ASSIST_THRESHOLD?}
    S -- yes --> W[Inject as top-priority APPROVED ANSWER<br/>source for the LLM]
    S -- no --> R[Normal hybrid RAG]
```

## 3.2 Product knowledge base workflow

```mermaid
flowchart TD
    A[Catalogue CSV/XLSX<br/>manuals, datasheets, FAQs] --> B[Loaders: records stay whole,<br/>PDF tables to Markdown]
    B --> C[Entity lexicon from<br/>product_name / code / short name]
    C --> D[(products collection)]
    Q[Question] --> E[Entity detection<br/>e.g. Product A -> Orion Analytics]
    E --> F{Comparison intent?}
    F -- yes, >=2 products --> G[Per-product retrieval<br/>guarantees context for each]
    F -- yes, 0 products --> CL[Ask which products]
    F -- no --> H[Hybrid retrieval + entity focus<br/>demote other products]
    G --> I[Rerank]
    H --> I
    I --> J[LLM: Markdown table, Not specified for gaps<br/>offline: table built from catalogue records]
    J --> K[Grounding: prices / numbers must exist in sources]
```

## 3.3 HR policies workflow

```mermaid
flowchart TD
    A[HR policy documents<br/>DOCX / PDF / MD] --> B[_access.yaml<br/>default roles: employee, hr, admin<br/>HR-INT-*: hr, admin]
    B --> C[(hr_policies collection)]
    Q[Question] --> R{User role allowed<br/>for HR KB?}
    R -- no --> D[Access denied message + audit log]
    R -- yes --> E[Hybrid retrieval<br/>document ACL filter]
    E --> F[Rerank -> context]
    F --> G[Answer with policy name + section citations]
    G --> H[Grounding check:<br/>days, weeks, amounts must match]
```

## 3.4 Query routing workflow

```mermaid
flowchart TD
    Q[Query] --> S{Greeting / small talk?} -- yes --> GR[Greeting]
    S -- no --> X[Expand abbreviations<br/>WFH, PTO, SSO...]
    X --> K[Signal 1: routing keywords]
    X --> E[Signal 2: similarity to example questions]
    X --> C[Signal 3: corpus term-to-KB association<br/>IDF-weighted, coverage-scaled]
    X --> N[Entity detection]
    K & E & C & N --> SC[Weighted score per KB]
    SC --> O{best < min_route_score?}
    O -- yes --> OOS[Out of scope]
    O -- no --> A{Top-2 weak and close?}
    A -- yes --> AM[Search both KBs;<br/>clarify if retrieval is weak]
    A -- no --> M[Select best KB + KBs >= 0.75 x best]
    M --> RB{Role allowed?}
    RB -- none --> DEN[Access denied]
    RB -- some --> OK[Route to allowed KBs]
    AM -. optional .-> L[LLM classifier for ambiguous /<br/>low-confidence cases]
    OOS -. optional .-> L
```

## 3.5 Document ingestion workflow

```mermaid
flowchart TD
    U[Admin upload / CLI sync / folder drop] --> V[Upload validation<br/>extension, size, magic bytes,<br/>macros, zip bombs]
    V --> H[SHA-256 content hash]
    H --> D{Same path,<br/>same hash?}
    D -- yes --> UN[Unchanged - skip]
    D -- no --> DU{Same hash under<br/>another path?}
    DU -- yes --> DUP[Duplicate - skip]
    DU -- no --> L[Type detection + extraction<br/>text, tables, records]
    L --> OCR{Page has no text<br/>but images?}
    OCR -- yes --> T[Tesseract OCR<br/>if OCR_ENABLED]
    OCR -- no --> CL
    T --> CL[Clean & normalise<br/>headers/footers, soft wraps]
    CL --> CH[Structure-aware chunking<br/>+ contextual headers]
    CH --> INJ[Prompt-injection scan<br/>flag chunks]
    INJ --> EM[Embeddings]
    EM --> DEL[Delete previous chunks of doc]
    DEL --> UP[(Upsert to KB collection<br/>with full metadata)]
    UP --> REG[(Registry: version+1,<br/>ACL, hash, chunk count)]
    REG --> REV[Bump KB revision -><br/>indexes refresh everywhere]
```

## 3.6 Complete chatbot execution

```mermaid
sequenceDiagram
    actor User
    participant UI as Portal / Streamlit
    participant API as FastAPI
    participant Auth as Auth + RBAC
    participant Svc as ChatService
    participant Router
    participant Golden as Golden matcher
    participant Ret as Hybrid retriever
    participant RR as Reranker
    participant LLM
    participant DB as PostgreSQL

    User->>UI: question
    UI->>API: POST /api/v1/chat (Bearer token / SSO headers)
    API->>Auth: authenticate, rate limit
    API->>Svc: answer(question, user, session)
    Svc->>Svc: validate, injection screen, rewrite follow-up
    Svc->>Router: route(query, roles)
    Router-->>Svc: intent, KBs, entities
    alt greeting / out of scope / clarify / denied
        Svc-->>API: canned response
    else knowledge question
        Svc->>Golden: match(query)
        alt strong approved match
            Golden-->>Svc: approved answer (verbatim)
        else
            Svc->>Ret: retrieve per KB (ACL filtered)
            Ret->>RR: candidates
            RR-->>Svc: top-K context
            Svc->>LLM: system rules + sources + question
            LLM-->>Svc: answer with [S#] citations
            Svc->>Svc: grounding check, citation validation, fallback if needed
        end
    end
    Svc->>DB: store messages, route, citations, latency, tokens, cost
    API-->>UI: answer, citations, KBs, message_id
    User->>UI: Helpful / Not helpful
    UI->>API: POST /api/v1/feedback
```

## Routing classifier trade-offs (prompt §5)

| Option | Pros | Cons | Used |
|---|---|---|---|
| Rules / keywords only | Fast, transparent | Brittle, needs maintenance | As one signal |
| Lightweight similarity (examples + corpus statistics) | No training data needed, adapts automatically when documents change, ~ms latency, free | Weaker on ambiguous phrasing | **Default (hybrid)** |
| Fine-tuned classifier | Accurate with labelled data | Needs labelled traffic and retraining | Later, from feedback logs |
| LLM classifier | Best semantic understanding, can ask clarifying questions | Latency + cost per query, non-deterministic | Only for ambiguous/low-confidence queries (`ROUTER_MODE=hybrid`), or always (`ROUTER_MODE=llm`) |
