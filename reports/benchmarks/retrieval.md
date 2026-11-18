# Benchmark: retrieval

Generated 2026-11-13 14:39 on the live knowledge base (isolated copies). Retrieval metrics: approved-answer shortcut off, expected KB fixed, k=5. End-to-end: full pipeline without an LLM.

| Configuration | Recall@5 | MRR | NDCG@5 | P@5 | Dest. acc. | Answer correct. | In-scope refusals | Retr. p50 ms | Retr. p95 ms | E2E p50 ms | Chunks | Tok/chunk med | Dim | Index s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hybrid 50/50 + rerank (current) | 0.948 | 0.888 | 0.892 | 0.394 | 0.880 | 0.816 | 0.103 | 1199.2 | 1570.1 | 1078.5 | 240 | 149.5 | 384 | 17.5 |
| dense only + rerank | 0.948 | 0.888 | 0.893 | 0.396 | 0.880 | 0.816 | 0.103 | 1158.5 | 1551.9 | 1117.1 | 240 | 149.5 | 384 | 30.9 |
| BM25 only + rerank | 0.948 | 0.893 | 0.896 | 0.394 | 0.880 | 0.816 | 0.103 | 1262.7 | 4609.3 | 923.6 | 240 | 149.5 | 384 | 56.5 |
