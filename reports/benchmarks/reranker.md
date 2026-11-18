# Benchmark: reranker

Generated 2026-11-13 15:03 on the live knowledge base (isolated copies). Retrieval metrics: approved-answer shortcut off, expected KB fixed, k=5. End-to-end: full pipeline without an LLM.

| Configuration | Recall@5 | MRR | NDCG@5 | P@5 | Dest. acc. | Answer correct. | In-scope refusals | Retr. p50 ms | Retr. p95 ms | E2E p50 ms | Chunks | Tok/chunk med | Dim | Index s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ms-marco-MiniLM-L-6 (current) | 0.948 | 0.888 | 0.892 | 0.394 | 0.880 | 0.816 | 0.103 | 1333.7 | 2444.8 | 637.8 | 240 | 149.5 | 384 | 14.4 |
| bge-reranker-base | 0.927 | 0.845 | 0.859 | 0.390 | 0.863 | 0.782 | 0.120 | 4656.6 | 16384.6 | 3555.7 | 240 | 149.5 | 384 | 21.7 |
| no reranking | 0.948 | 0.872 | 0.876 | 0.417 | 0.795 | 0.715 | 0.111 | 15.4 | 18.6 | 39.3 | 379 | 144 | 384 | 24.7 |
