# Benchmark: embedding

Generated 2026-11-13 15:43 on the live knowledge base (isolated copies). Retrieval metrics: approved-answer shortcut off, expected KB fixed, k=5. End-to-end: full pipeline without an LLM.

| Configuration | Recall@5 | MRR | NDCG@5 | P@5 | Dest. acc. | Answer correct. | In-scope refusals | Retr. p50 ms | Retr. p95 ms | E2E p50 ms | Chunks | Tok/chunk med | Dim | Index s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bge-small-en-v1.5 (current) | 0.938 | 0.878 | 0.882 | 0.392 | 0.821 | 0.737 | 0.162 | 860.4 | 1424.5 | 774.7 | 379 | 144 | 384 | 35.7 |
| bge-base-en-v1.5 | 0.938 | 0.878 | 0.881 | 0.390 | 0.846 | 0.781 | 0.137 | 1908.4 | 2615.6 | 668.9 | 379 | 144 | 768 | 81.9 |
| bge-large-en-v1.5 | 0.938 | 0.878 | 0.881 | 0.390 | 0.863 | 0.803 | 0.111 | 716.7 | 2322.4 | 713.0 | 379 | 144 | 1024 | 294.6 |
| e5-large-v2 | 0.729 | 0.675 | 0.684 | 0.271 | 0.761 | 0.662 | 0.179 | 1267.6 | 1742.4 | 1088.3 | 357 | 145 | 1024 | 251.8 |
| bge-m3 | 0.729 | 0.674 | 0.683 | 0.273 | 0.735 | 0.662 | 0.231 | 952.9 | 2596.4 | 1041.4 | 357 | 160 | 1024 | 357.7 |
