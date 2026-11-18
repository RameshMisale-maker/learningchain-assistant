# Sapphirus AI Assistant - Evaluation Report

Generated: 2026-11-13T07:25:14+00:00  
K = 5  
Configuration: llm=extractive-fallback, embedding=sentence_transformers:BAAI/bge-small-en-v1.5, reranker=cross_encoder, vector_store=chroma

> Knowledge: LearningChain help centre (products) and sapphirus.in (company). The test set is small,
> so treat these numbers as indicative and grow it with real user questions over time.

> OFFLINE MODE: answers were produced by the extractive (no-LLM) answerer and/or the local hashed
> embedder. Extractive answers copy source text, so faithfulness is ~1.0 by construction and token
> cost is 0. Set OPENAI_API_KEY (and EMBEDDING_BACKEND=openai) and re-run for production-representative
> generation, relevancy and cost metrics.

## Metrics by knowledge base

| Metric | overall | company | products | faq_exact | faq_paraphrase | out_of_scope |
|---|---|---|---|---|---|---|
| questions | 88 | 18 | 20 | 18 | 18 | 14 |
| routing_accuracy | 0.977 | 0.944 | 0.950 | 1.000 | 1.000 | 1.000 |
| recall_at_k | 0.946 | 0.944 | 0.900 | 1.000 | 0.944 | n/a |
| precision_at_k | 0.708 | 0.678 | 0.350 | 1.000 | 0.844 | n/a |
| mrr | 0.899 | 0.917 | 0.825 | 1.000 | 0.861 | n/a |
| ndcg_at_k | 0.903 | 0.905 | 0.828 | 1.000 | 0.889 | n/a |
| context_precision | 0.881 | 0.872 | 0.790 | 1.000 | 0.869 | n/a |
| context_recall | 0.946 | 0.944 | 0.900 | 1.000 | 0.944 | n/a |
| faithfulness | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | n/a |
| answer_relevancy | 0.719 | 0.634 | 0.722 | 0.802 | 0.718 | n/a |
| correctness | 0.837 | 0.773 | 0.633 | 1.000 | 0.837 | 1.000 |
| hallucination_rate | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| unanswered_rate | 0.068 | 0.056 | 0.100 | 0.000 | 0.111 | n/a |
| out_of_scope_refusal_rate | 1.000 | n/a | n/a | n/a | n/a | 1.000 |
| retrieval_failures | 4 | 1 | 2 | 0 | 1 | 0 |
| golden_direct_answers | 44 | 7 | 4 | 18 | 15 | 0 |
| latency_p50_ms | 371.650 | 391.200 | 540.350 | 36.400 | 490.250 | 14.200 |
| latency_p95_ms | 724.080 | 652.510 | 609.640 | 53.925 | 744.155 | 550.045 |
| avg_prompt_tokens | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| avg_completion_tokens | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| avg_cost_usd | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| total_cost_usd | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

## Confusion matrices

### 1. Destination: where the question should go vs. where it went

Rows = expected, columns = actual. `refuse` = out of scope, access denied, not in knowledge base or clarification. The diagonal is correct.

| Expected \ Actual | company | products | refuse | Total |
|---|---|---|---|---|
| **company** | **17** | 0 | 1 | 18 |
| **products** | 0 | **52** | 4 | 56 |
| **refuse** | 0 | 0 | **14** | 14 |
| **Total** | 17 | 52 | 19 | 88 |

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| company | 1.000 | 0.944 | 0.971 | 18 |
| products | 1.000 | 0.929 | 0.963 | 56 |
| refuse | 0.737 | 1.000 | 0.848 | 14 |

Accuracy **0.943** · Macro F1 **0.928** · Weighted F1 **0.946**

### 2. Answer vs. refuse decision

Positive = the bot answered.

| | Bot answered | Bot refused |
|---|---|---|
| **Should answer** | TP = 69 | FN = 5 (missed answer) |
| **Should refuse** | FP = 0 (answered when it shouldn't) | TN = 14 |

Precision **1.000** · Recall **0.932** · F1 **0.965** · Accuracy **0.943** · Specificity (correct refusals) **1.000**


## Metric definitions

- **Recall@K**: question has at least one relevant source in the top K retrieved chunks.
- **Precision@K**: share of the top K chunks that are relevant.
- **MRR**: 1 / rank of the first relevant chunk. **NDCG@K**: rank-discounted relevance.
- **Context precision**: RAGAS-style average precision of relevant chunks in the ranked context.
- **Context recall**: share of expected key facts present in the retrieved context.
- **Faithfulness**: share of answer sentences supported by the sources (lexical grounding check; numbers must match).
- **Answer relevancy**: embedding similarity between question and answer.
- **Correctness**: 0.8 x key-fact coverage + 0.2 x token F1 vs. expected answer (out-of-scope: 1 if correctly refused).
- **Hallucination rate**: answered questions with unsupported numbers or faithfulness < 0.5.
- **Unanswered rate**: in-scope questions that were refused / clarified.

## Failures and issues

| ID | Question | Expected KB | Routed | Type | Correctness |
|---|---|---|---|---|---|
| C06 | Do you have any office outside India? | company | company | extractive | 0.016 |
| C14 | Name some universities you have worked with. | company | products | refusal | 0.000 |
| P01 | What makes LearningChain certificates tamper-proof? | products | products | extractive | 0.021 |
| P02 | Why should a university pick LearningChain? | products | products | refusal | 0.000 |
| P03 | How does LearningChain fit into a university's process? | products | products | golden_semantic | 0.010 |
| P05 | How does an employer check a student's certificate? | products | products | extractive | 0.010 |
| P09 | I can't remember my password, what do I do? | products | products | refusal | 0.000 |
| P20 | Can I book a LearningChain demo? | products | company | golden_direct | 0.877 |
| F03p | What is the purpose of LearningChain for colleges? | products | company|products | golden_semantic | 0.070 |
| F04p | How does blockchain stop someone from altering my certificates? | products | products | refusal | 0.000 |
| F07p | Why can't a LearningChain QR code be faked like an ordinary one? | products | products | refusal | 0.000 |
