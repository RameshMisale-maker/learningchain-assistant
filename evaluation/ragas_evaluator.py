"""RAGAS evaluation of the chatbot's ACTUAL answers and the ACTUAL contexts it sent to the LLM.

* Evaluator LLM: a separate judge model (config.EVAL_LLM_MODEL) - never the production model's own output grading
  itself. Any OpenAI-compatible endpoint works (Groq, OpenAI, Azure OpenAI ...).
* Evaluator embeddings (Answer Relevancy): the chatbot's embedding model, imported read-only.
* RAGAS and LangChain live in evaluation/.venv only (see README) so the chatbot's Python environment is untouched.
"""
from __future__ import annotations

import math
import re
import time
from typing import Any

from evaluation import config

RAGAS_METRICS = ["faithfulness", "answer_relevancy", "answer_correctness", "context_precision", "context_recall"]


class RagasUnavailable(RuntimeError):
    pass


def ragas_status() -> tuple[bool, str]:
    try:
        import ragas  # noqa: F401
        from langchain_openai import ChatOpenAI  # noqa: F401
    except Exception as exc:
        return False, (f"RAGAS is not installed in this Python environment ({type(exc).__name__}: {exc}). "
                       "Start the dashboard with evaluation\\run_dashboard.bat (uses evaluation/.venv).")
    key, _ = config.evaluator_credentials()
    if not key:
        return False, "No evaluator API key: set EVAL_LLM_API_KEY (or OPENAI_API_KEY) to run RAGAS."
    return True, f"RAGAS ready - evaluator model {config.EVAL_LLM_MODEL}"


def _chatbot_embeddings():
    """LangChain Embeddings adapter over the chatbot's own embedder (read-only use)."""
    from langchain_core.embeddings import Embeddings

    from ingestion.embedding import get_embedder

    embedder = get_embedder()

    class ChatbotEmbeddings(Embeddings):
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [list(map(float, v)) for v in embedder.embed_documents(list(texts))]

        def embed_query(self, text: str) -> list[float]:
            return list(map(float, embedder.embed_query(text)))

    return ChatbotEmbeddings()


def build_evaluator():
    ok, msg = ragas_status()
    if not ok:
        raise RagasUnavailable(msg)
    from langchain_openai import ChatOpenAI
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    key, base = config.evaluator_credentials()
    chat = ChatOpenAI(model=config.EVAL_LLM_MODEL, api_key=key, base_url=base or None,
                      temperature=config.EVAL_LLM_TEMPERATURE, timeout=config.EVAL_LLM_TIMEOUT, max_retries=6,
                      extra_body=config.evaluator_extra_body())
    return LangchainLLMWrapper(chat), LangchainEmbeddingsWrapper(_chatbot_embeddings())


def judge_contexts(answer: str, contexts: list, max_chars: int = config.EVAL_RAGAS_CONTEXT_CHARS) -> list[str]:
    """Contexts given to the judge: the sources the answer cites ([S1], [S3] ...) first, then the rest, capped at
    `max_chars` in total. Keeps each judge call small enough for per-minute token limits on hosted judges, while
    the evidence the answer relies on is always included."""
    items = [(c.get("sid", ""), c.get("text", "")) if isinstance(c, dict) else ("", str(c)) for c in contexts]
    cited = set(re.findall(r"\[(S\d+)\]", answer or ""))
    ordered = [t for s, t in items if s in cited] + [t for s, t in items if s not in cited]
    out, used = [], 0
    for text in ordered:
        if used >= max_chars:
            break
        piece = text[: max_chars - used]
        out.append(piece)
        used += len(piece)
    return out


def evaluate_rows(rows: list[dict[str, Any]], progress=None,
                  max_minutes: float = config.EVAL_RAGAS_MAX_MINUTES) -> dict[str, dict[str, float | None]]:
    """rows: [{question_id, question, answer, contexts: [str | {sid, text}], reference: str}]
    Returns {question_id: {metric: score | None, "ragas_error": str}}.

    Scored in small batches so progress is visible and a time budget can stop cleanly: questions not reached within
    `max_minutes` are reported as "not scored (time budget)" instead of the run waiting on a throttled judge."""
    from ragas import EvaluationDataset, RunConfig, SingleTurnSample, evaluate
    from ragas.metrics import (AnswerCorrectness, Faithfulness, LLMContextPrecisionWithReference, LLMContextRecall,
                               ResponseRelevancy)

    llm, emb = build_evaluator()
    out: dict[str, dict[str, Any]] = {}
    for r in rows:
        out[r["question_id"]] = {m: None for m in RAGAS_METRICS}
        if not r.get("contexts"):
            out[r["question_id"]]["ragas_error"] = "no retrieved context (answer did not use the knowledge base)"
    scorable = [r for r in rows if r.get("contexts") and r.get("answer")]

    # one call at a time, and a per-call timeout long enough for the client to wait out a 429 (retry-after) and
    # retry - a short timeout turns every throttled call into a TimeoutError loop
    run_cfg = RunConfig(max_workers=config.EVAL_RAGAS_WORKERS, timeout=600, max_retries=4, max_wait=60)
    full = [Faithfulness(), ResponseRelevancy(strictness=1), AnswerCorrectness(), LLMContextPrecisionWithReference(),
            LLMContextRecall()]
    no_ref_metrics = [Faithfulness(), ResponseRelevancy(strictness=1)]
    names = {"faithfulness": "faithfulness", "answer_relevancy": "answer_relevancy",
             "answer_correctness": "answer_correctness",
             "llm_context_precision_with_reference": "context_precision", "context_recall": "context_recall"}
    deadline = time.monotonic() + max_minutes * 60
    batch_size = max(1, config.EVAL_RAGAS_BATCH)
    done = 0
    for start in range(0, len(scorable), batch_size):
        chunk = scorable[start:start + batch_size]
        if time.monotonic() > deadline:
            for r in scorable[start:]:
                out[r["question_id"]]["ragas_error"] = f"not scored (RAGAS time budget of {max_minutes:g} min reached)"
            break
        for has_ref, metrics in ((True, full), (False, no_ref_metrics)):
            batch = [r for r in chunk if bool((r.get("reference") or "").strip()) == has_ref]
            if batch:
                _score_batch(batch, metrics, llm, emb, run_cfg, names, out, evaluate, EvaluationDataset,
                             SingleTurnSample)
        done += len(chunk)
        if progress:
            progress(done, len(scorable))
    return out


def _score_batch(batch, metrics, llm, emb, run_cfg, names, out, evaluate, EvaluationDataset, SingleTurnSample):
    samples = [SingleTurnSample(user_input=r["question"], response=r["answer"],
                                retrieved_contexts=judge_contexts(r["answer"], r["contexts"]),
                                reference=(r.get("reference") or None)) for r in batch]
    try:
        result = evaluate(EvaluationDataset(samples=samples), metrics=metrics, llm=llm, embeddings=emb,
                          run_config=run_cfg, show_progress=False, raise_exceptions=False)
        df = result.to_pandas()
    except Exception as exc:
        for r in batch:
            out[r["question_id"]]["ragas_error"] = f"RAGAS failed: {type(exc).__name__}: {exc}"[:400]
        return
    for r, (_, row) in zip(batch, df.iterrows()):
        scores = out[r["question_id"]]
        missing = []
        for col, key in names.items():
            if col in row.index:
                v = row[col]
                nan = v is None or (isinstance(v, float) and math.isnan(v))
                scores[key] = None if nan else round(float(v), 4)
                if nan:
                    missing.append(key)
        if missing:
            scores["ragas_error"] = (f"not scored: {', '.join(missing)} (evaluator timeout, rate limit or "
                                     "unparseable judge output) - re-run to retry")
