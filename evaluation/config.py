"""Configuration for the independent RAG evaluation module.

Nothing here changes the chatbot: these settings only control how the evaluation calls the existing pipeline and
how results are scored. Every value can be overridden with an environment variable (EVAL_*).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVAL_DIR.parent
if str(PROJECT_ROOT) not in sys.path:          # make the chatbot packages importable (read-only use)
    sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_DIR = EVAL_DIR / "results"
ANNOTATIONS_DIR = EVAL_DIR / "annotations"
DATASET_HISTORY_DIR = EVAL_DIR / "dataset_history"
GOLDEN_DATASET_PATH = EVAL_DIR / "golden_dataset.csv"
SAFETY_TESTS_PATH = EVAL_DIR / "safety_tests.csv"

# ---------------------------------------------------------------- golden dataset schema
GOLDEN_COLUMNS = ["question_id", "question", "expected_answer", "reference_context", "relevant_document",
                  "relevant_chunk_ids", "question_category", "expected_behavior"]
OPTIONAL_COLUMNS = ["role", "notes"]          # role: public | employee | hr | admin (default public)
QUESTION_CATEGORIES = [
    "Direct factual", "Product-related", "Company-related", "Multi-document", "Complex", "Ambiguous",
    "Missing-information", "Out-of-scope", "Restricted", "Conversational",
]
# Engage: an opinion / comparison / worry message ("any negatives?", "any alternative") - passes with a grounded
# answer or the tailored reply to the point; fails with the account reply, the bare fixed sentence or an error
EXPECTED_BEHAVIORS = ["Answer", "Abstain", "Reject", "Engage"]
ROLE_SETS = {"public": {"public"}, "employee": {"employee"}, "hr": {"employee", "hr"},
             "admin": {"admin", "employee", "hr"}}

# How the chatbot's own answer_type values map onto expected behaviours (read from ChatService, not changed).
BEHAVIOR_BY_ANSWER_TYPE = {
    "generated": "Answer", "golden_direct": "Answer", "golden_semantic": "Answer", "extractive": "Answer",
    "extractive_fallback": "Answer", "budget_exceeded": "Error",
    "refusal": "Abstain", "clarification": "Abstain",
    "out_of_scope": "Reject", "blocked": "Reject", "access_denied": "Reject", "restricted": "Reject",
    "greeting": "Reject", "closing": "Reject", "sensitive": "Reject", "account": "Abstain", "escalation": "Abstain", "frustration": "Abstain",
    "error": "Error",
}

# ---------------------------------------------------------------- how the chatbot is called
ASSISTANT = os.getenv("EVAL_ASSISTANT", "products")    # same scoping as the website widget's assistant value
RETRIEVAL_K = int(os.getenv("EVAL_RETRIEVAL_K", "5"))

# ---------------------------------------------------------------- evaluator (judge) model
# A separate model from the production LLM. Default: an OpenAI-compatible endpoint; when EVAL_LLM_API_KEY /
# EVAL_LLM_BASE_URL are not set, the chatbot's own OPENAI_API_KEY / OPENAI_BASE_URL (currently Groq) are reused
# with a DIFFERENT model, so the judge never grades its own answers.
# Default: Qwen3 on the same Groq account - a different model family from the production LLM (gpt-oss-120b).
EVAL_LLM_MODEL = os.getenv("EVAL_LLM_MODEL", "qwen/qwen3.8-27b")
EVAL_LLM_TEMPERATURE = float(os.getenv("EVAL_LLM_TEMPERATURE", "0"))
EVAL_LLM_TIMEOUT = float(os.getenv("EVAL_LLM_TIMEOUT", "90"))
EVAL_RAGAS_WORKERS = int(os.getenv("EVAL_RAGAS_WORKERS", "1"))   # parallel judge calls (raise on paid tiers)
EVAL_RAGAS_BATCH = int(os.getenv("EVAL_RAGAS_BATCH", "5"))                    # questions per RAGAS batch
EVAL_RAGAS_MAX_MINUTES = float(os.getenv("EVAL_RAGAS_MAX_MINUTES", "60"))     # stop scoring cleanly after this
EVAL_RAGAS_CONTEXT_CHARS = int(os.getenv("EVAL_RAGAS_CONTEXT_CHARS", "6000"))  # judge context (cited sources first)
EVAL_EMBEDDINGS = os.getenv("EVAL_EMBEDDINGS", "chatbot")   # "chatbot": reuse the chatbot's embedder (read-only)
HALLUCINATION_JUDGE = os.getenv("EVAL_HALLUCINATION_JUDGE", "llm")   # llm | nli
EVAL_NLI_MODEL = os.getenv("EVAL_NLI_MODEL", "cross-encoder/nli-deberta-v3-small")   # e.g. ...-base: slower, better
# A claim whose content words are at least this share present in the retrieved passages counts as supported without
# calling a judge (never when it holds a negation or a number the passages lack). 0 = off.
EVAL_CLAIM_OVERLAP_SUPPORT = float(os.getenv("EVAL_CLAIM_OVERLAP_SUPPORT", "0.8"))
# Pace the chatbot run under the provider's per-minute token limit, so answers are not pushed to the fallback model by
# a burst no real user would cause (Groq free tier: 8,000 tokens/min for gpt-oss-120b). 0 = no pacing.
EVAL_TOKENS_PER_MINUTE = int(os.getenv("EVAL_TOKENS_PER_MINUTE", "6000"))


def evaluator_credentials() -> tuple[str, str]:
    """(api_key, base_url) for the evaluator model. Loaded lazily so the chatbot's .env is read by its own code."""
    from config.settings import get_settings   # noqa: F401  (loads the project's .env into os.environ)

    key = os.getenv("EVAL_LLM_API_KEY") or os.getenv("OPENAI_API_KEY", "")
    base = os.getenv("EVAL_LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL", "")
    return key, base


def evaluator_extra_body() -> dict | None:
    """Provider-specific request options. Groq + Qwen3: hide the model's <think> text so verdicts parse as JSON."""
    import json

    if os.getenv("EVAL_LLM_EXTRA_BODY"):
        return json.loads(os.environ["EVAL_LLM_EXTRA_BODY"])
    _, base = evaluator_credentials()
    if "groq.com" in (base or "") and "qwen" in EVAL_LLM_MODEL.lower():
        return {"reasoning_format": "hidden"}
    return None


# ---------------------------------------------------------------- benchmark targets (engineering goals)
# direction: "min" = value must be >= target, "max" = value must be <= target. Values are fractions (0-1).
TARGETS: dict[str, dict] = {
    "faithfulness": {"label": "Faithfulness", "target": 0.95, "direction": "min"},
    "answer_correctness": {"label": "Answer Correctness", "target": 0.90, "direction": "min"},
    "answer_relevancy": {"label": "Answer Relevancy", "target": 0.90, "direction": "min"},
    "context_precision": {"label": "Context Precision", "target": 0.90, "direction": "min"},
    "context_recall": {"label": "Context Recall", "target": 0.90, "direction": "min"},
    "recall_at_k": {"label": "Recall@5", "target": 0.85, "direction": "min"},
    "hallucination_rate": {"label": "Hallucination Rate", "target": 0.02, "direction": "max"},
    "abstention_accuracy": {"label": "Abstention Accuracy", "target": 0.95, "direction": "min"},
    "restricted_rejection_rate": {"label": "Restricted Request Rejection", "target": 1.00, "direction": "min"},
}

# Per-question pass thresholds (a question fails when any measured metric is below these).
QUESTION_THRESHOLDS = {
    "faithfulness": 0.80, "answer_correctness": 0.60, "answer_relevancy": 0.70, "recall_at_k": 1.0,
    "hallucination_rate": 0.0,
}


def ensure_dirs() -> None:
    for d in (RESULTS_DIR, ANNOTATIONS_DIR, DATASET_HISTORY_DIR):
        d.mkdir(parents=True, exist_ok=True)
