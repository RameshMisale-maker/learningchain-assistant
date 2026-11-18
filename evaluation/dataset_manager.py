"""Golden dataset management (load / validate / save) and human annotation storage.

The golden dataset lives in evaluation/golden_dataset.csv. Every save first copies the previous version to
evaluation/dataset_history/, so edits made in the dashboard can always be traced and undone.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from evaluation import config


# ------------------------------------------------------------------------------------------------ dataset I/O
def empty_dataset() -> pd.DataFrame:
    return pd.DataFrame(columns=config.GOLDEN_COLUMNS + ["role", "notes"])


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    for col in config.GOLDEN_COLUMNS + config.OPTIONAL_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    df = df.fillna("")
    for col in df.columns:
        df[col] = df[col].astype(str).str.strip()
    df["role"] = df["role"].replace("", "public")
    # tolerate case / spacing differences in the controlled vocabularies
    cats = {c.lower(): c for c in config.QUESTION_CATEGORIES}
    df["question_category"] = df["question_category"].map(lambda v: cats.get(v.lower().replace("_", " "), v))
    behs = {b.lower(): b for b in config.EXPECTED_BEHAVIORS}
    df["expected_behavior"] = df["expected_behavior"].map(lambda v: behs.get(v.lower(), v))
    return df[config.GOLDEN_COLUMNS + config.OPTIONAL_COLUMNS + [c for c in df.columns
                                                                if c not in config.GOLDEN_COLUMNS + config.OPTIONAL_COLUMNS]]


def load(path: Path | None = None) -> pd.DataFrame:
    path = path or config.GOLDEN_DATASET_PATH
    if not path.exists():
        return empty_dataset()
    return normalize(pd.read_csv(path, dtype=str, keep_default_na=False))


def read_upload(file_obj) -> pd.DataFrame:
    return normalize(pd.read_csv(file_obj, dtype=str, keep_default_na=False))


def save(df: pd.DataFrame, path: Path | None = None) -> Path | None:
    """Save the dataset; the previous version is kept in dataset_history/. Returns the backup path (if any)."""
    config.ensure_dirs()
    path = path or config.GOLDEN_DATASET_PATH
    backup = None
    if path.exists():
        backup = config.DATASET_HISTORY_DIR / f"golden_dataset_{datetime.now():%Y%m%d_%H%M%S_%f}.csv"
        shutil.copy2(path, backup)
    normalize(df).to_csv(path, index=False, encoding="utf-8")
    return backup


def dataset_hash(df: pd.DataFrame) -> str:
    return hashlib.sha1(normalize(df)[config.GOLDEN_COLUMNS].to_csv(index=False).encode("utf-8")).hexdigest()[:12]


# ------------------------------------------------------------------------------------------------ validation
def validate(df: pd.DataFrame, inventory: dict[str, Any] | None = None) -> tuple[list[str], list[str]]:
    """Return (errors, warnings). Errors block an evaluation run; warnings are shown but allowed."""
    errors: list[str] = []
    warnings: list[str] = []
    if df is None or df.empty:
        return ["The dataset is empty."], []
    missing = [c for c in config.GOLDEN_COLUMNS if c not in df.columns]
    if missing:
        return [f"Missing required columns: {', '.join(missing)}"], []
    ids = df["question_id"].str.strip()
    if (ids == "").any():
        errors.append(f"{int((ids == '').sum())} row(s) have no question_id.")
    dupes = sorted(ids[ids.duplicated() & (ids != "")].unique())
    if dupes:
        errors.append(f"Duplicate question_id values: {', '.join(dupes[:10])}")
    if (df["question"].str.strip() == "").any():
        errors.append(f"{int((df['question'].str.strip() == '').sum())} row(s) have an empty question.")
    bad_cat = sorted(set(df.loc[~df["question_category"].isin(config.QUESTION_CATEGORIES), "question_category"]))
    if bad_cat:
        errors.append(f"Unknown question_category values: {bad_cat}. Allowed: {config.QUESTION_CATEGORIES}")
    bad_beh = sorted(set(df.loc[~df["expected_behavior"].isin(config.EXPECTED_BEHAVIORS), "expected_behavior"]))
    if bad_beh:
        errors.append(f"Unknown expected_behavior values: {bad_beh}. Allowed: {config.EXPECTED_BEHAVIORS}")
    bad_role = sorted(set(df.loc[~df["role"].str.lower().isin(config.ROLE_SETS), "role"]))
    if bad_role:
        errors.append(f"Unknown role values: {bad_role}. Allowed: {sorted(config.ROLE_SETS)}")

    answer_rows = df[df["expected_behavior"] == "Answer"]
    no_expected = answer_rows[answer_rows["expected_answer"].str.strip() == ""]
    if len(no_expected):
        warnings.append(f"{len(no_expected)} 'Answer' question(s) have no expected_answer - Answer Correctness and "
                        f"context precision/recall cannot be scored for them: {', '.join(no_expected['question_id'][:8])}")
    no_labels = answer_rows[(answer_rows["relevant_document"].str.strip() == "")
                            & (answer_rows["relevant_chunk_ids"].str.strip() == "")]
    if len(no_labels):
        warnings.append(f"{len(no_labels)} 'Answer' question(s) have neither relevant_document nor "
                        f"relevant_chunk_ids - retrieval metrics will need human annotation for them.")
    if inventory:
        from evaluation.retrieval_metrics import split_list

        known = inventory["chunk_ids"] | inventory["parent_ids"]
        stale = sorted({cid for v in df["relevant_chunk_ids"] for cid in split_list(v) if cid not in known})
        if stale:
            warnings.append(f"{len(stale)} relevant_chunk_ids are not in the current vector index (chunk ids change "
                            f"when the knowledge base is re-indexed): {', '.join(stale[:5])}")
        docs_lower = " ".join(sorted(inventory["documents"])).lower()
        unknown_docs = sorted({m for v in df["relevant_document"] for m in split_list(v)
                               if "=" not in m and m.lower() not in docs_lower})
        if unknown_docs:
            warnings.append(f"{len(unknown_docs)} relevant_document value(s) match no indexed document, file, "
                            f"section or approved-answer id - retrieval for those questions will score 0: "
                            f"{', '.join(unknown_docs[:6])}")
    counts = df["question_category"].value_counts().to_dict()
    for cat in config.QUESTION_CATEGORIES:
        if cat not in counts:
            warnings.append(f"No questions in category '{cat}'.")
    return errors, warnings


# ------------------------------------------------------------------------------------------------ annotations
def _ann_path(name: str) -> Path:
    config.ensure_dirs()
    return config.ANNOTATIONS_DIR / name


def load_json(name: str) -> dict:
    p = _ann_path(name)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def save_json(name: str, data: dict) -> None:
    p = _ann_path(name)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)


def relevance_labels() -> dict[str, dict[str, bool]]:
    """{question_id: {chunk text_key: relevant}} - human relevance judgements."""
    return load_json("relevance_labels.json")


def save_relevance_labels(question_id: str, labels: dict[str, bool]) -> None:
    data = relevance_labels()
    data.setdefault(question_id, {}).update(labels)
    save_json("relevance_labels.json", data)


def claim_reviews(run_id: str) -> dict[str, dict[str, str]]:
    """{question_id: {claim_key: verdict}} - human verification of flagged claims for one run."""
    return load_json("claim_reviews.json").get(run_id, {})


def save_claim_reviews(run_id: str, question_id: str, reviews: dict[str, str]) -> None:
    data = load_json("claim_reviews.json")
    data.setdefault(run_id, {}).setdefault(question_id, {}).update({k: v for k, v in reviews.items() if v})
    save_json("claim_reviews.json", data)


# ------------------------------------------------------------------------------------------------ seeding
LEGACY_CATEGORY_MAP = {
    "products": "Product-related", "faq_exact": "Direct factual", "faq_paraphrase": "Product-related",
    "exact_detail": "Direct factual", "multi_article": "Multi-document", "multi_part": "Complex",
    "comparison": "Complex", "conflict": "Complex", "ambiguous": "Ambiguous", "missing_answer": "Missing-information",
    "misleading": "Missing-information", "out_of_scope": "Out-of-scope", "company": "Company-related",
}


def import_legacy_dataset(path: Path | None = None, include_company: bool = False) -> pd.DataFrame:
    """Convert the project's existing datasets/evaluation_dataset/evaluation_dataset.csv into the golden format.
    Company rows are skipped by default: the Company knowledge base is disabled in config/knowledge_bases.yaml."""
    path = path or (config.PROJECT_ROOT / "datasets/evaluation_dataset/evaluation_dataset.csv")
    src = pd.read_csv(path, dtype=str, keep_default_na=False)
    rows = []
    for r in src.to_dict("records"):
        if r["category"] == "company" and not include_company:
            continue
        flag = r.get("expect_refusal", "false").lower()
        if flag == "true":
            behavior = "Reject" if r["category"] == "out_of_scope" else "Abstain"
        elif flag == "either":
            behavior = "Abstain"
        else:
            behavior = "Answer"
        rows.append({
            "question_id": r["eval_id"], "question": r["question"], "expected_answer": r["expected_answer"],
            "reference_context": r.get("key_facts", ""), "relevant_document": r.get("relevant_sources", ""),
            "relevant_chunk_ids": "", "question_category": LEGACY_CATEGORY_MAP.get(r["category"], "Complex"),
            "expected_behavior": behavior, "role": r.get("role") or "public",
            "notes": f"imported from evaluation_dataset.csv (legacy category: {r['category']})"
                     + ("; ambiguous - a clarifying question or a sensible answer may both be acceptable"
                        if flag == "either" else ""),
        })
    return normalize(pd.DataFrame(rows))
