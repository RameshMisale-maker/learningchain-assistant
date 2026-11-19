"""Review queue, approved answers (golden_qa) and eval cases (REVIEW_QUEUE_ENABLED).

Items are created automatically:
  * thumbs_down    - a user marked an answer "not helpful";
  * refusal        - an in-scope question the assistant could not answer from its sources;
  * low_confidence - an answer whose confidence is below REVIEW_LOW_CONFIDENCE;
  * stale_doc      - a source page changed or disappeared (document tracking).
Similar questions are clustered (same content words), so one item collects every occurrence.

Resolving an item with an approved answer adds it to `golden_qa` and to `eval_cases`, exports the product's approved
answers to `<DATA_DIR>/golden/<kb>_reviewed.csv` and re-ingests that file, so the next user gets the approved answer.
"""
from __future__ import annotations

import csv
import hashlib
import io
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config.settings import get_settings

logger = logging.getLogger(__name__)

AUTO_SOURCES = ("thumbs_down", "refusal", "low_confidence", "stale_doc", "manual")
GOLDEN_COLUMNS = ["Question ID", "Category", "Question", "Approved Answer", "Keywords", "Alternate Questions",
                  "Access Roles", "Status", "Source URL"]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def cluster_key(question: str) -> str:
    """Questions with the same content words (stemmed, order-free) share a cluster."""
    from backend.common.text_utils import STOPWORDS, content_terms

    terms = sorted({t for t in content_terms(question) if t not in STOPWORDS})[:10]
    basis = " ".join(terms) or question.strip().lower()
    return hashlib.sha1(basis.encode("utf-8")).hexdigest()[:16]


def add_item(source: str, question: str, answer: str = "", message_id: int | None = None,
             product_id: str | None = None, details: dict[str, Any] | None = None) -> int:
    """Create a review item, or count one more occurrence on the open item of the same cluster."""
    from sqlalchemy import select

    from backend.db.models import ReviewItem
    from backend.db.session import db_session
    from backend.products import current_product

    product_id = product_id or current_product().product_id
    key = cluster_key(question) if question else hashlib.sha1(str(details).encode()).hexdigest()[:16]
    with db_session() as s:
        item = s.scalars(select(ReviewItem).where(ReviewItem.product_id == product_id, ReviewItem.status == "open",
                                                  ReviewItem.cluster_key == key, ReviewItem.source == source)).first()
        if item is not None:
            item.occurrences += 1
            info = dict(item.details or {})
            info["message_ids"] = (info.get("message_ids", []) + [message_id])[-50:] if message_id else info.get("message_ids", [])
            info.setdefault("variants", [])
            if question and question not in info["variants"] and question != item.question:
                info["variants"] = (info["variants"] + [question])[-20:]
            item.details = info
            return item.id
        item = ReviewItem(product_id=product_id, source=source, question=question[:4000], answer=answer[:8000],
                          message_id=message_id, cluster_key=key,
                          details=(details or {}) | ({"message_ids": [message_id]} if message_id else {}))
        s.add(item)
        s.flush()
        return item.id


def auto_review(resp, question: str) -> int | None:
    """After an answer: queue refusals and low-confidence answers."""
    settings = get_settings()
    if resp.answer_type == "refusal":
        source = "refusal"
    elif resp.answer_type in ("generated", "extractive", "extractive_fallback") and \
            resp.confidence < settings.review_low_confidence:
        source = "low_confidence"
    else:
        return None
    return add_item(source, question, resp.answer, resp.message_id,
                    details={"confidence": resp.confidence, "answer_type": resp.answer_type,
                             "knowledge_bases": list(resp.knowledge_bases)})


def on_thumbs_down(message_id: int, comment: str = "") -> int:
    from sqlalchemy import select

    from backend.db.models import ChatMessage
    from backend.db.session import db_session

    with db_session() as s:
        msg = s.get(ChatMessage, message_id)
        question, answer, product_id, kbs = "", "", None, ""
        if msg is not None:
            answer, product_id, kbs = msg.content, msg.product_id or None, msg.knowledge_bases
            prev = s.scalars(select(ChatMessage).where(ChatMessage.session_id == msg.session_id,
                                                       ChatMessage.id < msg.id, ChatMessage.role == "user")
                             .order_by(ChatMessage.id.desc())).first()
            question = prev.content if prev else ""
    return add_item("thumbs_down", question, answer, message_id, product_id,
                    details={"comment": comment[:1000], "knowledge_bases": kbs})


def list_items(status: str | None = "open", source: str | None = None, product_id: str | None = None,
               limit: int = 200) -> list[dict[str, Any]]:
    from sqlalchemy import select

    from backend.db.models import ReviewItem
    from backend.db.session import db_session

    with db_session() as s:
        stmt = select(ReviewItem).order_by(ReviewItem.occurrences.desc(), ReviewItem.id.desc()).limit(limit)
        if status:
            stmt = stmt.where(ReviewItem.status == status)
        if source:
            stmt = stmt.where(ReviewItem.source == source)
        if product_id:
            stmt = stmt.where(ReviewItem.product_id == product_id)
        return [{"id": i.id, "product_id": i.product_id, "created_at": i.created_at.isoformat() if i.created_at else None,
                 "source": i.source, "status": i.status, "question": i.question, "answer": i.answer,
                 "message_id": i.message_id, "cluster_key": i.cluster_key, "occurrences": i.occurrences,
                 "details": i.details, "resolved_by": i.resolved_by, "resolution": i.resolution}
                for i in s.scalars(stmt)]


def resolve(item_id: int, by: str, action: str, answer: str = "", question: str = "", kb: str = "",
            add_eval: bool = True, note: str = "") -> dict[str, Any]:
    """action: "approve" (answer becomes an approved answer + eval case) or "dismiss"."""
    from backend.db.models import ReviewItem
    from backend.db.repository import write_audit
    from backend.db.session import db_session
    from backend.products import get_product, product_for_kb

    if action not in ("approve", "dismiss"):
        raise ValueError("action must be 'approve' or 'dismiss'")
    with db_session() as s:
        item = s.get(ReviewItem, item_id)
        if item is None:
            raise ValueError(f"Unknown review item {item_id}")
        if item.status != "open":
            raise ValueError(f"Review item {item_id} is already {item.status}")
        product_id = item.product_id
        question = (question or item.question).strip()
        variants = list((item.details or {}).get("variants", []))
        item.status = "resolved" if action == "approve" else "dismissed"
        item.resolved_by, item.resolved_at, item.resolution = by, _utcnow(), (answer if action == "approve" else note)
    out: dict[str, Any] = {"id": item_id, "status": "resolved" if action == "approve" else "dismissed"}
    if action == "approve":
        if not question or not answer.strip():
            raise ValueError("approving needs a question and an approved answer")
        kb = kb or get_product(product_id or None).kb
        product = product_for_kb(kb)             # approved answers belong to the product that owns the KB
        qid = f"RV{item_id:05d}"
        upsert_golden(product.product_id, kb, qid, question, answer.strip(), alternates=variants, origin="review")
        if add_eval:
            add_eval_case(product.product_id, kb, question, answer.strip(), [qid])
        out |= {"qid": qid, "kb": kb, "ingest": publish_golden(product.product_id, kb)}
    write_audit(by, f"review_{action}", str(item_id), details={"product_id": product_id})
    return out


# ---------------------------------------------------------------- approved answers (golden_qa)
def upsert_golden(product_id: str, kb: str, qid: str, question: str, answer: str, alternates: list[str] | None = None,
                  category: str = "Reviewed", keywords: str = "", source_url: str = "", status: str = "active",
                  origin: str = "csv") -> None:
    from sqlalchemy import select

    from backend.db.models import GoldenQA
    from backend.db.session import db_session

    with db_session() as s:
        row = s.scalars(select(GoldenQA).where(GoldenQA.product_id == product_id, GoldenQA.qid == qid)).first()
        if row is None:
            row = GoldenQA(product_id=product_id, qid=qid)
            s.add(row)
        row.kb, row.question, row.answer, row.alternates = kb, question, answer, list(alternates or [])
        row.category, row.keywords, row.source_url, row.status, row.origin = category, keywords, source_url, status, origin


def import_golden_csv(product_id: str, kb: str, content: str, publish: bool = True) -> dict[str, Any]:
    """Import approved answers from CSV (columns as the golden datasets: Question ID, Question, Approved Answer,
    Alternate Questions (|-separated), Category, Keywords, Status, Source URL)."""
    reader = csv.DictReader(io.StringIO(content.lstrip("﻿")))
    imported, skipped = 0, []
    for idx, row in enumerate(reader, start=2):
        question, answer = (row.get("Question") or "").strip(), (row.get("Approved Answer") or "").strip()
        if not question or not answer:
            skipped.append(f"line {idx}: missing Question or Approved Answer")
            continue
        qid = (row.get("Question ID") or "").strip() or f"CSV{idx:05d}"
        alternates = [a.strip() for a in (row.get("Alternate Questions") or "").split("|") if a.strip()]
        upsert_golden(product_id, kb, qid, question, answer, alternates, category=(row.get("Category") or "").strip(),
                      keywords=(row.get("Keywords") or "").strip(), source_url=(row.get("Source URL") or "").strip(),
                      status=(row.get("Status") or "active").strip().lower() or "active", origin="csv")
        imported += 1
    out: dict[str, Any] = {"imported": imported, "skipped": skipped}
    if publish and imported:
        out["ingest"] = publish_golden(product_id, kb)
    return out


def list_golden(product_id: str | None = None) -> list[dict[str, Any]]:
    from sqlalchemy import select

    from backend.db.models import GoldenQA
    from backend.db.session import db_session

    with db_session() as s:
        stmt = select(GoldenQA).order_by(GoldenQA.id)
        if product_id:
            stmt = stmt.where(GoldenQA.product_id == product_id)
        return [{"id": g.id, "product_id": g.product_id, "kb": g.kb, "qid": g.qid, "question": g.question,
                 "answer": g.answer, "alternates": g.alternates, "status": g.status, "origin": g.origin}
                for g in s.scalars(stmt)]


def reviewed_golden_path(kb: str) -> Path:
    """The KB's database-managed approved answers (picked up by IngestionPipeline.discover)."""
    return get_settings().data_dir / "golden" / f"{kb}_reviewed.csv"


def publish_golden(product_id: str, kb: str) -> dict[str, Any]:
    """Write the KB's approved answers from golden_qa to its reviewed golden file and re-ingest it."""
    from sqlalchemy import select

    from backend.db.models import GoldenQA
    from backend.db.session import db_session
    from ingestion.pipeline import IngestionPipeline

    with db_session() as s:
        rows = list(s.scalars(select(GoldenQA).where(GoldenQA.kb == kb).order_by(GoldenQA.id)))
        data = [{"Question ID": g.qid, "Category": g.category, "Question": g.question, "Approved Answer": g.answer,
                 "Keywords": g.keywords, "Alternate Questions": " | ".join(g.alternates or []), "Access Roles": "",
                 "Status": "active" if g.status == "active" else "inactive", "Source URL": g.source_url}
                for g in rows]
    path = reviewed_golden_path(kb)
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=GOLDEN_COLUMNS)
    writer.writeheader()
    writer.writerows(data)
    path.write_text(buf.getvalue(), encoding="utf-8")
    result = IngestionPipeline().ingest_file(kb, path, uploaded_by=f"review:{product_id}", golden=True)
    return {"file": str(path), "rows": len(data), "status": result.status, "message": result.message}


# ---------------------------------------------------------------- eval cases
def add_eval_case(product_id: str, kb: str, question: str, expected_answer: str = "",
                  expected_sources: list[str] | None = None, origin: str = "review") -> int:
    from backend.db.models import EvalCase
    from backend.db.session import db_session

    with db_session() as s:
        case = EvalCase(product_id=product_id, kb=kb, question=question, expected_answer=expected_answer,
                        expected_sources=list(expected_sources or []), origin=origin)
        s.add(case)
        s.flush()
        return case.id


def list_eval_cases(product_id: str | None = None, kb: str | None = None, active_only: bool = True) -> list[dict]:
    from sqlalchemy import select

    from backend.db.models import EvalCase
    from backend.db.session import db_session

    with db_session() as s:
        stmt = select(EvalCase).order_by(EvalCase.id)
        if product_id:
            stmt = stmt.where(EvalCase.product_id == product_id)
        if kb:
            stmt = stmt.where(EvalCase.kb == kb)
        if active_only:
            stmt = stmt.where(EvalCase.active.is_(True))
        return [{"id": c.id, "product_id": c.product_id, "kb": c.kb, "question": c.question,
                 "expected_answer": c.expected_answer, "expected_sources": c.expected_sources, "origin": c.origin}
                for c in s.scalars(stmt)]
