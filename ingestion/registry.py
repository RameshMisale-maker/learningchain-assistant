"""Document registry: versions, content hashes (duplicate detection), ACLs, status."""
from __future__ import annotations

from typing import Any

from sqlalchemy import select

from backend.db.models import Document, utcnow
from backend.db.session import db_session


def _to_dict(doc: Document) -> dict[str, Any]:
    return {
        "doc_id": doc.doc_id, "kb": doc.kb, "doc_name": doc.doc_name, "source_path": doc.source_path,
        "file_type": doc.file_type, "doc_type": doc.doc_type, "category": doc.category,
        "access_roles": [r for r in doc.access_roles.split(",") if r], "confidentiality": doc.confidentiality,
        "content_hash": doc.content_hash, "version": doc.version, "num_chunks": doc.num_chunks,
        "status": doc.status, "is_sample": doc.is_sample, "uploaded_by": doc.uploaded_by,
        "uploaded_at": doc.uploaded_at.isoformat() if doc.uploaded_at else None,
        "updated_at": doc.updated_at.isoformat() if doc.updated_at else None, "notes": doc.notes,
    }


def get_document(doc_id: str) -> dict[str, Any] | None:
    with db_session() as s:
        doc = s.get(Document, doc_id)
        return _to_dict(doc) if doc else None


def find_by_hash(kb: str, content_hash: str) -> dict[str, Any] | None:
    with db_session() as s:
        doc = s.scalar(select(Document).where(Document.kb == kb, Document.content_hash == content_hash,
                                              Document.status == "active"))
        return _to_dict(doc) if doc else None


def list_documents(kb: str | None = None, include_deleted: bool = False) -> list[dict[str, Any]]:
    with db_session() as s:
        stmt = select(Document).order_by(Document.kb, Document.doc_name)
        if kb:
            stmt = stmt.where(Document.kb == kb)
        if not include_deleted:
            stmt = stmt.where(Document.status == "active")
        return [_to_dict(d) for d in s.scalars(stmt)]


def upsert_document(**fields: Any) -> dict[str, Any]:
    """Insert a new document or bump the version of an existing one."""
    with db_session() as s:
        doc = s.get(Document, fields["doc_id"])
        if doc is None:
            doc = Document(**{**fields, "version": 1, "status": "active"})
            s.add(doc)
        else:
            for key, value in fields.items():
                setattr(doc, key, value)
            doc.version = (doc.version or 0) + 1
            doc.status = "active"
            doc.updated_at = utcnow()
        s.flush()
        return _to_dict(doc)


def mark_deleted(doc_id: str) -> bool:
    with db_session() as s:
        doc = s.get(Document, doc_id)
        if not doc:
            return False
        doc.status = "deleted"
        doc.num_chunks = 0
        doc.updated_at = utcnow()
        return True


def purge_kb(kb: str) -> int:
    with db_session() as s:
        docs = list(s.scalars(select(Document).where(Document.kb == kb)))
        for d in docs:
            s.delete(d)
        return len(docs)
