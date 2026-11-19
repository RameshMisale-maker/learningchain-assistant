"""Data-access helpers for chat history, feedback, audit logs, KB revisions and users."""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import delete, func, select

from backend.db.models import AuditLog, ChatMessage, ChatSession, Feedback, KBState, User
from backend.db.session import db_session


# ---------------------------------------------------------------- audit
def write_audit(username: str, action: str, resource: str = "", outcome: str = "success",
                details: dict[str, Any] | None = None, ip_address: str = "") -> None:
    with db_session() as s:
        s.add(AuditLog(username=username or "anonymous", action=action, resource=resource[:300], outcome=outcome,
                       details=details or {}, ip_address=ip_address or ""))


def list_audit(limit: int = 200, action: str | None = None) -> list[dict[str, Any]]:
    with db_session() as s:
        stmt = select(AuditLog).order_by(AuditLog.id.desc()).limit(limit)
        if action:
            stmt = stmt.where(AuditLog.action == action)
        return [{"id": a.id, "timestamp": a.timestamp.isoformat(), "username": a.username, "action": a.action,
                 "resource": a.resource, "outcome": a.outcome, "details": a.details, "ip_address": a.ip_address}
                for a in s.scalars(stmt)]


# ---------------------------------------------------------------- KB revision
def get_kb_revision(kb: str) -> int:
    with db_session() as s:
        state = s.get(KBState, kb)
        return state.revision if state else 0


def bump_kb_revision(kb: str, embedding_model: str = "") -> int:
    with db_session() as s:
        state = s.get(KBState, kb)
        if state is None:
            state = KBState(kb=kb, revision=0, embedding_model=embedding_model)
            s.add(state)
        state.revision += 1
        if embedding_model:
            state.embedding_model = embedding_model
        s.flush()
        return state.revision


def get_kb_embedding_model(kb: str) -> str:
    with db_session() as s:
        state = s.get(KBState, kb)
        return state.embedding_model if state else ""


# ---------------------------------------------------------------- chat
def _product_id() -> str:
    """Product of the request being handled (rows are keyed by product - config/products/<id>.yaml)."""
    try:
        from backend.products import current_product
        return current_product().product_id
    except Exception:          # product config unavailable (e.g. a bare maintenance script): leave it empty
        return ""


def ensure_session(session_id: str | None, username: str) -> str:
    with db_session() as s:
        if session_id:
            existing = s.get(ChatSession, session_id)
            if existing and existing.username == username and not existing.cleared:
                return existing.id
        new_id = uuid.uuid4().hex
        s.add(ChatSession(id=new_id, username=username, product_id=_product_id()))
        return new_id


CONVERSATION_STATES = ("active", "closing", "closed")


def get_conversation_state(session_id: str) -> str:
    with db_session() as s:
        session = s.get(ChatSession, session_id)
        return (session.conversation_state or "active") if session else "active"


def set_conversation_state(session_id: str, state: str) -> None:
    if state not in CONVERSATION_STATES:
        raise ValueError(f"conversation_state must be one of {CONVERSATION_STATES}")
    with db_session() as s:
        session = s.get(ChatSession, session_id)
        if session is not None:
            session.conversation_state = state


def session_owner(session_id: str) -> str | None:
    with db_session() as s:
        session = s.get(ChatSession, session_id)
        return session.username if session else None


def message_session(message_id: int) -> str | None:
    with db_session() as s:
        msg = s.get(ChatMessage, message_id)
        return msg.session_id if msg else None


def session_product(session_id: str) -> str | None:
    if not session_id:
        return None
    with db_session() as s:
        session = s.get(ChatSession, session_id)
        return session.product_id if session else None


def last_assistant_message_id(session_id: str) -> int | None:
    with db_session() as s:
        return s.scalar(select(ChatMessage.id).where(ChatMessage.session_id == session_id,
                                                     ChatMessage.role == "assistant")
                        .order_by(ChatMessage.id.desc()).limit(1))


def add_message(session_id: str, role: str, content: str, **fields: Any) -> int:
    fields.setdefault("product_id", _product_id())
    with db_session() as s:
        msg = ChatMessage(session_id=session_id, role=role, content=content, **fields)
        s.add(msg)
        s.flush()
        return msg.id


def get_history(session_id: str, username: str, limit: int = 20) -> list[dict[str, Any]]:
    with db_session() as s:
        sess = s.get(ChatSession, session_id)
        if not sess or sess.username != username or sess.cleared:
            return []
        stmt = (select(ChatMessage).where(ChatMessage.session_id == session_id)
                .order_by(ChatMessage.id.desc()).limit(limit))
        rows = list(s.scalars(stmt))[::-1]
        return [{"id": m.id, "role": m.role, "content": m.content, "knowledge_bases": m.knowledge_bases,
                 "citations": m.citations, "answer_type": m.answer_type, "created_at": m.created_at.isoformat()}
                for m in rows]


def clear_session(session_id: str, username: str) -> bool:
    with db_session() as s:
        sess = s.get(ChatSession, session_id)
        if not sess or sess.username != username:
            return False
        s.execute(delete(ChatMessage).where(ChatMessage.session_id == session_id))
        sess.cleared = True
        return True


def message_owner(message_id: int) -> str | None:
    with db_session() as s:
        msg = s.get(ChatMessage, message_id)
        if not msg:
            return None
        return msg.session.username


def purge_old_history(retention_days: int) -> int:
    """Delete chat messages older than the retention window. Returns deleted message count."""
    if retention_days <= 0:
        return 0
    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
    with db_session() as s:
        result = s.execute(delete(ChatMessage).where(ChatMessage.created_at < cutoff))
        s.execute(delete(ChatSession).where(ChatSession.created_at < cutoff,
                                            ~ChatSession.messages.any()))
        return result.rowcount or 0


# ---------------------------------------------------------------- feedback
def add_feedback(message_id: int, username: str, rating: str, comment: str = "", session_id: str = "",
                 product_id: str = "", stars: int | None = None, details: dict[str, Any] | None = None) -> int:
    with db_session() as s:
        fb = Feedback(message_id=message_id, username=username, rating=rating, comment=comment[:2000],
                      session_id=session_id or "", product_id=product_id or "", stars=stars, details=details or {})
        s.add(fb)
        s.flush()
        return fb.id


def list_feedback(limit: int = 200) -> list[dict[str, Any]]:
    with db_session() as s:
        stmt = (select(Feedback, ChatMessage).join(ChatMessage, Feedback.message_id == ChatMessage.id)
                .order_by(Feedback.id.desc()).limit(limit))
        out = []
        for fb, msg in s.execute(stmt):
            out.append({"id": fb.id, "message_id": fb.message_id, "username": fb.username, "rating": fb.rating,
                        "comment": fb.comment, "created_at": fb.created_at.isoformat(),
                        "answer": msg.content[:500], "knowledge_bases": msg.knowledge_bases})
        return out


def feedback_summary() -> dict[str, int]:
    with db_session() as s:
        rows = s.execute(select(Feedback.rating, func.count()).group_by(Feedback.rating)).all()
        return {rating: count for rating, count in rows}


# ---------------------------------------------------------------- users
def get_user_by_token_hash(token_hash: str) -> User | None:
    with db_session() as s:
        return s.scalar(select(User).where(User.token_hash == token_hash, User.is_active.is_(True)))


def upsert_user(username: str, roles: list[str], token_hash: str | None, display_name: str = "") -> User:
    with db_session() as s:
        user = s.scalar(select(User).where(User.username == username))
        if user is None:
            user = User(username=username)
            s.add(user)
        user.roles = ",".join(sorted(set(roles)))
        user.display_name = display_name or username
        if token_hash:
            user.token_hash = token_hash
        user.is_active = True
        s.flush()
        return user


def list_users() -> list[dict[str, Any]]:
    with db_session() as s:
        return [{"username": u.username, "display_name": u.display_name, "roles": u.role_list,
                 "is_active": u.is_active} for u in s.scalars(select(User).order_by(User.username))]


def deactivate_user(username: str) -> bool:
    with db_session() as s:
        user = s.scalar(select(User).where(User.username == username))
        if not user:
            return False
        user.is_active = False
        user.token_hash = None
        return True


def write_request_metric(**fields: Any) -> int:
    from backend.db.models import RequestMetric

    fields.setdefault("product_id", _product_id())
    with db_session() as s:
        row = RequestMetric(**fields)
        s.add(row)
        s.flush()
        return row.id
