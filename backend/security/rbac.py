"""Role-based access control helpers (knowledge-base level). Document-level ACL is enforced in retrieval."""
from __future__ import annotations

from backend.security.auth import AuthenticatedUser
from config.settings import get_settings


def kb_allowed_roles(kb: str) -> set[str]:
    cfg = get_settings().kb_config()["knowledge_bases"].get(kb, {})
    return set(cfg.get("allowed_roles", []))


def can_query_kb(user: AuthenticatedUser, kb: str) -> bool:
    return user.is_admin or bool(user.roles & kb_allowed_roles(kb))


def accessible_kbs(user: AuthenticatedUser) -> list[str]:
    cfgs = get_settings().kb_config()["knowledge_bases"]
    return [kb for kb, cfg in cfgs.items() if cfg.get("enabled", True) and can_query_kb(user, kb)]


def can_manage_documents(user: AuthenticatedUser) -> bool:
    return user.is_admin


def suggested_questions(user: AuthenticatedUser) -> list[str]:
    items = get_settings().kb_config().get("suggested_questions", [])
    return [i["question"] for i in items if user.is_admin or user.roles & set(i.get("roles", []))]
