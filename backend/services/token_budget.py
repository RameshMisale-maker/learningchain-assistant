"""Token budget per user and per product (TOKEN_BUDGET_ENABLED).

The provider quota is shared (Groq: tokens per day, per model), and RATE_LIMIT_PER_MINUTE only counts requests -
30 short questions and 30 long ones cost very differently. Every request's AI tokens are already stored in
request_metrics with the username and product, so the budget is the sum over a rolling window
(TOKEN_BUDGET_WINDOW_HOURS, default 24 h - the same kind of window the provider uses):

  * user budget    TOKEN_BUDGET_USER_TOKENS     (all products together)
  * product budget TOKEN_BUDGET_PRODUCT_TOKENS  (or the product's entry in TOKEN_BUDGET_PRODUCT_LIMITS)

Over budget, the request runs WITHOUT the AI model (`llm_suspended()`): greetings, approved answers and cached answers
still work at no token cost; a question that would need the model gets the "limit reached" reply instead of using
the shared quota. Usernames in TOKEN_BUDGET_EXEMPT_USERS (e.g. the evaluation service account) are never limited.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterator

from config.settings import get_settings

logger = logging.getLogger(__name__)

# True while the current request must not call the AI model (set by ChatService.answer for over-budget requests)
_LLM_SUSPENDED: ContextVar[bool] = ContextVar("llm_suspended", default=False)
BUDGET_ANSWER_TYPE = "budget_exceeded"


@dataclass
class BudgetCheck:
    allowed: bool = True
    scope: str = ""            # "" | "user" | "product"
    used: int = 0
    limit: int = 0
    window_hours: float = 24.0

    def to_dict(self) -> dict:
        return {"allowed": self.allowed, "scope": self.scope, "used": self.used, "limit": self.limit,
                "window_hours": self.window_hours}


def llm_suspended() -> bool:
    return _LLM_SUSPENDED.get()


@contextmanager
def suspend_llm(active: bool) -> Iterator[None]:
    """Within the block, `llm_suspended()` is `active` (per request / thread - contextvars)."""
    token = _LLM_SUSPENDED.set(bool(active))
    try:
        yield
    finally:
        _LLM_SUSPENDED.reset(token)


def product_limit(product_id: str) -> int:
    s = get_settings()
    for part in (s.token_budget_product_limits or "").split(","):
        name, _, value = part.partition("=")
        if name.strip() == product_id and value.strip().isdigit():
            return int(value.strip())
    return s.token_budget_product_tokens


def tokens_used(username: str | None = None, product_id: str | None = None, hours: float | None = None) -> int:
    """AI tokens (input + output) recorded in the rolling window, for one user and/or one product."""
    from sqlalchemy import func, select

    from backend.db.models import RequestMetric
    from backend.db.session import db_session

    hours = get_settings().token_budget_window_hours if hours is None else hours
    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    q = select(func.coalesce(func.sum(RequestMetric.input_tokens + RequestMetric.output_tokens), 0)) \
        .where(RequestMetric.created_at >= since)
    if username is not None:
        q = q.where(RequestMetric.username == username)
    if product_id is not None:
        q = q.where(RequestMetric.product_id == product_id)
    with db_session() as s:
        return int(s.scalar(q) or 0)


def check(username: str, product_id: str) -> BudgetCheck:
    """Is this user (and their product) still within budget? Never raises: a failed lookup allows the request."""
    s = get_settings()
    window = s.token_budget_window_hours
    if not s.token_budget_enabled or username in s.token_budget_exempt_users:
        return BudgetCheck(window_hours=window)
    try:
        if s.token_budget_user_tokens > 0:
            used = tokens_used(username=username, hours=window)
            if used >= s.token_budget_user_tokens:
                return BudgetCheck(False, "user", used, s.token_budget_user_tokens, window)
        limit = product_limit(product_id)
        if limit > 0:
            used = tokens_used(product_id=product_id, hours=window)
            if used >= limit:
                return BudgetCheck(False, "product", used, limit, window)
    except Exception:          # the budget must never break a user request
        logger.exception("Token budget lookup failed - request allowed")
    return BudgetCheck(window_hours=window)


def budget_reply(check_: BudgetCheck, support: str) -> str:
    hours = int(check_.window_hours) if float(check_.window_hours).is_integer() else check_.window_hours
    who = "your account has" if check_.scope == "user" else "this assistant has"
    return (f"Sorry, {who} reached its limit for detailed answers for now (the limit resets over {hours} hours). "
            "I can still answer common questions from our approved answers - try asking in a different way, or "
            f"please try again later. For anything urgent, contact our support team: {support}.")
