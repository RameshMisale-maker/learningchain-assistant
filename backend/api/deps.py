"""FastAPI dependencies: authentication, admin guard, rate limiting."""
from __future__ import annotations

from fastapi import Depends, HTTPException, Request

from backend.db import repository
from backend.security.auth import AuthenticatedUser, AuthError, authenticate
from backend.security.rate_limit import RateLimiter
from config.settings import get_settings

_limiter: RateLimiter | None = None


def client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")
    return forwarded.split(",")[0].strip() if forwarded else (request.client.host if request.client else "")


def current_user(request: Request) -> AuthenticatedUser:
    try:
        return authenticate(dict(request.headers))
    except AuthError as exc:
        repository.write_audit("anonymous", "auth_failed", request.url.path, outcome="denied",
                               details={"reason": str(exc)}, ip_address=client_ip(request))
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


def admin_user(request: Request, user: AuthenticatedUser = Depends(current_user)) -> AuthenticatedUser:
    if not user.is_admin:
        repository.write_audit(user.username, "admin_access_denied", request.url.path, outcome="denied",
                               ip_address=client_ip(request))
        raise HTTPException(status_code=403, detail="Administrator role required")
    return user


def rate_limited(request: Request, user: AuthenticatedUser = Depends(current_user)) -> AuthenticatedUser:
    global _limiter
    if _limiter is None:
        _limiter = RateLimiter(get_settings().rate_limit_per_minute)
    key = user.username if user.authenticated else f"anon:{client_ip(request)}"
    if not _limiter.allow(key):
        raise HTTPException(status_code=429, detail="Too many requests - please wait a moment")
    return user
