"""Authentication (pluggable) and role model.

AUTH_MODE=
  token          - Bearer API tokens (hashed with HMAC-SHA256 in the users table). Default.
  proxy_headers  - the company portal / API gateway authenticates the user (SSO, OIDC, SAML) and forwards
                   X-Portal-User and X-Portal-Roles plus a shared secret header (X-Proxy-Secret).
  disabled       - local development only: every request runs as user 'dev' with DEV_ROLES. Refused in production.

Unauthenticated requests are treated as role 'public' when ALLOW_ANONYMOUS_PUBLIC=true.
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import secrets
from dataclasses import dataclass, field

from backend.db import repository
from config.settings import get_settings

logger = logging.getLogger(__name__)

VALID_ROLES = {"public", "employee", "hr", "admin"}
_DEV_SECRET = "dev-only-insecure-secret-change-me"


class AuthError(Exception):
    def __init__(self, message: str, status_code: int = 401) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass
class AuthenticatedUser:
    username: str
    roles: set[str] = field(default_factory=set)
    display_name: str = ""
    authenticated: bool = True

    @property
    def is_admin(self) -> bool:
        return "admin" in self.roles

    def to_dict(self) -> dict:
        return {"username": self.username, "display_name": self.display_name or self.username,
                "roles": sorted(self.roles), "authenticated": self.authenticated, "is_admin": self.is_admin}


ANONYMOUS = AuthenticatedUser(username="anonymous", roles={"public"}, display_name="Guest", authenticated=False)


def _secret() -> str:
    settings = get_settings()
    if settings.token_hash_secret:
        return settings.token_hash_secret
    if settings.app_env == "production":
        raise RuntimeError("TOKEN_HASH_SECRET must be set in production")
    return _DEV_SECRET


def hash_token(token: str) -> str:
    return hmac.new(_secret().encode(), token.encode(), hashlib.sha256).hexdigest()


def generate_token() -> str:
    return "sap_" + secrets.token_urlsafe(32)


def normalize_roles(roles) -> set[str]:
    if isinstance(roles, str):
        roles = roles.replace(";", ",").split(",")
    return {r.strip().lower() for r in roles if r and r.strip().lower() in VALID_ROLES}


def bootstrap_users() -> list[str]:
    """Create/refresh the bootstrap users whose tokens are provided through environment variables."""
    settings = get_settings()
    created = []
    for username, roles, token in (("admin", ["admin", "employee", "hr"], settings.bootstrap_admin_token),
                                   ("employee", ["employee"], settings.bootstrap_employee_token),
                                   ("hr_user", ["employee", "hr"], settings.bootstrap_hr_token),
                                   ("customer", ["public"], settings.bootstrap_public_token)):
        if token:
            if len(token) < 16:
                logger.warning("Bootstrap token for %s is shorter than 16 characters - use a longer random token", username)
            repository.upsert_user(username, roles, hash_token(token), display_name=username.replace("_", " ").title())
            created.append(username)
    return created


def authenticate(headers: dict[str, str]) -> AuthenticatedUser:
    settings = get_settings()
    h = {k.lower(): v for k, v in headers.items()}
    mode = settings.auth_mode
    if mode == "disabled":
        if settings.app_env == "production":
            raise AuthError("AUTH_MODE=disabled is not allowed in production", 500)
        return AuthenticatedUser(username="dev", roles=normalize_roles(settings.dev_roles) or {"employee"},
                                 display_name="Developer")
    if mode == "proxy_headers":
        if not settings.proxy_shared_secret:
            raise AuthError("PROXY_SHARED_SECRET is not configured", 500)
        if not hmac.compare_digest(h.get("x-proxy-secret", ""), settings.proxy_shared_secret):
            raise AuthError("Invalid proxy credentials")
        username = h.get(settings.proxy_user_header.lower(), "").strip()
        if not username:
            if settings.allow_anonymous_public:
                return ANONYMOUS
            raise AuthError("Missing user identity header")
        roles = normalize_roles(h.get(settings.proxy_roles_header.lower(), "")) or {"employee"}
        return AuthenticatedUser(username=username[:120], roles=roles, display_name=username)
    if mode == "token":
        auth = h.get("authorization", "")
        if not auth:
            if settings.allow_anonymous_public:
                return ANONYMOUS
            raise AuthError("Authentication required")
        if not auth.lower().startswith("bearer "):
            raise AuthError("Use 'Authorization: Bearer <token>'")
        user = repository.get_user_by_token_hash(hash_token(auth[7:].strip()))
        if user is None:
            raise AuthError("Invalid or revoked token")
        return AuthenticatedUser(username=user.username, roles=normalize_roles(user.roles) or {"public"},
                                 display_name=user.display_name)
    raise AuthError(f"Unknown AUTH_MODE '{mode}'", 500)
