"""User/token management CLI (token auth mode).

  python -m backend.security.manage_users create --username priya --roles employee,hr
  python -m backend.security.manage_users list
  python -m backend.security.manage_users deactivate --username priya

The generated token is printed ONCE; only its HMAC hash is stored.
"""
from __future__ import annotations

import argparse
import sys

from backend.db import repository
from backend.security.auth import generate_token, hash_token, normalize_roles


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Manage assistant users")
    sub = parser.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("create")
    c.add_argument("--username", required=True)
    c.add_argument("--roles", default="employee")
    c.add_argument("--display-name", default="")
    sub.add_parser("list")
    d = sub.add_parser("deactivate")
    d.add_argument("--username", required=True)
    args = parser.parse_args(argv)
    if args.cmd == "create":
        roles = normalize_roles(args.roles)
        if not roles:
            print("No valid roles. Use: public, employee, hr, admin")
            return 1
        token = generate_token()
        repository.upsert_user(args.username, sorted(roles), hash_token(token), args.display_name)
        repository.write_audit("cli", "user_created", args.username, details={"roles": sorted(roles)})
        print(f"User '{args.username}' roles={sorted(roles)}\nAPI token (store securely, shown once): {token}")
    elif args.cmd == "list":
        for u in repository.list_users():
            print(u)
    elif args.cmd == "deactivate":
        ok = repository.deactivate_user(args.username)
        repository.write_audit("cli", "user_deactivated", args.username, outcome="success" if ok else "not_found")
        print("Deactivated" if ok else "User not found")
    return 0


if __name__ == "__main__":
    sys.exit(main())
