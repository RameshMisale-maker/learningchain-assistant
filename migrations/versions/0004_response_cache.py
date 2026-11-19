"""response cache (exact + semantic answers keyed by index / prompt version)

Revision ID: 0004
Revises: 0003
Create Date: 2026-11-19
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None

_INDEXED = ("product_id", "scope_key", "source_message_id", "last_message_id", "expires_at")


def upgrade() -> None:
    op.create_table(
        "response_cache",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("product_id", sa.String(length=64), nullable=False, server_default=""),
        sa.Column("cache_key", sa.String(length=64), nullable=False),
        sa.Column("scope_key", sa.String(length=64), nullable=False),
        sa.Column("normalized_query", sa.Text(), nullable=False, server_default=""),
        sa.Column("embedding", sa.JSON(), nullable=True),
        sa.Column("index_version", sa.String(length=200), nullable=False, server_default=""),
        sa.Column("prompt_version", sa.String(length=64), nullable=False, server_default=""),
        sa.Column("response", sa.JSON(), nullable=True),
        sa.Column("hits", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("source_message_id", sa.Integer(), nullable=True),
        sa.Column("last_message_id", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_hit_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_response_cache_cache_key", "response_cache", ["cache_key"], unique=True)
    for col in _INDEXED:
        op.create_index(f"ix_response_cache_{col}", "response_cache", [col])


def downgrade() -> None:
    for col in (*_INDEXED, "cache_key"):
        op.drop_index(f"ix_response_cache_{col}", table_name="response_cache")
    op.drop_table("response_cache")
