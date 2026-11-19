"""document freshness tracking (source_url, last seen / changed, missed crawls, tombstone)

Revision ID: 0006
Revises: 0005
Create Date: 2026-11-19
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None

_COLUMNS = ("last_crawl_id", "tombstoned_at", "missed_crawls", "last_changed_at", "last_seen_at", "source_url")


def upgrade() -> None:
    with op.batch_alter_table("documents", schema=None) as batch_op:
        batch_op.add_column(sa.Column("source_url", sa.String(length=600), nullable=False, server_default=""))
        batch_op.add_column(sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=True))
        batch_op.add_column(sa.Column("last_changed_at", sa.DateTime(timezone=True), nullable=True))
        batch_op.add_column(sa.Column("missed_crawls", sa.Integer(), nullable=False, server_default="0"))
        batch_op.add_column(sa.Column("tombstoned_at", sa.DateTime(timezone=True), nullable=True))
        batch_op.add_column(sa.Column("last_crawl_id", sa.String(length=64), nullable=False, server_default=""))
        batch_op.create_index("ix_documents_source_url", ["source_url"])
    # backfill: every existing document counts as seen / changed at its last update
    op.execute("UPDATE documents SET last_seen_at = updated_at, last_changed_at = updated_at")


def downgrade() -> None:
    with op.batch_alter_table("documents", schema=None) as batch_op:
        batch_op.drop_index("ix_documents_source_url")
        for col in _COLUMNS:
            batch_op.drop_column(col)
