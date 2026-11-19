"""index versions + aliases; full request logging columns on request_metrics

Revision ID: 0003
Revises: 0002
Create Date: 2026-11-19
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None

_METRIC_COLUMNS = [
    sa.Column("index_version", sa.String(length=200), nullable=False, server_default=""),
    sa.Column("prompt_version", sa.String(length=64), nullable=False, server_default=""),
    sa.Column("provider", sa.String(length=64), nullable=False, server_default=""),
    sa.Column("intent", sa.String(length=32), nullable=False, server_default=""),
    sa.Column("cache_hit", sa.Boolean(), nullable=False, server_default=sa.false()),
    sa.Column("golden_hit", sa.Boolean(), nullable=False, server_default=sa.false()),
    sa.Column("refused", sa.Boolean(), nullable=False, server_default=sa.false()),
    sa.Column("retrieved", sa.JSON(), nullable=True),
    sa.Column("message_id", sa.Integer(), nullable=True),
]


def upgrade() -> None:
    op.create_table(
        "index_versions",
        sa.Column("version_id", sa.Integer(), primary_key=True),
        sa.Column("product_id", sa.String(length=64), nullable=False, server_default=""),
        sa.Column("kb", sa.String(length=64), nullable=False),
        sa.Column("collection", sa.String(length=200), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("crawl_id", sa.String(length=64), nullable=False, server_default=""),
        sa.Column("embedding_model", sa.String(length=200), nullable=False, server_default=""),
        sa.Column("chunk_config", sa.JSON(), nullable=True),
        sa.Column("doc_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("chunk_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("eval_score", sa.Float(), nullable=True),
        sa.Column("eval_details", sa.JSON(), nullable=True),
        sa.Column("status", sa.String(length=16), nullable=False, server_default="building"),
        sa.Column("notes", sa.Text(), nullable=False, server_default=""),
    )
    op.create_index("ix_index_versions_product_id", "index_versions", ["product_id"])
    op.create_index("ix_index_versions_kb", "index_versions", ["kb"])
    op.create_index("ix_index_versions_status", "index_versions", ["status"])
    op.create_table(
        "index_aliases",
        sa.Column("kb", sa.String(length=64), primary_key=True),
        sa.Column("product_id", sa.String(length=64), nullable=False, server_default=""),
        sa.Column("live_version_id", sa.Integer(), nullable=True),
        sa.Column("previous_version_id", sa.Integer(), nullable=True),
        sa.Column("switched_at", sa.DateTime(timezone=True), nullable=True),
    )
    with op.batch_alter_table("request_metrics", schema=None) as batch_op:
        for column in _METRIC_COLUMNS:
            batch_op.add_column(column.copy())
        batch_op.create_index("ix_request_metrics_intent", ["intent"])
        batch_op.create_index("ix_request_metrics_refused", ["refused"])
        batch_op.create_index("ix_request_metrics_message_id", ["message_id"])


def downgrade() -> None:
    with op.batch_alter_table("request_metrics", schema=None) as batch_op:
        batch_op.drop_index("ix_request_metrics_message_id")
        batch_op.drop_index("ix_request_metrics_refused")
        batch_op.drop_index("ix_request_metrics_intent")
        for column in reversed(_METRIC_COLUMNS):
            batch_op.drop_column(column.name)
    op.drop_table("index_aliases")
    op.drop_index("ix_index_versions_status", table_name="index_versions")
    op.drop_index("ix_index_versions_kb", table_name="index_versions")
    op.drop_index("ix_index_versions_product_id", table_name="index_versions")
    op.drop_table("index_versions")
