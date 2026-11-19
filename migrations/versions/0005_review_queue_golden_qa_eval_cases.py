"""review queue, golden_qa, eval_cases

Revision ID: 0005
Revises: 0004
Create Date: 2026-11-19
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None

_REVIEW_INDEXED = ("product_id", "created_at", "source", "status", "message_id", "cluster_key")


def upgrade() -> None:
    op.create_table(
        "review_items",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("product_id", sa.String(length=64), nullable=False, server_default=""),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("source", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False, server_default="open"),
        sa.Column("question", sa.Text(), nullable=False, server_default=""),
        sa.Column("answer", sa.Text(), nullable=False, server_default=""),
        sa.Column("message_id", sa.Integer(), nullable=True),
        sa.Column("cluster_key", sa.String(length=64), nullable=False, server_default=""),
        sa.Column("occurrences", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("details", sa.JSON(), nullable=True),
        sa.Column("resolved_by", sa.String(length=120), nullable=False, server_default=""),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("resolution", sa.Text(), nullable=False, server_default=""),
    )
    for col in _REVIEW_INDEXED:
        op.create_index(f"ix_review_items_{col}", "review_items", [col])
    op.create_table(
        "golden_qa",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("product_id", sa.String(length=64), nullable=False, server_default=""),
        sa.Column("kb", sa.String(length=64), nullable=False, server_default=""),
        sa.Column("qid", sa.String(length=64), nullable=False),
        sa.Column("category", sa.String(length=120), nullable=False, server_default=""),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("answer", sa.Text(), nullable=False),
        sa.Column("alternates", sa.JSON(), nullable=True),
        sa.Column("keywords", sa.String(length=500), nullable=False, server_default=""),
        sa.Column("source_url", sa.String(length=600), nullable=False, server_default=""),
        sa.Column("status", sa.String(length=16), nullable=False, server_default="active"),
        sa.Column("origin", sa.String(length=32), nullable=False, server_default="csv"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("product_id", "qid", name="uq_golden_product_qid"),
    )
    op.create_index("ix_golden_qa_product_id", "golden_qa", ["product_id"])
    op.create_index("ix_golden_qa_status", "golden_qa", ["status"])
    op.create_table(
        "eval_cases",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("product_id", sa.String(length=64), nullable=False, server_default=""),
        sa.Column("kb", sa.String(length=64), nullable=False, server_default=""),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("expected_answer", sa.Text(), nullable=False, server_default=""),
        sa.Column("expected_sources", sa.JSON(), nullable=True),
        sa.Column("origin", sa.String(length=32), nullable=False, server_default="review"),
        sa.Column("active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_eval_cases_product_id", "eval_cases", ["product_id"])
    op.create_index("ix_eval_cases_active", "eval_cases", ["active"])


def downgrade() -> None:
    op.drop_index("ix_eval_cases_active", table_name="eval_cases")
    op.drop_index("ix_eval_cases_product_id", table_name="eval_cases")
    op.drop_table("eval_cases")
    op.drop_index("ix_golden_qa_status", table_name="golden_qa")
    op.drop_index("ix_golden_qa_product_id", table_name="golden_qa")
    op.drop_table("golden_qa")
    for col in _REVIEW_INDEXED:
        op.drop_index(f"ix_review_items_{col}", table_name="review_items")
    op.drop_table("review_items")
