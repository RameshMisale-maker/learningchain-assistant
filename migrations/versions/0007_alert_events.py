"""alert events

Revision ID: 0007
Revises: 0006
Create Date: 2026-11-19
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "alert_events",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("rule", sa.String(length=64), nullable=False),
        sa.Column("severity", sa.String(length=16), nullable=False, server_default="warning"),
        sa.Column("product_id", sa.String(length=64), nullable=False, server_default=""),
        sa.Column("value", sa.Float(), nullable=False, server_default="0"),
        sa.Column("threshold", sa.Float(), nullable=False, server_default="0"),
        sa.Column("message", sa.Text(), nullable=False, server_default=""),
        sa.Column("notified", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.create_index("ix_alert_events_created_at", "alert_events", ["created_at"])
    op.create_index("ix_alert_events_rule", "alert_events", ["rule"])


def downgrade() -> None:
    op.drop_index("ix_alert_events_rule", table_name="alert_events")
    op.drop_index("ix_alert_events_created_at", table_name="alert_events")
    op.drop_table("alert_events")
