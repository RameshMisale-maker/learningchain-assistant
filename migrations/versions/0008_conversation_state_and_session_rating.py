"""conversation state per session; session-level rating fields on feedback

Revision ID: 0008
Revises: 0007
Create Date: 2026-11-19
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0008"
down_revision = "0007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("chat_sessions", schema=None) as batch_op:
        batch_op.add_column(sa.Column("conversation_state", sa.String(length=16), nullable=False,
                                      server_default="active"))
    with op.batch_alter_table("feedback", schema=None) as batch_op:
        batch_op.add_column(sa.Column("session_id", sa.String(length=64), nullable=False, server_default=""))
        batch_op.add_column(sa.Column("product_id", sa.String(length=64), nullable=False, server_default=""))
        batch_op.add_column(sa.Column("stars", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("details", sa.JSON(), nullable=True))
        batch_op.create_index("ix_feedback_session_id", ["session_id"])
        batch_op.create_index("ix_feedback_product_id", ["product_id"])
    # backfill: existing conversation ratings ("stars_N") get their star value
    for n in range(1, 6):
        op.execute(f"UPDATE feedback SET stars = {n} WHERE rating = 'stars_{n}'")


def downgrade() -> None:
    with op.batch_alter_table("feedback", schema=None) as batch_op:
        batch_op.drop_index("ix_feedback_product_id")
        batch_op.drop_index("ix_feedback_session_id")
        for col in ("details", "stars", "product_id", "session_id"):
            batch_op.drop_column(col)
    with op.batch_alter_table("chat_sessions", schema=None) as batch_op:
        batch_op.drop_column("conversation_state")
