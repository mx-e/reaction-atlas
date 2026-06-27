"""Add TS PBE0 Hessian columns + claim slots to reactions.

Optional dataset-only Hessian on the transition-state geometry, computed by a
flagged cpu-worker (COMPUTE_TS_HESSIAN=1) when its DFT and the PES queues are
both empty. Not part of any experiment-driving pipeline — pure post-hoc
ground-truth backfill for dataset releases.

Schema:
  - ts_hessian_pbe0          LargeBinary  the (3N,3N) float64 blob (no symmetrize)
  - ts_hessian_pbe0_at       TimestampTZ  completion timestamp
  - ts_hessian_pbe0_claimed_by  Text      worker_id holding the claim
  - ts_hessian_pbe0_claimed_at  TimestampTZ  claim time (for stale-recovery)
  - ts_hessian_pbe0_failed   Boolean      sticky failure flag (skipped by future picks)

Indexing:
  Partial btree on `id` filtered to eligible rows so the random-pick claim
  (ORDER BY random() LIMIT 1) scans only the work-pool. Cheap to maintain
  because rows leave the index permanently once `ts_hessian_pbe0` is populated.

Revision ID: 006
Revises: 005
Create Date: 2026-05-06
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "006"
down_revision: Union[str, None] = "005"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("reactions", sa.Column("ts_hessian_pbe0", sa.LargeBinary, nullable=True))
    op.add_column("reactions", sa.Column("ts_hessian_pbe0_at", sa.TIMESTAMP(timezone=True), nullable=True))
    op.add_column("reactions", sa.Column("ts_hessian_pbe0_claimed_by", sa.Text, nullable=True))
    op.add_column("reactions", sa.Column("ts_hessian_pbe0_claimed_at", sa.TIMESTAMP(timezone=True), nullable=True))
    op.add_column(
        "reactions",
        sa.Column("ts_hessian_pbe0_failed", sa.Boolean, nullable=False, server_default=sa.false()),
    )
    op.alter_column("reactions", "ts_hessian_pbe0_failed", server_default=None)

    op.create_index(
        "idx_reactions_ts_hessian_pending",
        "reactions",
        ["id"],
        postgresql_where=sa.text("ts_hessian_pbe0 IS NULL AND NOT ts_hessian_pbe0_failed"),
    )


def downgrade() -> None:
    op.drop_index("idx_reactions_ts_hessian_pending", table_name="reactions")
    op.drop_column("reactions", "ts_hessian_pbe0_failed")
    op.drop_column("reactions", "ts_hessian_pbe0_claimed_at")
    op.drop_column("reactions", "ts_hessian_pbe0_claimed_by")
    op.drop_column("reactions", "ts_hessian_pbe0_at")
    op.drop_column("reactions", "ts_hessian_pbe0")
