"""Add TS-correction columns + Hessian/correction wall-time tracking.

Two related additions:

1. `ts_ml_invalid` — boolean flag set when the PBE0 Hessian of the ML-predicted
   TS does not have exactly one imaginary mode (i.e. the ML geometry is not a
   true saddle at the DFT level). Populated as the Hessian backfill lands a
   new row; can also be backfilled offline from existing Hessians.

2. The corrected-TS columns — populated by a separate cpu-worker mode that
   runs a PBE0/def2-TZVPP saddle-point optimization starting from the ML TS
   geometry whenever `ts_ml_invalid` is true. Records the optimized geometry,
   its energy, the ΔE vs. the ML single-point, the Kabsch-aligned RMSD to the
   ML geometry, and the wall-clock cost. Mirrors the claim/failure plumbing
   of the Hessian backfill so multiple workers can drain the queue safely.

3. `ts_hessian_pbe0_wall_s` — added in passing so we can report DFT-Hessian
   compute cost in the dataset paper alongside the new TS-correction cost.

Revision ID: 007
Revises: 006
Create Date: 2026-05-21
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "007"
down_revision: Union[str, None] = "006"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Hessian wall-time (added so the paper can report DFT compute cost).
    op.add_column("reactions", sa.Column("ts_hessian_pbe0_wall_s", sa.Float, nullable=True))

    # Validity flag derived from the Hessian.
    op.add_column("reactions", sa.Column("ts_ml_invalid", sa.Boolean, nullable=True))

    # Corrected-TS payload.
    op.add_column("reactions", sa.Column("ts_pbe0_corrected_positions", sa.LargeBinary, nullable=True))
    op.add_column("reactions", sa.Column("ts_pbe0_corrected_energy", sa.Float, nullable=True))
    op.add_column("reactions", sa.Column("ts_pbe0_corrected_at", sa.TIMESTAMP(timezone=True), nullable=True))
    op.add_column("reactions", sa.Column("ts_pbe0_corrected_de", sa.Float, nullable=True))
    op.add_column("reactions", sa.Column("ts_pbe0_corrected_rmsd", sa.Float, nullable=True))
    op.add_column("reactions", sa.Column("ts_pbe0_corrected_wall_s", sa.Float, nullable=True))
    op.add_column(
        "reactions",
        sa.Column("ts_pbe0_corrected_failed", sa.Boolean, nullable=False, server_default=sa.false()),
    )
    op.alter_column("reactions", "ts_pbe0_corrected_failed", server_default=None)
    op.add_column("reactions", sa.Column("ts_pbe0_corrected_claimed_by", sa.Text, nullable=True))
    op.add_column("reactions", sa.Column("ts_pbe0_corrected_claimed_at", sa.TIMESTAMP(timezone=True), nullable=True))

    # Worker claim-pool index: only rows that need a correction and don't have one yet.
    op.create_index(
        "idx_reactions_ts_corrected_pending",
        "reactions",
        ["id"],
        postgresql_where=sa.text(
            "ts_ml_invalid = TRUE "
            "AND ts_pbe0_corrected_positions IS NULL "
            "AND NOT ts_pbe0_corrected_failed"
        ),
    )


def downgrade() -> None:
    op.drop_index("idx_reactions_ts_corrected_pending", table_name="reactions")
    op.drop_column("reactions", "ts_pbe0_corrected_claimed_at")
    op.drop_column("reactions", "ts_pbe0_corrected_claimed_by")
    op.drop_column("reactions", "ts_pbe0_corrected_failed")
    op.drop_column("reactions", "ts_pbe0_corrected_wall_s")
    op.drop_column("reactions", "ts_pbe0_corrected_rmsd")
    op.drop_column("reactions", "ts_pbe0_corrected_de")
    op.drop_column("reactions", "ts_pbe0_corrected_at")
    op.drop_column("reactions", "ts_pbe0_corrected_energy")
    op.drop_column("reactions", "ts_pbe0_corrected_positions")
    op.drop_column("reactions", "ts_ml_invalid")
    op.drop_column("reactions", "ts_hessian_pbe0_wall_s")
