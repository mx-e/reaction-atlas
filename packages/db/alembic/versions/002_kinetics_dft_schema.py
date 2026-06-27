"""Kinetics + DFT schema additions.

Adds the foundation columns and tables for:
  - separated/IRC-extremum ML barriers (Phase 1, computed by GPU worker at reaction creation)
  - PBE0 single-point energy slots on every relevant table (Phase 1; populated
    by the cpu-worker DFT job in Phase 4 for compounds + reactions; left null
    elsewhere for on-demand frontend triggers / dataset builder later)
  - manual equilibrium rate constant columns on Reaction (Phase 2)
  - WorkerHeartbeat.current_job_kind so cpu-workers can report dft vs crest
  - dft_work_queue table (Phase 4 cpu-worker claim queue)
  - kinetics_snapshots table (Phase 3 kinetics solver output cache)

No backfill — fresh deploy. All new columns are nullable.

Revision ID: 002
Revises: 001
Create Date: 2026-04-07
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB as _PG_JSONB

JSONColumn = sa.JSON().with_variant(_PG_JSONB, "postgresql")

revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # --- compounds: denormalized DFT cache ---
    op.add_column("compounds", sa.Column("energy_pbe0", sa.Float, nullable=True))
    op.add_column("compounds", sa.Column("energy_pbe0_method", sa.Text, nullable=True))
    op.add_column("compounds", sa.Column("energy_pbe0_at", sa.DateTime(timezone=True), nullable=True))

    # --- minima: per-conformer DFT slot ---
    op.add_column("minima", sa.Column("energy_pbe0", sa.Float, nullable=True))
    op.add_column("minima", sa.Column("energy_pbe0_method", sa.Text, nullable=True))
    op.add_column("minima", sa.Column("energy_pbe0_at", sa.DateTime(timezone=True), nullable=True))

    # --- intra_transition_states: dataset DFT slots (never auto-populated) ---
    op.add_column("intra_transition_states", sa.Column("energy_pbe0", sa.Float, nullable=True))
    op.add_column("intra_transition_states", sa.Column("barrier_fwd_pbe0", sa.Float, nullable=True))
    op.add_column("intra_transition_states", sa.Column("barrier_bwd_pbe0", sa.Float, nullable=True))
    op.add_column("intra_transition_states", sa.Column("energy_pbe0_method", sa.Text, nullable=True))
    op.add_column("intra_transition_states", sa.Column("energy_pbe0_at", sa.DateTime(timezone=True), nullable=True))

    # --- reactions: separated/ex ML barriers + DFT slots + manual rate constants ---
    op.add_column("reactions", sa.Column("barrier_forward_separated", sa.Float, nullable=True))
    op.add_column("reactions", sa.Column("barrier_backward_separated", sa.Float, nullable=True))
    op.add_column("reactions", sa.Column("barrier_forward_ex", sa.Float, nullable=True))
    op.add_column("reactions", sa.Column("barrier_backward_ex", sa.Float, nullable=True))

    op.add_column("reactions", sa.Column("energy_R_pbe0", sa.Float, nullable=True))
    op.add_column("reactions", sa.Column("energy_TS_pbe0", sa.Float, nullable=True))
    op.add_column("reactions", sa.Column("energy_P_pbe0", sa.Float, nullable=True))
    op.add_column("reactions", sa.Column("barrier_forward_pbe0", sa.Float, nullable=True))
    op.add_column("reactions", sa.Column("barrier_backward_pbe0", sa.Float, nullable=True))
    op.add_column("reactions", sa.Column("barrier_forward_separated_pbe0", sa.Float, nullable=True))
    op.add_column("reactions", sa.Column("barrier_backward_separated_pbe0", sa.Float, nullable=True))
    op.add_column("reactions", sa.Column("energy_pbe0_method", sa.Text, nullable=True))
    op.add_column("reactions", sa.Column("energy_pbe0_at", sa.DateTime(timezone=True), nullable=True))

    op.add_column("reactions", sa.Column("manual_k_fwd", sa.Float, nullable=True))
    op.add_column("reactions", sa.Column("manual_k_bwd", sa.Float, nullable=True))

    # --- worker_heartbeats: track which job kind a cpu-worker is running ---
    op.add_column("worker_heartbeats", sa.Column("current_job_kind", sa.Text, nullable=True))

    # --- dft_work_queue (new) ---
    op.create_table(
        "dft_work_queue",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("reaction_id", sa.Integer, sa.ForeignKey("reactions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="pending"),
        sa.Column("worker_id", sa.Text, nullable=True),
        sa.Column("claimed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_msg", sa.Text, nullable=True),
        sa.UniqueConstraint("reaction_id", name="uq_dft_work_reaction"),
    )
    op.create_index(
        "idx_dft_work_pending", "dft_work_queue", ["status"],
        postgresql_where=sa.text("status = 'pending'"),
    )

    # --- kinetics_snapshots (new) ---
    op.create_table(
        "kinetics_snapshots",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("network_version", sa.Integer, nullable=False),
        sa.Column("n_reactions_dft", sa.Integer, nullable=False, server_default="0"),
        sa.Column("temperature", sa.Float, nullable=False),
        sa.Column("payload_jsonb", JSONColumn, nullable=False),
        sa.Column("solve_wall_time_s", sa.Float, nullable=True),
        sa.Column("computed_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("idx_kinetics_snapshots_recent", "kinetics_snapshots", ["computed_at"])


def downgrade() -> None:
    op.drop_index("idx_kinetics_snapshots_recent", table_name="kinetics_snapshots")
    op.drop_table("kinetics_snapshots")

    op.drop_index("idx_dft_work_pending", table_name="dft_work_queue")
    op.drop_table("dft_work_queue")

    op.drop_column("worker_heartbeats", "current_job_kind")

    op.drop_column("reactions", "manual_k_bwd")
    op.drop_column("reactions", "manual_k_fwd")
    op.drop_column("reactions", "energy_pbe0_at")
    op.drop_column("reactions", "energy_pbe0_method")
    op.drop_column("reactions", "barrier_backward_separated_pbe0")
    op.drop_column("reactions", "barrier_forward_separated_pbe0")
    op.drop_column("reactions", "barrier_backward_pbe0")
    op.drop_column("reactions", "barrier_forward_pbe0")
    op.drop_column("reactions", "energy_P_pbe0")
    op.drop_column("reactions", "energy_TS_pbe0")
    op.drop_column("reactions", "energy_R_pbe0")
    op.drop_column("reactions", "barrier_backward_ex")
    op.drop_column("reactions", "barrier_forward_ex")
    op.drop_column("reactions", "barrier_backward_separated")
    op.drop_column("reactions", "barrier_forward_separated")

    op.drop_column("intra_transition_states", "energy_pbe0_at")
    op.drop_column("intra_transition_states", "energy_pbe0_method")
    op.drop_column("intra_transition_states", "barrier_bwd_pbe0")
    op.drop_column("intra_transition_states", "barrier_fwd_pbe0")
    op.drop_column("intra_transition_states", "energy_pbe0")

    op.drop_column("minima", "energy_pbe0_at")
    op.drop_column("minima", "energy_pbe0_method")
    op.drop_column("minima", "energy_pbe0")

    op.drop_column("compounds", "energy_pbe0_at")
    op.drop_column("compounds", "energy_pbe0_method")
    op.drop_column("compounds", "energy_pbe0")
