"""Experiments tagging for multi-experiment support.

Adds an `experiments TEXT[]` column to entity-style tables (compounds, minima,
intra_transition_states, reactions, graph_edges, annotations, saved_layouts)
and an `experiment TEXT` column to work-state tables (pes/crest/dft work
queues, worker_heartbeats, kinetics_snapshots, batch_log, exploration_stats).

Existing rows are backfilled to 'main' / ARRAY['main'] via server_default at
ADD COLUMN time, then the default is dropped so application code must set
experiment(s) explicitly going forward (no silent drift).

Migration B (005) tags the formose-drilldown subset; this migration is purely
schema.

Revision ID: 004
Revises: 003
Create Date: 2026-05-06
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "004"
down_revision: Union[str, None] = "003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# Tables whose rows can belong to multiple experiments.
_MULTI_TABLES = (
    "compounds",
    "minima",
    "intra_transition_states",
    "reactions",
    "graph_edges",
    "annotations",
    "saved_layouts",
)

# Tables whose rows are scoped to exactly one experiment.
_SINGLE_TABLES = (
    "pes_work_queue",
    "crest_work_queue",
    "dft_work_queue",
    "worker_heartbeats",
    "kinetics_snapshots",
    "batch_log",
    "exploration_stats",
)


def upgrade() -> None:
    # --- multi-experiment columns + GIN indexes ---
    for tbl in _MULTI_TABLES:
        op.add_column(
            tbl,
            sa.Column(
                "experiments",
                postgresql.ARRAY(sa.Text),
                nullable=False,
                server_default="{main}",
            ),
        )
        op.alter_column(tbl, "experiments", server_default=None)
        op.create_index(
            f"idx_{tbl}_experiments",
            tbl,
            ["experiments"],
            postgresql_using="gin",
        )

    # --- single-experiment columns + targeted btree indexes ---
    for tbl in _SINGLE_TABLES:
        op.add_column(
            tbl,
            sa.Column(
                "experiment",
                sa.Text,
                nullable=False,
                server_default="main",
            ),
        )
        op.alter_column(tbl, "experiment", server_default=None)

    # Worker poll path: WHERE status='pending' AND experiment=:exp.
    # Composite index restricted to pending rows mirrors the existing
    # idx_<queue>_pending partials and keeps them tight.
    op.create_index(
        "idx_pes_work_pending_exp",
        "pes_work_queue",
        ["experiment", "status"],
        postgresql_where=sa.text("status = 'pending'"),
    )
    op.create_index(
        "idx_crest_work_pending_exp",
        "crest_work_queue",
        ["experiment", "status"],
        postgresql_where=sa.text("status = 'pending'"),
    )
    op.create_index(
        "idx_dft_work_pending_exp",
        "dft_work_queue",
        ["experiment", "status"],
        postgresql_where=sa.text("status = 'pending'"),
    )

    op.create_index(
        "idx_heartbeats_experiment",
        "worker_heartbeats",
        ["experiment", "last_heartbeat"],
    )

    # "Latest snapshot per experiment" is the hot read.
    op.create_index(
        "idx_kinetics_snapshots_exp_recent",
        "kinetics_snapshots",
        ["experiment", "computed_at"],
    )

    op.create_index(
        "idx_batch_log_experiment",
        "batch_log",
        ["experiment", "created_at"],
    )

    # exploration_stats was a singleton (id=1) — now one row per experiment.
    # Unique index on experiment turns it into an upsert key.
    op.create_index(
        "uq_exploration_stats_experiment",
        "exploration_stats",
        ["experiment"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index("uq_exploration_stats_experiment", table_name="exploration_stats")
    op.drop_index("idx_batch_log_experiment", table_name="batch_log")
    op.drop_index("idx_kinetics_snapshots_exp_recent", table_name="kinetics_snapshots")
    op.drop_index("idx_heartbeats_experiment", table_name="worker_heartbeats")
    op.drop_index("idx_dft_work_pending_exp", table_name="dft_work_queue")
    op.drop_index("idx_crest_work_pending_exp", table_name="crest_work_queue")
    op.drop_index("idx_pes_work_pending_exp", table_name="pes_work_queue")

    for tbl in _SINGLE_TABLES:
        op.drop_column(tbl, "experiment")

    for tbl in _MULTI_TABLES:
        op.drop_index(f"idx_{tbl}_experiments", table_name=tbl)
        op.drop_column(tbl, "experiments")
