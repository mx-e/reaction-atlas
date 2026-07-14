"""Add `frontier_in TEXT[]` column to compounds.

Records the experiments in which a compound is treated as a frontier
(boundary) species — visible in that experiment's graph view but excluded
from its sampling/exploration pool. Used by closed-subgraph experiments
(those running with `RESTRICT_TO_EXISTING_COMPOUNDS=true`, e.g.
formose-drilldown) when a reaction lands on a product compound that
isn't part of the experiment's curated explorable set: the reaction is
still recorded and the product compound is tagged with the experiment
(so the equation node renders), but the experiment is also added to
`frontier_in` so the worker's sampling query skips this compound and no
PES/CREST exploration work is queued for it.

Schema:
  - frontier_in   TEXT[] NOT NULL DEFAULT '{}'
  - GIN index for the `~ANY(frontier_in)` lookup at sample time

Default `'{}'` means existing compounds are explorable in every
experiment they're already tagged in — no data migration needed. New
worker code only writes non-empty values for compounds it specifically
classifies as frontier products.

Idempotent (uses ADD COLUMN IF NOT EXISTS / CREATE INDEX IF NOT EXISTS).

Revision ID: 008
Revises: 007
Create Date: 2026-05-08
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "008"
down_revision: Union[str, None] = "007"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "compounds",
        sa.Column(
            "frontier_in",
            sa.dialects.postgresql.ARRAY(sa.Text),
            nullable=False,
            server_default="{}",
        ),
    )
    # GIN index supports the ANY(frontier_in) and array containment
    # filters used by the worker's _load_compounds_for_sampling.
    op.create_index(
        "idx_compounds_frontier_in",
        "compounds",
        ["frontier_in"],
        postgresql_using="gin",
    )


def downgrade() -> None:
    op.drop_index("idx_compounds_frontier_in", table_name="compounds")
    op.drop_column("compounds", "frontier_in")
