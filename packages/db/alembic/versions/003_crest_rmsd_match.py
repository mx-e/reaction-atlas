"""Add rmsd_match JSONB column to crest_results.

Stores per-CREST-conformer Kabsch RMSD against our PES minima, computed by
the cpu-worker as a post-processing step after each successful CREST run
(and via /api/admin/backfill-rmsd-match for existing rows).

Revision ID: 003
Revises: 002
Create Date: 2026-04-08
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB as _PG_JSONB

JSONColumn = sa.JSON().with_variant(_PG_JSONB, "postgresql")

revision: str = "003"
down_revision: Union[str, None] = "002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("crest_results", sa.Column("rmsd_match", JSONColumn, nullable=True))


def downgrade() -> None:
    op.drop_column("crest_results", "rmsd_match")
