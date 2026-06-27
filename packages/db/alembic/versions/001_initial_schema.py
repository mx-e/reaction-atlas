"""Initial schema for CRN exploration database.

Revision ID: 001
Revises: None
Create Date: 2026-03-23
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONColumn as _PG_JSONColumn

# Use JSONColumn on PostgreSQL, plain JSON elsewhere
JSONColumn = sa.JSON().with_variant(_PG_JSONColumn, "postgresql")
TIMESTAMP = sa.DateTime

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # --- compounds ---
    op.create_table(
        "compounds",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("smiles", sa.Text, nullable=False, unique=True),
        sa.Column("formula", sa.Text, nullable=False),
        sa.Column("charge", sa.Integer, nullable=False, server_default="0"),
        sa.Column("n_atoms", sa.Integer, nullable=False),
        sa.Column("sorted_atomic_numbers", sa.LargeBinary, nullable=False),
        sa.Column("is_seed", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )

    # --- minima ---
    op.create_table(
        "minima",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("compound_id", sa.Integer, sa.ForeignKey("compounds.id"), nullable=False),
        sa.Column("local_id", sa.Integer, nullable=False),
        sa.Column("positions", sa.LargeBinary, nullable=False),
        sa.Column("energy", sa.Float, nullable=False),
        sa.Column("hessian", sa.LargeBinary, nullable=True),
        sa.Column("explored", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("name", sa.Text, nullable=False, server_default=""),
        sa.Column("n_merged", sa.Integer, nullable=False, server_default="0"),
        sa.Column("max_merge_rmsd", sa.Float, nullable=False, server_default="0.0"),
        sa.Column("discovery_timestamp", sa.Float, nullable=False, server_default="0.0"),
        sa.UniqueConstraint("compound_id", "local_id", name="uq_minima_compound_local"),
    )
    op.create_index("idx_minima_compound", "minima", ["compound_id"])
    op.create_index(
        "idx_minima_unexplored", "minima", ["compound_id"],
        postgresql_where=sa.text("NOT explored"),
    )

    # --- intra_transition_states ---
    op.create_table(
        "intra_transition_states",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("compound_id", sa.Integer, sa.ForeignKey("compounds.id"), nullable=False),
        sa.Column("local_id", sa.Integer, nullable=False),
        sa.Column("positions", sa.LargeBinary, nullable=False),
        sa.Column("energy", sa.Float, nullable=False),
        sa.Column("eigenvalue", sa.Float, nullable=False, server_default="0.0"),
        sa.Column("hessian", sa.LargeBinary, nullable=True),
        sa.Column("min_fwd_id", sa.Integer, sa.ForeignKey("minima.id"), nullable=False),
        sa.Column("min_bwd_id", sa.Integer, sa.ForeignKey("minima.id"), nullable=False),
        sa.Column("barrier_fwd", sa.Float, nullable=False),
        sa.Column("barrier_bwd", sa.Float, nullable=False),
        sa.Column("rmsd_to_fwd_min", sa.Float, nullable=False, server_default="0.0"),
        sa.Column("rmsd_to_bwd_min", sa.Float, nullable=False, server_default="0.0"),
        sa.Column("endpoint_to_endpoint_rmsd", sa.Float, nullable=False, server_default="0.0"),
        sa.Column("fwd_trajectory", sa.LargeBinary, nullable=True),
        sa.Column("bwd_trajectory", sa.LargeBinary, nullable=True),
        sa.Column("name", sa.Text, nullable=False, server_default=""),
        sa.Column("discovery_timestamp", sa.Float, nullable=False, server_default="0.0"),
        sa.UniqueConstraint("compound_id", "local_id", name="uq_intra_ts_compound_local"),
    )
    op.create_index("idx_intra_ts_compound", "intra_transition_states", ["compound_id"])

    # --- reactions ---
    op.create_table(
        "reactions",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("ts_id", sa.Integer, nullable=False, unique=True),
        sa.Column("ts_conformer_positions", sa.LargeBinary, nullable=False),
        sa.Column("ts_conformer_atomic_numbers", sa.LargeBinary, nullable=False),
        sa.Column("ts_conformer_charge", sa.Integer, nullable=False, server_default="0"),
        sa.Column("ts_energy", sa.Float, nullable=False),
        sa.Column("barrier_forward", sa.Float, nullable=False, server_default="0.0"),
        sa.Column("barrier_backward", sa.Float, nullable=False, server_default="0.0"),
        sa.Column("reactant_trajectory", sa.LargeBinary, nullable=True),
        sa.Column("product_trajectory", sa.LargeBinary, nullable=True),
        sa.Column("discovery_method", sa.Text, nullable=True),
        sa.Column("discovery_noise_level", sa.Integer, nullable=True),
        sa.Column("discovery_timestamp", sa.Float, nullable=True),
        sa.Column("name", sa.Text, nullable=False, server_default=""),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )

    # --- reaction_reactants ---
    op.create_table(
        "reaction_reactants",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("reaction_id", sa.Integer, sa.ForeignKey("reactions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("compound_id", sa.Integer, sa.ForeignKey("compounds.id"), nullable=False),
        sa.Column("conformer_local_id", sa.Integer, nullable=True),
        sa.UniqueConstraint("reaction_id", "compound_id", "conformer_local_id", name="uq_reaction_reactant"),
    )
    op.create_index("idx_reaction_reactants_compound", "reaction_reactants", ["compound_id"])

    # --- reaction_products ---
    op.create_table(
        "reaction_products",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("reaction_id", sa.Integer, sa.ForeignKey("reactions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("compound_id", sa.Integer, sa.ForeignKey("compounds.id"), nullable=False),
        sa.Column("conformer_local_id", sa.Integer, nullable=False),
        sa.Column("energy", sa.Float, nullable=False),
    )
    op.create_index("idx_reaction_products_compound", "reaction_products", ["compound_id"])

    # --- graph_edges ---
    op.create_table(
        "graph_edges",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("source_node", sa.Text, nullable=False),
        sa.Column("target_node", sa.Text, nullable=False),
        sa.Column("source_type", sa.Text, nullable=False),
        sa.Column("target_type", sa.Text, nullable=False),
        sa.Column("direction", sa.Text, nullable=True),
        sa.Column("stoichiometry", sa.Integer, nullable=False, server_default="1"),
        sa.Column("energy_diff", sa.Float, nullable=True),
        sa.Column("reaction_id", sa.Integer, sa.ForeignKey("reactions.id"), nullable=True),
    )
    op.create_index("idx_graph_edges_source", "graph_edges", ["source_node"])
    op.create_index("idx_graph_edges_target", "graph_edges", ["target_node"])

    # --- pes_work_queue ---
    op.create_table(
        "pes_work_queue",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("compound_id", sa.Integer, sa.ForeignKey("compounds.id"), nullable=False),
        sa.Column("minimum_id", sa.Integer, sa.ForeignKey("minima.id"), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="pending"),
        sa.Column("worker_id", sa.Text, nullable=True),
        sa.Column("claimed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("minimum_id", name="uq_pes_work_minimum"),
    )
    op.create_index(
        "idx_pes_work_pending", "pes_work_queue", ["status"],
        postgresql_where=sa.text("status = 'pending'"),
    )

    # --- exploration_stats ---
    op.create_table(
        "exploration_stats",
        sa.Column("id", sa.Integer, primary_key=True, server_default="1"),
        sa.Column("stats_json", JSONColumn, nullable=False, server_default="{}"),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )

    # --- batch_log ---
    op.create_table(
        "batch_log",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("batch_idx", sa.Integer, nullable=True),
        sa.Column("summary_json", JSONColumn, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )

    # --- annotations ---
    op.create_table(
        "annotations",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("entity_type", sa.Text, nullable=False),
        sa.Column("entity_key", sa.Text, nullable=False),
        sa.Column("label", sa.Text, nullable=True),
        sa.Column("notes", sa.Text, nullable=True),
        sa.UniqueConstraint("entity_type", "entity_key", name="uq_annotation_entity"),
    )

    # --- saved_layouts ---
    op.create_table(
        "saved_layouts",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("name", sa.Text, nullable=False, unique=True),
        sa.Column("layout_data", JSONColumn, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )

    # Insert default exploration stats row
    op.execute("INSERT INTO exploration_stats (id, stats_json) VALUES (1, '{}')")


def downgrade() -> None:
    op.drop_table("saved_layouts")
    op.drop_table("annotations")
    op.drop_table("batch_log")
    op.drop_table("exploration_stats")
    op.drop_table("pes_work_queue")
    op.drop_table("graph_edges")
    op.drop_table("reaction_products")
    op.drop_table("reaction_reactants")
    op.drop_table("reactions")
    op.drop_table("intra_transition_states")
    op.drop_table("minima")
    op.drop_table("compounds")
