"""SQLAlchemy models for the CRN exploration database."""

from datetime import datetime, timezone

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, relationship

# Use JSONB on PostgreSQL, plain JSON elsewhere (SQLite)
from sqlalchemy.dialects.postgresql import ARRAY as _PG_ARRAY, JSONB as _PG_JSONB

JSONColumn = JSON().with_variant(_PG_JSONB, "postgresql")

# Multi-experiment tag column. Postgres-native TEXT[] is the primary type
# so the comparator exposes the array operators (in particular `.any()`,
# which compiles to `:val = ANY(column)`). SQLite gets a JSON fallback so
# unit tests / local scratch DBs still load the schema, but JSON-on-SQLite
# does NOT support array containment queries — application code that uses
# `.any()` is therefore Postgres-only.
ExperimentsArray = _PG_ARRAY(Text).with_variant(JSON(), "sqlite")

# Timezone-aware datetime column type (works on both PostgreSQL and SQLite)
TimestampTZ = DateTime(timezone=True)


class Base(DeclarativeBase):
    pass


class Compound(Base):
    __tablename__ = "compounds"

    id = Column(Integer, primary_key=True)
    smiles = Column(Text, nullable=False, unique=True)
    formula = Column(Text, nullable=False)
    charge = Column(Integer, nullable=False, default=0)
    n_atoms = Column(Integer, nullable=False)
    sorted_atomic_numbers = Column(LargeBinary, nullable=False)
    is_seed = Column(Boolean, nullable=False, default=False)
    created_at = Column(
        TimestampTZ,
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    # Denormalized DFT cache: PBE0 single-point energy at the lowest-energy
    # minimum's geometry. Written by the cpu-worker DFT job; reused across all
    # reactions touching this compound (separated-barrier reference energy).
    energy_pbe0 = Column(Float, nullable=True)
    energy_pbe0_method = Column(Text, nullable=True)  # e.g. 'PBE0/def2-TZVPP'
    energy_pbe0_at = Column(TimestampTZ, nullable=True)

    experiments = Column(ExperimentsArray, nullable=False, default=lambda: ["main"])
    # Subset of experiments where this compound is treated as a frontier
    # (boundary) species: visible in the experiment's graph view but
    # excluded from sampling/exploration. Set when a closed-subgraph
    # worker (RESTRICT_TO_EXISTING_COMPOUNDS=true) discovers a reaction
    # whose product compound isn't in the experiment's curated set —
    # we record the reaction without widening the explorable pool.
    frontier_in = Column(ExperimentsArray, nullable=False, default=list)

    minima = relationship("Minimum", back_populates="compound", cascade="all, delete-orphan")
    intra_transition_states = relationship(
        "IntraTransitionState", back_populates="compound", cascade="all, delete-orphan"
    )
    pes_work_items = relationship("PESWorkQueue", back_populates="compound")


class Minimum(Base):
    __tablename__ = "minima"
    __table_args__ = (
        UniqueConstraint("compound_id", "local_id", name="uq_minima_compound_local"),
        Index("idx_minima_compound", "compound_id"),
        Index("idx_minima_unexplored", "compound_id", postgresql_where="NOT explored"),
    )

    id = Column(Integer, primary_key=True)
    compound_id = Column(Integer, ForeignKey("compounds.id"), nullable=False)
    local_id = Column(Integer, nullable=False)  # Sequential per compound (0, 1, 2...)
    positions = Column(LargeBinary, nullable=False)
    energy = Column(Float, nullable=False)
    hessian = Column(LargeBinary, nullable=True)
    explored = Column(Boolean, nullable=False, default=False)
    name = Column(Text, nullable=False, default="")
    n_merged = Column(Integer, nullable=False, default=0)
    max_merge_rmsd = Column(Float, nullable=False, default=0.0)
    discovery_timestamp = Column(Float, nullable=False, default=0.0)

    # Per-conformer DFT slot. Auto-populated only for the lowest-E minimum of
    # each compound when the cpu-worker hits its parent compound; left null on
    # other minima for on-demand frontend triggers / dataset builder later.
    energy_pbe0 = Column(Float, nullable=True)
    energy_pbe0_method = Column(Text, nullable=True)
    energy_pbe0_at = Column(TimestampTZ, nullable=True)

    experiments = Column(ExperimentsArray, nullable=False, default=lambda: ["main"])

    compound = relationship("Compound", back_populates="minima")


class IntraTransitionState(Base):
    __tablename__ = "intra_transition_states"
    __table_args__ = (
        UniqueConstraint("compound_id", "local_id", name="uq_intra_ts_compound_local"),
        Index("idx_intra_ts_compound", "compound_id"),
    )

    id = Column(Integer, primary_key=True)
    compound_id = Column(Integer, ForeignKey("compounds.id"), nullable=False)
    local_id = Column(BigInteger, nullable=False)
    positions = Column(LargeBinary, nullable=False)
    energy = Column(Float, nullable=False)
    eigenvalue = Column(Float, nullable=False, default=0.0)
    hessian = Column(LargeBinary, nullable=True)
    min_fwd_id = Column(Integer, ForeignKey("minima.id"), nullable=False)
    min_bwd_id = Column(Integer, ForeignKey("minima.id"), nullable=False)
    barrier_fwd = Column(Float, nullable=False)
    barrier_bwd = Column(Float, nullable=False)
    rmsd_to_fwd_min = Column(Float, nullable=False, default=0.0)
    rmsd_to_bwd_min = Column(Float, nullable=False, default=0.0)
    endpoint_to_endpoint_rmsd = Column(Float, nullable=False, default=0.0)
    fwd_trajectory = Column(LargeBinary, nullable=True)
    bwd_trajectory = Column(LargeBinary, nullable=True)
    name = Column(Text, nullable=False, default="")
    discovery_timestamp = Column(Float, nullable=False, default=0.0)

    # DFT slots — never auto-populated, dataset builder / on-demand only.
    energy_pbe0 = Column(Float, nullable=True)
    barrier_fwd_pbe0 = Column(Float, nullable=True)
    barrier_bwd_pbe0 = Column(Float, nullable=True)
    energy_pbe0_method = Column(Text, nullable=True)
    energy_pbe0_at = Column(TimestampTZ, nullable=True)

    experiments = Column(ExperimentsArray, nullable=False, default=lambda: ["main"])

    min_fwd = relationship("Minimum", foreign_keys=[min_fwd_id])
    min_bwd = relationship("Minimum", foreign_keys=[min_bwd_id])
    compound = relationship("Compound", back_populates="intra_transition_states")


class Reaction(Base):
    __tablename__ = "reactions"
    __table_args__ = (
        Index("idx_reactions_name", "name"),
        Index("idx_reactions_discovery_timestamp", "discovery_timestamp"),
    )

    id = Column(Integer, primary_key=True)
    ts_id = Column(BigInteger, nullable=False, unique=True)
    ts_conformer_positions = Column(LargeBinary, nullable=False)
    ts_conformer_atomic_numbers = Column(LargeBinary, nullable=False)
    ts_conformer_charge = Column(Integer, nullable=False, default=0)
    ts_energy = Column(Float, nullable=False)

    # In-box ML barriers (TS - trajectory endpoint energy, integrated via forces).
    # Susceptible to ML long-distance artifacts; kept for the dataset / comparison
    # but NOT used by the kinetics solver.
    barrier_forward = Column(Float, nullable=False, default=0.0)
    barrier_backward = Column(Float, nullable=False, default=0.0)

    # Separated ML barriers (TS - sum of reference-conformer energies of each
    # reactant/product compound). The principled choice for kinetics — bypasses
    # ML long-distance artifacts. Computed at reaction creation time.
    barrier_forward_separated = Column(Float, nullable=True)
    barrier_backward_separated = Column(Float, nullable=True)

    # IRC-extremum ML barriers (TS - min(E along IRC trajectory on each side)).
    # A weaker fix for the same long-distance artifact issue. Stored for comparison.
    barrier_forward_ex = Column(Float, nullable=True)
    barrier_backward_ex = Column(Float, nullable=True)

    # DFT (PBE0/def2-TZVPP) energies and derived barriers, populated by the
    # cpu-worker DFT job after the reaction is discovered. Same in-box vs
    # separated split as the ML barriers above.
    energy_R_pbe0 = Column(Float, nullable=True)   # PBE0 on reactant_trajectory[0]
    energy_TS_pbe0 = Column(Float, nullable=True)  # PBE0 on ts_conformer_positions
    energy_P_pbe0 = Column(Float, nullable=True)   # PBE0 on product_trajectory[0]
    barrier_forward_pbe0 = Column(Float, nullable=True)            # in-box DFT
    barrier_backward_pbe0 = Column(Float, nullable=True)
    barrier_forward_separated_pbe0 = Column(Float, nullable=True)  # separated DFT — kinetics solver primary
    barrier_backward_separated_pbe0 = Column(Float, nullable=True)
    energy_pbe0_method = Column(Text, nullable=True)  # 'PBE0/def2-TZVPP'
    energy_pbe0_at = Column(TimestampTZ, nullable=True)

    # Optional PBE0 Hessian on the TS geometry (post-hoc dataset backfill,
    # gated by COMPUTE_TS_HESSIAN flag on a cpu-worker). Stored as the raw
    # (3N, 3N) float64 blob — caller symmetrizes if desired. Method/basis
    # share the energy_pbe0_method slot above.
    ts_hessian_pbe0 = Column(LargeBinary, nullable=True)
    ts_hessian_pbe0_at = Column(TimestampTZ, nullable=True)
    # Inline atomic-claim slots; the reactions table itself is the work pool.
    ts_hessian_pbe0_claimed_by = Column(Text, nullable=True)
    ts_hessian_pbe0_claimed_at = Column(TimestampTZ, nullable=True)
    ts_hessian_pbe0_failed = Column(Boolean, nullable=False, default=False)
    # Wall-clock cost of the Hessian compute (so the paper can quote it).
    ts_hessian_pbe0_wall_s = Column(Float, nullable=True)

    # Set True by the Hessian backfill when the ML-predicted TS is *not* a true
    # saddle at the DFT level (exactly-one-imaginary-mode criterion). When True
    # the corrected-TS worker picks the row up and runs a PBE0 saddle-point
    # optimization to produce a real DFT TS nearby.
    ts_ml_invalid = Column(Boolean, nullable=True)

    # Corrected-TS payload — populated only when ts_ml_invalid is True. The
    # geometry is the result of a PBE0 saddle-point opt starting from the ML
    # TS; ts_pbe0_corrected_de is E(corrected) - E(ML at ML geom), and
    # ts_pbe0_corrected_rmsd is the Kabsch-aligned RMSD to the ML geometry.
    ts_pbe0_corrected_positions = Column(LargeBinary, nullable=True)
    ts_pbe0_corrected_energy = Column(Float, nullable=True)
    ts_pbe0_corrected_at = Column(TimestampTZ, nullable=True)
    ts_pbe0_corrected_de = Column(Float, nullable=True)
    ts_pbe0_corrected_rmsd = Column(Float, nullable=True)
    ts_pbe0_corrected_wall_s = Column(Float, nullable=True)
    ts_pbe0_corrected_failed = Column(Boolean, nullable=False, default=False)
    ts_pbe0_corrected_claimed_by = Column(Text, nullable=True)
    ts_pbe0_corrected_claimed_at = Column(TimestampTZ, nullable=True)

    # Manual equilibrium reactions (water autoionization, CO2 hydration, etc.)
    # use literal rate constants instead of Eyring-from-barriers. When set, the
    # kinetics solver uses these directly (temperature-independent).
    manual_k_fwd = Column(Float, nullable=True)
    manual_k_bwd = Column(Float, nullable=True)

    reactant_trajectory = Column(LargeBinary, nullable=True)
    product_trajectory = Column(LargeBinary, nullable=True)
    discovery_method = Column(Text, nullable=True)
    discovery_noise_level = Column(Integer, nullable=True)
    discovery_timestamp = Column(Float, nullable=True)
    name = Column(Text, nullable=False, default="")
    created_at = Column(
        TimestampTZ,
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    experiments = Column(ExperimentsArray, nullable=False, default=lambda: ["main"])

    reactants = relationship("ReactionReactant", back_populates="reaction", cascade="all, delete-orphan")
    products = relationship("ReactionProduct", back_populates="reaction", cascade="all, delete-orphan")


class ReactionReactant(Base):
    __tablename__ = "reaction_reactants"
    __table_args__ = (
        UniqueConstraint("reaction_id", "compound_id", "conformer_local_id", name="uq_reaction_reactant"),
        Index("idx_reaction_reactants_compound", "compound_id"),
    )

    id = Column(Integer, primary_key=True)
    reaction_id = Column(Integer, ForeignKey("reactions.id", ondelete="CASCADE"), nullable=False)
    compound_id = Column(Integer, ForeignKey("compounds.id"), nullable=False)
    conformer_local_id = Column(Integer, nullable=True)

    reaction = relationship("Reaction", back_populates="reactants")
    compound = relationship("Compound")


class ReactionProduct(Base):
    __tablename__ = "reaction_products"
    __table_args__ = (
        Index("idx_reaction_products_compound", "compound_id"),
    )

    id = Column(Integer, primary_key=True)
    reaction_id = Column(Integer, ForeignKey("reactions.id", ondelete="CASCADE"), nullable=False)
    compound_id = Column(Integer, ForeignKey("compounds.id"), nullable=False)
    conformer_local_id = Column(Integer, nullable=False)
    energy = Column(Float, nullable=False)

    reaction = relationship("Reaction", back_populates="products")
    compound = relationship("Compound")


class GraphEdge(Base):
    __tablename__ = "graph_edges"
    __table_args__ = (
        Index("idx_graph_edges_source", "source_node"),
        Index("idx_graph_edges_target", "target_node"),
    )

    id = Column(Integer, primary_key=True)
    source_node = Column(Text, nullable=False)
    target_node = Column(Text, nullable=False)
    source_type = Column(Text, nullable=False)
    target_type = Column(Text, nullable=False)
    direction = Column(Text, nullable=True)
    stoichiometry = Column(Integer, nullable=False, default=1)
    energy_diff = Column(Float, nullable=True)
    reaction_id = Column(Integer, ForeignKey("reactions.id"), nullable=True)
    experiments = Column(ExperimentsArray, nullable=False, default=lambda: ["main"])


class PESWorkQueue(Base):
    __tablename__ = "pes_work_queue"
    __table_args__ = (
        UniqueConstraint("minimum_id", name="uq_pes_work_minimum"),
        Index("idx_pes_work_pending", "status", postgresql_where="status = 'pending'"),
    )

    id = Column(Integer, primary_key=True)
    compound_id = Column(Integer, ForeignKey("compounds.id"), nullable=False)
    minimum_id = Column(Integer, ForeignKey("minima.id"), nullable=False)
    status = Column(String(20), nullable=False, default="pending")
    job_kind = Column(String(20), nullable=False, default="explore")  # "explore" or "dedup"
    worker_id = Column(Text, nullable=True)
    claimed_at = Column(TimestampTZ, nullable=True)
    completed_at = Column(TimestampTZ, nullable=True)
    experiment = Column(Text, nullable=False, default="main")

    compound = relationship("Compound", back_populates="pes_work_items")
    minimum = relationship("Minimum")


class WorkerHeartbeat(Base):
    """Live worker registry — workers upsert a heartbeat row periodically."""
    __tablename__ = "worker_heartbeats"
    __table_args__ = (
        Index("idx_heartbeat_active", "last_heartbeat"),
    )

    worker_id = Column(Text, primary_key=True)
    worker_type = Column(Text, nullable=False)  # "exploration", "cpu" (formerly "crest")
    status = Column(Text, nullable=False, default="idle")  # "idle", "pes", "generative", "dft", "crest"
    current_task = Column(Text, nullable=True)  # e.g. "compound 42" or "batch 5"
    # For cpu workers: which kind of job is currently being processed.
    # One of: "dft", "crest", "idle". NULL for non-cpu workers.
    current_job_kind = Column(Text, nullable=True)
    started_at = Column(TimestampTZ, nullable=False, default=lambda: datetime.now(timezone.utc))
    last_heartbeat = Column(TimestampTZ, nullable=False, default=lambda: datetime.now(timezone.utc))
    batches_completed = Column(Integer, nullable=False, default=0)
    pes_completed = Column(Integer, nullable=False, default=0)
    total_wall_time_s = Column(Float, nullable=False, default=0.0)
    experiment = Column(Text, nullable=False, default="main")


class CrestWorkQueue(Base):
    __tablename__ = "crest_work_queue"
    __table_args__ = (
        UniqueConstraint("compound_id", name="uq_crest_work_compound"),
        Index("idx_crest_work_pending", "status", postgresql_where="status = 'pending'"),
    )

    id = Column(Integer, primary_key=True)
    compound_id = Column(Integer, ForeignKey("compounds.id"), nullable=False)
    status = Column(String(20), nullable=False, default="pending")
    worker_id = Column(Text, nullable=True)
    claimed_at = Column(TimestampTZ, nullable=True)
    completed_at = Column(TimestampTZ, nullable=True)
    experiment = Column(Text, nullable=False, default="main")

    compound = relationship("Compound")


class CrestResult(Base):
    __tablename__ = "crest_results"
    __table_args__ = (
        UniqueConstraint("compound_id", name="uq_crest_result_compound"),
    )

    id = Column(Integer, primary_key=True)
    compound_id = Column(Integer, ForeignKey("compounds.id"), nullable=False)
    n_conformers = Column(Integer, nullable=False, default=0)
    s_conf = Column(Float, nullable=True)  # Conformational entropy (cal/mol·K)
    conformers_xyz = Column(LargeBinary, nullable=True)  # Raw XYZ file contents
    crest_output = Column(Text, nullable=True)  # Last lines of crest.out for debugging
    charge = Column(Integer, nullable=False, default=0)
    # RMSD comparison vs our PES minima — populated by the cpu-worker as a
    # post-processing step right after CREST completes (or via the backfill
    # admin endpoint for existing rows). Schema: {
    #   "best_rmsds": [float, ...],   # min Kabsch RMSD from each CREST conformer to any of our minima
    #   "threshold": 0.125,           # default match threshold (Å); the frontend slider overrides this
    #   "n_our_minima": int,
    # }
    rmsd_match = Column(JSONColumn, nullable=True)
    created_at = Column(
        TimestampTZ,
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    compound = relationship("Compound")


class ExplorationStats(Base):
    """Per-experiment singleton: one row per experiment, keyed by `experiment`.

    Pre-experiment-tagging this was a single (id=1) singleton. The unique index
    on `experiment` is the upsert key now; primary key is still autoincrement
    so multiple experiments can coexist.
    """
    __tablename__ = "exploration_stats"

    id = Column(Integer, primary_key=True)
    stats_json = Column(JSONColumn, nullable=False, default=dict)
    updated_at = Column(
        TimestampTZ,
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    experiment = Column(Text, nullable=False, unique=True)


class BatchLog(Base):
    __tablename__ = "batch_log"
    __table_args__ = (
        Index("idx_batch_log_created_at", "created_at"),
    )

    id = Column(Integer, primary_key=True)
    batch_idx = Column(Integer, nullable=True)
    summary_json = Column(JSONColumn, nullable=False)
    created_at = Column(
        TimestampTZ,
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    experiment = Column(Text, nullable=False, default="main")


class Annotation(Base):
    __tablename__ = "annotations"
    __table_args__ = (
        UniqueConstraint("entity_type", "entity_key", name="uq_annotation_entity"),
        Index("idx_annotations_entity_type", "entity_type"),
    )

    id = Column(Integer, primary_key=True)
    entity_type = Column(Text, nullable=False)
    entity_key = Column(Text, nullable=False)
    label = Column(Text, nullable=True)
    notes = Column(Text, nullable=True)
    experiments = Column(ExperimentsArray, nullable=False, default=lambda: ["main"])


class SavedLayout(Base):
    __tablename__ = "saved_layouts"

    id = Column(Integer, primary_key=True)
    name = Column(Text, nullable=False, unique=True)
    layout_data = Column(JSONColumn, nullable=False)
    created_at = Column(
        TimestampTZ,
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    updated_at = Column(TimestampTZ, nullable=True)
    experiments = Column(ExperimentsArray, nullable=False, default=lambda: ["main"])


class DftWorkQueue(Base):
    """Work queue for cpu-worker DFT jobs (one per Reaction).

    Auto-enqueued by db.create_reaction(); claimed by cpu-worker in
    barrier_forward ASC order so kinetically relevant reactions are refined first.
    """
    __tablename__ = "dft_work_queue"
    __table_args__ = (
        UniqueConstraint("reaction_id", name="uq_dft_work_reaction"),
        Index("idx_dft_work_pending", "status", postgresql_where="status = 'pending'"),
    )

    id = Column(Integer, primary_key=True)
    reaction_id = Column(Integer, ForeignKey("reactions.id", ondelete="CASCADE"), nullable=False)
    status = Column(String(20), nullable=False, default="pending")
    worker_id = Column(Text, nullable=True)
    claimed_at = Column(TimestampTZ, nullable=True)
    completed_at = Column(TimestampTZ, nullable=True)
    error_msg = Column(Text, nullable=True)
    experiment = Column(Text, nullable=False, default="main")


class KineticsSnapshot(Base):
    """Cached output of the kinetics solver loop running in the API container.

    The kinetics solver runs as a background asyncio task (advisory-locked
    singleton across Cloud Run instances), polls every ~60s, and writes a new
    row when the network has materially changed since the last solve.
    """
    __tablename__ = "kinetics_snapshots"
    __table_args__ = (
        Index("idx_kinetics_snapshots_recent", "computed_at"),
    )

    id = Column(Integer, primary_key=True)
    network_version = Column(Integer, nullable=False)  # COUNT(reactions WHERE discovery_method != 'manual_equilibrium')
    n_reactions_dft = Column(Integer, nullable=False, default=0)  # of which N had PBE0 separated barriers
    temperature = Column(Float, nullable=False)
    payload_jsonb = Column(JSONColumn, nullable=False)  # serialized KineticsSnapshot dataclass
    solve_wall_time_s = Column(Float, nullable=True)
    computed_at = Column(
        TimestampTZ,
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    experiment = Column(Text, nullable=False, default="main")
