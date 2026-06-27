"""Database access layer for CRN exploration workers.

Replaces pickle + file lock with PostgreSQL queries. Provides CRUD operations
for compounds, minima, transition states, reactions, and work queue management.
"""

import os
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Optional

import numpy as np
from loguru import logger
from sqlalchemy import create_engine, text, update, select, func
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from packages.db.models import (
    Annotation,
    Base,
    BatchLog,
    Compound as CompoundRow,
    CrestWorkQueue,
    DftWorkQueue,
    ExplorationStats,
    GraphEdge,
    IntraTransitionState,
    Minimum,
    PESWorkQueue,
    Reaction as ReactionRow,
    ReactionProduct,
    ReactionReactant,
)
from packages.db.serialization import (
    serialize_ndarray,
    deserialize_ndarray,
    serialize_ndarray_optional,
    deserialize_ndarray_optional,
    serialize_trajectory,
    deserialize_trajectory,
)


class DB:
    """Database access layer for worker processes.

    Every instance is bound to a single ``experiment`` string. All inserts
    are tagged with this experiment; all queue reads (claim_pes_work,
    backlog counts, exploration completion checks) are filtered by it.
    Compounds are multi-experiment (TEXT[]) — when an existing compound is
    re-encountered, the constructor's experiment is appended to its
    `experiments` array if not already present.
    """

    # Timeout for stale in-progress work items (seconds).
    # Workers handle SIGTERM gracefully and release work back to pending,
    # so this is a fallback for hard kills (OOM, node failure).
    WORK_TIMEOUT_S = 3600  # 1 hour (fallback — workers actively free allocs on completion)

    def __init__(
        self,
        database_url: str | None = None,
        pool_size: int = 2,
        experiment: str | None = None,
    ):
        url = database_url or os.environ["DATABASE_URL"]
        self.engine = create_engine(
            url,
            pool_size=pool_size,
            max_overflow=3,
            pool_pre_ping=True,
        )
        self._session_factory = sessionmaker(bind=self.engine)
        self.worker_id = f"worker-{uuid.uuid4().hex[:8]}"
        # `experiment` is required; we accept Optional only so the DB can be
        # constructed for ad-hoc tooling that explicitly supplies one. The
        # worker entrypoint must pass it; bare DB() calls will fail loudly
        # the first time a tagging operation runs.
        self.experiment = experiment

    @contextmanager
    def session(self):
        """Provide a transactional session scope."""
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def init_db(self):
        """Create all tables (for development/testing)."""
        try:
            Base.metadata.create_all(self.engine)
        except Exception as e:
            logger.debug(f"init_db DDL race (harmless): {e}")

    # =========================================================================
    # Compound CRUD
    # =========================================================================

    def get_or_create_compound(
        self,
        session: Session,
        smiles: str,
        formula: str,
        charge: int,
        n_atoms: int,
        sorted_atomic_numbers: np.ndarray,
        is_seed: bool = False,
        frontier: bool = False,
    ) -> tuple[int, bool]:
        """Get existing compound by SMILES or create new one.

        Returns (compound_id, is_new). Tags the row with `self.experiment`:
        new rows get `experiments=[self.experiment]`; existing rows get the
        experiment appended to their array if not already present (so a
        compound first discovered under main is also visible under a
        secondary experiment that re-encounters it).

        `frontier=True` (used by closed-subgraph workers when a reaction
        produces a compound outside the experiment's curated explorable
        set) additionally appends `self.experiment` to the row's
        `frontier_in` array and skips CREST enqueueing. Sampling queries
        filter on `~frontier_in.any(experiment)`, so the compound stays
        visible in the experiment's graph view but is never picked as a
        sampling/exploration starting point.
        """
        row = session.query(CompoundRow).filter(CompoundRow.smiles == smiles).first()
        if row is not None:
            if is_seed and not row.is_seed:
                row.is_seed = True
            if self.experiment and self.experiment not in (row.experiments or []):
                row.experiments = list(row.experiments or []) + [self.experiment]
                # Only mark as frontier when the experiment is being added
                # for the first time. An already-explorable compound stays
                # explorable even if a later worker re-encounters it as a
                # product (matching the "be conservative about scope
                # changes" contract).
                if frontier and self.experiment not in (row.frontier_in or []):
                    row.frontier_in = list(row.frontier_in or []) + [self.experiment]
            return row.id, False

        row = CompoundRow(
            smiles=smiles,
            formula=formula,
            charge=charge,
            n_atoms=n_atoms,
            sorted_atomic_numbers=serialize_ndarray(sorted_atomic_numbers),
            is_seed=is_seed,
            experiments=[self.experiment] if self.experiment else [],
            frontier_in=[self.experiment] if (frontier and self.experiment) else [],
        )
        nested = session.begin_nested()
        try:
            session.add(row)
            nested.commit()
        except IntegrityError:
            nested.rollback()
            session.expire_all()
            row = session.query(CompoundRow).filter(CompoundRow.smiles == smiles).first()
            if self.experiment and self.experiment not in (row.experiments or []):
                row.experiments = list(row.experiments or []) + [self.experiment]
                if frontier and self.experiment not in (row.frontier_in or []):
                    row.frontier_in = list(row.frontier_in or []) + [self.experiment]
            return row.id, False

        # Auto-enqueue CREST conformer search for any new compound with enough
        # atoms to actually have rotatable bonds. Single atoms / diatomics have
        # no conformers to find — skip them. Frontier compounds skip CREST too:
        # they're terminal nodes for this experiment, not worth conformer-search
        # compute. CrestWorkQueue uniqueness on compound_id makes this
        # idempotent across worker races.
        if n_atoms > 2 and not frontier:
            crest_nested = session.begin_nested()
            try:
                session.add(CrestWorkQueue(
                    compound_id=row.id,
                    status="pending",
                    experiment=self.experiment,
                ))
                crest_nested.commit()
            except IntegrityError:
                crest_nested.rollback()  # already enqueued — fine

        return row.id, True

    def get_compound_by_smiles(self, session: Session, smiles: str) -> Optional[CompoundRow]:
        """Look up a compound by SMILES, scoped to this DB's experiment.

        Returns None if the compound exists but isn't tagged with this
        experiment — defense-in-depth so any code path that resolves a
        SMILES into a compound row can't accidentally cross experiments.
        """
        q = session.query(CompoundRow).filter(CompoundRow.smiles == smiles)
        if self.experiment:
            q = q.filter(CompoundRow.experiments.any(self.experiment))
        return q.first()

    def get_compound_count(self, session: Session) -> int:
        """Count compounds in this DB's experiment scope.

        Used as the exploration progress metric — workers stop when this
        hits MAX_VALID_NODES. Scoped so two experiments don't see each
        other's progress.
        """
        q = session.query(func.count(CompoundRow.id))
        if self.experiment:
            q = q.filter(CompoundRow.experiments.any(self.experiment))
        return q.scalar()

    # =========================================================================
    # Minimum CRUD
    # =========================================================================

    def next_minimum_local_id(self, session: Session, compound_id: int) -> int:
        """Return the next sequential positive local_id for a compound's minima."""
        max_id = (
            session.query(func.max(Minimum.local_id))
            .filter(Minimum.compound_id == compound_id, Minimum.local_id >= 0)
            .scalar()
        )
        # Explicit `is None` — `(max_id or -1)` is wrong because max_id=0 is
        # falsy and would short-circuit to -1, returning 0 (collision with
        # the existing zeroth minimum) instead of 1.
        return (-1 if max_id is None else int(max_id)) + 1

    def next_temp_local_id(self, session: Session, compound_id: int) -> int:
        """Return the next negative temp local_id for a compound's minima.

        Temp minima use negative local_ids (-1, -2, ...) to avoid collisions
        with PES exploration's positive sequential IDs. They're resolved by
        dedup PES jobs into either merges or promotions to positive IDs.
        """
        min_neg = (
            session.query(func.min(Minimum.local_id))
            .filter(Minimum.compound_id == compound_id, Minimum.local_id < 0)
            .scalar()
        )
        return (min_neg or 0) - 1

    def add_minimum(
        self,
        session: Session,
        compound_id: int,
        local_id: Optional[int],
        positions: np.ndarray,
        energy: float,
        hessian: Optional[np.ndarray] = None,
        explored: bool = False,
        discovery_timestamp: float = 0.0,
        name: str = "",
        n_merged: int = 0,
        max_merge_rmsd: float = 0.0,
        enqueue_pes: bool = True,
    ) -> tuple[int, bool]:
        """Add a minimum to a compound's PES. Returns (minimum_db_id, is_new).

        If local_id is None, auto-allocates the next sequential ID with a
        retry loop to handle concurrent workers racing on the same compound.
        If local_id is provided (e.g., from save_pes_results where PESGraph
        IDs == DB local_ids), uses it directly.

        If a minimum with this (compound_id, local_id) already exists, enriches
        it with any new data (hessian, merge stats) and returns (existing_id, False).
        """
        energy = float(energy) if energy is not None else None
        max_merge_rmsd = float(max_merge_rmsd)
        discovery_timestamp = float(discovery_timestamp)

        def _try_insert(lid: int) -> tuple[Optional[int], bool]:
            """Attempt insert with given local_id. Returns (db_id, is_new) or (None, False) on collision."""
            existing = (
                session.query(Minimum)
                .filter(Minimum.compound_id == compound_id, Minimum.local_id == lid)
                .first()
            )
            if existing is not None:
                if hessian is not None and existing.hessian is None:
                    existing.hessian = serialize_ndarray_optional(hessian)
                if n_merged > existing.n_merged:
                    existing.n_merged = n_merged
                    existing.max_merge_rmsd = max(max_merge_rmsd, existing.max_merge_rmsd)
                if name and not existing.name:
                    existing.name = name
                return existing.id, False

            row = Minimum(
                compound_id=compound_id,
                local_id=lid,
                positions=serialize_ndarray(positions),
                energy=energy,
                hessian=serialize_ndarray_optional(hessian),
                explored=explored,
                discovery_timestamp=discovery_timestamp,
                name=name,
                n_merged=n_merged,
                max_merge_rmsd=max_merge_rmsd,
                experiments=[self.experiment] if self.experiment else [],
            )
            nested = session.begin_nested()
            try:
                session.add(row)
                nested.commit()
            except IntegrityError:
                nested.rollback()
                session.expire_all()
                return None, False  # signal retry

            # Cache invalidation: Compound.energy_pbe0 caches the PBE0 of
            # whatever was the lowest-E minimum at fill time. If this newly
            # inserted minimum is now the new lowest-E, the cache is stale —
            # NULL it so the next DFT job touching this compound recomputes
            # against the correct reference geometry.
            if energy is not None:
                other_min_energy = (
                    session.query(func.min(Minimum.energy))
                    .filter(
                        Minimum.compound_id == compound_id,
                        Minimum.id != row.id,
                    )
                    .scalar()
                )
                if other_min_energy is None or energy < other_min_energy:
                    session.query(CompoundRow).filter(
                        CompoundRow.id == compound_id,
                        CompoundRow.energy_pbe0.isnot(None),
                    ).update(
                        {
                            "energy_pbe0": None,
                            "energy_pbe0_method": None,
                            "energy_pbe0_at": None,
                        },
                        synchronize_session=False,
                    )

            # Enqueue PES work for new unexplored minima (> 2 atoms).
            # `enqueue_pes=False` is used by callers registering a frontier
            # minimum (a product of a closed-subgraph reaction we don't
            # want to spend further GPU on). In that case we mark the row
            # `explored=True` so the dashboard's unexplored counter
            # doesn't tick for an item that will never be claimed.
            n_atoms = session.query(CompoundRow.n_atoms).filter(CompoundRow.id == compound_id).scalar()
            if not explored and n_atoms is not None and n_atoms > 2 and enqueue_pes:
                pes_nested = session.begin_nested()
                try:
                    session.add(PESWorkQueue(
                        compound_id=compound_id,
                        minimum_id=row.id,
                        status="pending",
                        experiment=self.experiment,
                    ))
                    pes_nested.commit()
                except IntegrityError:
                    pes_nested.rollback()
            elif not explored and n_atoms is not None and (n_atoms <= 2 or not enqueue_pes):
                row.explored = True

            return row.id, True

        if local_id is not None:
            # Caller specified an exact local_id (save_pes_results path)
            result_id, is_new = _try_insert(local_id)
            if result_id is not None:
                return result_id, is_new
            # Collision on specified local_id — re-query the existing row
            existing = (
                session.query(Minimum)
                .filter(Minimum.compound_id == compound_id, Minimum.local_id == local_id)
                .first()
            )
            return (existing.id if existing else 0), False

        # Auto-allocate local_id with retry loop for concurrent workers
        for _attempt in range(10):
            lid = self.next_minimum_local_id(session, compound_id)
            result_id, is_new = _try_insert(lid)
            if result_id is not None:
                return result_id, is_new
            # Another worker grabbed this local_id — retry with fresh max+1
            logger.debug(f"local_id {lid} collision for compound {compound_id}, retrying")
        raise RuntimeError(f"Failed to allocate local_id for compound {compound_id} after 10 attempts")

    def add_temp_minimum(
        self,
        session: Session,
        compound_id: int,
        positions: np.ndarray,
        energy: float,
        discovery_timestamp: float = 0.0,
        name: str = "",
        enqueue_dedup_pes: bool = True,
    ) -> tuple[int, int]:
        """Add a temporary minimum with negative local_id for dedup.

        Used by register_compound when adding a conformer to an EXISTING
        compound. The temp minimum is resolved later by a dedup PES job
        (which has the full PES graph + NEB calculator to properly dedup).

        Returns (minimum_db_id, temp_local_id).
        """
        from packages.db.serialization import serialize_ndarray
        energy = float(energy) if energy is not None else None
        discovery_timestamp = float(discovery_timestamp)

        for _attempt in range(10):
            lid = self.next_temp_local_id(session, compound_id)
            row = Minimum(
                compound_id=compound_id,
                local_id=lid,
                positions=serialize_ndarray(positions),
                energy=energy,
                explored=True,  # temp minima are not explored directly
                discovery_timestamp=discovery_timestamp,
                name=name,
                experiments=[self.experiment] if self.experiment else [],
            )
            nested = session.begin_nested()
            try:
                session.add(row)
                nested.commit()
            except IntegrityError:
                nested.rollback()
                session.expire_all()
                logger.debug(f"temp local_id {lid} collision for compound {compound_id}, retrying")
                continue

            # Enqueue dedup PES job (skipped for frontier minima — products
            # of closed-subgraph reactions we don't want to spend GPU on).
            if enqueue_dedup_pes:
                dedup_nested = session.begin_nested()
                try:
                    session.add(PESWorkQueue(
                        compound_id=compound_id,
                        minimum_id=row.id,
                        status="pending",
                        job_kind="dedup",
                        experiment=self.experiment,
                    ))
                    dedup_nested.commit()
                except IntegrityError:
                    dedup_nested.rollback()

            return row.id, lid

        raise RuntimeError(f"Failed to allocate temp local_id for compound {compound_id}")

    def resolve_temp_minimum_as_duplicate(
        self,
        session: Session,
        compound_id: int,
        temp_min_db_id: int,
        temp_local_id: int,
        canonical_local_id: int,
    ):
        """Resolve a temp minimum as a duplicate of an existing one.

        Updates all reaction references from temp_local_id to canonical_local_id,
        then deletes the PES work queue entry and the temp minimum row.
        """
        from packages.db.models import ReactionReactant, ReactionProduct

        # Update reaction references
        session.query(ReactionReactant).filter(
            ReactionReactant.compound_id == compound_id,
            ReactionReactant.conformer_local_id == temp_local_id,
        ).update({"conformer_local_id": canonical_local_id})

        session.query(ReactionProduct).filter(
            ReactionProduct.compound_id == compound_id,
            ReactionProduct.conformer_local_id == temp_local_id,
        ).update({"conformer_local_id": canonical_local_id})

        # Delete PES work queue entry (FK to minima.id)
        session.query(PESWorkQueue).filter(
            PESWorkQueue.minimum_id == temp_min_db_id,
        ).delete()

        # Delete temp minimum
        session.query(Minimum).filter(
            Minimum.id == temp_min_db_id,
        ).delete()

    def promote_temp_minimum(
        self,
        session: Session,
        compound_id: int,
        temp_min_db_id: int,
        temp_local_id: int,
    ) -> int:
        """Promote a temp minimum to a positive local_id (genuinely new conformer).

        Assigns next positive local_id, updates reaction references,
        and enqueues an explore PES job.

        Returns the new positive local_id.

        Wraps the minimum UPDATE in a savepoint + retry: under load we
        otherwise hit `uq_minima_compound_local` violations (separate-
        session reads of `next_minimum_local_id` can race against
        concurrent sibling promotes / explore-side `add_minimum` inserts
        for the same compound). Same retry pattern as `add_minimum`.
        """
        from packages.db.models import ReactionReactant, ReactionProduct

        new_lid: Optional[int] = None
        for _attempt in range(10):
            candidate = self.next_minimum_local_id(session, compound_id)
            nested = session.begin_nested()
            try:
                session.query(Minimum).filter(
                    Minimum.id == temp_min_db_id,
                ).update({"local_id": candidate, "explored": False},
                         synchronize_session=False)
                nested.commit()
                new_lid = candidate
                break
            except IntegrityError:
                nested.rollback()
                session.expire_all()
                logger.debug(
                    f"promote_temp_minimum: local_id {candidate} collision for "
                    f"compound {compound_id} (temp_min_db_id={temp_min_db_id}), retrying"
                )
        if new_lid is None:
            raise RuntimeError(
                f"Failed to promote temp minimum {temp_min_db_id} of compound "
                f"{compound_id} after 10 attempts"
            )

        # Update reaction references
        session.query(ReactionReactant).filter(
            ReactionReactant.compound_id == compound_id,
            ReactionReactant.conformer_local_id == temp_local_id,
        ).update({"conformer_local_id": new_lid})

        session.query(ReactionProduct).filter(
            ReactionProduct.compound_id == compound_id,
            ReactionProduct.conformer_local_id == temp_local_id,
        ).update({"conformer_local_id": new_lid})

        # Update the existing dedup PES job to an explore job
        session.query(PESWorkQueue).filter(
            PESWorkQueue.minimum_id == temp_min_db_id,
        ).update({"job_kind": "explore", "status": "pending",
                  "worker_id": None, "claimed_at": None})

        return new_lid

    def mark_minimum_explored(self, session: Session, minimum_db_id: int):
        """Mark a minimum as explored."""
        session.query(Minimum).filter(Minimum.id == minimum_db_id).update(
            {"explored": True}
        )

    def get_minima_for_compound(self, session: Session, compound_id: int) -> list[Minimum]:
        return session.query(Minimum).filter(Minimum.compound_id == compound_id).all()

    def get_unexplored_minima_for_compound(self, session: Session, compound_id: int) -> list[Minimum]:
        return (
            session.query(Minimum)
            .filter(Minimum.compound_id == compound_id, Minimum.explored == False)
            .all()
        )

    # =========================================================================
    # Intra-molecular Transition State CRUD
    # =========================================================================

    def add_intra_ts(
        self,
        session: Session,
        compound_id: int,
        local_id: int,
        positions: np.ndarray,
        energy: float,
        eigenvalue: float,
        min_fwd_db_id: int,
        min_bwd_db_id: int,
        barrier_fwd: float,
        barrier_bwd: float,
        hessian: Optional[np.ndarray] = None,
        rmsd_to_fwd_min: float = 0.0,
        rmsd_to_bwd_min: float = 0.0,
        endpoint_to_endpoint_rmsd: float = 0.0,
        fwd_trajectory=None,
        bwd_trajectory=None,
        discovery_timestamp: float = 0.0,
    ) -> tuple[int, bool]:
        """Add an intramolecular TS. Returns (ts_db_id, is_new)."""
        # Coerce numpy scalars — psycopg2 stringifies np.float64 as
        # "np.float64(...)" which postgres parses as schema.column.
        energy = float(energy) if energy is not None else 0.0
        eigenvalue = float(eigenvalue)
        barrier_fwd = float(barrier_fwd)
        barrier_bwd = float(barrier_bwd)
        rmsd_to_fwd_min = float(rmsd_to_fwd_min)
        rmsd_to_bwd_min = float(rmsd_to_bwd_min)
        endpoint_to_endpoint_rmsd = float(endpoint_to_endpoint_rmsd)
        discovery_timestamp = float(discovery_timestamp)
        # Coerce numpy scalars to plain Python floats — psycopg2 stringifies
        # np.float64 as "np.float64(...)" which postgres can't parse.
        energy = float(energy) if energy is not None else 0.0
        eigenvalue = float(eigenvalue)
        barrier_fwd = float(barrier_fwd)
        barrier_bwd = float(barrier_bwd)
        rmsd_to_fwd_min = float(rmsd_to_fwd_min)
        rmsd_to_bwd_min = float(rmsd_to_bwd_min)
        endpoint_to_endpoint_rmsd = float(endpoint_to_endpoint_rmsd)
        discovery_timestamp = float(discovery_timestamp)
        existing = (
            session.query(IntraTransitionState)
            .filter(
                IntraTransitionState.compound_id == compound_id,
                IntraTransitionState.local_id == local_id,
            )
            .first()
        )
        if existing is not None:
            # Enrich existing TS with hessian if not yet stored
            if hessian is not None and existing.hessian is None:
                existing.hessian = serialize_ndarray_optional(hessian)
            return existing.id, False

        row = IntraTransitionState(
            compound_id=compound_id,
            local_id=local_id,
            positions=serialize_ndarray(positions),
            energy=energy,
            eigenvalue=eigenvalue,
            hessian=serialize_ndarray_optional(hessian),
            min_fwd_id=min_fwd_db_id,
            min_bwd_id=min_bwd_db_id,
            barrier_fwd=barrier_fwd,
            barrier_bwd=barrier_bwd,
            rmsd_to_fwd_min=rmsd_to_fwd_min,
            rmsd_to_bwd_min=rmsd_to_bwd_min,
            endpoint_to_endpoint_rmsd=endpoint_to_endpoint_rmsd,
            fwd_trajectory=serialize_trajectory(fwd_trajectory),
            bwd_trajectory=serialize_trajectory(bwd_trajectory),
            discovery_timestamp=discovery_timestamp,
            experiments=[self.experiment] if self.experiment else [],
        )
        nested = session.begin_nested()
        try:
            session.add(row)
            nested.commit()
        except IntegrityError:
            nested.rollback()
            session.expire_all()
            existing = (
                session.query(IntraTransitionState)
                .filter(
                    IntraTransitionState.compound_id == compound_id,
                    IntraTransitionState.local_id == local_id,
                )
                .first()
            )
            return existing.id, False
        return row.id, True

    def get_intra_ts_for_compound(self, session: Session, compound_id: int) -> list[IntraTransitionState]:
        return (
            session.query(IntraTransitionState)
            .filter(IntraTransitionState.compound_id == compound_id)
            .all()
        )

    # =========================================================================
    # Reaction CRUD
    # =========================================================================

    def get_reaction_by_ts_id(self, session: Session, ts_id: int) -> Optional[ReactionRow]:
        # defer the 4 LargeBinary blobs — callers typically only read id or
        # a few scalar attrs; blob columns will lazy-load if actually used.
        from sqlalchemy.orm import defer
        return (
            session.query(ReactionRow)
            .options(
                defer(ReactionRow.ts_conformer_positions),
                defer(ReactionRow.ts_conformer_atomic_numbers),
                defer(ReactionRow.reactant_trajectory),
                defer(ReactionRow.product_trajectory),
            )
            .filter(ReactionRow.ts_id == ts_id)
            .first()
        )

    def create_reaction(
        self,
        session: Session,
        ts_id: int,
        ts_conformer_positions: np.ndarray,
        ts_conformer_atomic_numbers: np.ndarray,
        ts_conformer_charge: int,
        ts_energy: float,
        barrier_forward: float,
        barrier_backward: float,
        reactant_compound_ids: list[tuple[int, Optional[int]]],  # [(compound_id, conformer_local_id)]
        product_compound_ids: list[tuple[int, int, float]],  # [(compound_id, conformer_local_id, energy)]
        reactant_trajectory=None,
        product_trajectory=None,
        discovery_method: Optional[str] = None,
        discovery_noise_level: Optional[int] = None,
        discovery_timestamp: Optional[float] = None,
        name: str = "",
        # ML barrier variants computed by worker at reaction creation.
        # Separated = TS - sum(reference-conformer energies); the principled
        # choice for kinetics. Ex = TS - min(E along IRC trajectory side);
        # weaker fix for the same long-distance ML artifact issue.
        barrier_forward_separated: Optional[float] = None,
        barrier_backward_separated: Optional[float] = None,
        barrier_forward_ex: Optional[float] = None,
        barrier_backward_ex: Optional[float] = None,
    ) -> tuple[int, bool]:
        """Create a reaction. Returns (reaction_db_id, is_new).

        If a reaction with this ts_id already exists, adds new products to it
        (for fragmentation) and returns (existing_id, False).
        """
        # Defensive: coerce numpy scalars (np.float64 from force-integration
        # code) to plain Python floats. psycopg2 stringifies np.float64 as
        # "np.float64(...)" which the postgres parser reads as schema.column
        # → InvalidSchemaName error.
        def _f(x):
            return float(x) if x is not None else None
        ts_energy = _f(ts_energy)
        barrier_forward = _f(barrier_forward)
        barrier_backward = _f(barrier_backward)
        barrier_forward_separated = _f(barrier_forward_separated)
        barrier_backward_separated = _f(barrier_backward_separated)
        barrier_forward_ex = _f(barrier_forward_ex)
        barrier_backward_ex = _f(barrier_backward_ex)
        # product_compound_ids contains (compound_id, conformer_local_id, energy) — coerce energy
        product_compound_ids = [(cid, clid, _f(e)) for cid, clid, e in product_compound_ids]
        existing = self.get_reaction_by_ts_id(session, ts_id)
        if existing is not None:
            # Add new products to existing reaction (fragmentation case)
            for compound_id, conformer_local_id, energy in product_compound_ids:
                product = ReactionProduct(
                    reaction_id=existing.id,
                    compound_id=compound_id,
                    conformer_local_id=conformer_local_id,
                    energy=energy,
                )
                session.add(product)
            # Update barrier_backward
            total_product_energy = sum(p.energy for p in existing.products) + sum(
                e for _, _, e in product_compound_ids
            )
            existing.barrier_backward = ts_energy - total_product_energy
            return existing.id, False

        row = ReactionRow(
            ts_id=ts_id,
            ts_conformer_positions=serialize_ndarray(ts_conformer_positions),
            ts_conformer_atomic_numbers=serialize_ndarray(ts_conformer_atomic_numbers),
            ts_conformer_charge=ts_conformer_charge,
            ts_energy=ts_energy,
            barrier_forward=barrier_forward,
            barrier_backward=barrier_backward,
            barrier_forward_separated=barrier_forward_separated,
            barrier_backward_separated=barrier_backward_separated,
            barrier_forward_ex=barrier_forward_ex,
            barrier_backward_ex=barrier_backward_ex,
            reactant_trajectory=serialize_trajectory(reactant_trajectory),
            product_trajectory=serialize_trajectory(product_trajectory),
            discovery_method=discovery_method,
            discovery_noise_level=discovery_noise_level,
            discovery_timestamp=discovery_timestamp,
            name=name,
            experiments=[self.experiment] if self.experiment else [],
        )
        nested = session.begin_nested()
        try:
            session.add(row)
            nested.commit()
        except IntegrityError:
            nested.rollback()
            session.expire_all()
            existing = self.get_reaction_by_ts_id(session, ts_id)
            return existing.id, False

        # Add reactants
        for compound_id, conformer_local_id in reactant_compound_ids:
            reactant = ReactionReactant(
                reaction_id=row.id,
                compound_id=compound_id,
                conformer_local_id=conformer_local_id,
            )
            session.add(reactant)

        # Add products
        for compound_id, conformer_local_id, energy in product_compound_ids:
            product = ReactionProduct(
                reaction_id=row.id,
                compound_id=compound_id,
                conformer_local_id=conformer_local_id,
                energy=energy,
            )
            session.add(product)

        # Auto-enqueue DFT work for this reaction so the cpu-worker fleet can
        # refine its barriers with PBE0 single-points. Skip manual equilibria
        # (they use literal rate constants, not Eyring-from-barriers) — and
        # skip on UNIQUE collision in case of a race / replay.
        if discovery_method != "manual_equilibrium":
            dft_nested = session.begin_nested()
            try:
                session.add(DftWorkQueue(
                    reaction_id=row.id,
                    status="pending",
                    experiment=self.experiment,
                ))
                dft_nested.commit()
            except IntegrityError:
                dft_nested.rollback()  # already enqueued — fine

        return row.id, True

    # =========================================================================
    # Graph Edges
    # =========================================================================

    def add_graph_edge(
        self,
        session: Session,
        source_node: str,
        target_node: str,
        source_type: str,
        target_type: str,
        direction: Optional[str] = None,
        stoichiometry: int = 1,
        energy_diff: Optional[float] = None,
        reaction_id: Optional[int] = None,
    ):
        """Add a graph edge for the viewer."""
        edge = GraphEdge(
            source_node=source_node,
            target_node=target_node,
            source_type=source_type,
            target_type=target_type,
            direction=direction,
            stoichiometry=stoichiometry,
            energy_diff=energy_diff,
            reaction_id=reaction_id,
            experiments=[self.experiment] if self.experiment else [],
        )
        session.add(edge)

    # =========================================================================
    # Work Queue (PES Exploration)
    # =========================================================================

    def claim_pes_work(
        self,
        session: Session,
        min_compound_age_s: float = 0.0,
        conc_lookup: Optional[dict] = None,
        min_conc: float = 0.0,
        job_kind_filter: Optional[str] = None,
    ) -> Optional[dict]:
        """Claim a pending PES work item using SELECT ... FOR UPDATE SKIP LOCKED.

        Skips compounds that already have an in_progress item, so only one
        worker operates on a given compound at a time (prevents local_id
        collisions and dedup races).

        Ordering: FIFO by queue-id within compounds. The compound's
        `created_at` (first ever sighting) gates eligibility via
        `min_compound_age_s` so newly-discovered compounds have time to
        appear in the kinetics snapshot before we spend GPU on them.

        Concentration gate: if `min_conc > 0` and `conc_lookup` is supplied,
        eligible rows are filtered Python-side to require
        `conc_lookup.get(compound.smiles, 0.0) >= min_conc`. If the lookup
        is None or empty (e.g. kinetics snapshot not ready), the filter is
        not applied — falls back to previous behavior (take any eligible
        item). We scan up to `_PES_CLAIM_SCAN_LIMIT` candidates to find one
        that passes the concentration gate before giving up for this poll.

        Returns dict with compound_id, minimum_id, work_id or None.
        """
        timeout_threshold = datetime.now(timezone.utc).timestamp() - self.WORK_TIMEOUT_S

        # Wait-gate on compound age: compound.created_at is a TimestampTZ.
        age_clause = ""
        if min_compound_age_s > 0.0:
            age_clause = "AND c.created_at < now() - make_interval(secs => :max_age_s)"

        gate_on = bool(conc_lookup and min_conc > 0.0)

        # Optional job_kind restriction (e.g. dedup-only workers).
        kind_clause = ""
        if job_kind_filter:
            kind_clause = "AND q.job_kind = :job_kind_filter"

        if not gate_on:
            # Fast path: single-row locking claim, identical to pre-gate behavior.
            # Scoped to this worker's experiment — workers across experiments
            # never see each other's queue rows.
            sql = f"""
                UPDATE pes_work_queue
                   SET status = 'in_progress',
                       worker_id = :worker_id,
                       claimed_at = now()
                 WHERE id = (
                     SELECT q.id FROM pes_work_queue q
                       JOIN compounds c ON c.id = q.compound_id
                      WHERE q.experiment = :experiment
                        AND (q.status = 'pending'
                             OR (q.status = 'in_progress'
                                 AND q.claimed_at < to_timestamp(:timeout_threshold)))
                        AND NOT EXISTS (
                            SELECT 1 FROM pes_work_queue other
                             WHERE other.compound_id = q.compound_id
                               AND other.experiment = :experiment
                               AND other.status = 'in_progress'
                               AND other.claimed_at >= to_timestamp(:timeout_threshold)
                        )
                        {age_clause}
                        {kind_clause}
                      ORDER BY q.id
                      LIMIT 1
                      -- Per-compound serialization: locking BOTH q and c
                      -- + SKIP LOCKED means a second worker skips a
                      -- candidate when its compound row is locked by a
                      -- sibling claim. This is *semantically* required
                      -- for dedup — two workers running NEB on two temp
                      -- minima of the same compound only check against
                      -- the current positives, never against each other,
                      -- so genuinely-duplicate temps would both get
                      -- promoted as "new" minima. (The NOT EXISTS in-
                      -- progress clause above doesn't prevent this on
                      -- its own: under READ COMMITTED two concurrent
                      -- claims for the same compound can both pass it.)
                      FOR UPDATE OF q, c SKIP LOCKED
                 )
                 RETURNING id, compound_id, minimum_id, job_kind
            """
            params = {
                "worker_id": self.worker_id,
                "timeout_threshold": timeout_threshold,
                "experiment": self.experiment,
            }
            if min_compound_age_s > 0.0:
                params["max_age_s"] = float(min_compound_age_s)
            if job_kind_filter:
                params["job_kind_filter"] = job_kind_filter
            result = session.execute(text(sql), params)
            row = result.fetchone()
            if row is None:
                return None
            return {"work_id": row[0], "compound_id": row[1], "minimum_id": row[2],
                    "job_kind": row[3] if len(row) > 3 else "explore"}

        # Gated path: scan the full eligible-pending set (no LIMIT — the
        # scan was previously capped at 2000, which starved PES whenever
        # the queue head was dominated by old dead-end compounds while
        # the eligible ones sat further back).  No row locks; the narrow
        # UPDATE below handles races.  We need every eligible row visible
        # because the conc-gate sort+filter happens Python-side — a
        # LIMIT here silently drops candidates that would have passed
        # the gate.
        scan_sql = f"""
            SELECT q.id, c.smiles
              FROM pes_work_queue q
              JOIN compounds c ON c.id = q.compound_id
             WHERE q.experiment = :experiment
               AND (q.status = 'pending'
                    OR (q.status = 'in_progress'
                        AND q.claimed_at < to_timestamp(:timeout_threshold)))
               AND NOT EXISTS (
                   SELECT 1 FROM pes_work_queue other
                    WHERE other.compound_id = q.compound_id
                      AND other.experiment = :experiment
                      AND other.status = 'in_progress'
                      AND other.claimed_at >= to_timestamp(:timeout_threshold)
               )
               {age_clause}
               {kind_clause}
             ORDER BY q.id
        """
        scan_params = {
            "timeout_threshold": timeout_threshold,
            "experiment": self.experiment,
        }
        if min_compound_age_s > 0.0:
            scan_params["max_age_s"] = float(min_compound_age_s)
        if job_kind_filter:
            scan_params["job_kind_filter"] = job_kind_filter

        scan_rows = session.execute(text(scan_sql), scan_params).fetchall()
        if not scan_rows:
            return None

        # Narrow claim: UPDATE by id AND status='pending' (plus the timeout
        # equivalent for stuck in_progress). Returns nothing if another worker
        # grabbed it in the meantime.
        claim_sql = """
            UPDATE pes_work_queue
               SET status = 'in_progress',
                   worker_id = :worker_id,
                   claimed_at = now()
             WHERE id = :wid
               AND experiment = :experiment
               AND (status = 'pending'
                    OR (status = 'in_progress'
                        AND claimed_at < to_timestamp(:timeout_threshold)))
               AND NOT EXISTS (
                   SELECT 1 FROM pes_work_queue other
                    WHERE other.compound_id = (SELECT compound_id FROM pes_work_queue WHERE id = :wid)
                      AND other.experiment = :experiment
                      AND other.status = 'in_progress'
                      AND other.claimed_at >= to_timestamp(:timeout_threshold)
                      AND other.id != :wid
               )
         RETURNING id, compound_id, minimum_id, job_kind
        """

        # Sort by concentration descending — explore highest-conc compounds first.
        # Compounds absent from snapshot get -inf so they're always last.
        scan_rows_sorted = sorted(
            scan_rows,
            key=lambda row: float(conc_lookup.get(row[1], float("-inf"))),
            reverse=True,
        )
        for wid, smi in scan_rows_sorted:
            if float(conc_lookup.get(smi, 0.0)) < min_conc:
                break  # sorted desc, so all remaining are below threshold
            res = session.execute(text(claim_sql), {
                "worker_id": self.worker_id,
                "wid": wid,
                "timeout_threshold": timeout_threshold,
                "experiment": self.experiment,
            })
            claimed = res.fetchone()
            if claimed is None:
                continue  # raced with another worker; try next eligible
            return {
                "work_id": claimed[0], "compound_id": claimed[1],
                "minimum_id": claimed[2],
                "job_kind": claimed[3] if len(claimed) > 3 else "explore",
            }
        return None

    def complete_pes_work(self, session: Session, work_id: int):
        """Mark a PES work item as completed."""
        session.query(PESWorkQueue).filter(PESWorkQueue.id == work_id).update(
            {
                "status": "completed",
                "completed_at": datetime.now(timezone.utc),
            }
        )

    def fail_pes_work(self, session: Session, work_id: int):
        """Mark a PES work item as failed."""
        session.query(PESWorkQueue).filter(PESWorkQueue.id == work_id).update(
            {
                "status": "failed",
                "completed_at": datetime.now(timezone.utc),
            }
        )

    def release_pes_work(self, session: Session, work_id: int):
        """Release a claimed PES work item back to pending (e.g. on spot preemption)."""
        session.query(PESWorkQueue).filter(PESWorkQueue.id == work_id).update(
            {
                "status": "pending",
                "worker_id": None,
                "claimed_at": None,
            }
        )

    # Short TTL cache for the per-loop gating counts. These counts drive
    # whether we skip PES for this poll or declare exploration complete —
    # freshness to within a fraction of a second is plenty, and dedupes the
    # two call sites (`is_exploration_complete` + `get_compound_to_postprocess`)
    # that each used to issue their own COUNT(*) queries every loop iter.
    _LOOP_COUNTS_TTL_S = 0.5

    def _get_loop_counts(self, session: Session) -> tuple[int, int, int]:
        """Return (n_compounds, n_pending, n_in_progress) in one query.

        Scoped to the worker's experiment: compound count uses the GIN
        ANY-of-experiments filter; queue counts filter on the singular
        experiment column. Cached for `_LOOP_COUNTS_TTL_S` seconds.
        """
        now = time.monotonic()
        cached = getattr(self, "_loop_counts_cache", None)
        if cached is not None and now - cached[0] < self._LOOP_COUNTS_TTL_S:
            return cached[1]
        row = session.execute(text("""
            SELECT
              (SELECT COUNT(*) FROM compounds
                WHERE :experiment = ANY(experiments)) AS n_compounds,
              COUNT(*) FILTER (WHERE status = 'pending'
                AND experiment = :experiment) AS n_pending,
              COUNT(*) FILTER (WHERE status = 'in_progress'
                AND experiment = :experiment) AS n_in_progress
            FROM pes_work_queue
            WHERE experiment = :experiment
        """), {"experiment": self.experiment}).fetchone()
        counts = (int(row[0]), int(row[1]), int(row[2]))
        self._loop_counts_cache = (now, counts)
        return counts

    def get_pes_backlog(self, session: Session) -> tuple[int, int, int]:
        """Return (n_pending, n_in_progress, n_total_compounds)."""
        n_compounds, n_pending, n_in_progress = self._get_loop_counts(session)
        return n_pending, n_in_progress, n_compounds

    # =========================================================================
    # Exploration Stats & Batch Log
    # =========================================================================

    def update_exploration_stats(self, session: Session, updates: dict):
        """Atomically merge updates into this experiment's stats JSON.

        ExplorationStats is per-experiment now (one row per experiment,
        unique on `experiment`); legacy id=1 singleton became a (possibly
        multi-row) table keyed off the experiment column.
        """
        row = (
            session.query(ExplorationStats)
            .filter(ExplorationStats.experiment == self.experiment)
            .with_for_update()
            .first()
        )
        if row is None:
            row = ExplorationStats(stats_json={}, experiment=self.experiment)
            session.add(row)
            session.flush()

        stats = dict(row.stats_json)
        for key, value in updates.items():
            if key == "noise_histogram":
                hist = stats.setdefault("noise_histogram", {})
                for noise_key, noise_val in value.items():
                    entry = hist.setdefault(str(noise_key), {"batches": 0, "reactions": 0})
                    entry["batches"] += noise_val.get("batches", 0)
                    entry["reactions"] += noise_val.get("reactions", 0)
            elif isinstance(value, (int, float)):
                stats[key] = stats.get(key, 0) + value
            else:
                stats[key] = value

        row.stats_json = stats
        row.updated_at = datetime.now(timezone.utc)

    def append_batch_log(self, session: Session, batch_summary: dict):
        """Append a batch summary to the batch log, tagged with experiment."""
        log = BatchLog(
            summary_json=batch_summary,
            batch_idx=batch_summary.get("batch_idx"),
            experiment=self.experiment,
        )
        session.add(log)

    # =========================================================================
    # Exploration Completion Check
    # =========================================================================

    def is_exploration_complete(self, session: Session, max_compounds: int) -> bool:
        """Check if exploration target is met and all PES work is done."""
        n_compounds, n_pending, _ = self._get_loop_counts(session)
        if n_compounds >= max_compounds and n_pending == 0:
            logger.info(
                f"Exploration complete: {n_compounds} compounds (target {max_compounds}), "
                f"no pending PES work"
            )
            return True
        return False

    # =========================================================================
    # Worker Heartbeat
    # =========================================================================

    def heartbeat(
        self,
        session: Session,
        worker_id: str,
        worker_type: str,
        status: str = "idle",
        current_task: str | None = None,
        batches_completed: int = 0,
        pes_completed: int = 0,
        total_wall_time_s: float = 0.0,
    ):
        """Upsert a heartbeat for this worker."""
        from packages.db.models import WorkerHeartbeat

        now = datetime.now(timezone.utc)
        row = session.query(WorkerHeartbeat).filter(WorkerHeartbeat.worker_id == worker_id).first()
        if row is None:
            row = WorkerHeartbeat(
                worker_id=worker_id,
                worker_type=worker_type,
                status=status,
                current_task=current_task,
                started_at=now,
                last_heartbeat=now,
                batches_completed=batches_completed,
                pes_completed=pes_completed,
                total_wall_time_s=total_wall_time_s,
                experiment=self.experiment,
            )
            session.add(row)
        else:
            row.status = status
            row.current_task = current_task
            row.last_heartbeat = now
            row.batches_completed = batches_completed
            row.pes_completed = pes_completed
            row.total_wall_time_s = total_wall_time_s

    def remove_heartbeat(self, session: Session, worker_id: str):
        """Remove heartbeat on shutdown."""
        from packages.db.models import WorkerHeartbeat
        session.query(WorkerHeartbeat).filter(WorkerHeartbeat.worker_id == worker_id).delete()
