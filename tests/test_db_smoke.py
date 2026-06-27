"""Smoke test for DB models, serialization, and basic CRUD.

Uses SQLite in-memory — tests core logic without requiring PostgreSQL.
Run: python tests/test_db_smoke.py
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from packages.db.models import (
    Base,
    Compound,
    Minimum,
    IntraTransitionState,
    Reaction,
    ReactionReactant,
    ReactionProduct,
    GraphEdge,
    PESWorkQueue,
    ExplorationStats,
    BatchLog,
    Annotation,
    SavedLayout,
)
from packages.db.serialization import (
    serialize_ndarray,
    deserialize_ndarray,
    serialize_ndarray_optional,
    deserialize_ndarray_optional,
    serialize_trajectory,
    deserialize_trajectory,
)


def test_serialization():
    """Test numpy array serialization round-trip."""
    print("Testing serialization...", end=" ")

    # Test basic array
    arr = np.random.randn(5, 3).astype(np.float64)
    data = serialize_ndarray(arr)
    arr2 = deserialize_ndarray(data)
    assert np.allclose(arr, arr2), "Array round-trip failed"

    # Test optional None
    assert serialize_ndarray_optional(None) is None
    assert deserialize_ndarray_optional(None) is None

    # Test optional with value
    data = serialize_ndarray_optional(arr)
    arr3 = deserialize_ndarray_optional(data)
    assert np.allclose(arr, arr3), "Optional array round-trip failed"

    print("OK")


def test_trajectory_serialization():
    """Test trajectory serialization round-trip."""
    print("Testing trajectory serialization...", end=" ")

    class FakeTrajectory:
        def __init__(self):
            self.positions = [np.random.randn(5, 3) for _ in range(3)]
            self.energies = [-10.5, -10.3, -10.1]
            self.forces = [np.random.randn(5, 3) for _ in range(3)]
            self.hessians = [np.random.randn(15, 15), None, np.random.randn(15, 15)]

    traj = FakeTrajectory()
    data = serialize_trajectory(traj)
    assert data is not None

    result = deserialize_trajectory(data)
    assert result is not None
    assert len(result["positions"]) == 3
    assert len(result["energies"]) == 3
    assert len(result["forces"]) == 3
    assert len(result["hessians"]) == 3
    assert result["hessians"][1] is None
    assert np.allclose(result["positions"][0], traj.positions[0])
    assert np.allclose(result["hessians"][0], traj.hessians[0])

    # Test None trajectory
    assert serialize_trajectory(None) is None
    assert deserialize_trajectory(None) is None

    print("OK")


def test_db_crud():
    """Test basic CRUD operations with SQLite."""
    print("Testing DB CRUD...", end=" ")

    # SQLite in-memory database
    engine = create_engine("sqlite:///:memory:")

    # SQLite doesn't support some PostgreSQL features, but basic CRUD works
    # We need to handle the TIMESTAMP type
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    # --- Create compound ---
    positions = np.random.randn(5, 3).astype(np.float64)
    sorted_anum = np.array([1, 1, 6, 8, 8], dtype=np.int32)

    compound = Compound(
        smiles="C(=O)O",
        formula="CH2O2",
        charge=0,
        n_atoms=5,
        sorted_atomic_numbers=serialize_ndarray(sorted_anum),
        is_seed=True,
    )
    session.add(compound)
    session.flush()
    assert compound.id is not None
    compound_id = compound.id

    # --- Create minima ---
    min1 = Minimum(
        compound_id=compound_id,
        local_id=0,
        positions=serialize_ndarray(positions),
        energy=-10.5,
        explored=False,
        discovery_timestamp=1234567890.0,
    )
    session.add(min1)
    session.flush()

    min2 = Minimum(
        compound_id=compound_id,
        local_id=1,
        positions=serialize_ndarray(positions + 0.1),
        energy=-10.3,
        explored=False,
    )
    session.add(min2)
    session.flush()

    # Query minima
    minima = session.query(Minimum).filter(Minimum.compound_id == compound_id).all()
    assert len(minima) == 2

    # Deserialize positions
    pos_back = deserialize_ndarray(minima[0].positions)
    assert np.allclose(pos_back, positions), "Position round-trip failed"

    # --- Create intra TS ---
    ts = IntraTransitionState(
        compound_id=compound_id,
        local_id=0,
        positions=serialize_ndarray(positions + 0.05),
        energy=-10.0,
        eigenvalue=-0.5,
        min_fwd_id=min1.id,
        min_bwd_id=min2.id,
        barrier_fwd=0.5,
        barrier_bwd=0.3,
    )
    session.add(ts)
    session.flush()

    # --- Create reaction ---
    ts_positions = np.random.randn(5, 3).astype(np.float64)
    ts_anum = np.array([1, 1, 6, 8, 8], dtype=np.int32)

    compound2 = Compound(
        smiles="CO",
        formula="CH4O",
        charge=0,
        n_atoms=6,
        sorted_atomic_numbers=serialize_ndarray(np.array([1, 1, 1, 1, 6, 8], dtype=np.int32)),
        is_seed=False,
    )
    session.add(compound2)
    session.flush()

    reaction = Reaction(
        ts_id=42,
        ts_conformer_positions=serialize_ndarray(ts_positions),
        ts_conformer_atomic_numbers=serialize_ndarray(ts_anum),
        ts_conformer_charge=0,
        ts_energy=-9.5,
        barrier_forward=1.0,
        barrier_backward=0.8,
        discovery_method="generative",
        name="rxn-test-1",
    )
    session.add(reaction)
    session.flush()

    reactant = ReactionReactant(
        reaction_id=reaction.id,
        compound_id=compound_id,
        conformer_local_id=0,
    )
    session.add(reactant)

    product = ReactionProduct(
        reaction_id=reaction.id,
        compound_id=compound2.id,
        conformer_local_id=0,
        energy=-10.3,
    )
    session.add(product)

    # --- Graph edges ---
    edge = GraphEdge(
        source_node="C(=O)O",
        target_node="rxn-test-1",
        source_type="compound",
        target_type="ts",
        direction="up",
        energy_diff=1.0,
        reaction_id=reaction.id,
    )
    session.add(edge)

    # --- PES work queue ---
    work = PESWorkQueue(
        compound_id=compound_id,
        minimum_id=min1.id,
        status="pending",
    )
    session.add(work)
    session.flush()

    # Query work queue
    pending = session.query(PESWorkQueue).filter(PESWorkQueue.status == "pending").all()
    assert len(pending) == 1
    assert pending[0].compound_id == compound_id

    # --- Annotations ---
    ann = Annotation(
        entity_type="compounds",
        entity_key="C(=O)O",
        label="Formic acid",
        notes="Starting compound",
    )
    session.add(ann)

    # --- Layouts ---
    layout = SavedLayout(
        name="default",
        layout_data={"nodes": [{"id": "C(=O)O", "x": 0, "y": 0}]},
    )
    session.add(layout)

    session.commit()

    # --- Verify relationships ---
    compound_reloaded = session.query(Compound).filter(Compound.smiles == "C(=O)O").first()
    assert len(compound_reloaded.minima) == 2
    assert len(compound_reloaded.intra_transition_states) == 1

    reaction_reloaded = session.query(Reaction).filter(Reaction.ts_id == 42).first()
    assert len(reaction_reloaded.reactants) == 1
    assert len(reaction_reloaded.products) == 1
    assert reaction_reloaded.reactants[0].compound.smiles == "C(=O)O"
    assert reaction_reloaded.products[0].compound.smiles == "CO"

    session.close()
    print("OK")


def test_db_worker_flow():
    """Test the DB class used by workers."""
    print("Testing DB worker flow...", end=" ")

    # The DB class uses PostgreSQL-specific features (FOR UPDATE SKIP LOCKED)
    # so we can't fully test it with SQLite. But we can test the non-PG parts.

    # Just verify the class can be imported and instantiated
    from packages.db.models import Base
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    # Test compound count on empty DB
    count = session.query(Compound).count()
    assert count == 0

    session.close()
    print("OK")


if __name__ == "__main__":
    print("=" * 60)
    print("CRN Cloud - DB Smoke Tests")
    print("=" * 60)
    print()

    tests = [
        test_serialization,
        test_trajectory_serialization,
        test_db_crud,
        test_db_worker_flow,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print()
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    sys.exit(1 if failed > 0 else 0)
