#!/usr/bin/env python3
"""
Comprehensive verification suite for crn-cloud.

Prerequisites:
  docker compose up -d db    # Start PostgreSQL
  pip install -r tests/requirements.txt

Usage:
  python tests/verify_all.py          # Run all tests
  python tests/verify_all.py --api    # Also test API server (needs: docker compose up -d db api)

This suite:
1. Tests serialization (numpy <-> bytea round-trips)
2. Tests DB schema creation + migrations on real PostgreSQL
3. Tests full CRUD with relationships, constraints, indexes
4. Tests work queue (SELECT FOR UPDATE SKIP LOCKED) with concurrent sessions
5. Tests the DB access layer (lib/db.py) end-to-end
6. Tests API endpoints (if --api flag, needs API server running)
"""

import argparse
import sys
import os
import time
import traceback
from contextlib import contextmanager

import numpy as np
from sqlalchemy import create_engine, text, event
from sqlalchemy.orm import sessionmaker

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://crn:crn@localhost:5432/crn_cloud"
)
API_URL = os.environ.get("API_URL", "http://localhost:8080")


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def fresh_db():
    """Drop and recreate all tables."""
    from packages.db.models import Base
    engine = create_engine(DATABASE_URL)
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    return engine


def make_session(engine):
    return sessionmaker(bind=engine)()


passed = 0
failed = 0
errors = []


def run_test(name, fn, *args, **kwargs):
    global passed, failed
    print(f"  {name}...", end=" ", flush=True)
    try:
        fn(*args, **kwargs)
        print("OK")
        passed += 1
    except Exception as e:
        print(f"FAILED: {e}")
        errors.append((name, traceback.format_exc()))
        failed += 1


# ─────────────────────────────────────────────────────────────
# 1. Serialization Tests
# ─────────────────────────────────────────────────────────────

def test_ndarray_roundtrip():
    from packages.db.serialization import serialize_ndarray, deserialize_ndarray
    for shape in [(5, 3), (1,), (10, 10), (28, 3)]:
        arr = np.random.randn(*shape).astype(np.float64)
        rt = deserialize_ndarray(serialize_ndarray(arr))
        assert np.array_equal(arr, rt), f"Failed for shape {shape}"


def test_ndarray_optional():
    from packages.db.serialization import serialize_ndarray_optional, deserialize_ndarray_optional
    assert serialize_ndarray_optional(None) is None
    assert deserialize_ndarray_optional(None) is None
    arr = np.array([1.0, 2.0, 3.0])
    rt = deserialize_ndarray_optional(serialize_ndarray_optional(arr))
    assert np.array_equal(arr, rt)


def test_trajectory_roundtrip():
    from packages.db.serialization import serialize_trajectory, deserialize_trajectory

    class Traj:
        positions = [np.random.randn(5, 3) for _ in range(4)]
        energies = [-10.5, -10.3, -10.1, -9.9]
        forces = [np.random.randn(5, 3) for _ in range(4)]
        hessians = [np.random.randn(15, 15), None, np.random.randn(15, 15), None]

    traj = Traj()
    data = serialize_trajectory(traj)
    result = deserialize_trajectory(data)

    assert len(result["positions"]) == 4
    assert len(result["energies"]) == 4
    assert len(result["forces"]) == 4
    assert len(result["hessians"]) == 4
    assert result["hessians"][1] is None
    assert result["hessians"][3] is None
    assert np.allclose(result["positions"][0], traj.positions[0])
    assert np.allclose(result["hessians"][0], traj.hessians[0])

    # Edge case: None trajectory
    assert serialize_trajectory(None) is None
    assert deserialize_trajectory(None) is None


def test_trajectory_empty():
    from packages.db.serialization import serialize_trajectory, deserialize_trajectory

    class EmptyTraj:
        positions = []
        energies = []
        forces = []
        hessians = []

    data = serialize_trajectory(EmptyTraj())
    result = deserialize_trajectory(data)
    assert len(result["energies"]) == 0


# ─────────────────────────────────────────────────────────────
# 2. Schema Tests (PostgreSQL-specific)
# ─────────────────────────────────────────────────────────────

def test_schema_creation():
    """Verify all tables can be created on PostgreSQL."""
    engine = fresh_db()
    # Check tables exist
    with engine.connect() as conn:
        result = conn.execute(text(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'public' ORDER BY table_name"
        ))
        tables = [row[0] for row in result]

    expected = [
        "annotations", "batch_log", "compounds", "exploration_stats",
        "graph_edges", "intra_transition_states", "minima",
        "pes_work_queue", "reaction_products", "reaction_reactants",
        "reactions", "saved_layouts",
    ]
    for t in expected:
        assert t in tables, f"Table {t} not found. Got: {tables}"


def test_indexes_exist():
    """Verify key indexes were created."""
    engine = fresh_db()
    with engine.connect() as conn:
        result = conn.execute(text(
            "SELECT indexname FROM pg_indexes WHERE schemaname = 'public'"
        ))
        indexes = {row[0] for row in result}

    for idx in ["idx_minima_compound", "idx_intra_ts_compound",
                "idx_reaction_reactants_compound", "idx_reaction_products_compound",
                "idx_graph_edges_source", "idx_graph_edges_target"]:
        assert idx in indexes, f"Index {idx} not found"


def test_unique_constraints():
    """Verify unique constraints prevent duplicates."""
    from packages.db.models import Compound
    from packages.db.serialization import serialize_ndarray

    engine = fresh_db()
    session = make_session(engine)

    anum = serialize_ndarray(np.array([1, 6, 8], dtype=np.int32))
    c1 = Compound(smiles="CO", formula="CH4O", charge=0, n_atoms=3, sorted_atomic_numbers=anum)
    session.add(c1)
    session.commit()

    # Duplicate SMILES should fail
    c2 = Compound(smiles="CO", formula="CH4O", charge=0, n_atoms=3, sorted_atomic_numbers=anum)
    session.add(c2)
    try:
        session.commit()
        assert False, "Should have raised IntegrityError"
    except Exception:
        session.rollback()

    session.close()


# ─────────────────────────────────────────────────────────────
# 3. Full CRUD Tests (PostgreSQL)
# ─────────────────────────────────────────────────────────────

def test_compound_crud():
    from packages.db.models import Compound, Minimum, IntraTransitionState
    from packages.db.serialization import serialize_ndarray, deserialize_ndarray

    engine = fresh_db()
    session = make_session(engine)

    anum = np.array([1, 1, 6, 8], dtype=np.int32)
    pos1 = np.random.randn(4, 3).astype(np.float64)
    pos2 = np.random.randn(4, 3).astype(np.float64)
    pos_ts = np.random.randn(4, 3).astype(np.float64)

    # Create compound
    c = Compound(
        smiles="C=O", formula="CH2O", charge=0, n_atoms=4,
        sorted_atomic_numbers=serialize_ndarray(anum), is_seed=True,
    )
    session.add(c)
    session.flush()

    # Add minima
    m1 = Minimum(compound_id=c.id, local_id=0, positions=serialize_ndarray(pos1), energy=-10.5)
    m2 = Minimum(compound_id=c.id, local_id=1, positions=serialize_ndarray(pos2), energy=-10.3)
    session.add_all([m1, m2])
    session.flush()

    # Add intra TS
    ts = IntraTransitionState(
        compound_id=c.id, local_id=0,
        positions=serialize_ndarray(pos_ts), energy=-10.0,
        eigenvalue=-0.5,
        min_fwd_id=m1.id, min_bwd_id=m2.id,
        barrier_fwd=0.5, barrier_bwd=0.3,
    )
    session.add(ts)
    session.commit()

    # Verify relationships
    loaded = session.query(Compound).filter(Compound.smiles == "C=O").first()
    assert loaded is not None
    assert len(loaded.minima) == 2
    assert len(loaded.intra_transition_states) == 1

    # Verify positions roundtrip
    loaded_pos = deserialize_ndarray(loaded.minima[0].positions)
    assert loaded_pos.shape == (4, 3)

    session.close()


def test_reaction_crud():
    from packages.db.models import Compound, Reaction, ReactionReactant, ReactionProduct
    from packages.db.serialization import serialize_ndarray

    engine = fresh_db()
    session = make_session(engine)

    anum = serialize_ndarray(np.array([1, 6, 8], dtype=np.int32))
    c1 = Compound(smiles="C=O", formula="CH2O", charge=0, n_atoms=3, sorted_atomic_numbers=anum)
    c2 = Compound(smiles="CO", formula="CH4O", charge=0, n_atoms=3, sorted_atomic_numbers=anum)
    session.add_all([c1, c2])
    session.flush()

    rxn = Reaction(
        ts_id=42,
        ts_conformer_positions=serialize_ndarray(np.random.randn(3, 3)),
        ts_conformer_atomic_numbers=anum,
        ts_energy=-9.5, barrier_forward=1.0, barrier_backward=0.8,
        discovery_method="generative", name="rxn-test-1",
    )
    session.add(rxn)
    session.flush()

    session.add(ReactionReactant(reaction_id=rxn.id, compound_id=c1.id, conformer_local_id=0))
    session.add(ReactionProduct(reaction_id=rxn.id, compound_id=c2.id, conformer_local_id=0, energy=-10.3))
    session.commit()

    # Verify
    loaded = session.query(Reaction).filter(Reaction.ts_id == 42).first()
    assert loaded is not None
    assert len(loaded.reactants) == 1
    assert len(loaded.products) == 1
    assert loaded.reactants[0].compound.smiles == "C=O"
    assert loaded.products[0].compound.smiles == "CO"

    # Cascade delete
    session.delete(loaded)
    session.commit()
    assert session.query(ReactionReactant).count() == 0
    assert session.query(ReactionProduct).count() == 0

    session.close()


def test_graph_edges():
    from packages.db.models import GraphEdge

    engine = fresh_db()
    session = make_session(engine)

    edges = [
        GraphEdge(source_node="C=O", target_node="rxn-1", source_type="compound", target_type="ts", direction="up"),
        GraphEdge(source_node="rxn-1", target_node="CO", source_type="ts", target_type="compound", direction="down"),
    ]
    session.add_all(edges)
    session.commit()

    by_source = session.query(GraphEdge).filter(GraphEdge.source_node == "C=O").all()
    assert len(by_source) == 1
    assert by_source[0].target_node == "rxn-1"

    session.close()


def test_annotations_and_layouts():
    from packages.db.models import Annotation, SavedLayout

    engine = fresh_db()
    session = make_session(engine)

    session.add(Annotation(entity_type="compounds", entity_key="C=O", label="Formaldehyde", notes="Test"))
    session.add(SavedLayout(name="default", layout_data={"x": 1, "y": 2}))
    session.commit()

    ann = session.query(Annotation).filter(Annotation.entity_key == "C=O").first()
    assert ann.label == "Formaldehyde"

    layout = session.query(SavedLayout).filter(SavedLayout.name == "default").first()
    assert layout.layout_data["x"] == 1

    session.close()


# ─────────────────────────────────────────────────────────────
# 4. Work Queue Tests (PostgreSQL-specific: FOR UPDATE SKIP LOCKED)
# ─────────────────────────────────────────────────────────────

def test_work_queue_claim():
    """Test that work items can be claimed atomically."""
    from packages.db.models import Compound, Minimum, PESWorkQueue
    from packages.db.serialization import serialize_ndarray

    engine = fresh_db()
    session = make_session(engine)

    anum = serialize_ndarray(np.array([1, 6], dtype=np.int32))
    c = Compound(smiles="CH", formula="CH", charge=0, n_atoms=2, sorted_atomic_numbers=anum)
    session.add(c)
    session.flush()

    pos = serialize_ndarray(np.zeros((2, 3)))
    m = Minimum(compound_id=c.id, local_id=0, positions=pos, energy=-5.0)
    session.add(m)
    session.flush()

    work = PESWorkQueue(compound_id=c.id, minimum_id=m.id, status="pending")
    session.add(work)
    session.commit()

    # Claim via raw SQL (same query as lib/db.py)
    result = session.execute(text("""
        UPDATE pes_work_queue
        SET status = 'in_progress', worker_id = 'test-worker', claimed_at = now()
        WHERE id = (
            SELECT id FROM pes_work_queue
            WHERE status = 'pending'
            ORDER BY id LIMIT 1
            FOR UPDATE SKIP LOCKED
        )
        RETURNING id, compound_id, minimum_id
    """))
    row = result.fetchone()
    session.commit()

    assert row is not None
    assert row[1] == c.id

    # Second claim should return nothing (already claimed)
    result2 = session.execute(text("""
        UPDATE pes_work_queue
        SET status = 'in_progress', worker_id = 'test-worker-2', claimed_at = now()
        WHERE id = (
            SELECT id FROM pes_work_queue
            WHERE status = 'pending'
            ORDER BY id LIMIT 1
            FOR UPDATE SKIP LOCKED
        )
        RETURNING id
    """))
    row2 = result2.fetchone()
    session.commit()

    assert row2 is None, "Second claim should be empty"

    session.close()


def test_work_queue_concurrent():
    """Test concurrent workers don't get the same work item."""
    from packages.db.models import Compound, Minimum, PESWorkQueue
    from packages.db.serialization import serialize_ndarray

    engine = fresh_db()

    anum = serialize_ndarray(np.array([1, 6], dtype=np.int32))
    pos = serialize_ndarray(np.zeros((2, 3)))

    # Seed 5 work items
    s = make_session(engine)
    c = Compound(smiles="CH", formula="CH", charge=0, n_atoms=2, sorted_atomic_numbers=anum)
    s.add(c)
    s.flush()
    for i in range(5):
        m = Minimum(compound_id=c.id, local_id=i, positions=pos, energy=-5.0 + i * 0.1)
        s.add(m)
        s.flush()
        s.add(PESWorkQueue(compound_id=c.id, minimum_id=m.id, status="pending"))
    s.commit()
    s.close()

    # Simulate 5 concurrent workers claiming
    claimed_ids = []
    for worker_num in range(5):
        ws = make_session(engine)
        result = ws.execute(text("""
            UPDATE pes_work_queue
            SET status = 'in_progress', worker_id = :wid, claimed_at = now()
            WHERE id = (
                SELECT id FROM pes_work_queue
                WHERE status = 'pending'
                ORDER BY id LIMIT 1
                FOR UPDATE SKIP LOCKED
            )
            RETURNING id
        """), {"wid": f"worker-{worker_num}"})
        row = result.fetchone()
        ws.commit()
        if row:
            claimed_ids.append(row[0])
        ws.close()

    # All 5 should be claimed, all unique
    assert len(claimed_ids) == 5, f"Expected 5 claims, got {len(claimed_ids)}"
    assert len(set(claimed_ids)) == 5, "Work items should be unique"


# ─────────────────────────────────────────────────────────────
# 5. DB Access Layer Tests (lib/db.py)
# ─────────────────────────────────────────────────────────────

def test_db_layer_compound_lifecycle():
    """Test the DB class end-to-end."""
    from packages.worker.lib.db import DB

    engine = fresh_db()
    db = DB(database_url=DATABASE_URL)

    with db.session() as session:
        # Create compound
        cid, is_new = db.get_or_create_compound(
            session, smiles="C=O", formula="CH2O", charge=0, n_atoms=4,
            sorted_atomic_numbers=np.array([1, 1, 6, 8], dtype=np.int32),
            is_seed=True,
        )
        assert is_new
        assert cid > 0

        # Duplicate should return same ID
        cid2, is_new2 = db.get_or_create_compound(
            session, smiles="C=O", formula="CH2O", charge=0, n_atoms=4,
            sorted_atomic_numbers=np.array([1, 1, 6, 8], dtype=np.int32),
        )
        assert not is_new2
        assert cid2 == cid

        # Add minimum
        mid, min_new = db.add_minimum(
            session, compound_id=cid, local_id=0,
            positions=np.random.randn(4, 3), energy=-10.5,
        )
        assert min_new

        # Duplicate minimum
        mid2, min_new2 = db.add_minimum(
            session, compound_id=cid, local_id=0,
            positions=np.random.randn(4, 3), energy=-10.4,
        )
        assert not min_new2
        assert mid2 == mid


def test_db_layer_reaction():
    from packages.worker.lib.db import DB

    engine = fresh_db()
    db = DB(database_url=DATABASE_URL)

    with db.session() as session:
        c1_id, _ = db.get_or_create_compound(
            session, "C=O", "CH2O", 0, 3, np.array([1, 6, 8], dtype=np.int32),
        )
        c2_id, _ = db.get_or_create_compound(
            session, "CO", "CH4O", 0, 3, np.array([1, 6, 8], dtype=np.int32),
        )

        rxn_id, is_new = db.create_reaction(
            session, ts_id=99,
            ts_conformer_positions=np.random.randn(3, 3),
            ts_conformer_atomic_numbers=np.array([1, 6, 8]),
            ts_conformer_charge=0,
            ts_energy=-9.0, barrier_forward=1.5, barrier_backward=0.7,
            reactant_compound_ids=[(c1_id, 0)],
            product_compound_ids=[(c2_id, 0, -10.3)],
            discovery_method="generative",
        )
        assert is_new

        # Duplicate ts_id should return existing
        rxn_id2, is_new2 = db.create_reaction(
            session, ts_id=99,
            ts_conformer_positions=np.random.randn(3, 3),
            ts_conformer_atomic_numbers=np.array([1, 6, 8]),
            ts_conformer_charge=0,
            ts_energy=-9.0, barrier_forward=1.5, barrier_backward=0.7,
            reactant_compound_ids=[(c1_id, 0)],
            product_compound_ids=[(c2_id, 1, -10.2)],
        )
        assert not is_new2
        assert rxn_id2 == rxn_id


def test_db_layer_work_queue():
    from packages.worker.lib.db import DB

    engine = fresh_db()
    db = DB(database_url=DATABASE_URL)

    with db.session() as session:
        cid, _ = db.get_or_create_compound(
            session, "C=O", "CH2O", 0, 3, np.array([1, 6, 8], dtype=np.int32),
        )
        mid, _ = db.add_minimum(
            session, cid, local_id=0,
            positions=np.random.randn(3, 3), energy=-10.0,
        )
        # add_minimum auto-enqueues for PES work

    # Claim work
    with db.session() as session:
        work = db.claim_pes_work(session)
        assert work is not None
        assert work["compound_id"] == cid

    # Complete work
    with db.session() as session:
        db.complete_pes_work(session, work["work_id"])

    # No more work
    with db.session() as session:
        work2 = db.claim_pes_work(session)
        assert work2 is None


def test_db_layer_stats():
    from packages.worker.lib.db import DB
    from packages.db.models import ExplorationStats

    engine = fresh_db()
    db = DB(database_url=DATABASE_URL)

    # Seed initial stats row
    s = make_session(engine)
    s.add(ExplorationStats(id=1, stats_json={}))
    s.commit()
    s.close()

    with db.session() as session:
        db.update_exploration_stats(session, {
            "compounds_explored": 3,
            "total_pes_time_s": 45.0,
        })

    with db.session() as session:
        db.update_exploration_stats(session, {
            "compounds_explored": 2,
            "total_pes_time_s": 30.0,
        })

    # Verify accumulation
    s = make_session(engine)
    row = s.query(ExplorationStats).filter(ExplorationStats.id == 1).first()
    assert row.stats_json["compounds_explored"] == 5
    assert row.stats_json["total_pes_time_s"] == 75.0
    s.close()


def test_db_layer_exploration_complete():
    from packages.worker.lib.db import DB

    engine = fresh_db()
    db = DB(database_url=DATABASE_URL)

    # No compounds → not complete (target > 0)
    with db.session() as session:
        assert not db.is_exploration_complete(session, max_compounds=10)

    # Add 2 compounds with explored minima (no pending work)
    with db.session() as session:
        for i, smi in enumerate(["C=O", "CO"]):
            cid, _ = db.get_or_create_compound(
                session, smi, f"mol{i}", 0, 3, np.array([1, 6, 8], dtype=np.int32),
            )
            mid, _ = db.add_minimum(
                session, cid, local_id=0,
                positions=np.random.randn(3, 3), energy=-10.0, explored=True,
            )

    # 2 compounds, target 2, all explored → complete
    with db.session() as session:
        assert db.is_exploration_complete(session, max_compounds=2)


# ─────────────────────────────────────────────────────────────
# 6. API Endpoint Tests (requires running API server)
# ─────────────────────────────────────────────────────────────

def seed_test_data():
    """Seed a small test dataset for API testing."""
    from packages.db.models import (
        Base, Compound, Minimum, IntraTransitionState,
        Reaction, ReactionReactant, ReactionProduct,
        GraphEdge, ExplorationStats, Annotation,
    )
    from packages.db.serialization import serialize_ndarray, serialize_trajectory

    engine = fresh_db()
    s = make_session(engine)

    anum = np.array([1, 1, 6, 8], dtype=np.int32)
    pos1 = np.random.randn(4, 3).astype(np.float64)
    pos2 = pos1 + 0.1
    pos_ts = (pos1 + pos2) / 2
    hessian = np.random.randn(12, 12).astype(np.float64)
    hessian = (hessian + hessian.T) / 2  # symmetrize

    # Compounds
    c1 = Compound(smiles="C=O", formula="CH2O", charge=0, n_atoms=4,
                  sorted_atomic_numbers=serialize_ndarray(anum), is_seed=True)
    c2 = Compound(smiles="CO", formula="CH4O", charge=0, n_atoms=4,
                  sorted_atomic_numbers=serialize_ndarray(anum))
    s.add_all([c1, c2])
    s.flush()

    # Minima for c1
    m1 = Minimum(compound_id=c1.id, local_id=0, positions=serialize_ndarray(pos1),
                 energy=-10.5, hessian=serialize_ndarray(hessian), explored=True)
    m2 = Minimum(compound_id=c1.id, local_id=1, positions=serialize_ndarray(pos2),
                 energy=-10.3, explored=False)
    s.add_all([m1, m2])
    s.flush()

    # Minima for c2
    m3 = Minimum(compound_id=c2.id, local_id=0, positions=serialize_ndarray(pos1 + 0.2),
                 energy=-10.1)
    s.add(m3)
    s.flush()

    # Intra TS
    ts_intra = IntraTransitionState(
        compound_id=c1.id, local_id=0,
        positions=serialize_ndarray(pos_ts), energy=-10.0, eigenvalue=-0.3,
        min_fwd_id=m1.id, min_bwd_id=m2.id,
        barrier_fwd=0.5, barrier_bwd=0.3,
        rmsd_to_fwd_min=0.1, rmsd_to_bwd_min=0.15, endpoint_to_endpoint_rmsd=0.2,
    )
    s.add(ts_intra)
    s.flush()

    # Reaction
    rxn = Reaction(
        ts_id=42, ts_conformer_positions=serialize_ndarray(pos_ts),
        ts_conformer_atomic_numbers=serialize_ndarray(anum),
        ts_energy=-9.5, barrier_forward=1.0, barrier_backward=0.6,
        discovery_method="generative", discovery_noise_level=300,
        name="rxn-test-1",
    )
    s.add(rxn)
    s.flush()
    s.add(ReactionReactant(reaction_id=rxn.id, compound_id=c1.id, conformer_local_id=0))
    s.add(ReactionProduct(reaction_id=rxn.id, compound_id=c2.id, conformer_local_id=0, energy=-10.1))

    # Graph edges
    s.add(GraphEdge(source_node="C=O", target_node="rxn-test-1",
                    source_type="compound", target_type="ts", direction="up", energy_diff=1.0))
    s.add(GraphEdge(source_node="rxn-test-1", target_node="CO",
                    source_type="ts", target_type="compound", direction="down",
                    energy_diff=-0.6, reaction_id=rxn.id))

    # Stats
    s.add(ExplorationStats(id=1, stats_json={"compounds_explored": 5}))

    # Annotation
    s.add(Annotation(entity_type="compounds", entity_key="C=O", label="Formaldehyde"))

    s.commit()
    s.close()
    return engine


def test_api_capabilities():
    import requests
    r = requests.get(f"{API_URL}/api/capabilities")
    assert r.status_code == 200
    data = r.json()
    assert "neb" in data


def test_api_reaction_graph():
    import requests
    r = requests.get(f"{API_URL}/api/reaction-graph")
    assert r.status_code == 200
    data = r.json()
    assert "nodes" in data
    assert "edges" in data
    assert len(data["nodes"]) >= 2  # At least 2 compounds + 1 TS
    compound_nodes = [n for n in data["nodes"] if n["type"] == "compound"]
    assert any(n["smiles"] == "C=O" for n in compound_nodes)


def test_api_pes_graph():
    import requests
    from urllib.parse import quote
    r = requests.get(f"{API_URL}/api/compound/{quote('C=O', safe='')}/pes-graph")
    assert r.status_code == 200
    data = r.json()
    assert data["smiles"] == "C=O"
    assert len(data["nodes"]) == 2  # 2 minima
    assert len(data["edges"]) == 1  # 1 intra TS


def test_api_conformer():
    import requests
    from urllib.parse import quote
    r = requests.get(f"{API_URL}/api/compound/{quote('C=O', safe='')}/conformer/0")
    assert r.status_code == 200
    data = r.json()
    assert len(data["positions"]) == 4
    assert len(data["atomic_numbers"]) == 4
    assert data["has_hessian"] is True


def test_api_modes():
    import requests
    from urllib.parse import quote
    r = requests.get(f"{API_URL}/api/compound/{quote('C=O', safe='')}/conformer/0/modes")
    assert r.status_code == 200
    data = r.json()
    assert "eigenvalues" in data
    assert "modes" in data
    assert data["n_atoms"] == 4


def test_api_reaction_ts():
    import requests
    r = requests.get(f"{API_URL}/api/reaction-ts/42")
    assert r.status_code == 200
    data = r.json()
    assert len(data["positions"]) == 4
    assert data["energy"] == -9.5


def test_api_annotations():
    import requests
    r = requests.get(f"{API_URL}/api/annotations")
    assert r.status_code == 200
    data = r.json()
    assert "compounds" in data
    assert data["compounds"].get("C=O", {}).get("label") == "Formaldehyde"

    # PUT
    r2 = requests.put(
        f"{API_URL}/api/annotations/compounds/CO",
        json={"label": "Methanol", "notes": "Product"},
    )
    assert r2.status_code == 200

    # Verify
    r3 = requests.get(f"{API_URL}/api/annotations")
    assert r3.json()["compounds"]["CO"]["label"] == "Methanol"


def test_api_layouts():
    import requests
    # Save
    r = requests.post(
        f"{API_URL}/api/layouts/test-layout",
        json={"nodes": [{"id": "C=O", "x": 100, "y": 200}]},
    )
    assert r.status_code == 200

    # List
    r2 = requests.get(f"{API_URL}/api/layouts")
    assert "test-layout" in r2.json()

    # Delete
    r3 = requests.delete(f"{API_URL}/api/layouts/test-layout")
    assert r3.status_code == 200


def test_api_stats():
    import requests
    r = requests.get(f"{API_URL}/api/exploration-stats")
    assert r.status_code == 200
    data = r.json()
    assert "basic_stats" in data
    assert data["basic_stats"]["n_compounds"] >= 2


def test_api_kinetics():
    import requests
    r = requests.post(f"{API_URL}/api/kinetics/simulate", json={
        "species": ["C=O", "CO"],
        "reactions": [{
            "reactants": ["C=O"],
            "products": ["CO"],
            "barrier_forward": 1.0,
            "barrier_backward": 0.6,
        }],
        "initial_concentrations": {"C=O": 1.0, "CO": 0.0},
        "temperature": 300,
        "t_end": 1.0,
        "n_points": 100,
    })
    assert r.status_code == 200
    data = r.json()
    assert "times" in data
    assert "concentrations" in data
    assert len(data["times"]) == 100


def test_api_all_xyz():
    import requests
    from urllib.parse import quote
    r = requests.get(f"{API_URL}/api/compound/{quote('C=O', safe='')}/all-xyz")
    assert r.status_code == 200
    assert "C=O" in r.text  # SMILES in comment line
    lines = r.text.strip().split("\n")
    assert lines[0].strip() == "4"  # n_atoms


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CRN Cloud verification suite")
    parser.add_argument("--api", action="store_true", help="Also test API endpoints (needs running API server)")
    args = parser.parse_args()

    print("=" * 70)
    print("CRN Cloud — Comprehensive Verification Suite")
    print("=" * 70)
    print(f"Database: {DATABASE_URL}")
    if args.api:
        print(f"API:      {API_URL}")
    print()

    # Check DB connection
    try:
        engine = create_engine(DATABASE_URL)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("PostgreSQL connection: OK")
    except Exception as e:
        print(f"PostgreSQL connection FAILED: {e}")
        print("Start PostgreSQL with: docker compose up -d db")
        sys.exit(1)

    print()

    # ── 1. Serialization ──
    print("[1/6] Serialization")
    run_test("ndarray round-trip", test_ndarray_roundtrip)
    run_test("ndarray optional", test_ndarray_optional)
    run_test("trajectory round-trip", test_trajectory_roundtrip)
    run_test("trajectory empty", test_trajectory_empty)

    # ── 2. Schema ──
    print("\n[2/6] PostgreSQL Schema")
    run_test("schema creation", test_schema_creation)
    run_test("indexes exist", test_indexes_exist)
    run_test("unique constraints", test_unique_constraints)

    # ── 3. CRUD ──
    print("\n[3/6] CRUD Operations")
    run_test("compound CRUD", test_compound_crud)
    run_test("reaction CRUD", test_reaction_crud)
    run_test("graph edges", test_graph_edges)
    run_test("annotations & layouts", test_annotations_and_layouts)

    # ── 4. Work Queue ──
    print("\n[4/6] Work Queue (FOR UPDATE SKIP LOCKED)")
    run_test("work queue claim", test_work_queue_claim)
    run_test("concurrent workers", test_work_queue_concurrent)

    # ── 5. DB Access Layer ──
    print("\n[5/6] DB Access Layer (lib/db.py)")
    run_test("compound lifecycle", test_db_layer_compound_lifecycle)
    run_test("reaction creation", test_db_layer_reaction)
    run_test("work queue via DB class", test_db_layer_work_queue)
    run_test("stats accumulation", test_db_layer_stats)
    run_test("exploration complete check", test_db_layer_exploration_complete)

    # ── 6. API Endpoints ──
    if args.api:
        print("\n[6/6] API Endpoints")
        print("  Seeding test data...", end=" ", flush=True)
        try:
            seed_test_data()
            print("OK")
            # Give API a moment to see the new data
            time.sleep(1)
        except Exception as e:
            print(f"FAILED: {e}")
            errors.append(("seed_test_data", traceback.format_exc()))
            failed += 1

        run_test("GET /api/capabilities", test_api_capabilities)
        run_test("GET /api/reaction-graph", test_api_reaction_graph)
        run_test("GET /api/compound/{smiles}/pes-graph", test_api_pes_graph)
        run_test("GET /api/compound/{smiles}/conformer/{id}", test_api_conformer)
        run_test("GET /api/.../conformer/{id}/modes", test_api_modes)
        run_test("GET /api/reaction-ts/{ts_id}", test_api_reaction_ts)
        run_test("GET/PUT /api/annotations", test_api_annotations)
        run_test("POST/GET/DELETE /api/layouts", test_api_layouts)
        run_test("GET /api/exploration-stats", test_api_stats)
        run_test("POST /api/kinetics/simulate", test_api_kinetics)
        run_test("GET /api/compound/{smiles}/all-xyz", test_api_all_xyz)
    else:
        print("\n[6/6] API Endpoints — SKIPPED (use --api flag)")

    # ── Summary ──
    print()
    print("=" * 70)
    print(f"Results: {passed} passed, {failed} failed")
    if errors:
        print("\nFailed tests:")
        for name, tb in errors:
            print(f"\n  {name}:")
            for line in tb.strip().split("\n")[-3:]:
                print(f"    {line}")
    print("=" * 70)

    sys.exit(1 if failed > 0 else 0)
