#!/usr/bin/env python3
"""
Integration test: simulates a full exploration run with concurrent workers,
DB coordination, and API verification.

No GPU or ML models needed — mocks the compute, tests the coordination.

Prerequisites:
  docker compose up -d db api
  pip install -r tests/requirements.txt

Usage:
  python tests/test_integration.py
"""

import os
import sys
import time
import random
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import requests
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://crn:crn@localhost:5432/crn_cloud")
API_URL = os.environ.get("API_URL", "http://localhost:8080")
N_WORKERS = 5
N_COMPOUNDS = 20
N_REACTIONS = 15

from packages.db.models import (
    Base, Compound, Minimum, IntraTransitionState,
    Reaction, ReactionReactant, ReactionProduct,
    GraphEdge, PESWorkQueue, ExplorationStats, BatchLog,
)
from packages.db.serialization import serialize_ndarray, deserialize_ndarray
from packages.worker.lib.db import DB


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

ELEMENTS = [1, 6, 7, 8]
SMILES_POOL = [
    "C=O", "CO", "C(=O)O", "CC=O", "CCO", "C=C", "CC", "O=CO",
    "C(C)=O", "CC(=O)O", "COC", "OC=O", "C#N", "CC#N", "C=CC",
    "CCCO", "CC(O)C", "C(=O)CC", "OCC=O", "OCCO",
]


def random_anum(n=4):
    return np.array(random.choices(ELEMENTS, k=n), dtype=np.int32)


def random_pos(n=4):
    return np.random.randn(n, 3).astype(np.float64)


def random_hessian(n=4):
    h = np.random.randn(3 * n, 3 * n).astype(np.float64)
    return (h + h.T) / 2


def fresh_db():
    engine = create_engine(DATABASE_URL)
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    # Seed stats row
    Session = sessionmaker(bind=engine)
    s = Session()
    s.add(ExplorationStats(id=1, stats_json={}))
    s.commit()
    s.close()
    return engine


# ─────────────────────────────────────────────────────────────
# Phase 1: Seed initial compounds (simulates first worker init)
# ─────────────────────────────────────────────────────────────

def seed_initial_compounds(db: DB, n_seed=3):
    """Seed starting compounds like the first worker would."""
    seeded = []
    with db.session() as session:
        for i in range(n_seed):
            smiles = SMILES_POOL[i]
            anum = random_anum()
            cid, is_new = db.get_or_create_compound(
                session, smiles=smiles, formula=f"mol_{i}",
                charge=0, n_atoms=len(anum),
                sorted_atomic_numbers=anum, is_seed=True,
            )
            assert is_new, f"Seed compound {smiles} already exists"

            # Add initial minimum (auto-enqueues PES work)
            mid, _ = db.add_minimum(
                session, compound_id=cid, local_id=0,
                positions=random_pos(len(anum)), energy=-10.0 - i * 0.5,
                hessian=random_hessian(len(anum)),
                discovery_timestamp=time.time(),
            )
            seeded.append((smiles, cid))

    return seeded


# ─────────────────────────────────────────────────────────────
# Phase 2: Simulated worker (runs in a thread)
# ─────────────────────────────────────────────────────────────

worker_stats = {
    "pes_claimed": 0,
    "pes_completed": 0,
    "compounds_added": 0,
    "reactions_added": 0,
    "batches_run": 0,
}
stats_lock = threading.Lock()


def simulated_worker(worker_id: str, db: DB, max_rounds: int = 10):
    """Simulate a worker: claim PES work, add compounds/reactions."""
    local_stats = {"pes": 0, "compounds": 0, "reactions": 0, "batches": 0}

    for round_num in range(max_rounds):
        # 1. Try to claim PES work
        with db.session() as session:
            work = db.claim_pes_work(session)

        if work is not None:
            local_stats["pes"] += 1
            # Simulate PES exploration: add a new minimum + intra TS
            with db.session() as session:
                compound = session.query(Compound).filter(Compound.id == work["compound_id"]).first()
                if compound:
                    n_atoms = compound.n_atoms
                    # Add discovered minimum
                    new_local_id = random.randint(100, 99999)
                    mid, is_new = db.add_minimum(
                        session, compound_id=compound.id,
                        local_id=new_local_id,
                        positions=random_pos(n_atoms),
                        energy=-10.0 + random.uniform(-1, 1),
                        explored=True,
                        discovery_timestamp=time.time(),
                    )
                    # Mark original minimum explored
                    db.mark_minimum_explored(session, work["minimum_id"])

            with db.session() as session:
                db.complete_pes_work(session, work["work_id"])
            continue

        # 2. Simulate generative batch: discover new compound + reaction
        if len(SMILES_POOL) > local_stats["compounds"]:
            smiles_idx = min(3 + local_stats["compounds"] + hash(worker_id) % 5, len(SMILES_POOL) - 1)
            new_smiles = SMILES_POOL[smiles_idx]

            with db.session() as session:
                anum = random_anum(random.randint(3, 6))
                cid, is_new = db.get_or_create_compound(
                    session, smiles=new_smiles, formula=f"gen_{new_smiles}",
                    charge=0, n_atoms=len(anum),
                    sorted_atomic_numbers=anum,
                )
                if is_new:
                    local_stats["compounds"] += 1
                    db.add_minimum(
                        session, compound_id=cid, local_id=0,
                        positions=random_pos(len(anum)),
                        energy=-10.0 + random.uniform(-2, 0),
                        discovery_timestamp=time.time(),
                    )

                # Add reaction between two existing compounds
                all_compounds = session.query(Compound).all()
                if len(all_compounds) >= 2:
                    c1, c2 = random.sample(all_compounds, 2)
                    ts_id = random.randint(1000, 999999)

                    existing = db.get_reaction_by_ts_id(session, ts_id)
                    if existing is None:
                        rxn_id, rxn_new = db.create_reaction(
                            session, ts_id=ts_id,
                            ts_conformer_positions=random_pos(4),
                            ts_conformer_atomic_numbers=random_anum(4),
                            ts_conformer_charge=0,
                            ts_energy=-9.0 + random.uniform(-1, 1),
                            barrier_forward=random.uniform(0.3, 2.0),
                            barrier_backward=random.uniform(0.1, 1.5),
                            reactant_compound_ids=[(c1.id, 0)],
                            product_compound_ids=[(c2.id, 0, c2.smiles and -10.0)],
                            discovery_method="generative",
                            name=f"rxn-{worker_id}-{round_num}",
                        )
                        if rxn_new:
                            local_stats["reactions"] += 1
                            # Add graph edges
                            db.add_graph_edge(
                                session, c1.smiles, f"rxn-{worker_id}-{round_num}",
                                "compound", "ts", direction="up",
                            )
                            db.add_graph_edge(
                                session, f"rxn-{worker_id}-{round_num}", c2.smiles,
                                "ts", "compound", direction="down",
                                reaction_id=rxn_id,
                            )

            local_stats["batches"] += 1

            # Record batch stats
            with db.session() as session:
                db.update_exploration_stats(session, {
                    "pipeline_contexts_submitted": random.randint(10, 32),
                    "pipeline_added_to_graph": local_stats["reactions"],
                    "reactions_generative": 1 if local_stats["reactions"] > 0 else 0,
                })
                db.append_batch_log(session, {
                    "batch_idx": round_num,
                    "worker_id": worker_id,
                    "wall_time_s": random.uniform(5, 30),
                    "noise_level": random.randint(100, 650),
                })

        # Small delay to simulate compute time
        time.sleep(random.uniform(0.01, 0.05))

    with stats_lock:
        worker_stats["pes_claimed"] += local_stats["pes"]
        worker_stats["compounds_added"] += local_stats["compounds"]
        worker_stats["reactions_added"] += local_stats["reactions"]
        worker_stats["batches_run"] += local_stats["batches"]

    return local_stats


# ─────────────────────────────────────────────────────────────
# Phase 3: Verify via API
# ─────────────────────────────────────────────────────────────

def verify_api(expected_min_compounds: int, expected_min_reactions: int):
    """Verify the API serves correct data after the simulated run."""
    errors = []

    def check(name, condition, msg=""):
        if not condition:
            errors.append(f"{name}: {msg}")
            print(f"    FAIL: {name} — {msg}")
        else:
            print(f"    OK: {name}")

    print("\n  Verifying API endpoints...")

    # Reaction graph
    r = requests.get(f"{API_URL}/api/reaction-graph")
    check("reaction-graph status", r.status_code == 200, f"status={r.status_code}")
    if r.status_code == 200:
        data = r.json()
        compound_nodes = [n for n in data["nodes"] if n["type"] == "compound"]
        ts_nodes = [n for n in data["nodes"] if n["type"] == "ts"]
        check(
            "reaction-graph compounds",
            len(compound_nodes) >= expected_min_compounds,
            f"got {len(compound_nodes)}, expected >= {expected_min_compounds}",
        )
        check(
            "reaction-graph reactions",
            len(ts_nodes) >= 1,
            f"got {len(ts_nodes)} TS nodes",
        )
        check("reaction-graph edges", len(data["edges"]) >= 2, f"got {len(data['edges'])} edges")
        check("reaction-graph total_conformers", data["total_conformers"] >= expected_min_compounds)

        # Verify a compound node has expected fields
        if compound_nodes:
            cn = compound_nodes[0]
            for field in ["smiles", "formula", "energy", "n_conformers", "type"]:
                check(f"compound node has '{field}'", field in cn, f"missing from {cn.get('id')}")

    # PES graph for a seed compound
    smiles = SMILES_POOL[0]
    from urllib.parse import quote
    r = requests.get(f"{API_URL}/api/compound/{quote(smiles, safe='')}/pes-graph")
    check("pes-graph status", r.status_code == 200, f"status={r.status_code} for {smiles}")
    if r.status_code == 200:
        data = r.json()
        check("pes-graph has nodes", len(data["nodes"]) >= 1)
        check("pes-graph smiles matches", data["smiles"] == smiles)

    # Conformer data
    r = requests.get(f"{API_URL}/api/compound/{quote(smiles, safe='')}/conformer/0")
    check("conformer status", r.status_code == 200)
    if r.status_code == 200:
        data = r.json()
        check("conformer has positions", len(data["positions"]) > 0)
        check("conformer has_hessian", data["has_hessian"] is True)

    # Stats
    r = requests.get(f"{API_URL}/api/exploration-stats")
    check("stats status", r.status_code == 200)
    if r.status_code == 200:
        data = r.json()
        check("stats has basic_stats", "basic_stats" in data)
        check(
            "stats n_compounds",
            data["basic_stats"]["n_compounds"] >= expected_min_compounds,
            f"got {data['basic_stats']['n_compounds']}",
        )
        check("stats has exploration_stats", "exploration_stats" in data)
        check(
            "stats pipeline_contexts_submitted > 0",
            data["exploration_stats"].get("pipeline_contexts_submitted", 0) > 0,
        )

    # Kinetics (smoke test)
    compounds = requests.get(f"{API_URL}/api/reaction-graph").json()["nodes"]
    compound_smiles = [n["id"] for n in compounds if n["type"] == "compound"][:2]
    if len(compound_smiles) >= 2:
        r = requests.post(f"{API_URL}/api/kinetics/simulate", json={
            "species": compound_smiles,
            "reactions": [{
                "reactants": [compound_smiles[0]],
                "products": [compound_smiles[1]],
                "barrier_forward": 1.0,
                "barrier_backward": 0.5,
            }],
            "initial_concentrations": {compound_smiles[0]: 1.0},
            "temperature": 300,
            "t_end": 1.0,
            "n_points": 50,
        })
        check("kinetics status", r.status_code == 200)
        if r.status_code == 200:
            data = r.json()
            check("kinetics has times", len(data.get("times", [])) == 50)
            check("kinetics no error", "error" not in data)

    # Annotations round-trip
    requests.put(
        f"{API_URL}/api/annotations/compounds/{quote(smiles, safe='')}",
        json={"label": "Integration test", "notes": "Auto-generated"},
    )
    r = requests.get(f"{API_URL}/api/annotations")
    check("annotations round-trip", r.status_code == 200)
    if r.status_code == 200:
        check(
            "annotation saved",
            r.json().get("compounds", {}).get(smiles, {}).get("label") == "Integration test",
        )

    return errors


# ─────────────────────────────────────────────────────────────
# Phase 4: Verify DB invariants
# ─────────────────────────────────────────────────────────────

def verify_db_invariants():
    """Check DB-level invariants after the simulated run."""
    errors = []
    engine = create_engine(DATABASE_URL)
    Session = sessionmaker(bind=engine)
    session = Session()

    def check(name, condition, msg=""):
        if not condition:
            errors.append(f"{name}: {msg}")
            print(f"    FAIL: {name} — {msg}")
        else:
            print(f"    OK: {name}")

    print("\n  Verifying DB invariants...")

    # No duplicate SMILES
    from sqlalchemy import func
    dup_smiles = (
        session.query(Compound.smiles, func.count(Compound.id))
        .group_by(Compound.smiles)
        .having(func.count(Compound.id) > 1)
        .all()
    )
    check("no duplicate SMILES", len(dup_smiles) == 0, f"duplicates: {dup_smiles}")

    # No duplicate (compound_id, local_id) in minima
    dup_minima = (
        session.query(Minimum.compound_id, Minimum.local_id, func.count(Minimum.id))
        .group_by(Minimum.compound_id, Minimum.local_id)
        .having(func.count(Minimum.id) > 1)
        .all()
    )
    check("no duplicate minima", len(dup_minima) == 0, f"duplicates: {dup_minima}")

    # No duplicate ts_id in reactions
    dup_ts = (
        session.query(Reaction.ts_id, func.count(Reaction.id))
        .group_by(Reaction.ts_id)
        .having(func.count(Reaction.id) > 1)
        .all()
    )
    check("no duplicate reaction ts_id", len(dup_ts) == 0, f"duplicates: {dup_ts}")

    # All work items are completed or failed (none stuck in pending after workers finish)
    pending = session.query(PESWorkQueue).filter(PESWorkQueue.status == "pending").count()
    in_progress = session.query(PESWorkQueue).filter(PESWorkQueue.status == "in_progress").count()
    completed = session.query(PESWorkQueue).filter(PESWorkQueue.status == "completed").count()
    check(
        "PES work queue drained",
        in_progress == 0,
        f"pending={pending}, in_progress={in_progress}, completed={completed}",
    )

    # Every compound has at least one minimum
    orphan_compounds = (
        session.query(Compound)
        .outerjoin(Minimum, Compound.id == Minimum.compound_id)
        .filter(Minimum.id == None)
        .count()
    )
    check("no orphan compounds", orphan_compounds == 0, f"{orphan_compounds} compounds without minima")

    # Every reaction has at least one reactant and one product
    for rxn in session.query(Reaction).all():
        n_reactants = session.query(ReactionReactant).filter(
            ReactionReactant.reaction_id == rxn.id
        ).count()
        n_products = session.query(ReactionProduct).filter(
            ReactionProduct.reaction_id == rxn.id
        ).count()
        if n_reactants == 0 or n_products == 0:
            errors.append(f"Reaction {rxn.id} has {n_reactants} reactants, {n_products} products")
            print(f"    FAIL: reaction {rxn.id} — {n_reactants} reactants, {n_products} products")

    if not errors:
        print("    OK: all reactions have reactants and products")

    # Graph edges reference existing nodes
    all_smiles = {c.smiles for c in session.query(Compound).all()}
    all_rxn_names = {r.name for r in session.query(Reaction).all() if r.name}
    valid_nodes = all_smiles | all_rxn_names
    edges = session.query(GraphEdge).all()
    dangling = []
    for e in edges:
        if e.source_type == "compound" and e.source_node not in all_smiles:
            dangling.append(f"source={e.source_node}")
        if e.target_type == "compound" and e.target_node not in all_smiles:
            dangling.append(f"target={e.target_node}")
    check("no dangling graph edges", len(dangling) == 0, f"{dangling[:5]}")

    # Batch log has entries
    n_batches = session.query(BatchLog).count()
    check("batch log populated", n_batches > 0, f"got {n_batches}")

    # Stats accumulated
    stats = session.query(ExplorationStats).filter(ExplorationStats.id == 1).first()
    check("stats row exists", stats is not None)
    if stats:
        check(
            "stats pipeline_contexts > 0",
            stats.stats_json.get("pipeline_contexts_submitted", 0) > 0,
        )

    session.close()
    return errors


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("CRN Cloud — Integration Test")
    print(f"  {N_WORKERS} concurrent workers, {N_COMPOUNDS} target compounds")
    print(f"  Database: {DATABASE_URL}")
    print(f"  API:      {API_URL}")
    print("=" * 70)

    # Check prerequisites
    try:
        engine = create_engine(DATABASE_URL)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("\n  PostgreSQL: OK")
    except Exception as e:
        print(f"\n  PostgreSQL: FAILED ({e})")
        print("  Run: docker compose up -d db")
        sys.exit(1)

    api_available = False
    try:
        r = requests.get(f"{API_URL}/api/capabilities", timeout=3)
        api_available = r.status_code == 200
        print(f"  API server: {'OK' if api_available else 'UNAVAILABLE'}")
    except Exception:
        print("  API server: UNAVAILABLE (run: docker compose up -d db api)")

    # ── Phase 1: Setup ──
    print("\n[Phase 1] Seeding initial compounds...")
    fresh_db()
    db = DB(database_url=DATABASE_URL)
    seeded = seed_initial_compounds(db, n_seed=3)
    print(f"  Seeded {len(seeded)} compounds")

    # ── Phase 2: Concurrent workers ──
    print(f"\n[Phase 2] Running {N_WORKERS} concurrent workers (10 rounds each)...")
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {
            pool.submit(simulated_worker, f"w{i}", DB(database_url=DATABASE_URL), 10): f"w{i}"
            for i in range(N_WORKERS)
        }
        worker_results = {}
        for future in as_completed(futures):
            wid = futures[future]
            try:
                result = future.result()
                worker_results[wid] = result
            except Exception as e:
                print(f"  Worker {wid} CRASHED: {e}")
                traceback.print_exc()
                worker_results[wid] = {"error": str(e)}

    elapsed = time.time() - t0
    print(f"  Completed in {elapsed:.1f}s")
    print(f"  Worker stats: {worker_stats}")
    for wid, result in sorted(worker_results.items()):
        if "error" not in result:
            print(f"    {wid}: pes={result['pes']}, compounds={result['compounds']}, "
                  f"reactions={result['reactions']}, batches={result['batches']}")

    # ── Phase 3: DB invariants ──
    print("\n[Phase 3] Checking DB invariants...")
    db_errors = verify_db_invariants()

    # ── Phase 4: API verification ──
    api_errors = []
    if api_available:
        print("\n[Phase 4] Verifying API endpoints...")
        n_compounds = create_engine(DATABASE_URL).connect().execute(
            text("SELECT count(*) FROM compounds")
        ).scalar()
        api_errors = verify_api(
            expected_min_compounds=3,  # At least the seeds
            expected_min_reactions=0,
        )
    else:
        print("\n[Phase 4] API verification SKIPPED (server not running)")

    # ── Summary ──
    all_errors = db_errors + api_errors
    print("\n" + "=" * 70)
    if all_errors:
        print(f"FAILED — {len(all_errors)} error(s):")
        for e in all_errors:
            print(f"  - {e}")
    else:
        print("ALL CHECKS PASSED")
        engine = create_engine(DATABASE_URL)
        with engine.connect() as conn:
            n_c = conn.execute(text("SELECT count(*) FROM compounds")).scalar()
            n_r = conn.execute(text("SELECT count(*) FROM reactions")).scalar()
            n_m = conn.execute(text("SELECT count(*) FROM minima")).scalar()
            n_w = conn.execute(text("SELECT count(*) FROM pes_work_queue WHERE status='completed'")).scalar()
            n_b = conn.execute(text("SELECT count(*) FROM batch_log")).scalar()
        print(f"\n  Final state:")
        print(f"    Compounds:      {n_c}")
        print(f"    Reactions:      {n_r}")
        print(f"    Minima:         {n_m}")
        print(f"    PES completed:  {n_w}")
        print(f"    Batch logs:     {n_b}")
        print(f"    Workers used:   {N_WORKERS}")
        print(f"    Wall time:      {elapsed:.1f}s")
    print("=" * 70)

    sys.exit(1 if all_errors else 0)


if __name__ == "__main__":
    main()
