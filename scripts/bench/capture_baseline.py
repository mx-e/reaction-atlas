"""Capture query result baselines for consistency checking after optimizations.

Saves deterministic summaries of the outputs of hot queries to
/tmp/crn-bench/baseline.json. After optimizations are applied the same
script can run in verify mode and compare against the stored baseline —
any drift indicates a correctness bug.

Usage:
    /tmp/crn-bench/venv/bin/python scripts/bench/capture_baseline.py         # save
    /tmp/crn-bench/venv/bin/python scripts/bench/capture_baseline.py verify  # compare
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

from sqlalchemy import create_engine, text

DB_URL = os.environ.get("DATABASE_URL", "postgresql://crn:crn@localhost:5433/crn_cloud")
BASELINE_PATH = Path("/tmp/crn-bench/baseline.json")

WORK_TIMEOUT_S = 1800.0
# Use a fixed "now" for reproducibility so the timestamp-dependent queries
# return identical results pre- and post-fix.
FIXED_NOW_EPOCH = 1_766_000_000.0  # 2025-12-18 — safely in the past vs all claim timestamps
TIMEOUT_THRESHOLD = FIXED_NOW_EPOCH - WORK_TIMEOUT_S


def _hash(obj) -> str:
    """Stable SHA-256 of any JSON-serializable structure."""
    s = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha256(s.encode()).hexdigest()


def capture(engine) -> dict:
    out: dict = {}
    with engine.connect() as conn:
        # --- Counts ---
        out["count_compounds"] = conn.execute(text("SELECT COUNT(*) FROM compounds")).scalar()
        out["count_reactions"] = conn.execute(text("SELECT COUNT(*) FROM reactions")).scalar()
        out["count_minima"] = conn.execute(text("SELECT COUNT(*) FROM minima")).scalar()
        out["count_pes_pending"] = conn.execute(text(
            "SELECT COUNT(*) FROM pes_work_queue WHERE status = 'pending'"
        )).scalar()
        out["count_pes_in_progress"] = conn.execute(text(
            "SELECT COUNT(*) FROM pes_work_queue WHERE status = 'in_progress'"
        )).scalar()
        out["count_pes_total"] = conn.execute(text("SELECT COUNT(*) FROM pes_work_queue")).scalar()
        out["count_kinetics_snapshots"] = conn.execute(text("SELECT COUNT(*) FROM kinetics_snapshots")).scalar()

        # --- Latest snapshot ---
        row = conn.execute(text("""
            SELECT id, network_version, n_reactions_dft, temperature,
                   payload_jsonb::text AS payload
            FROM kinetics_snapshots
            ORDER BY computed_at DESC LIMIT 1
        """)).fetchone()
        if row:
            payload = json.loads(row[4])
            out["latest_snapshot_id"] = row[0]
            out["latest_snapshot_network_version"] = row[1]
            out["latest_snapshot_n_reactions_dft"] = row[2]
            out["latest_snapshot_temperature"] = float(row[3])
            # Summaries of the payload — we don't store the whole thing (270 KB) but
            # hash what matters so we can detect any drift.
            out["latest_snapshot_n_species"] = len(payload.get("smiles_list", []))
            out["latest_snapshot_steady_state_log_concs_keys_count"] = len(payload.get("steady_state_log_concs", {}))
            out["latest_snapshot_steady_state_log_concs_hash"] = _hash(payload.get("steady_state_log_concs", {}))
            out["latest_snapshot_steady_state_distribution_hash"] = _hash(payload.get("steady_state_distribution", {}))
            out["latest_snapshot_n_decades"] = len(payload.get("decade_distributions", []))
            out["latest_snapshot_n_reactions"] = payload.get("n_reactions")
            out["latest_snapshot_payload_top_keys"] = sorted(payload.keys())
            out["latest_snapshot_payload_size_bytes"] = len(row[4])

        # --- PES queue scan state (first 2000 items from the gated scan query) ---
        rows = conn.execute(text(f"""
            SELECT q.id, c.smiles
              FROM pes_work_queue q
              JOIN compounds c ON c.id = q.compound_id
             WHERE (q.status = 'pending'
                    OR (q.status = 'in_progress'
                        AND q.claimed_at < to_timestamp(:t)))
               AND NOT EXISTS (
                   SELECT 1 FROM pes_work_queue other
                    WHERE other.compound_id = q.compound_id
                      AND other.status = 'in_progress'
                      AND other.claimed_at >= to_timestamp(:t)
               )
             ORDER BY q.id
             LIMIT 2000
        """), {"t": TIMEOUT_THRESHOLD}).fetchall()
        scan_items = [(int(r[0]), r[1]) for r in rows]
        out["pes_scan_count"] = len(scan_items)
        out["pes_scan_first_20"] = scan_items[:20]
        out["pes_scan_last_20"] = scan_items[-20:]
        out["pes_scan_hash"] = _hash(scan_items)

        # --- Pair reaction counts (the joinedload query) ---
        rows = conn.execute(text("""
            SELECT r.id,
                   rr.compound_id AS r_cid, rc.smiles AS r_smi,
                   rp.compound_id AS p_cid, pc.smiles AS p_smi
              FROM reactions r
              LEFT JOIN reaction_reactants rr ON rr.reaction_id = r.id
              LEFT JOIN compounds rc ON rc.id = rr.compound_id
              LEFT JOIN reaction_products rp ON rp.reaction_id = r.id
              LEFT JOIN compounds pc ON pc.id = rp.compound_id
             WHERE (r.discovery_method != 'manual_equilibrium' OR r.discovery_method IS NULL)
             ORDER BY r.id, rr.id, rp.id
        """)).fetchall()
        # Reconstruct pair_rxn_counts deterministically (matches worker logic)
        from collections import defaultdict
        pair_counts: dict[frozenset, int] = defaultdict(int)
        by_rxn: dict[int, dict[str, set]] = {}
        for rid, rcid, rsmi, pcid, psmi in rows:
            d = by_rxn.setdefault(rid, {"smi": set()})
            if rsmi:
                d["smi"].add(rsmi)
            if psmi:
                d["smi"].add(psmi)
        for rid, d in by_rxn.items():
            if d["smi"]:
                pair_counts[frozenset(d["smi"])] += 1
        # Convert frozenset keys into sorted tuples for deterministic serialization
        pair_summary = sorted([(sorted(list(k)), v) for k, v in pair_counts.items()])
        out["pair_rxn_counts_n_distinct_pairs"] = len(pair_summary)
        out["pair_rxn_counts_hash"] = _hash(pair_summary)
        out["pair_rxn_counts_total_reactions"] = sum(v for _, v in pair_summary)

        # --- Compounds for sampling (bulk join) ---
        rows = conn.execute(text("""
            SELECT m.id, m.compound_id, m.local_id,
                   c.id AS cid, c.smiles, c.formula, c.n_atoms, c.charge
              FROM minima m
              JOIN compounds c ON c.id = m.compound_id
             WHERE m.local_id >= 0
             ORDER BY m.id
        """)).fetchall()
        # Group by compound, mirror the worker's compound_minima structure
        by_cid: dict[int, dict] = {}
        for m_id, cid, lid, c_id, smi, formula, n_atoms, charge in rows:
            d = by_cid.setdefault(
                cid,
                {"smiles": smi, "formula": formula, "n_atoms": n_atoms, "charge": charge, "minima": []},
            )
            d["minima"].append((int(m_id), int(lid)))
        out["compounds_for_sampling_n_compounds"] = len(by_cid)
        out["compounds_for_sampling_total_minima"] = sum(len(v["minima"]) for v in by_cid.values())
        serialisable = sorted([(cid, v) for cid, v in by_cid.items()])
        out["compounds_for_sampling_hash"] = _hash(serialisable)

    # --- Endpoint-level aggregation: /reaction-graph shape (equation nodes) ---
    # Runs the actual _build_reaction_graph_locked to pin the aggregation logic.
    try:
        import sys, pathlib, types
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
        stub = types.ModuleType("packages.api.db_session")
        stub.get_session = None
        sys.modules["packages.api.db_session"] = stub

        from packages.api.routers.compounds import (
            _build_reaction_graph_locked, _graph_cache, equation_ts_list,
        )
        from sqlalchemy.orm import sessionmaker
        Session = sessionmaker(bind=engine)
        with _graph_cache.lock:
            _graph_cache.invalidate()
        with Session() as s:
            result = _build_reaction_graph_locked(s, _graph_cache)

        types_ct: dict[str, int] = {}
        for n in result["nodes"]:
            types_ct[n["type"]] = types_ct.get(n["type"], 0) + 1
        out["graph_endpoint_compound_count"] = types_ct.get("compound", 0)
        out["graph_endpoint_equation_count"] = types_ct.get("equation", 0)
        out["graph_endpoint_edge_count"] = len(result["edges"])
        out["graph_endpoint_sum_n_ts"] = sum(
            n.get("n_ts", 0) for n in result["nodes"] if n["type"] == "equation"
        )
        eq_ids = sorted(n["id"] for n in result["nodes"] if n["type"] == "equation")
        out["graph_endpoint_equation_ids_hash"] = _hash(eq_ids)
        out["graph_endpoint_equations_with_dft"] = sum(
            1 for n in result["nodes"] if n["type"] == "equation" and n.get("has_dft")
        )
        merge_eq = [n for n in result["nodes"] if n["type"] == "equation"
                    and sum(1 for e in result["edges"] if e["target"] == n["id"]) > 1]
        out["graph_endpoint_merge_equations"] = len(merge_eq)

        # Sanity: drill-in for the biggest equation returns the expected count
        biggest = max((n for n in result["nodes"] if n["type"] == "equation"),
                      key=lambda n: n.get("n_ts", 0), default=None)
        if biggest is not None:
            with Session() as s:
                tss = equation_ts_list(biggest["id"], s)
            out["graph_endpoint_biggest_eq_n_ts"] = biggest["n_ts"]
            out["graph_endpoint_biggest_eq_drillin_count"] = len(tss)
    except Exception as e:
        out["graph_endpoint_error"] = str(e)

    out["_baseline_captured_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    out["_schema_notes"] = "DB_URL points to the local bench DB with the restored prod snapshot."
    return out


def _normalize(v):
    """Recursively convert tuples→lists for apples-to-apples comparison with
    JSON-roundtripped baseline (which only has lists)."""
    if isinstance(v, tuple):
        return [_normalize(x) for x in v]
    if isinstance(v, list):
        return [_normalize(x) for x in v]
    if isinstance(v, dict):
        return {k: _normalize(x) for k, x in v.items()}
    return v


def verify(engine) -> int:
    if not BASELINE_PATH.exists():
        print(f"ERROR: no baseline at {BASELINE_PATH}")
        return 1
    prior = json.loads(BASELINE_PATH.read_text())
    now = capture(engine)

    # Compare every key that exists in both (skip _meta keys). Normalize both
    # sides through the JSON-compatible form so tuples/lists match.
    diff_keys = []
    for k in sorted(set(prior) | set(now)):
        if k.startswith("_"):
            continue
        if _normalize(prior.get(k)) != _normalize(now.get(k)):
            diff_keys.append(k)

    if not diff_keys:
        print(f"OK — all {len([k for k in prior if not k.startswith('_')])} baseline fields match.")
        return 0

    print(f"DRIFT in {len(diff_keys)} fields:")
    for k in diff_keys:
        p, n = prior.get(k), now.get(k)
        if isinstance(p, (list, dict)) or isinstance(n, (list, dict)):
            print(f"  {k}: prior={str(p)[:80]}... now={str(n)[:80]}...")
        else:
            print(f"  {k}: prior={p!r} now={n!r}")
    return 2


def main():
    engine = create_engine(DB_URL)
    if len(sys.argv) > 1 and sys.argv[1] == "verify":
        sys.exit(verify(engine))
    BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = capture(engine)
    BASELINE_PATH.write_text(json.dumps(data, indent=2, sort_keys=True, default=str))
    print(f"Captured {len([k for k in data if not k.startswith('_')])} fields → {BASELINE_PATH}")
    # Print a compact summary
    for k in sorted(data):
        if k.startswith("_"):
            continue
        v = data[k]
        if isinstance(v, (list, dict)):
            print(f"  {k}: {type(v).__name__} (len={len(v) if hasattr(v, '__len__') else '?'})")
        elif isinstance(v, str) and len(v) > 80:
            print(f"  {k}: {v[:77]}...")
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
