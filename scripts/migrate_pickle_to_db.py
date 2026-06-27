#!/usr/bin/env python3
"""Migrate a pickle-based exploration state + GEXF graph into the PostgreSQL DB.

Usage:
    python scripts/migrate_pickle_to_db.py test_data/n111/

Reads:
    - reaction_graph.gexf (compound/TS nodes + edges)
    - .reaction_graph_state.pkl (full state with CompoundRegistry, PES graphs)
    - pes_graphs/*.gexf (individual PES graphs)

Writes to the DB specified by DATABASE_URL.
"""

import os
import sys
import time
import pickle
import numpy as np
import networkx as nx
from pathlib import Path
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, str(Path(__file__).parent.parent))

# Stub classes needed for unpickling the pickle file
# The pickle references lib.reaction_graph classes — we create a fake module
# with stub classes so pickle.load() can reconstruct the objects.
import types

class _Stub:
    """Generic stub that accepts any attribute access for unpickling."""
    def __setstate__(self, state):
        self.__dict__.update(state)

class ReactionRegistry(_Stub): pass
class ReactionEntry(_Stub): pass
class ReactantEntry(_Stub): pass
class _Reaction(_Stub): pass
class NameGenerator(_Stub): pass
class ReactionGraph(_Stub): pass

# Create auto-stubbing module hierarchy for pickle deserialization
class _StubModule(types.ModuleType):
    """Module that returns a new _Stub subclass for any undefined attribute."""
    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        cls = type(name, (_Stub,), {})
        setattr(self, name, cls)
        return cls

rg_module = _StubModule("lib.reaction_graph")
rg_module.ReactionRegistry = ReactionRegistry
rg_module.ReactionEntry = ReactionEntry
rg_module.ReactantEntry = ReactantEntry
rg_module.Reaction = _Reaction
rg_module.NameGenerator = NameGenerator
rg_module.ReactionGraph = ReactionGraph

lib_module = _StubModule("lib")
lib_module.reaction_graph = rg_module

sys.modules["lib"] = lib_module
sys.modules["lib.reaction_graph"] = rg_module

# Register stub modules for all lib subpackages the pickle might reference
for submod in ["compound", "types", "pes_explorer", "pes_explorer.pes_graph",
               "pes_explorer.prfo", "energy", "fragment_mols", "naming", "utils",
               "md", "merge_mols", "md_et_calculator", "constants", "graph_analysis"]:
    mod = _StubModule(f"lib.{submod}")
    sys.modules[f"lib.{submod}"] = mod
    # Wire into parent module
    parts = submod.split(".")
    parent = lib_module
    for i, p in enumerate(parts[:-1]):
        full = ".".join(parts[:i+1])
        if full not in sys.modules:
            sys.modules[full] = _StubModule(full)
        parent = sys.modules[full]
    setattr(parent, parts[-1], mod)

from packages.db.models import Base, Compound, Minimum, IntraTransitionState, Reaction, \
    ReactionReactant, ReactionProduct, GraphEdge, PESWorkQueue, ExplorationStats, BatchLog
from packages.db.serialization import serialize_ndarray, serialize_trajectory

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql+psycopg://crn:crn@localhost:5432/crn_cloud")


def _to_numpy(obj):
    """Convert a value to numpy array, handling torch tensors and plain arrays."""
    if hasattr(obj, 'cpu'):  # torch tensor
        return np.array(obj.cpu())
    return np.array(obj)


def migrate(data_dir: str):
    data_path = Path(data_dir)

    # Read GEXF
    gexf_path = data_path / "reaction_graph.gexf"
    print(f"Loading GEXF from {gexf_path}...")
    g = nx.read_gexf(str(gexf_path))

    compounds = {n: d for n, d in g.nodes(data=True) if d.get("type") == "compound"}
    ts_nodes = {n: d for n, d in g.nodes(data=True) if d.get("type") == "ts"}
    print(f"  {len(compounds)} compounds, {len(ts_nodes)} TS nodes, {g.number_of_edges()} edges")

    # Try loading pickle for PES data
    pkl_path = data_path / ".reaction_graph_state.pkl"
    state = None
    compound_registry = None
    # Build SMILES->Compound lookup from pickle
    smiles_to_compound_obj = {}
    if pkl_path.exists():
        print(f"Loading pickle state from {pkl_path}...")
        with open(pkl_path, "rb") as f:
            state = pickle.load(f)
        if isinstance(state, dict):
            compound_registry = state.get("compound_registry")
        elif hasattr(state, "compound_registry"):
            compound_registry = state.compound_registry
        else:
            compound_registry = None
        if compound_registry is not None:
            comp_list = getattr(compound_registry, "compounds", None) or getattr(compound_registry, "_compounds", None)
            if isinstance(comp_list, list):
                for c in comp_list:
                    smiles_to_compound_obj[c.smiles] = c
            elif isinstance(comp_list, dict):
                smiles_to_compound_obj = comp_list
            print(f"  CompoundRegistry: {len(smiles_to_compound_obj)} compounds")

    # Connect to DB
    print(f"Connecting to {DATABASE_URL}...")
    engine = create_engine(DATABASE_URL)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    # Clear existing data
    print("Clearing existing data...")
    for table in [GraphEdge, ReactionProduct, ReactionReactant, Reaction,
                  IntraTransitionState, PESWorkQueue, Minimum, BatchLog,
                  ExplorationStats, Compound]:
        session.query(table).delete()
    session.commit()

    # Ensure ExplorationStats row
    session.add(ExplorationStats(id=1, stats_json={}))
    session.flush()

    # Phase 1: Insert compounds
    print("Migrating compounds...")
    smiles_to_db_id = {}
    n_with_pes = 0
    for smiles, data in compounds.items():
        # Get real data from pickle if available
        comp_obj = smiles_to_compound_obj.get(smiles)

        if comp_obj is not None:
            sorted_anum = _to_numpy(comp_obj.sorted_atomic_numbers).astype(np.int32)
            n_atoms = len(sorted_anum)
            formula = comp_obj.formula
        else:
            sorted_anum = np.array([6], dtype=np.int32)  # fallback
            n_atoms = int(data.get("n_atoms", 1))
            formula = data.get("formula", smiles)

        energy = float(data.get("energy", 0.0))
        is_seed = str(data.get("is_seed", "false")).lower() == "true"

        # Compute formal charge from SMILES via RDKit
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        charge = Chem.GetFormalCharge(mol) if mol is not None else 0

        compound = Compound(
            smiles=smiles,
            formula=formula,
            charge=charge,
            n_atoms=n_atoms,
            sorted_atomic_numbers=serialize_ndarray(sorted_anum),
            is_seed=is_seed,
        )
        session.add(compound)
        session.flush()
        smiles_to_db_id[smiles] = compound.id

        # Add minima and intra TS from PES graph
        if comp_obj is not None and hasattr(comp_obj, "pes_graph") and comp_obj.pes_graph is not None:
            n_with_pes += 1
            # Insert minima and build local_id -> db_id mapping
            pes_local_to_db = {}
            for min_id, minimum in comp_obj.pes_graph.minima.items():
                m = Minimum(
                    compound_id=compound.id,
                    local_id=min_id,
                    positions=serialize_ndarray(_to_numpy(minimum.positions)),
                    energy=float(minimum.energy),
                    explored=minimum.explored if hasattr(minimum, "explored") else False,
                    discovery_timestamp=float(getattr(minimum, "discovery_timestamp", 0.0)),
                    name=getattr(minimum, "name", ""),
                    n_merged=int(getattr(minimum, "n_merged", 0)),
                    max_merge_rmsd=float(getattr(minimum, "max_merge_rmsd", 0.0)),
                )
                if hasattr(minimum, "hessian") and minimum.hessian is not None:
                    m.hessian = serialize_ndarray(_to_numpy(minimum.hessian))
                session.add(m)
                session.flush()
                pes_local_to_db[min_id] = m.id

            if hasattr(comp_obj.pes_graph, "transition_states"):
                for ts_id, ts in comp_obj.pes_graph.transition_states.items():
                    fwd_id = getattr(ts, "min_fwd_id", 0)
                    bwd_id = getattr(ts, "min_bwd_id", 0)
                    # Map PES-local IDs to DB IDs; skip if mapping missing
                    db_fwd = pes_local_to_db.get(fwd_id)
                    db_bwd = pes_local_to_db.get(bwd_id)
                    if db_fwd is None or db_bwd is None:
                        continue
                    its = IntraTransitionState(
                        compound_id=compound.id,
                        local_id=ts_id,
                        positions=serialize_ndarray(_to_numpy(ts.positions)),
                        energy=float(ts.energy),
                        eigenvalue=float(ts.eigenvalue) if hasattr(ts, "eigenvalue") else 0.0,
                        hessian=serialize_ndarray(_to_numpy(ts.hessian)) if getattr(ts, "hessian", None) is not None else None,
                        min_fwd_id=db_fwd,
                        min_bwd_id=db_bwd,
                        barrier_fwd=float(ts.barrier_fwd) if hasattr(ts, "barrier_fwd") else 0.0,
                        barrier_bwd=float(ts.barrier_bwd) if hasattr(ts, "barrier_bwd") else 0.0,
                        rmsd_to_fwd_min=float(getattr(ts, "rmsd_to_fwd_min", 0.0)),
                        rmsd_to_bwd_min=float(getattr(ts, "rmsd_to_bwd_min", 0.0)),
                        endpoint_to_endpoint_rmsd=float(getattr(ts, "endpoint_to_endpoint_rmsd", 0.0)),
                        fwd_trajectory=serialize_trajectory(getattr(ts, "fwd_trajectory", None)),
                        bwd_trajectory=serialize_trajectory(getattr(ts, "bwd_trajectory", None)),
                        discovery_timestamp=float(getattr(ts, "discovery_timestamp", 0.0)),
                    )
                    session.add(its)
        else:
            # No PES data — add a minimum using initial positions if available
            if comp_obj is not None:
                pos = _to_numpy(comp_obj.initial_positions)
            else:
                pos = np.zeros((n_atoms, 3))
            m = Minimum(
                compound_id=compound.id,
                local_id=0,
                positions=serialize_ndarray(pos),
                energy=energy,
                explored=True,
                discovery_timestamp=float(getattr(comp_obj, "discovery_timestamp", 0.0)) if comp_obj else 0.0,
            )
            session.add(m)

    session.flush()
    print(f"  Inserted {len(smiles_to_db_id)} compounds ({n_with_pes} with PES data)")

    # Build reaction registry lookup from pickle (by name and ts_id)
    pkl_reactions_by_name = {}
    pkl_reactions_by_ts_id = {}
    if isinstance(state, dict) and "reaction_registry" in state:
        rr = state["reaction_registry"]
        rxn_list = rr._reactions if hasattr(rr, "_reactions") else {}
        if isinstance(rxn_list, dict):
            for rxn_obj in rxn_list.values():
                if hasattr(rxn_obj, "name") and rxn_obj.name:
                    pkl_reactions_by_name[rxn_obj.name] = rxn_obj
                if hasattr(rxn_obj, "ts_id"):
                    pkl_reactions_by_ts_id[str(rxn_obj.ts_id)] = rxn_obj
        print(f"  ReactionRegistry: {len(pkl_reactions_by_name)} reactions by name")

    # Phase 2: Insert reactions from TS nodes + edges
    print("Migrating reactions...")
    n_reactions = 0
    ts_name_to_db_id = {}

    for ts_name, ts_data in ts_nodes.items():
        ts_id = int(ts_data.get("ts_id", hash(ts_name) % (2**31)))
        ts_energy = float(ts_data.get("energy", 0.0))

        # Find reactants and products from edges
        # Edges: compound -> ts (reactant, direction=up/merge)
        #        ts -> compound (product, direction=down)
        reactant_smiles = []
        product_smiles = []
        for u, v, edata in g.edges(data=True):
            if v == ts_name and u in compounds:
                reactant_smiles.append(u)
            elif u == ts_name and v in compounds:
                product_smiles.append(v)

        if not reactant_smiles or not product_smiles:
            continue

        # Get real TS conformer data from pickle if available
        pkl_rxn = pkl_reactions_by_name.get(ts_name) or pkl_reactions_by_ts_id.get(str(ts_id))

        if pkl_rxn is not None and hasattr(pkl_rxn, "ts_conformer") and pkl_rxn.ts_conformer is not None:
            tc = pkl_rxn.ts_conformer
            if hasattr(tc, "to_numpy"):
                tc = tc.to_numpy()
            ts_positions = _to_numpy(tc.positions)
            ts_anum = _to_numpy(tc.atomic_numbers).astype(np.int32).flatten()
            ts_charge = getattr(pkl_rxn.ts_conformer, "charge", 0)
            barrier_fwd = float(getattr(pkl_rxn, "barrier_forward", 0.0))
            barrier_bwd = float(getattr(pkl_rxn, "barrier_backward", 0.0))
            discovery_method = getattr(pkl_rxn, "discovery_method", "generative")
            discovery_noise = getattr(pkl_rxn, "discovery_noise_level", None)
            discovery_ts = getattr(pkl_rxn, "discovery_timestamp", None)
        else:
            n_atoms_ts = max(
                int(compounds.get(s, {}).get("n_atoms", 4))
                for s in reactant_smiles
            )
            ts_positions = np.zeros((n_atoms_ts, 3))
            ts_anum = np.zeros(n_atoms_ts, dtype=np.int32)
            ts_charge = 0
            barrier_fwd = 0.0
            barrier_bwd = 0.0
            discovery_method = ts_data.get("discovery_method", "generative")
            discovery_noise = None
            discovery_ts = None

        # Trajectories from pickle
        reactant_traj = None
        product_traj = None
        if pkl_rxn is not None:
            reactant_traj = serialize_trajectory(getattr(pkl_rxn, "reactant_trajectory", None))
            product_traj = serialize_trajectory(getattr(pkl_rxn, "product_trajectory", None))

        reaction = Reaction(
            ts_id=ts_id,
            ts_conformer_positions=serialize_ndarray(ts_positions),
            ts_conformer_atomic_numbers=serialize_ndarray(ts_anum),
            ts_conformer_charge=ts_charge,
            ts_energy=ts_energy,
            barrier_forward=barrier_fwd,
            barrier_backward=barrier_bwd,
            reactant_trajectory=reactant_traj,
            product_trajectory=product_traj,
            name=ts_name,
            discovery_method=discovery_method,
            discovery_noise_level=discovery_noise,
            discovery_timestamp=discovery_ts,
        )
        session.add(reaction)
        session.flush()
        ts_name_to_db_id[ts_name] = reaction.id

        # Get conformer IDs from pickle if available
        pkl_reactants = {}  # smiles -> conformer_id
        pkl_products = {}   # smiles -> (conformer_id, energy)
        if pkl_rxn is not None:
            for r in getattr(pkl_rxn, "reactants", []):
                pkl_reactants[r.smiles] = getattr(r, "conformer_id", 0)
            for p in getattr(pkl_rxn, "products", []):
                pkl_products[p.smiles] = (getattr(p, "conformer_id", 0), getattr(p, "energy", 0.0))

        for smiles in reactant_smiles:
            if smiles in smiles_to_db_id:
                session.add(ReactionReactant(
                    reaction_id=reaction.id,
                    compound_id=smiles_to_db_id[smiles],
                    conformer_local_id=pkl_reactants.get(smiles, 0),
                ))

        for smiles in product_smiles:
            if smiles in smiles_to_db_id:
                pkl_prod = pkl_products.get(smiles, (0, 0.0))
                c_energy = float(pkl_prod[1]) if pkl_prod[1] else float(compounds.get(smiles, {}).get("energy", 0.0))
                session.add(ReactionProduct(
                    reaction_id=reaction.id,
                    compound_id=smiles_to_db_id[smiles],
                    conformer_local_id=pkl_prod[0],
                    energy=c_energy,
                ))

        n_reactions += 1

    session.flush()
    print(f"  Inserted {n_reactions} reactions")

    # Phase 3: Insert graph edges
    print("Migrating graph edges...")
    n_edges = 0
    for u, v, edata in g.edges(data=True):
        direction = edata.get("direction", "")
        stoichiometry = int(edata.get("stoichiometry", 1))

        if u in compounds:
            source_type = "compound"
        elif u in ts_nodes:
            source_type = "ts"
        else:
            continue

        if v in compounds:
            target_type = "compound"
        elif v in ts_nodes:
            target_type = "ts"
        else:
            continue

        reaction_id = ts_name_to_db_id.get(u) or ts_name_to_db_id.get(v)

        edge = GraphEdge(
            source_node=u,
            target_node=v,
            source_type=source_type,
            target_type=target_type,
            direction=direction,
            stoichiometry=stoichiometry,
            reaction_id=reaction_id,
        )
        session.add(edge)
        n_edges += 1

    session.flush()
    print(f"  Inserted {n_edges} graph edges")

    # Phase 4: Migrate exploration stats
    n_minima = session.query(Minimum).count()
    n_intra_ts = session.query(IntraTransitionState).count()

    exploration_stats_json = {}
    if isinstance(state, dict):
        # Merge all stat dicts from pickle
        for key in ["exploration_stats", "decomposition_stats", "energy_validation_stats", "stats"]:
            d = state.get(key, {})
            if isinstance(d, dict):
                exploration_stats_json.update(d)
    exploration_stats_json["migrated_from"] = str(data_path)
    exploration_stats_json["conformers_total"] = n_minima
    exploration_stats_json["intramolecular_ts_total"] = n_intra_ts

    stats_row = session.query(ExplorationStats).filter(ExplorationStats.id == 1).first()
    stats_row.stats_json = exploration_stats_json

    # Phase 5: Migrate batch log
    n_batches = 0
    if isinstance(state, dict) and "batch_log" in state:
        batch_log = state["batch_log"]
        print(f"Migrating {len(batch_log)} batch log entries...")
        for entry in batch_log:
            if isinstance(entry, dict):
                bl = BatchLog(
                    summary_json=entry,
                    batch_idx=entry.get("batch_idx"),
                )
                session.add(bl)
                n_batches += 1
        session.flush()
        print(f"  Inserted {n_batches} batch log entries")

    session.commit()
    session.close()

    print()
    print("=" * 60)
    print(f"Migration complete!")
    print(f"  Compounds:    {len(compounds)}")
    print(f"  Reactions:    {n_reactions}")
    print(f"  Minima:       {n_minima}")
    print(f"  Intra TS:     {n_intra_ts}")
    print(f"  Graph edges:  {n_edges}")
    print("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <data_dir>")
        sys.exit(1)
    migrate(sys.argv[1])
