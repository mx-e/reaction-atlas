"""Seed manual buffer-equilibrium reactions into the database.

The kinetic ODE simulation diverges to nonsense without proton/water buffering
because the diffusion model can't discover acid-base equilibria. We seed four
of them at startup with literature rate constants — discovered reactions that
follow Eyring/TST then layer on top.

Called once at GPU worker startup from inside _seed_initial_compounds() under
the same advisory lock that seeds the start molecule + fragments. Idempotent:
checks for existing rows by synthetic negative ts_id.

Rate constants and SMILES match the upstream BUFFER_EQUILIBRIA constant in
crn-exploration/lib/kinetic_sampler.py exactly.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger
from sqlalchemy.orm import Session

from packages.db.models import (
    Compound,
    Minimum,
    Reaction,
    ReactionProduct,
    ReactionReactant,
    GraphEdge,
)
from packages.db.serialization import serialize_ndarray


# H+ in our graph is [HH] — single H atom; the MLFF has no electrons concept,
# so "neutral H" and "proton" share the same canonical SMILES.
H_PLUS_SMILES = "[HH]"

# Atomic symbol → number for parsing XYZ files
_SYM_TO_Z = {
    "H": 1, "C": 6, "N": 7, "O": 8, "F": 9,
    "P": 15, "S": 16, "Cl": 17,
}

# Placeholder buffer compounds. Each entry:
#   (smiles, formula, charge, atomic_numbers, xyz_filename)
# The xyz_filename is resolved against $FRAGMENT_PATH (defaults to
# /app/data/fragments inside the container, ./data/fragments locally).
# atomic_numbers must match the XYZ file content.
BUFFER_COMPOUNDS = [
    ("[OH-]",       "HO",    -1, [1, 8],                  "OH.xyz"),
    ("[OH3+]",      "H3O",   +1, [1, 1, 1, 8],            "H3O+.xyz"),
    ("O=C=O",       "CO2",    0, [6, 8, 8],               "CO2.xyz"),
    ("OC(O)O",      "CH2O3",  0, [1, 1, 6, 8, 8, 8],      "H2CO3.xyz"),
    ("O=C([O-])O",  "CHO3",  -1, [1, 6, 8, 8, 8],         "HCO3-.xyz"),
]


def _load_xyz_geometry(xyz_path: Path, expected_atomic_numbers: list[int]) -> np.ndarray:
    """Parse an XYZ file and return positions sorted by atomic number to match
    the canonical sorted_atomic_numbers convention.

    Raises FileNotFoundError or ValueError on a malformed file. The expected
    atomic_numbers list is used purely for sanity checking — it must match
    the file's element count after sorting.
    """
    text = xyz_path.read_text()
    lines = [l for l in text.strip().splitlines() if l.strip()]
    n_atoms = int(lines[0].strip())
    pairs: list[tuple[int, list[float]]] = []
    for line in lines[2:2 + n_atoms]:
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"Malformed XYZ line: {line}")
        sym = parts[0]
        z = _SYM_TO_Z.get(sym)
        if z is None:
            raise ValueError(f"Unknown element symbol: {sym}")
        pairs.append((z, [float(parts[1]), float(parts[2]), float(parts[3])]))

    pairs.sort(key=lambda p: p[0])
    sorted_anum = [p[0] for p in pairs]
    if sorted_anum != sorted(expected_atomic_numbers):
        raise ValueError(
            f"XYZ file {xyz_path.name} atoms {sorted_anum} don't match "
            f"expected {sorted(expected_atomic_numbers)}"
        )
    return np.array([p[1] for p in pairs], dtype=np.float64)

# Buffer equilibrium reactions. Synthetic negative ts_id values are used so
# they're stable across reseeds and never collide with discovered reactions
# (which have positive 64-bit hash IDs).
BUFFER_EQUILIBRIA = [
    # (ts_id, name, reactants, products, k_fwd, k_bwd)
    (-1, "water_autoionization",
        ["O"], [H_PLUS_SMILES, "[OH-]"],
        1.8e-3, 1e10),
    (-2, "CO2_hydration",
        ["O=C=O", "O"], ["OC(O)O"],
        5.5e-3, 178.0),
    (-3, "carbonic_acid_dissociation",
        ["OC(O)O"], [H_PLUS_SMILES, "O=C([O-])O"],
        2.5e6, 1e10),
    (-4, "proton_solvation",
        [H_PLUS_SMILES, "O"], ["[OH3+]"],
        1e10, 1.0),
]


def _canonicalize_smiles(smiles: str) -> str:
    """Re-canonicalize a SMILES through RDKit so buffer seeds use the same
    string the discovery pipeline produces. Without this, e.g. our seeded
    'OC(O)O' (carbonic acid) and the discovered 'O=C(O)O' end up as separate
    Compound rows even though they're the same molecule.
    """
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return Chem.MolToSmiles(mol, isomericSmiles=False)
    except Exception:
        pass
    return smiles


def _ensure_placeholder_compound(
    session: Session,
    fragment_dir: Path,
    smiles: str,
    formula: str,
    charge: int,
    atomic_numbers: list[int],
    xyz_filename: str,
) -> int:
    """Ensure a buffer compound exists; create with a single Minimum holding
    a real 3D geometry loaded from the matching XYZ file. Returns compound_id.

    The Minimum is required to satisfy foreign key constraints from
    reaction_reactants/reaction_products. The kinetic solver uses manual_k_*
    directly for buffer reactions, so this Minimum's energy is never used
    for rate-constant calculations — but we want a real geometry so the 3D
    viewer renders these compounds correctly.
    """
    xyz_path = fragment_dir / xyz_filename
    positions = _load_xyz_geometry(xyz_path, atomic_numbers)

    # Re-canonicalize so we match what the discovery pipeline will produce
    # (avoids duplicate Compound rows for e.g. 'OC(O)O' vs 'O=C(O)O').
    smiles = _canonicalize_smiles(smiles)

    existing = session.query(Compound).filter(Compound.smiles == smiles).first()
    if existing is not None:
        # Migration: rewrite zero-position placeholder Minima from older seed
        # runs that didn't have real geometries.
        old_min = (
            session.query(Minimum)
            .filter(Minimum.compound_id == existing.id, Minimum.local_id == 0)
            .first()
        )
        if old_min is not None:
            from packages.db.serialization import deserialize_ndarray
            try:
                old_pos = deserialize_ndarray(old_min.positions)
                if np.allclose(old_pos, 0.0):
                    old_min.positions = serialize_ndarray(positions)
                    logger.info(f"Migrated zero-position placeholder for {smiles}")
            except Exception:
                pass
        return existing.id

    n_atoms = len(atomic_numbers)
    anum_arr = np.array(sorted(atomic_numbers), dtype=np.int32)
    compound = Compound(
        smiles=smiles,
        formula=formula,
        charge=charge,
        n_atoms=n_atoms,
        sorted_atomic_numbers=serialize_ndarray(anum_arr),
        is_seed=True,  # buffer compounds count as seeded — they're not discovered
    )
    session.add(compound)
    session.flush()  # populate compound.id

    # Mark small buffer species (≤2 atoms) as explored — no conformers to find.
    # Larger ones (H2CO3, HCO3-) DO have conformational flexibility and should
    # be explored via PES + CREST like any other compound.
    should_explore = n_atoms > 2
    minimum = Minimum(
        compound_id=compound.id,
        local_id=0,
        positions=serialize_ndarray(positions),
        energy=0.0,
        explored=not should_explore,
        name=f"buffer_{smiles}",
        discovery_timestamp=0.0,
    )
    session.add(minimum)
    session.flush()

    # Enqueue PES + CREST work for explorable buffer compounds
    if should_explore:
        from packages.db.models import PESWorkQueue, CrestWorkQueue
        try:
            session.add(PESWorkQueue(
                compound_id=compound.id, minimum_id=minimum.id, status="pending",
            ))
            session.flush()
        except Exception:
            session.rollback()
        try:
            session.add(CrestWorkQueue(compound_id=compound.id, status="pending"))
            session.flush()
        except Exception:
            session.rollback()

    logger.info(f"Seeded buffer compound: {smiles} ({formula}, charge={charge}, explore={should_explore})")
    return compound.id


def _ensure_equilibrium_reaction(
    session: Session,
    ts_id: int,
    name: str,
    reactant_smiles: list[str],
    product_smiles: list[str],
    k_fwd: float,
    k_bwd: float,
) -> Optional[int]:
    """Idempotently insert one manual equilibrium Reaction. Returns reaction
    row id, or None if the reaction was already present.
    """
    existing = session.query(Reaction).filter(Reaction.ts_id == ts_id).first()
    if existing is not None:
        return None

    # Re-canonicalize reactant/product SMILES through RDKit so they match
    # the canonical strings the buffer compounds were stored under.
    reactant_smiles = [_canonicalize_smiles(s) for s in reactant_smiles]
    product_smiles = [_canonicalize_smiles(s) for s in product_smiles]

    # Resolve compound IDs (must already exist — placeholder compounds are
    # seeded by _ensure_placeholder_compound before this is called, and the
    # natural buffer compounds [HH]/O/C=O are seeded by the fragment loader).
    def _lookup(smi: str) -> Optional[Compound]:
        return session.query(Compound).filter(Compound.smiles == smi).first()

    reactant_compounds = [_lookup(s) for s in reactant_smiles]
    product_compounds = [_lookup(s) for s in product_smiles]

    missing = [
        s for s, c in zip(reactant_smiles + product_smiles,
                          reactant_compounds + product_compounds)
        if c is None
    ]
    if missing:
        logger.warning(
            f"Skipping equilibrium '{name}': missing compound(s) {missing}. "
            f"Make sure these are seeded as fragments or buffer compounds first."
        )
        return None

    # Placeholder TS geometry — never rendered (the API skips ts_conformer for
    # discovery_method='manual_equilibrium' nodes). Single dummy atom.
    placeholder_positions = np.zeros((1, 3), dtype=np.float64)
    placeholder_anum = np.array([1], dtype=np.int32)

    reaction = Reaction(
        ts_id=ts_id,
        ts_conformer_positions=serialize_ndarray(placeholder_positions),
        ts_conformer_atomic_numbers=serialize_ndarray(placeholder_anum),
        ts_conformer_charge=0,
        ts_energy=0.0,
        # Barriers are unused for manual equilibria — solver reads manual_k_*
        # directly. We still set sensible zeros so any code that reads
        # barrier_forward defensively doesn't blow up.
        barrier_forward=0.0,
        barrier_backward=0.0,
        manual_k_fwd=k_fwd,
        manual_k_bwd=k_bwd,
        discovery_method="manual_equilibrium",
        discovery_timestamp=0.0,
        name=f"eq_{name}",
    )
    session.add(reaction)
    session.flush()

    # Reactant join rows (conformer_local_id=0 → the placeholder minimum on each)
    for c in reactant_compounds:
        session.add(ReactionReactant(
            reaction_id=reaction.id,
            compound_id=c.id,
            conformer_local_id=0,
        ))

    # Product join rows. The reaction_products schema requires conformer_local_id
    # NOT NULL and a real energy column — use 0 for both (the kinetic solver
    # only ever reads manual_k_* for these).
    for c in product_compounds:
        session.add(ReactionProduct(
            reaction_id=reaction.id,
            compound_id=c.id,
            conformer_local_id=0,
            energy=0.0,
        ))

    # Graph edges so the viewer renders connectivity. The frontend treats
    # discovery_method='manual_equilibrium' specially (no TS geometry render).
    ts_node_name = reaction.name
    for c in reactant_compounds:
        session.add(GraphEdge(
            source_node=c.smiles,
            target_node=ts_node_name,
            source_type="compound",
            target_type="ts",
            direction="up",
            stoichiometry=1,
            reaction_id=reaction.id,
        ))
    for c in product_compounds:
        session.add(GraphEdge(
            source_node=ts_node_name,
            target_node=c.smiles,
            source_type="ts",
            target_type="compound",
            direction="down",
            stoichiometry=1,
            reaction_id=reaction.id,
        ))

    logger.info(
        f"Seeded equilibrium '{name}': "
        f"{' + '.join(reactant_smiles)} ⇌ {' + '.join(product_smiles)} "
        f"(k_f={k_fwd:.2e}, k_b={k_bwd:.2e})"
    )
    return reaction.id


def seed_buffer_equilibria(session: Session, fragment_dir: Path) -> dict:
    """Idempotently seed the 4 manual buffer equilibria + their placeholder
    compounds. Returns a summary dict.

    Args:
        session: SQLAlchemy session
        fragment_dir: directory containing the buffer-compound XYZ files
            (CO2.xyz, H2CO3.xyz, OH.xyz, H3O+.xyz, HCO3-.xyz). In the
            container this is /app/data/fragments; locally it's data/fragments.

    MUST be called inside the same advisory-locked block as the start
    molecule / fragment seeding (worker.py:_seed_initial_compounds), AFTER
    fragments are seeded — depends on the natural buffer compounds (water,
    H+, optionally formaldehyde) being present.
    """
    n_compounds_seeded = 0
    for smiles, formula, charge, anum, xyz_filename in BUFFER_COMPOUNDS:
        before = session.query(Compound).filter(Compound.smiles == smiles).count()
        _ensure_placeholder_compound(
            session, fragment_dir, smiles, formula, charge, anum, xyz_filename
        )
        after = session.query(Compound).filter(Compound.smiles == smiles).count()
        if after > before:
            n_compounds_seeded += 1

    n_reactions_seeded = 0
    for ts_id, name, reactants, products, kf, kb in BUFFER_EQUILIBRIA:
        result = _ensure_equilibrium_reaction(session, ts_id, name, reactants, products, kf, kb)
        if result is not None:
            n_reactions_seeded += 1

    summary = {
        "buffer_compounds_seeded": n_compounds_seeded,
        "buffer_reactions_seeded": n_reactions_seeded,
    }
    logger.info(f"Buffer equilibria seeding complete: {summary}")
    return summary
