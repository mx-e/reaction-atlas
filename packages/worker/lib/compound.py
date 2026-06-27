"""Compound and CompoundRegistry for tracking chemical species by SMILES."""

import os
import re
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, TYPE_CHECKING
import numpy as np
import torch
from loguru import logger
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from ase import Atoms
from ase.neighborlist import natural_cutoffs, NeighborList
from lib.types import Conformer
from lib.pes_explorer.pes_graph import PESGraph
from lib.constants import ATOMIC_SYMBOLS


if TYPE_CHECKING:
    from lib.pes_explorer.pes_graph import PESGraph, Minimum
    from lib.pes_explorer import ExploreConfig
    from ase.calculators.calculator import Calculator


def get_sorted_atomic_numbers(
    atomic_numbers: np.ndarray | torch.Tensor | list,
) -> tuple[int, ...]:
    """Get sorted tuple of atomic numbers (canonical ordering for comparison)."""
    if isinstance(atomic_numbers, torch.Tensor):
        atomic_numbers = atomic_numbers.detach().cpu().numpy()
    elif isinstance(atomic_numbers, list):
        atomic_numbers = np.array(atomic_numbers)
    return tuple(sorted(atomic_numbers.flatten().astype(int).tolist()))


def canonicalize_structure(
    atomic_numbers: np.ndarray | torch.Tensor | list,
    positions: np.ndarray | torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reorder atoms using atomic number + RDKit canonical rank as tie-breaker.

    Uses RDKit's CanonicalRankAtoms to produce a deterministic ordering for
    atoms of the same element, so that conformers of the same molecule always
    have identical atom ordering regardless of input order.

    Returns:
        (sorted_atomic_numbers, reordered_positions, sort_indices)
    """
    atomic_numbers, positions = _prepare_inputs(atomic_numbers, positions)

    if len(atomic_numbers) <= 1:
        # Single atom: nothing to rank
        return atomic_numbers.copy(), positions.copy(), np.arange(len(atomic_numbers))

    try:
        mol = _create_mol_with_conformer(atomic_numbers, positions)
        for charge in [0, -1, 1, -2, 2]:
            try:
                rdDetermineBonds.DetermineBonds(mol, charge=charge)
                break
            except Exception:
                mol = _create_mol_with_conformer(atomic_numbers, positions)
        ranks = np.array(Chem.CanonicalRankAtoms(mol, breakTies=True))
    except Exception as e:
        logger.error(
            f"RDKit canonical ranking failed for molecule with {len(atomic_numbers)} atoms: {e}. "
            f"Falling back to atomic-number-only ordering (same-element atoms will have arbitrary order)."
        )
        ranks = np.arange(len(atomic_numbers))

    sort_indices = np.lexsort((ranks, atomic_numbers))
    return atomic_numbers[sort_indices], positions[sort_indices], sort_indices


def _get_bonds_from_ase(
    atomic_numbers: np.ndarray, positions: np.ndarray
) -> list[tuple[int, int]]:
    """Use ASE's neighbor list with natural cutoffs to determine bonds."""
    atoms = Atoms(numbers=atomic_numbers, positions=positions)
    cutoffs = natural_cutoffs(atoms, mult=1.2)
    nl = NeighborList(cutoffs, self_interaction=False, bothways=False)
    nl.update(atoms)
    bonds = []
    for i in range(len(atoms)):
        indices, _ = nl.get_neighbors(i)
        for j in indices:
            if i < j:
                bonds.append((i, j))
    return bonds


def _prepare_inputs(
    atomic_numbers: np.ndarray | torch.Tensor | list,
    positions: np.ndarray | torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert atomic_numbers and positions to numpy arrays."""
    if isinstance(atomic_numbers, torch.Tensor):
        atomic_numbers = atomic_numbers.detach().cpu().numpy()
    elif isinstance(atomic_numbers, list):
        atomic_numbers = np.array(atomic_numbers)
    atomic_numbers = atomic_numbers.flatten().astype(int)

    if isinstance(positions, torch.Tensor):
        positions = positions.detach().cpu().numpy()
    positions = np.array(positions).reshape(-1, 3)
    return atomic_numbers, positions


def _create_mol_with_conformer(
    atomic_numbers: np.ndarray, positions: np.ndarray
) -> Chem.RWMol:
    """Create an RDKit RWMol with a 3D conformer from atomic numbers and positions."""
    mol = Chem.RWMol()
    for z in atomic_numbers:
        mol.AddAtom(Chem.Atom(int(z)))
    conf = Chem.Conformer(len(atomic_numbers))
    for i, pos in enumerate(positions):
        conf.SetAtomPosition(i, pos.tolist())
    mol.AddConformer(conf, assignId=True)
    return mol


def get_smiles_and_charge_from_structure(
    atomic_numbers: np.ndarray | torch.Tensor | list,
    positions: np.ndarray | torch.Tensor,
) -> Optional[tuple[str, int]]:
    """Generate canonical SMILES and net formal charge from atomic numbers and 3D positions.

    Tries multiple charge values for DetermineBonds to handle charged species.
    Returns (canonical_smiles, net_formal_charge) or None if all attempts fail.
    """
    atomic_numbers, positions = _prepare_inputs(atomic_numbers, positions)

    # Try DetermineBonds with multiple charge values (0 first for fast path)
    for charge in [0, -1, 1, -2, 2, -3, 3]:
        try:
            mol = _create_mol_with_conformer(atomic_numbers, positions)
            rdDetermineBonds.DetermineBonds(mol, charge=charge)
            Chem.SanitizeMol(mol)
            net_charge = Chem.GetFormalCharge(mol)
            mol = Chem.RemoveHs(mol)
            smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
            return smiles, net_charge
        except Exception:
            continue

    # Final fallback: allowChargedFragments with charge=0
    try:
        mol = _create_mol_with_conformer(atomic_numbers, positions)
        rdDetermineBonds.DetermineBonds(mol, charge=0, allowChargedFragments=True)
        Chem.SanitizeMol(mol)
        net_charge = Chem.GetFormalCharge(mol)
        mol = Chem.RemoveHs(mol)
        smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        return smiles, net_charge
    except Exception as e:
        logger.debug(f"rdDetermineBonds(allowCharged) failed: {e}")

    logger.debug(
        f"SMILES determination failed for molecule with {len(atomic_numbers)} atoms"
    )
    return None


def get_smiles_from_structure(
    atomic_numbers: np.ndarray | torch.Tensor | list,
    positions: np.ndarray | torch.Tensor,
) -> Optional[str]:
    """Generate canonical SMILES from atomic numbers and 3D positions.

    Uses RDKit's rdDetermineBonds to determine bond orders from geometry.
    Returns None if bond determination fails (unreliable geometry).
    """
    result = get_smiles_and_charge_from_structure(atomic_numbers, positions)
    if result is None:
        return None
    return result[0]


def get_display_smiles_from_structure(
    atomic_numbers: np.ndarray | torch.Tensor | list,
    positions: np.ndarray | torch.Tensor,
) -> str:
    """Generate SMILES for display/logging purposes (never None).

    Uses rdDetermineBonds first, falls back to ASE-based connectivity.
    NOT suitable for compound identity matching — use get_smiles_from_structure instead.
    """
    # Try the reliable method first
    result = get_smiles_from_structure(atomic_numbers, positions)
    if result is not None:
        return result

    atomic_numbers, positions = _prepare_inputs(atomic_numbers, positions)

    # Fallback: ASE-based connectivity with partial sanitization
    try:
        mol = _create_mol_with_conformer(atomic_numbers, positions)
        bonds = _get_bonds_from_ase(atomic_numbers, positions)
        for i, j in bonds:
            mol.AddBond(int(i), int(j), Chem.BondType.SINGLE)
        Chem.SanitizeMol(
            mol,
            sanitizeOps=Chem.SanitizeFlags.SANITIZE_FINDRADICALS
            | Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION
            | Chem.SanitizeFlags.SANITIZE_SETCONJUGATION
            | Chem.SanitizeFlags.SANITIZE_SETAROMATICITY,
        )
        mol = Chem.RemoveHs(mol, sanitize=False)
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception as e:
        logger.debug(f"ASE fallback failed: {e}")

    # Last resort: raw connectivity SMILES
    mol = _create_mol_with_conformer(atomic_numbers, positions)
    bonds = _get_bonds_from_ase(atomic_numbers, positions)
    for i, j in bonds:
        mol.AddBond(int(i), int(j), Chem.BondType.SINGLE)
    try:
        mol = Chem.RemoveHs(mol, sanitize=False)
    except Exception:
        pass
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def get_smiles_from_conformer(conformer: Conformer) -> Optional[str]:
    """Get canonical SMILES from a Conformer. Returns None if bond determination fails."""
    conf = conformer.to_numpy()
    return get_smiles_from_structure(conf.atomic_numbers, conf.positions)


def get_smiles_and_charge_from_conformer(conformer: Conformer) -> Optional[tuple[str, int]]:
    """Get canonical SMILES and net formal charge from a Conformer.

    Returns (smiles, charge) or None if bond determination fails.
    """
    conf = conformer.to_numpy()
    return get_smiles_and_charge_from_structure(conf.atomic_numbers, conf.positions)


def get_charge_from_smiles(smiles: str) -> int:
    """Get net formal charge from a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0
    return Chem.GetFormalCharge(mol)


def get_sorted_atomic_numbers_from_conformer(conformer: Conformer) -> tuple[int, ...]:
    """Get composition ID from a Conformer."""
    conf = conformer.to_numpy()
    return get_sorted_atomic_numbers(conf.atomic_numbers)


def get_chemical_formula(sorted_atomic_numbers: tuple[int, ...]) -> str:
    """Convert composition ID to chemical formula string."""
    counts = Counter(sorted_atomic_numbers)
    parts = []
    # C first, then H, then rest alphabetically
    for z in [6, 1]:
        if z in counts:
            sym = ATOMIC_SYMBOLS[z]
            parts.append(f"{sym}{counts[z]}" if counts[z] > 1 else sym)
            del counts[z]
    for z in sorted(counts.keys(), key=lambda x: ATOMIC_SYMBOLS.get(x, f"X{x}")):
        sym = ATOMIC_SYMBOLS.get(z, f"[{z}]")
        parts.append(f"{sym}{counts[z]}" if counts[z] > 1 else sym)
    return "".join(parts)


@dataclass
class Compound:
    """Chemical species identified by canonical SMILES, with PES graph containing at least one conformer."""

    smiles: str
    sorted_atomic_numbers: tuple[int, ...] = field(repr=False)
    initial_positions: np.ndarray = field(
        repr=False
    )  # Must be canonicalized (sorted by atomic number)
    initial_energy: float = field(repr=False)
    is_seed: bool = False
    formula: str = field(default="", init=False)
    pes_graph: "PESGraph" = field(default=None, init=False, repr=False)
    reference_conformer_id: int = field(default=0, init=False)
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.formula = get_chemical_formula(self.sorted_atomic_numbers)
        self.pes_graph = PESGraph(
            atomic_numbers=np.array(self.sorted_atomic_numbers, dtype=np.int32)
        )
        self.reference_conformer_id, _, _, _ = self.pes_graph.add_minimum(
            positions=self.initial_positions,
            energy=self.initial_energy,
        )
        if len(self.sorted_atomic_numbers) == 1:
            self.pes_graph.mark_explored(
                self.reference_conformer_id
            )  # Single atom, no exploration needed

    def __hash__(self):
        return hash(self.smiles)

    def __eq__(self, other):
        return isinstance(other, Compound) and self.smiles == other.smiles

    @property
    def n_atoms(self) -> int:
        return len(self.sorted_atomic_numbers)

    @property
    def n_conformers(self) -> int:
        return len(self.pes_graph.minima)

    @property
    def n_intramolecular_ts(self) -> int:
        return len(self.pes_graph.transition_states)

    @classmethod
    def from_conformer(cls, conformer: Conformer, energy: float, is_seed: bool = False) -> "Compound":
        """Create a Compound from a Conformer.

        Raises ValueError if SMILES determination fails.
        """
        smiles = get_smiles_from_conformer(conformer)
        if smiles is None:
            raise ValueError("Cannot create Compound: SMILES determination failed")
        conf = conformer.to_numpy()
        # Canonicalize: reorder positions to match sorted atomic numbers
        _, canonicalized_positions, _ = canonicalize_structure(
            conf.atomic_numbers, conf.positions
        )
        return cls(
            smiles=smiles,
            sorted_atomic_numbers=get_sorted_atomic_numbers(conf.atomic_numbers),
            initial_positions=canonicalized_positions,
            initial_energy=energy,
            is_seed=is_seed,
        )

    def add_conformer(self, conformer: Conformer, energy: float, calc=None) -> tuple[int, bool]:
        """Add conformer from Conformer. Returns (conformer_id, is_new).

        If calc is provided, temporarily sets it on the PES graph so NEB
        deduplication can run even outside of PES exploration.
        """
        conf = conformer.to_numpy()
        # Canonicalize: reorder positions to match sorted atomic numbers
        _, canonicalized_positions, _ = canonicalize_structure(
            conf.atomic_numbers, conf.positions
        )
        old_calc = self.pes_graph.calc
        if calc is not None:
            self.pes_graph.calc = calc
        try:
            conformer_id, is_new, _, _ = self.pes_graph.add_minimum(
                positions=canonicalized_positions,
                energy=energy,
            )
        finally:
            self.pes_graph.calc = old_calc
        return conformer_id, is_new

    def get_lowest_energy_conformer(self) -> "Minimum":
        return min(self.pes_graph.minima.values(), key=lambda m: m.energy)

    def get_unexplored_conformers(self) -> list["Minimum"]:
        return self.pes_graph.get_unexplored_minima()

    def explore_pes(
        self,
        calc: "Calculator",
        conformer_id: Optional[int] = None,
        config: Optional["ExploreConfig"] = None,
        verbose: bool = True,
    ) -> tuple[int, list[dict], dict]:
        """Explore PES from conformer. If conformer_id is None, uses lowest energy.

        Returns:
            (n_new_ts, escaped_validations, timings): Number of new intramolecular TSs added,
            list of validation dicts where a TS endpoint escaped to a different compound,
            and per-step timing dict.
        """
        from lib.pes_explorer.pes_explorer import ExploreConfig, _explore_from_minimum

        if config is None:
            config = ExploreConfig()

        if conformer_id is None:
            conformer = self.get_lowest_energy_conformer()
            conformer_id = conformer.id
        else:
            conformer = self.pes_graph.minima[conformer_id]

        if conformer.explored:
            return 0, [], {}

        # Filter: only accept TSs where both endpoints have the same SMILES as this compound.
        # TSs where an endpoint "escapes" to a different compound are collected separately.
        def smiles_filter(validation: dict) -> bool:
            anum = self.pes_graph.atomic_numbers
            fwd_smiles = get_smiles_from_structure(anum, validation["fwd_positions"])
            bwd_smiles = get_smiles_from_structure(anum, validation["bwd_positions"])

            # Endpoints are fully relaxed, so rdDetermineBonds should
            # succeed on any well-defined molecular geometry. Failure
            # means the geometry converged to something with ambiguous
            # bonds — e.g. a tautomer or ring-opening product where bond
            # lengths sit between single/double. Reject these outright
            # rather than guessing whether they're intramolecular.
            n_failed = (fwd_smiles is None) + (bwd_smiles is None)
            if n_failed > 0:
                # Relaxed endpoints should always have determinable bonds.
                # If rdDetermineBonds fails, the geometry converged to
                # something with ambiguous bond lengths (tautomer,
                # ring-opening intermediate, etc.) — reject the TS.
                logger.info(
                    f"Rejecting TS: rdDetermineBonds failed for {n_failed} "
                    f"endpoint(s) of {self.smiles}: fwd={fwd_smiles}, "
                    f"bwd={bwd_smiles}"
                )
                return False

            is_same = fwd_smiles == self.smiles and bwd_smiles == self.smiles
            if not is_same:
                logger.info(
                    f"TS endpoint escaped compound {self.smiles}: "
                    f"fwd={fwd_smiles}, bwd={bwd_smiles}"
                )
            return is_same

        # Set calc on PES graph for NEB dedup during exploration
        old_calc = self.pes_graph.calc
        self.pes_graph.calc = calc
        self.pes_graph.neb_fire_steps = config.neb_fire_steps
        self.pes_graph.neb_barrier_threshold = config.neb_barrier_threshold
        self.pes_graph.neb_spring_constant = config.neb_spring_constant
        try:
            n_new_ts, escaped, timings = _explore_from_minimum(
                minimum_positions=conformer.positions,
                atomic_numbers=conformer.atomic_numbers,
                calc=calc,
                config=config,
                pes_graph=self.pes_graph,
                ts_filter=smiles_filter,
                verbose=verbose,
                charge=get_charge_from_smiles(self.smiles),
            )
            self.pes_graph.mark_explored(conformer_id)
        finally:
            self.pes_graph.calc = old_calc
        return n_new_ts, escaped, timings

    @staticmethod
    def sanitize_smiles_for_filename(smiles: str) -> str:
        """Sanitize SMILES string for use as a filename.

        Replaces characters that are invalid or problematic in filenames.
        """
        # Replace problematic characters with safe alternatives
        replacements = {
            "/": "_slash_",
            "\\": "_backslash_",
            ":": "_colon_",
            "*": "_star_",
            "?": "_question_",
            '"': "_quote_",
            "<": "_lt_",
            ">": "_gt_",
            "|": "_pipe_",
            "#": "_hash_",
            "@": "_at_",
            "=": "_eq_",
            "+": "_plus_",
            "[": "_lb_",
            "]": "_rb_",
            "(": "_lp_",
            ")": "_rp_",
        }
        result = smiles
        for char, replacement in replacements.items():
            result = result.replace(char, replacement)
        return result

    def save_pes_graph(self, output_dir: os.PathLike) -> Path:
        """Save the PES graph as a GEXF file.

        Args:
            output_dir: Base output directory. File will be saved to
                        output_dir/pes_graphs/{sanitized_smiles}.gexf

        Returns:
            Path to the saved GEXF file.
        """
        from networkx.readwrite import write_gexf

        output_dir = Path(output_dir)
        pes_graphs_dir = output_dir / "pes_graphs"
        pes_graphs_dir.mkdir(parents=True, exist_ok=True)

        sanitized_smiles = self.sanitize_smiles_for_filename(self.smiles)
        gexf_path = pes_graphs_dir / f"{sanitized_smiles}.gexf"

        write_gexf(self.pes_graph.graph, gexf_path)
        logger.debug(f"Saved PES graph for {self.formula} to {gexf_path}")

        return gexf_path

    def __repr__(self):
        return f"Compound({self.formula}, conformers={self.n_conformers}, ts={self.n_intramolecular_ts})"


class CompoundRegistry:
    """Registry for compounds keyed by SMILES."""

    # Timeout for in-progress compounds (seconds). If a process crashes,
    # compounds will become available again after this timeout.
    IN_PROGRESS_TIMEOUT = 3600 * 5  # 5 hours

    def __init__(self):
        self._compounds: dict[str, Compound] = {}
        self._in_progress: dict[str, float] = {}  # SMILES -> checkout timestamp

    def _is_in_progress(self, smiles: str) -> bool:
        """Check if compound is in progress and not expired."""
        import time

        if smiles not in self._in_progress:
            return False
        checkout_time = self._in_progress[smiles]
        if time.time() - checkout_time > self.IN_PROGRESS_TIMEOUT:
            # Expired - clean up and return False
            del self._in_progress[smiles]
            return False
        return True

    def get_or_create_from_conformer(
        self, conformer: Conformer, energy: float, is_seed: bool = False,
        calc=None,
    ) -> Optional[tuple[Compound, int, bool]]:
        """Get existing compound or create new one from conformer.

        Returns (compound, conformer_id, is_new_compound), or None if
        SMILES determination failed (unreliable geometry).
        """
        result = get_smiles_and_charge_from_conformer(conformer)
        if result is None:
            logger.warning("Rejecting conformer: SMILES determination failed")
            return None
        smiles, charge = result
        conformer.charge = charge
        if smiles in self._compounds:
            compound = self._compounds[smiles]
            if is_seed:
                compound.is_seed = True
            conformer_id, _ = compound.add_conformer(conformer, energy, calc=calc)
            return compound, conformer_id, False
        compound = Compound.from_conformer(conformer, energy, is_seed=is_seed)
        self._compounds[smiles] = compound
        return compound, compound.reference_conformer_id, True

    def get(self, smiles: str) -> Optional[Compound]:
        return self._compounds.get(smiles)

    def contains_conformer(self, conformer: Conformer) -> bool:
        """Check if a conformer's compound exists in the registry."""
        smiles = get_smiles_from_conformer(conformer)
        if smiles is None:
            return False
        return smiles in self._compounds

    @property
    def compounds(self) -> list[Compound]:
        return list(self._compounds.values())

    def get_compounds_with_unexplored_conformers(self) -> list[Compound]:
        """Get compounds that have unexplored conformers."""
        return [c for c in self._compounds.values() if c.get_unexplored_conformers()]

    def get_pes_backlog(self) -> tuple[int, int, int]:
        """Return (n_unexplored_available, n_in_progress, n_total).

        n_unexplored_available: compounds needing PES exploration that are NOT
            currently checked out by another worker.
        n_in_progress: compounds currently being explored by some worker.
        n_total: total compounds in registry.
        """
        pending_all = self.get_compounds_with_unexplored_conformers()
        in_progress = sum(1 for c in pending_all if self._is_in_progress(c.smiles))
        available = len(pending_all) - in_progress
        return available, in_progress, len(self._compounds)

    def get_compound_for_postprocessing(self) -> Optional[Compound]:
        """Get a random compound with unexplored conformers (deep copy).

        Marks the compound as in-progress to prevent other processes from
        checking it out simultaneously. Call release_compound() when done.
        Compounds automatically become available again after IN_PROGRESS_TIMEOUT
        if the process crashes without releasing.
        """
        import random
        import time

        all_compounds = list(self._compounds.values())
        pending_all = self.get_compounds_with_unexplored_conformers()
        in_progress_count = sum(
            1 for c in pending_all if self._is_in_progress(c.smiles)
        )

        # Exclude compounds already being explored by another process (with timeout check)
        pending = [c for c in pending_all if not self._is_in_progress(c.smiles)]

        logger.info(
            f"Postprocessing check: {len(all_compounds)} compounds, "
            f"{len(pending_all)} need exploration, {in_progress_count} in-progress, "
            f"{len(pending)} available"
        )

        if pending_all and not pending:
            # All compounds with unexplored conformers are in-progress
            logger.warning(
                f"All {len(pending_all)} compounds with unexplored conformers are in-progress!"
            )

        if not pending:
            return None

        compound = random.choice(pending)
        self._in_progress[compound.smiles] = time.time()
        logger.info(
            f"Selected for exploration: {compound.formula} ({compound.smiles}) "
            f"with {len(compound.get_unexplored_conformers())} unexplored conformers"
        )
        return deepcopy(compound)

    def release_compound(self, smiles: str) -> None:
        """Release a compound from in-progress state (called after exploration completes)."""
        self._in_progress.pop(smiles, None)

    def checkpoint_compound(self, compound: Compound) -> None:
        """Update compound in registry without releasing the in-progress lock."""
        self._compounds[compound.smiles] = compound

    def update_compound(self, compound: Compound) -> None:
        self._compounds[compound.smiles] = compound
        self._in_progress.pop(compound.smiles, None)  # Release from in-progress

    def __len__(self):
        return len(self._compounds)

    def __iter__(self):
        return iter(self._compounds.values())

    def __repr__(self):
        n_pending = len(self.get_compounds_with_unexplored_conformers())
        return f"CompoundRegistry(n={len(self)}, pending={n_pending})"


def is_same_compound(conf1: Conformer, conf2: Conformer) -> bool:
    """Check if two conformers have same molecular structure (same SMILES).

    Returns False if SMILES determination fails for either conformer.
    """
    s1 = get_smiles_from_conformer(conf1)
    s2 = get_smiles_from_conformer(conf2)
    if s1 is None or s2 is None:
        return False
    return s1 == s2


def is_reaction(conf1: Conformer, conf2: Conformer) -> bool:
    """Check if transition between conformers is a reaction (different compounds)."""
    return not is_same_compound(conf1, conf2)
