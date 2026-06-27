"""
Exploration context for chemical reaction network exploration.

ExplorationContext is an ephemeral container that captures one round of exploration
(sampling -> TS -> IRC -> products), then gets consumed when adding to the reaction graph.
"""

import math
import os
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
from copy import copy

# Allow slightly-negative forward barriers through the gate. ML noise on the
# (E_ts - (E_A + E_B)) estimate for merged contexts commonly puts real saddles
# at a small negative value. Downstream Hessian + IRC filters reject non-saddles
# so letting them past the cheap gate mostly costs one extra Hessian eval.
BARRIER_FWD_MIN_EV = float(os.environ.get("BARRIER_FWD_MIN_EV", "0.0"))

import torch
from ase import Atoms

from lib.energy import compute_barrier_from_trajectory

from lib.types import Conformer
from lib.pes_explorer.pes_graph import RelaxationTrajectory


class PointType(Enum):
    """Type of stationary point on the PES."""

    MIN = "min"
    TS = "ts"


@dataclass
class ExplorationContext:
    """
    Ephemeral container for one exploration round's results.

    Created during exploration, consumed when adding to graph.
    Captures the full pathway: reactant(s) -> TS -> product(s).
    """

    # === STARTING POINT ===
    # Single molecule start
    start_conformer: Optional[Conformer] = None
    start_id: Optional[int] = None
    start_energy: Optional[float] = None
    source_compound_smiles: Optional[str] = None

    # Merge start (if was_merged=True)
    was_merged: bool = False
    merge_component_conformers: Optional[tuple[Conformer, Conformer]] = None
    merge_component_ids: Optional[tuple[int, int]] = None
    merge_component_energies: Optional[tuple[float, float]] = None
    merge_component_smiles: Optional[tuple[str, str]] = None  # SMILES of source compounds
    merge_atom_indices: Optional[torch.Tensor] = None  # Which component each atom came from
    merged_conformer: Optional[Conformer] = None  # The combined structure before TS
    merged_energy: Optional[float] = None

    # === TRANSITION STATE ===
    ts_conformer: Optional[Conformer] = None
    ts_id: Optional[int] = None
    ts_energy: Optional[float] = None

    # === ENDPOINT(S) ===
    # After IRC/relaxation - the product side
    min_conformer: Optional[Conformer] = None  # Single MIN (if no fragmentation)
    min_id: Optional[int] = None
    min_energy: Optional[float] = None

    # Fragments (if decomposition occurred)
    min_fragments: Optional[list[Conformer]] = None  # Fragment conformers
    min_fragment_ids: Optional[list[int]] = None
    min_fragment_energies: Optional[list[float]] = None

    # === IRC TRAJECTORIES ===
    # Relaxation trajectories from TS displacement (for inter-molecular reactions)
    reactant_trajectory: Optional[RelaxationTrajectory] = None  # TS -> reactant direction
    product_trajectory: Optional[RelaxationTrajectory] = None  # TS -> product direction

    # === DISCOVERY METHOD ===
    discovery_method: Optional[str] = None  # "generative" or "pes_exploration"
    discovery_noise_level: Optional[int] = None  # diffusion noise level t (generative only)
    discovery_timestamp: Optional[float] = None  # unix timestamp when this reaction was discovered

    # === VALIDATION ===
    is_valid: bool = True
    validation_reason: str = ""

    # === HELPERS ===

    @staticmethod
    def compute_id(conformer: Conformer) -> int:
        """Geometry-based hash for deduplication.

        Uses a deterministic hash (not Python's hash() which is randomized
        per-process via PYTHONHASHSEED) so that the same geometry produces
        the same ID across all workers.
        """
        import hashlib
        conf = conformer.to_torch()
        positions = torch.round(conf.positions.float() * 256).long()
        atom_types = conf.atomic_numbers.long()
        data = (
            tuple(positions.flatten().tolist()),
            tuple(atom_types.flatten().tolist()),
        )
        h = hashlib.blake2b(repr(data).encode(), digest_size=8)
        return int.from_bytes(h.digest(), byteorder="big", signed=True)

    def get_reactant_energy(self) -> Optional[float]:
        """Total energy of reactant(s)."""
        if self.was_merged:
            if self.merge_component_energies is None:
                return None
            return sum(self.merge_component_energies)
        return self.start_energy

    def get_product_energy(self) -> Optional[float]:
        """Total energy of product(s)."""
        if self.has_fragments:
            if self.min_fragment_energies is None:
                return None
            return sum(self.min_fragment_energies)
        return self.min_energy

    def get_ts_barrier_forward(self) -> Optional[float]:
        """Energy barrier from reactants to TS.

        Two cases:
          - **Post-IRC** (trajectories available): trajectory force-integral.
          - **Pre-IRC** or trajectory too short: endpoint energy difference.
        """
        if self.reactant_trajectory is not None:
            try:
                return compute_barrier_from_trajectory(self.reactant_trajectory, use_hessians=False)
            except (ValueError, AttributeError, IndexError):
                pass  # fall through to energy-diff
        if self.ts_energy is None:
            return None
        reactant_e = self.get_reactant_energy()
        if reactant_e is None:
            return None
        return self.ts_energy - reactant_e

    def get_ts_barrier_backward(self) -> Optional[float]:
        """Energy barrier from products to TS — see get_ts_barrier_forward."""
        if self.product_trajectory is not None:
            try:
                return compute_barrier_from_trajectory(self.product_trajectory, use_hessians=False)
            except (ValueError, AttributeError, IndexError):
                pass  # fall through to energy-diff
        if self.ts_energy is None:
            return None
        product_e = self.get_product_energy()
        if product_e is None:
            return None
        return self.ts_energy - product_e

    def validate_energy_profile(self, threshold: float) -> tuple[bool, str]:
        """Check if energy profile is physically reasonable."""
        barrier_fwd = self.get_ts_barrier_forward()
        barrier_bwd = self.get_ts_barrier_backward()

        if barrier_fwd is None or barrier_bwd is None:
            return False, "missing energy data"

        if barrier_fwd <= 0:
            return False, f"TS below reactants by {-barrier_fwd:.4f} eV"
        if barrier_bwd <= 0:
            return False, f"TS below products by {-barrier_bwd:.4f} eV"
        if barrier_fwd > threshold:
            return (
                False,
                f"Forward barrier {barrier_fwd:.4f} exceeds threshold {threshold}",
            )
        if barrier_bwd > threshold:
            return (
                False,
                f"Backward barrier {barrier_bwd:.4f} exceeds threshold {threshold}",
            )
        return True, "valid"

    def validate_forward_barrier(self, threshold: float) -> tuple[bool, str, str]:
        """
        Validate MIN -> TS energy profile.

        Returns:
            (is_valid, reason, stat_key) where stat_key indicates which statistic to update:
            - "min_to_ts_valid": valid forward barrier
            - "min_to_ts_invalid": invalid (TS below reactant or exceeds threshold)
            - "threshold_violations": also set if threshold exceeded
        """
        barrier_fwd = self.get_ts_barrier_forward()

        if barrier_fwd is None:
            return False, "missing energy data", "min_to_ts_invalid"
        if not math.isfinite(barrier_fwd):
            return False, f"non-finite barrier ({barrier_fwd})", "min_to_ts_invalid"

        if barrier_fwd < -BARRIER_FWD_MIN_EV:
            return (
                False,
                f"TS below reactants by {-barrier_fwd:.4f} eV "
                f"(tol={BARRIER_FWD_MIN_EV:.3f})",
                "min_to_ts_invalid",
            )
        if barrier_fwd > threshold:
            return (
                False,
                f"Forward barrier {barrier_fwd:.4f} exceeds threshold {threshold}",
                "threshold_violation",
            )
        return True, f"Forward barrier {barrier_fwd:.4f} eV", "min_to_ts_valid"

    def validate_backward_barrier(self, threshold: float) -> tuple[bool, str, str]:
        """
        Validate TS -> MIN/fragments energy profile.

        Returns:
            (is_valid, reason, stat_key) where stat_key indicates which statistic to update:
            - "ts_to_min_valid": valid backward barrier
            - "ts_to_min_invalid": invalid (products above TS or exceeds threshold)
            - "threshold_violations": also set if threshold exceeded
        """
        barrier_bwd = self.get_ts_barrier_backward()

        if barrier_bwd is None:
            return False, "missing energy data", "ts_to_min_invalid"
        if not math.isfinite(barrier_bwd):
            return False, f"non-finite barrier ({barrier_bwd})", "ts_to_min_invalid"

        if barrier_bwd <= 0:
            return (
                False,
                f"TS below products by {-barrier_bwd:.4f} eV",
                "ts_to_min_invalid",
            )
        if barrier_bwd > threshold:
            return (
                False,
                f"Backward barrier {barrier_bwd:.4f} exceeds threshold {threshold}",
                "threshold_violation",
            )
        return True, f"Backward barrier {barrier_bwd:.4f} eV", "ts_to_min_valid"

    @property
    def has_fragments(self) -> bool:
        """Check if decomposition produced fragments."""
        return self.min_fragments is not None and len(self.min_fragments) > 0

    @property
    def n_fragments(self) -> int:
        """Number of fragments (0 if no fragmentation)."""
        if self.min_fragments is None:
            return 0
        return len(self.min_fragments)

    def get_start_conformer(self) -> Conformer:
        """Get the starting conformer (merged or single)."""
        if self.was_merged:
            if self.merged_conformer is None:
                raise ValueError("Merged context has no merged_conformer")
            return self.merged_conformer
        if self.start_conformer is None:
            raise ValueError("Context has no start_conformer")
        return self.start_conformer

    def get_start_id(self) -> int:
        """Get the starting structure ID."""
        if self.was_merged:
            # For merged structures, compute ID from merged conformer
            if self.merged_conformer is not None:
                return self.compute_id(self.merged_conformer)
            raise ValueError("Merged context has no merged_conformer")
        if self.start_id is not None:
            return self.start_id
        if self.start_conformer is not None:
            return self.compute_id(self.start_conformer)
        raise ValueError("Context has no start info")

    def molecule_for_ml(self, which: str = "ts") -> dict:
        """
        Get SchNetPack-style batch dict for ML models.

        Args:
            which: One of "ts", "min", "merged", "start"

        Returns:
            Dict with _positions, _atomic_numbers, etc.
        """
        if which == "ts":
            conf = self.ts_conformer
        elif which == "min":
            conf = self.min_conformer
        elif which == "merged":
            conf = self.merged_conformer
        elif which == "start":
            conf = self.get_start_conformer()
        else:
            raise ValueError(f"Unknown molecule type: {which}")

        if conf is None:
            raise ValueError(f"No {which} conformer available")

        conf = conf.to_torch()
        device = (
            conf.positions.device
            if isinstance(conf.positions, torch.Tensor)
            else "cpu"
        )
        return {
            "_positions": conf.positions,
            "_atomic_numbers": conf.atomic_numbers,
            "_n_atoms": torch.tensor([conf.n_atoms], dtype=torch.long, device=device),
            "_idx": torch.tensor([0], dtype=torch.long, device=device),
            "_cell": torch.zeros((1, 3, 3), dtype=conf.positions.dtype, device=device),
            "_pbc": torch.zeros((1, 3), dtype=torch.bool, device=device),
            "_charge": torch.tensor([conf.charge], dtype=torch.long, device=device),
        }

    def to(self, device: str) -> "ExplorationContext":
        """Move all conformers to device for ML operations."""
        new_ctx = copy(self)

        if self.start_conformer is not None:
            new_ctx.start_conformer = self.start_conformer.to(device)
        if self.merged_conformer is not None:
            new_ctx.merged_conformer = self.merged_conformer.to(device)
        if self.ts_conformer is not None:
            new_ctx.ts_conformer = self.ts_conformer.to(device)
        if self.min_conformer is not None:
            new_ctx.min_conformer = self.min_conformer.to(device)
        if self.merge_component_conformers is not None:
            new_ctx.merge_component_conformers = (
                self.merge_component_conformers[0].to(device),
                self.merge_component_conformers[1].to(device),
            )
        if self.min_fragments is not None:
            new_ctx.min_fragments = [f.to(device) for f in self.min_fragments]
        if self.merge_atom_indices is not None:
            new_ctx.merge_atom_indices = self.merge_atom_indices.to(device)

        return new_ctx

    def get_merge_component_positions(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Extract positions for each merge component from current TS or MIN.

        Uses merge_atom_indices to map atoms back to their original components.
        """
        if not self.was_merged or self.merge_atom_indices is None:
            raise ValueError("Not a merge context")

        # Get positions from whichever structure we have (prefer min, then ts, then merged)
        if self.min_conformer is not None:
            pos = self.min_conformer.to_torch().positions
        elif self.ts_conformer is not None:
            pos = self.ts_conformer.to_torch().positions
        elif self.merged_conformer is not None:
            pos = self.merged_conformer.to_torch().positions
        else:
            raise ValueError("No conformer available to extract positions from")

        mask_0 = self.merge_atom_indices == 0
        mask_1 = self.merge_atom_indices == 1
        return pos[mask_0], pos[mask_1]


# === FACTORY FUNCTIONS ===


def create_single_mol_context(
    start_conformer: Conformer,
    source_compound_smiles: Optional[str] = None,
) -> ExplorationContext:
    """
    Create context for single-molecule exploration.

    Args:
        start_conformer: The starting conformer (should have energy set)
        source_compound_smiles: SMILES of the compound this was sampled from
    """
    ctx = ExplorationContext()
    ctx.start_conformer = start_conformer
    ctx.start_id = ExplorationContext.compute_id(start_conformer)
    ctx.start_energy = start_conformer.energy
    ctx.source_compound_smiles = source_compound_smiles
    return ctx


def create_merge_context(
    component_1: Conformer,
    component_2: Conformer,
    merged_conformer: Conformer,
    atom_indices: torch.Tensor,
    component_1_id: Optional[int] = None,
    component_2_id: Optional[int] = None,
    component_1_smiles: Optional[str] = None,
    component_2_smiles: Optional[str] = None,
) -> ExplorationContext:
    """
    Create context for bimolecular (merge) exploration.

    Args:
        component_1: First component conformer
        component_2: Second component conformer
        merged_conformer: The combined structure
        atom_indices: Tensor indicating which component each atom came from (0 or 1)
        component_1_id: Optional pre-computed ID for component 1
        component_2_id: Optional pre-computed ID for component 2
        component_1_smiles: Optional SMILES of source compound for component 1
        component_2_smiles: Optional SMILES of source compound for component 2
    """
    ctx = ExplorationContext()
    ctx.was_merged = True
    ctx.merge_component_conformers = (component_1, component_2)
    ctx.merge_component_ids = (
        component_1_id or ExplorationContext.compute_id(component_1),
        component_2_id or ExplorationContext.compute_id(component_2),
    )
    ctx.merge_component_energies = (component_1.energy, component_2.energy)
    if component_1_smiles is not None and component_2_smiles is not None:
        ctx.merge_component_smiles = (component_1_smiles, component_2_smiles)
    ctx.merge_atom_indices = atom_indices
    ctx.merged_conformer = merged_conformer
    ctx.merged_energy = merged_conformer.energy
    return ctx
