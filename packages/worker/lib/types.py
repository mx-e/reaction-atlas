"""
Core types for molecular structure representation.

The main type here is Conformer - a clean representation of molecular geometry.
ExplorationContext (in lib/exploration.py) handles exploration-specific metadata.
"""

from dataclasses import dataclass
from typing import Optional, Union
import torch
import numpy as np
from ase import Atoms


@dataclass
class MDOpts:
    """Configuration options for molecular dynamics simulations."""

    steps: int = 200_000
    save_interval: int = 20
    log_interval: int = 2000
    step_size_fs: float = 0.5
    temperature: float = 150.0
    bgfs_steps: int = 1000
    bgfs_fmax: float = 0.05
    thermostat: str = "bussi"  # "nose_hoover", "langevin", "bussi", "nve"
    thermostat_tau: float = 1000.0
    energy_threshold: float = 0.5  # in eV


@dataclass
class Conformer:
    """
    Represents a molecular geometry (conformer/structure).

    This is the fundamental representation of a molecular structure used
    throughout the codebase. It holds atomic positions and atomic numbers,
    and can work with both numpy arrays (for PES graph storage) and
    torch tensors (for ML model operations).

    Attributes:
        positions: Atomic positions, shape (n_atoms, 3). Can be numpy or torch.
        atomic_numbers: Atomic numbers for each atom. Can be numpy or torch.
        energy: Energy of this conformer in eV (optional).
    """

    positions: Union[np.ndarray, torch.Tensor]  # Shape: (n_atoms, 3)
    atomic_numbers: Union[np.ndarray, torch.Tensor]  # Shape: (n_atoms,)
    energy: Optional[float] = None
    charge: int = 0

    @property
    def n_atoms(self) -> int:
        """Number of atoms in this conformer."""
        if isinstance(self.positions, torch.Tensor):
            return self.positions.shape[0]
        return len(self.positions)

    def to(self, device: str) -> "Conformer":
        """Move data to the specified device, converting numpy arrays to tensors if needed."""
        if isinstance(self.positions, torch.Tensor):
            return Conformer(
                positions=self.positions.to(device),
                atomic_numbers=self.atomic_numbers.to(device),
                energy=self.energy,
                charge=self.charge,
            )
        # Convert numpy arrays to tensors on the target device
        return Conformer(
            positions=torch.tensor(self.positions, dtype=torch.float32, device=device),
            atomic_numbers=torch.tensor(
                self.atomic_numbers, dtype=torch.long, device=device
            ),
            energy=self.energy,
            charge=self.charge,
        )

    def to_numpy(self) -> "Conformer":
        """Convert tensors to numpy arrays."""
        positions = self.positions
        atomic_numbers = self.atomic_numbers

        if isinstance(positions, torch.Tensor):
            positions = positions.detach().cpu().numpy()
        if isinstance(atomic_numbers, torch.Tensor):
            atomic_numbers = atomic_numbers.detach().cpu().numpy()

        return Conformer(
            positions=positions,
            atomic_numbers=atomic_numbers,
            energy=self.energy,
            charge=self.charge,
        )

    def to_torch(self, device: str = "cpu") -> "Conformer":
        """Convert numpy arrays to torch tensors."""
        positions = self.positions
        atomic_numbers = self.atomic_numbers

        if isinstance(positions, np.ndarray):
            positions = torch.tensor(positions, dtype=torch.float32, device=device)
        if isinstance(atomic_numbers, np.ndarray):
            atomic_numbers = torch.tensor(
                atomic_numbers, dtype=torch.long, device=device
            )

        return Conformer(
            positions=positions,
            atomic_numbers=atomic_numbers,
            energy=self.energy,
            charge=self.charge,
        )

    @classmethod
    def from_batch(cls, batch: dict, energy: Optional[float] = None) -> "Conformer":
        """
        Create a Conformer from a SchNetPack-style batch dict.

        Args:
            batch: Dict with keys like '_positions' (or 'R'), '_atomic_numbers' (or 'Z')
            energy: Optional energy value

        Returns:
            Conformer instance (with centered positions)
        """
        # Handle both legacy Node.molecule format and SchNetPack format
        if "_positions" in batch:
            positions = batch["_positions"]
            atomic_numbers = batch["_atomic_numbers"]
        else:
            # SchNetPack properties format (R for positions, Z for atomic numbers)
            positions = batch.get("R", batch.get("_R"))
            atomic_numbers = batch.get("Z", batch.get("_Z"))

        if positions is None or atomic_numbers is None:
            raise ValueError(
                "Batch must contain positions (_positions or R) and "
                "atomic_numbers (_atomic_numbers or Z)"
            )

        charge = batch.get("_charge", 0)
        if isinstance(charge, torch.Tensor):
            charge = int(charge.item())

        conformer = cls(
            positions=positions,
            atomic_numbers=atomic_numbers,
            energy=energy,
            charge=charge,
        )
        # Always return centered conformer
        return conformer.center()

    def to_ase_atoms(self) -> Atoms:
        """Convert to ASE Atoms object."""
        conf = self.to_numpy()
        atoms = Atoms(numbers=conf.atomic_numbers.flatten(), positions=conf.positions)
        atoms.info['charge'] = self.charge
        return atoms

    def clone(self) -> "Conformer":
        """Create a deep copy of this conformer."""
        if isinstance(self.positions, torch.Tensor):
            return Conformer(
                positions=self.positions.clone().detach(),
                atomic_numbers=self.atomic_numbers.clone().detach(),
                energy=self.energy,
                charge=self.charge,
            )
        return Conformer(
            positions=self.positions.copy(),
            atomic_numbers=self.atomic_numbers.copy(),
            energy=self.energy,
            charge=self.charge,
        )

    def center(self) -> "Conformer":
        """Center the positions around the center of mass (geometric center)."""
        if isinstance(self.positions, torch.Tensor):
            centroid = self.positions.mean(dim=0, keepdim=True)
            return Conformer(
                positions=self.positions - centroid,
                atomic_numbers=self.atomic_numbers,
                energy=self.energy,
                charge=self.charge,
            )
        else:
            centroid = self.positions.mean(axis=0, keepdims=True)
            return Conformer(
                positions=self.positions - centroid,
                atomic_numbers=self.atomic_numbers,
                energy=self.energy,
                charge=self.charge,
            )

    def __setstate__(self, state):
        """Pickle backward compatibility: ensure charge field exists."""
        state.setdefault('charge', 0)
        self.__dict__.update(state)
