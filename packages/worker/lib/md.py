"""
Molecular dynamics simulation runner using ASE.

Provides MD simulation with configurable thermostats and observer attachments
for trajectory collection and console logging.
"""

import numpy as np
from ase import Atoms, units
from ase.calculators.calculator import Calculator
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from loguru import logger

from lib.types import MDOpts


class TrajectoryCollector:
    """ASE observer that collects full Atoms snapshots and energies during MD."""

    def __init__(self, atoms: Atoms):
        self.atoms = atoms
        self.trajectory: list[Atoms] = []
        self.energies: list[float] = []

    def __call__(self):
        from ase.calculators.singlepoint import SinglePointCalculator
        energy = float(self.atoms.get_potential_energy())
        forces = self.atoms.get_forces().copy()
        snapshot = self.atoms.copy()
        # Attach cached results so snapshot.get_forces()/get_momenta() work
        # without holding a reference to the live calculator
        snapshot.calc = SinglePointCalculator(snapshot, energy=energy, forces=forces)
        self.trajectory.append(snapshot)
        self.energies.append(energy)


class ConsoleLogger:
    """ASE observer that logs MD progress."""

    def __init__(self, atoms: Atoms, log_interval: int):
        self.atoms = atoms
        self.log_interval = log_interval
        self.step = 0

    def __call__(self):
        self.step += self.log_interval
        e = self.atoms.get_potential_energy()
        t = self.atoms.get_kinetic_energy() / (1.5 * units.kB * len(self.atoms))
        logger.debug(f"MD step {self.step}: E={e:.4f} eV, T={t:.0f} K")


def _get_thermostat(atoms: Atoms, opts: MDOpts):
    """Create an ASE MD integrator with the specified thermostat."""
    dt = opts.step_size_fs * units.fs
    temp_k = opts.temperature

    if opts.thermostat == "langevin":
        friction = 1.0 / (opts.thermostat_tau * units.fs)
        return Langevin(atoms, timestep=dt, temperature_K=temp_k, friction=friction)

    if opts.thermostat == "nose_hoover":
        from ase.md.nptberendsen import NPTBerendsen
        return NPTBerendsen(
            atoms, timestep=dt, temperature_K=temp_k,
            taut=opts.thermostat_tau * units.fs,
            pressure_au=0, taup=1e6 * units.fs, compressibility_au=0,
        )

    if opts.thermostat == "bussi":
        try:
            from ase.md.bussi import Bussi
            return Bussi(atoms, timestep=dt, temperature_K=temp_k,
                         taut=opts.thermostat_tau * units.fs)
        except ImportError:
            logger.warning("Bussi thermostat not available, falling back to Langevin")
            friction = 1.0 / (opts.thermostat_tau * units.fs)
            return Langevin(atoms, timestep=dt, temperature_K=temp_k, friction=friction)

    if opts.thermostat == "nve":
        from ase.md.verlet import VelocityVerlet
        return VelocityVerlet(atoms, timestep=dt)

    raise ValueError(f"Unknown thermostat: {opts.thermostat}")


def run_md_simulation(
    atoms: Atoms,
    calc: Calculator,
    opts: MDOpts,
    attachments: list[tuple],
):
    """Run an MD simulation with ASE.

    Args:
        atoms: ASE Atoms object (will be modified in place).
        calc: ASE calculator for forces/energies.
        opts: MD configuration options.
        attachments: List of (observer, interval) tuples. Each observer is
            called every `interval` steps.
    """
    atoms.calc = calc

    # Initialize velocities
    MaxwellBoltzmannDistribution(atoms, temperature_K=opts.temperature)

    # Create integrator
    dyn = _get_thermostat(atoms, opts)

    # Attach observers
    for observer, interval in attachments:
        dyn.attach(observer, interval=interval)

    # Run
    logger.debug(
        f"Starting MD: {opts.steps} steps, T={opts.temperature} K, "
        f"thermostat={opts.thermostat}, dt={opts.step_size_fs} fs"
    )
    dyn.run(steps=opts.steps)
