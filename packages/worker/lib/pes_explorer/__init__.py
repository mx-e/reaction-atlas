"""
PES Explorer: Tools for exploring molecular potential energy surfaces.

This package provides functionality for:
- Automated PES exploration via MD and transition state search
- Graph-based representation of minima and transition states
- Integration with ML force fields (MD-ET)

Main entry point:
    from lib.pes_explorer import explore_pes, ExploreConfig, PESGraph

Example:
    from lib.pes_explorer import explore_pes, ExploreConfig
    from lib.md_et_calculator import get_md_et_calculator

    # Create calculator
    calc = get_md_et_calculator(run_dir, device="cuda")

    # Run exploration
    graph = explore_pes(atoms, calc)
"""

from lib.pes_explorer.pes_explorer import explore_pes, ExploreConfig
from lib.pes_explorer.pes_graph import (
    PESGraph,
    Minimum,
    TransitionState,
    RelaxationTrajectory,
    compute_rmsd,
)
from lib.pes_explorer.newton_minimize import (
    optimize_minimum,
    optimize_minima_batched,
    MinimizationResult,
    MinimizationTrajectory,
)
from lib.pes_explorer.prfo import optimize_saddle_points_batched
from lib.types import Conformer

__all__ = [
    # Main interface
    "explore_pes",
    "ExploreConfig",
    # Graph classes
    "PESGraph",
    "Conformer",
    "Minimum",
    "TransitionState",
    "RelaxationTrajectory",
    "compute_rmsd",
    # Newton minimizer
    "optimize_minimum",
    "optimize_minima_batched",
    "MinimizationResult",
    "MinimizationTrajectory",
    # Batched saddle point optimization
    "optimize_saddle_points_batched",
]

__version__ = "0.1.0"
