"""
MD-ET Calculator: Wrapper around the md-et pip package.

Provides the same interface as the original MLCalculator but uses
`md_et.load_calculator()` instead of loading from a local training run.
"""

from pathlib import Path
from typing import Literal

from loguru import logger


def get_md_et_calculator(
    run_dir_path: Path = None,
    device: str = "cuda",
    checkpoint_name: str = "best_model",
    type: Literal["standard"] = "standard",
    filter_forces: bool = True,
    hessian_batch_size: int = 16,
    variant: str = "12l",
):
    """
    Load an MD-ET calculator from the md-et pip package.

    For backwards compatibility, accepts the same arguments as the original
    get_md_et_calculator, but ignores run_dir_path and checkpoint_name.

    The md-et package handles model loading and weight downloading from
    HuggingFace Hub (or cache).

    Args:
        run_dir_path: Ignored (kept for API compatibility)
        device: torch device
        checkpoint_name: Ignored
        type: Calculator type
        filter_forces: Whether to apply force filtering
        hessian_batch_size: Not used by md-et package
        variant: Model variant: "4l", "5l", or "12l" (default)

    Returns:
        MDETCalculator (ASE Calculator interface)
    """
    from md_et import load_calculator

    logger.info(f"Loading md-et calculator (variant={variant}, device={device})")
    calc = load_calculator(
        variant=variant,
        device=device,
        filter_forces=filter_forces,
    )

    # Override default vmap chunk size: 16 OOMs on L4 24GB,
    # and the built-in halving only applies within a single call.
    _orig_get_batched_hessians = calc.get_batched_hessians

    def _get_batched_hessians_sized(atoms_list, hessian_batch_size=hessian_batch_size):
        return _orig_get_batched_hessians(atoms_list, hessian_batch_size=hessian_batch_size)

    calc.get_batched_hessians = _get_batched_hessians_sized

    return calc
