"""Steady-state pair sampling from a KineticsSnapshot.

Samples two species from the steady-state distribution stored in the
snapshot. The distribution is softmax(log(conc)) computed at t_max.
Sampling is with replacement (allows A+A homodimers).

Returns None if the snapshot has no steady-state distribution with at
least 2 species.

Pure function — no DB, no PETSc, no side effects. Importable from the GPU
worker (Phase 5) with zero extra dependencies beyond numpy.
"""

import random
from typing import Optional

from packages.kinetics.snapshot import KineticsSnapshot


def sample_pair_from_snapshot(
    snapshot: KineticsSnapshot,
    max_carbon_count: Optional[int] = None,
    carbon_count_lookup: Optional[dict[str, int]] = None,
    rng: Optional[random.Random] = None,
) -> Optional[tuple[str, str]]:
    """Pick a 2-species reaction seed from the steady-state distribution.

    Args:
        snapshot: cached KineticsSnapshot (from kinetics_snapshots table)
        max_carbon_count: optional filter — both species must have <= this
            many carbons. Requires carbon_count_lookup to resolve.
        carbon_count_lookup: optional dict {smiles -> n_carbons}. If None and
            max_carbon_count is set, the filter is ignored.
        rng: optional seeded random.Random for determinism

    Returns:
        (smiles_1, smiles_2) or None if no sampleable distribution found
    """
    r = rng if rng is not None else random

    dist = snapshot.steady_state_distribution
    if len(dist) < 2:
        return None

    smiles = list(dist.keys())
    weights = list(dist.values())
    total = sum(weights)
    if total <= 0:
        return None

    max_attempts = 25
    for _ in range(max_attempts):
        s1 = _weighted_choice(r, weights, total)
        s2 = _weighted_choice(r, weights, total)
        smi1, smi2 = smiles[s1], smiles[s2]

        if max_carbon_count is not None and carbon_count_lookup is not None:
            n1 = carbon_count_lookup.get(smi1, 0)
            n2 = carbon_count_lookup.get(smi2, 0)
            if n1 > max_carbon_count or n2 > max_carbon_count:
                continue

        return smi1, smi2

    return None


def _weighted_choice(rng, weights: list[float], total: float) -> int:
    """Linear-scan weighted index sampling."""
    u = rng.random() * total
    acc = 0.0
    for i, w in enumerate(weights):
        acc += w
        if u < acc:
            return i
    return len(weights) - 1
