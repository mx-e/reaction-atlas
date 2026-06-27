"""KineticsSnapshot — JSON-serializable result of one solve.

Stored in the kinetics_snapshots table (payload_jsonb column). Read by:
  - GET /api/kinetics/snapshot (frontend display)
  - GPU worker sample_pair_from_snapshot (kinetic-weighted sampling, Phase 5)
  - GET /api/sbml/export indirectly (uses model.py instead, but shares the
    decade representation idea)

Decade-resolved structure:
  decade_times          = [10^-10, 10^-9, ..., 10^8] s
  decade_distributions  = [{smiles -> log_conc_weight}, ...] (one dict per decade)

Sampling is two-step: first pick a decade (entropy-weighted, Phase 5), then
pick two SMILES from that decade weighted by log_conc_weight.
"""

import dataclasses
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class KineticsSnapshot:
    # Species index — sorted SMILES list, used for the worker's lookup tables
    smiles_list: list[str]

    # Continuous time series for the frontend's concentration plot
    times: list[float]                              # ~400 log-spaced t in [1e-12, 1e8] s
    concentrations: dict[str, list[float]]          # smiles -> list aligned with times

    # Per-decade snapshots used by the sampler
    decade_times: list[float]                       # 19 points: 10^-10..10^8
    decade_exponents: list[int]                     # [-10, -9, ..., 8]
    decade_distributions: list[dict[str, float]]    # one dict per decade entry, smiles -> normalized log-conc weight

    temperature: float
    n_species: int
    n_reactions: int        # total inter-molecular reactions in the model
    n_reactions_dft: int    # of which had DFT separated barriers (drives quality story)
    n_manual_equilibria: int

    # Per-reaction detail for the frontend reaction table
    reactions_summary: list[dict] = field(default_factory=list)
    # Non-zero initial concentrations {smiles → M}
    initial_concentrations: dict[str, float] = field(default_factory=dict)
    # Pre-computed Shannon entropy per decade (bits)
    decade_entropies: list[float] = field(default_factory=list)

    # Steady-state sampling distribution: softmax(log(conc)) at the final
    # time point. This is what the GPU worker uses for merge-pair sampling.
    steady_state_distribution: dict[str, float] = field(default_factory=dict)
    # Raw log-concentrations at steady state (for display / diagnostics)
    steady_state_log_concs: dict[str, float] = field(default_factory=dict)

    solve_wall_time_s: Optional[float] = None
    computed_at: Optional[str] = None  # ISO 8601 string when present

    def to_json(self) -> dict:
        """Return a JSON-friendly dict (everything is already serializable)."""
        return asdict(self)

    @classmethod
    def from_json(cls, payload: dict) -> "KineticsSnapshot":
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in payload.items() if k in known})

    @property
    def is_ready(self) -> bool:
        """Whether this snapshot has a usable steady-state distribution."""
        return len(self.steady_state_distribution) >= 2
