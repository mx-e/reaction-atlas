"""Experiment registry — single source of truth for valid experiment ids.

An experiment is a logical scope on the otherwise-shared CRN database.
- "main"              = the original full graph (all rows pre-tagging end up here)
- "formose-drilldown" = a subgraph drilldown on formose chemistry; the worker
                        for this experiment runs with RESTRICT_TO_EXISTING_COMPOUNDS
                        so it generates new reactions only between already-tagged
                        compounds, never adds new compounds.

Adding an experiment id requires updating this set; the API dependency,
the kinetics solver loop, and the worker config all read from here so a
typo in env config is rejected before any data lands.
"""

DEFAULT_EXPERIMENT = "main"

EXPERIMENTS: frozenset[str] = frozenset({
    "main",
    "formose-drilldown",
})


def is_valid(experiment: str) -> bool:
    return experiment in EXPERIMENTS
