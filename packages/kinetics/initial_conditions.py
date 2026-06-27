"""Initial concentrations and physical constants for the kinetic ODE.

These match the upstream defaults in crn-exploration/lib/kinetic_sampler.py:
  - water = 55 M (pure water)
  - CH2O (formaldehyde) = 0.1 M
  - CO2 = 0.1 M
  - H2 = 1e-5 M (trace)
  - pH = 10 → [H+] = 1e-10 M
  - [OH-] = 0.1 M (held effectively fixed by buffer equilibria)

The kinetic solver injects these at t=0 for any species present in the
reaction graph. Species not in the graph are simply ignored. The solver also
applies the four manual buffer equilibria (water autoionization, CO2
hydration, H2CO3 dissociation, proton solvation) which keep these
concentrations near steady-state during the integration.
"""

# Physical constants used by Eyring rate computation
KB_EV = 8.617333262e-5   # eV / K
H_EV_S = 4.135668e-15    # eV * s
EV_TO_KCAL = 23.0605
DIFFUSION_LIMIT_M_PER_S = 1e10  # cap on bimolecular rate constants

# Default exploration temperature — lower than the previous 750 K to give
# more physically meaningful steady-state distributions for sampling.
DEFAULT_TEMPERATURE_K = 500.0

# Default pH for proton initial concentration
DEFAULT_PH = 10.0

# H+ in our graph is [HH] (single H, no electrons in MLFF = proton)
H_PLUS_SMILES = "[HH]"

# SMILES → initial concentration (M). All seed species start at a uniform
# 1 mM to increase diversity in the steady-state sampling distribution.
# The manual buffer equilibria keep acid-base chemistry self-consistent.
DEFAULT_INITIAL_CONCS = {
    "O":             1e-3,    # H2O
    "C=O":           1e-3,    # CH2O (formaldehyde)
    "O=C=O":         1e-3,    # CO2
    "[H+].[H+]":     1e-3,    # H2
    H_PLUS_SMILES:   1e-3,    # H+
    "[OH-]":         1e-3,    # hydroxide
    "OC(O)O":        1e-3,    # carbonic acid (H2CO3)
    "O=C([O-])O":    1e-3,    # bicarbonate (HCO3-)
    "[OH3+]":        1e-3,    # hydronium
}

# Uniform initial concentration applied to EVERY species in the ODE model
# (including compounds discovered at runtime). This gives newly-found species
# a chance to react at t=0 instead of sitting at exactly zero forever — which
# was producing a steady-state distribution dominated entirely by seeds.
# Any smiles appearing in DEFAULT_INITIAL_CONCS or the override dict passed
# to build_ode_model will take precedence over this uniform baseline.
UNIFORM_INITIAL_CONC_M = 1e-3

# Concentration floor below which a species is treated as "not active" for
# decade-sampling purposes (Shannon entropy weighting filters these out).
NOISE_FLOOR_M = 1e-20
