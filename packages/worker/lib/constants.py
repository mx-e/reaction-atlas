"""
Shared constants for the CRN exploration codebase.

Unit conversions, energy thresholds, and atomic data.
"""

# Unit conversions
BOHR_TO_ANG = 0.529177249
ANG_TO_BOHR = 1.0 / BOHR_TO_ANG
HARTREE_TO_EV = 27.211386245988
HARTREE_BOHR_TO_EV_ANG = HARTREE_TO_EV / BOHR_TO_ANG
HARTREE_BOHR2_TO_EV_ANG2 = HARTREE_TO_EV / (BOHR_TO_ANG**2)

# Energy threshold for validating reaction energy differences
ENERGY_THRESHOLD_HARTREE = 0.1  # Maximum allowed energy difference magnitude (hartree)
ENERGY_THRESHOLD_EV = ENERGY_THRESHOLD_HARTREE * HARTREE_TO_EV

# Atomic number -> symbol mapping
ATOMIC_SYMBOLS = {
    1: "H",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
    15: "P",
    16: "S",
    17: "Cl",
    35: "Br",
    53: "I",
}
