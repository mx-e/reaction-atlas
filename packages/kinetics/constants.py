"""Shared constants for the kinetics subsystem.

Kept in a dedicated module so solver, sampler, and snapshot builder can
all reference the same decade grid without import cycles.
"""

# Decade time grid for the sample-and-weight step of the sampler.
# 19 points spanning the kinetically relevant window: 10^-10 s (femtosecond
# chemistry) to 10^8 s (> 1 year — well past any realistic reactor residence).
DECADE_EXPONENTS = list(range(-10, 9))  # [-10, -9, ..., 8]
DECADE_TIMES = [10.0 ** e for e in DECADE_EXPONENTS]
