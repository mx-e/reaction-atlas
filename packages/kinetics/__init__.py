"""Kinetics core library — model builder, PETSc solver, snapshot, sampler.

Pure-Python library imported by the API (loop.py runs the solver as a
background asyncio task) and by SBML export. The PETSc/numba imports are
isolated in petsc_solver.py + numba_kernels.py so that callers wanting only
the model builder (e.g. SBML export) don't need PETSc installed.
"""
