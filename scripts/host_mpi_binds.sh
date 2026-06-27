#!/bin/bash
# Sourced by SLURM scripts that need host OpenMPI 4.1.2 inside the SIF.
# Sets the HOST_MPI_BINDS bash array with all the --bind args to mirror
# the host's Debian OpenMPI install at the same paths inside the
# container, so OMPI's hard-coded --prefix=/usr layout works as-is.
#
# Why this is what's needed: ORCA's orca_*_mpi binaries are linked
# against libmpi.so.40 (ABI 30.2). Debian trixie ships OpenMPI 5.0 with
# a bumped ABI; the host has 4.1.2 with the right ABI. Binding host's
# MPI into the original (no-MPI) SIF avoids a SIF rebuild and pins us
# to the exact MPI ORCA was built against.

LIBDIR=/usr/lib/x86_64-linux-gnu

# Pure files (each bound source → identical dest path)
_MPI_FILES=(
    /usr/bin/mpirun
    /usr/bin/orterun
    /usr/bin/orted
    /usr/bin/ompi_info
    "$LIBDIR/libmpi.so.40"
    "$LIBDIR/libmpi_cxx.so.40"
    "$LIBDIR/libopen-rte.so.40"
    "$LIBDIR/libopen-pal.so.40"
    "$LIBDIR/libhwloc.so.15"
    "$LIBDIR/libevent_core-2.1.so.7"
    "$LIBDIR/libevent_pthreads-2.1.so.7"
    "$LIBDIR/libudev.so.1"
    "$LIBDIR/libpmix.so.2"
    "$LIBDIR/libnl-3.so.200"
    "$LIBDIR/libnl-route-3.so.200"
)

# Whole-dir binds: plugin trees + config + help text
_MPI_DIRS=(
    "$LIBDIR/openmpi"   # /usr/lib/.../openmpi/{include,lib,lib/openmpi3/}
    "$LIBDIR/pmix2"     # PMIx runtime
    /etc/openmpi
    /usr/share/openmpi
)

HOST_MPI_BINDS=()
for p in "${_MPI_FILES[@]}"; do HOST_MPI_BINDS+=(--bind "$p:$p:ro"); done
for d in "${_MPI_DIRS[@]}";  do HOST_MPI_BINDS+=(--bind "$d:$d:ro"); done

export HOST_MPI_BINDS
