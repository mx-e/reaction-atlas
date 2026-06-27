"""Serialization helpers for numpy arrays <-> PostgreSQL bytea columns."""

import io
from typing import Optional

import numpy as np


def serialize_ndarray(arr: np.ndarray) -> bytes:
    """Serialize a numpy array to bytes for storage in a bytea column."""
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()


def deserialize_ndarray(data: bytes) -> np.ndarray:
    """Deserialize bytes from a bytea column back to a numpy array."""
    buf = io.BytesIO(data)
    return np.load(buf)


def serialize_ndarray_optional(arr: Optional[np.ndarray]) -> Optional[bytes]:
    """Serialize a numpy array or return None."""
    if arr is None:
        return None
    return serialize_ndarray(arr)


def deserialize_ndarray_optional(data: Optional[bytes]) -> Optional[np.ndarray]:
    """Deserialize bytes or return None."""
    if data is None:
        return None
    return deserialize_ndarray(data)


def serialize_trajectory(trajectory) -> Optional[bytes]:
    """Serialize a RelaxationTrajectory to bytes using npz format.

    Stores positions, energies, forces as arrays, and hessians as individual arrays.
    """
    if trajectory is None:
        return None

    buf = io.BytesIO()
    save_dict = {
        "energies": np.array(trajectory.energies, dtype=np.float64),
    }

    # Positions: list of (n_atoms, 3) arrays -> (n_frames, n_atoms, 3)
    if trajectory.positions:
        save_dict["positions"] = np.array(trajectory.positions, dtype=np.float64)

    # Forces: list of (n_atoms, 3) arrays -> (n_frames, n_atoms, 3)
    if trajectory.forces:
        save_dict["forces"] = np.array(trajectory.forces, dtype=np.float64)

    # Hessians: list of optional (3n, 3n) arrays
    if trajectory.hessians:
        for i, h in enumerate(trajectory.hessians):
            if h is not None:
                save_dict[f"hessian_{i}"] = np.array(h, dtype=np.float64)
        save_dict["n_hessians"] = np.array(len(trajectory.hessians))

    np.savez_compressed(buf, **save_dict)
    return buf.getvalue()


def deserialize_trajectory(data: Optional[bytes]):
    """Deserialize bytes back to a dict with trajectory arrays.

    Returns a dict with keys: positions, energies, forces, hessians.
    Caller is responsible for constructing the RelaxationTrajectory object.
    """
    if data is None:
        return None

    buf = io.BytesIO(data)
    npz = np.load(buf, allow_pickle=False)

    result = {
        "energies": npz["energies"].tolist(),
    }

    if "positions" in npz:
        result["positions"] = [frame for frame in npz["positions"]]
    else:
        result["positions"] = []

    if "forces" in npz:
        result["forces"] = [frame for frame in npz["forces"]]
    else:
        result["forces"] = []

    if "n_hessians" in npz:
        n = int(npz["n_hessians"])
        hessians = []
        for i in range(n):
            key = f"hessian_{i}"
            hessians.append(npz[key] if key in npz else None)
        result["hessians"] = hessians
    else:
        result["hessians"] = []

    return result
