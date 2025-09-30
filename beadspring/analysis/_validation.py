"""Validation helpers for trajectory analysis functions.

The functions in this module centralise validation logic that is shared
across multiple public analysis utilities.  Keeping the validation in a
single place makes it easier to maintain consistent behaviour and to
write focused unit tests.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike

__all__ = ["ensure_trajectory_array", "ensure_wave_vectors"]


def ensure_trajectory_array(positions: ArrayLike, *, min_frames: int = 2) -> np.ndarray:
    """Return a validated trajectory array.

    Parameters
    ----------
    positions:
        Array-like object containing the particle coordinates.  The array must
        be three dimensional with the shape ``(n_frames, n_particles, ndim)``
        and is converted to a :class:`numpy.ndarray` with ``float64`` dtype.
    min_frames:
        Minimum number of frames required for the analysis.  Many dynamical
        observables require at least two frames.  A :class:`ValueError` is
        raised when the input trajectory does not contain the requested number
        of frames.

    Returns
    -------
    numpy.ndarray
        A contiguous array with ``dtype=float``.  The returned array always has
        ``ndim == 3``.

    Raises
    ------
    ValueError
        If the input array does not contain exactly three dimensions or the
        number of frames is smaller than ``min_frames``.
    TypeError
        If the provided object cannot be interpreted as an array of floats.
    """

    array = np.asarray(positions, dtype=float)

    if array.ndim != 3:
        raise ValueError(
            "A trajectory array must have three dimensions ``(frames, particles, coords)``."
        )

    n_frames = array.shape[0]
    if n_frames < min_frames:
        raise ValueError(
            f"The trajectory must contain at least {min_frames} frames; "
            f"received {n_frames}."
        )

    return np.ascontiguousarray(array)


def ensure_wave_vectors(k_vectors: ArrayLike) -> np.ndarray:
    """Validate an array of reciprocal-space vectors.

    Parameters
    ----------
    k_vectors:
        Array-like object with shape ``(n_vectors, 3)`` containing wave vectors
        in Cartesian components.  Any iterable accepted by
        :func:`numpy.asarray` is valid.

    Returns
    -------
    numpy.ndarray
        A two dimensional array with shape ``(n_vectors, 3)`` and
        ``dtype=float``.

    Raises
    ------
    ValueError
        If the provided array does not contain exactly three columns.
    TypeError
        If the array cannot be converted to floating point numbers.
    """

    array = np.asarray(k_vectors, dtype=float)

    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(
            "Wave-vector arrays must be two dimensional with shape ``(n, 3)``."
        )

    return np.ascontiguousarray(array)
