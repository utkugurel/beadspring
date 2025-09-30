"""Time correlation utilities used throughout :mod:`beadspring`.

The original code base contained several bespoke correlation functions that
were reimplemented across different modules.  The helpers in this file provide
well-documented, vectorised routines that can be re-used whenever a new time
correlation needs to be computed.
"""

from __future__ import annotations

import numpy as np


def _validate_time_axis(data: np.ndarray) -> np.ndarray:
    """Ensure that *data* can be treated as a time series.

    Parameters
    ----------
    data:
        Input array whose first dimension corresponds to the time axis.

    Returns
    -------
    numpy.ndarray
        A read-only view of *data* with ``float64`` dtype for numerical
        stability.

    Raises
    ------
    ValueError
        If the array has fewer than two frames along the time axis.
    """

    series = np.asarray(data, dtype=float)
    if series.ndim == 0 or series.shape[0] < 2:
        raise ValueError("at least two time frames are required to compute a correlation")
    return series


def time_autocorrelation(
    data: np.ndarray,
    max_lag: int | None = None,
    *,
    normalize: bool = True,
    demean: bool = True,
) -> np.ndarray:
    """Compute the time autocorrelation of a trajectory.

    The function treats the first axis of ``data`` as the time dimension and
    flattens all remaining axes into one vector per frame.  The scalar product
    between the vectors separated by a lag ``tau`` is averaged over the valid
    time window and optionally normalised by the zero-lag value.  This closely
    mimics the analysis pipeline used in molecular simulations while remaining
    agnostic to the concrete physical observable.

    Parameters
    ----------
    data:
        Array of shape ``(n_frames, ...)`` describing the quantity of interest.
    max_lag:
        Maximum time lag to evaluate.  When ``None`` (the default) all possible
        lags up to ``n_frames - 1`` are computed.
    normalize:
        If ``True`` (default) the correlation is divided by the zero-lag value
        so that ``C(0) == 1``.
    demean:
        If ``True`` (default) the mean value of each component is subtracted
        prior to computing the correlation.  This is often required to obtain a
        stationary signal.

    Returns
    -------
    numpy.ndarray
        Array of length ``max_lag + 1`` containing the autocorrelation values
        ``C(tau)`` for ``tau = 0, 1, ..., max_lag``.
    """

    series = _validate_time_axis(data)
    n_frames = series.shape[0]

    if max_lag is None:
        max_lag = n_frames - 1
    if max_lag < 0 or max_lag >= n_frames:
        raise ValueError("max_lag must satisfy 0 <= max_lag < n_frames")

    flattened = series.reshape(n_frames, -1)
    if demean:
        flattened = flattened - flattened.mean(axis=0, keepdims=True)

    correlations = np.empty(max_lag + 1, dtype=float)
    for lag in range(max_lag + 1):
        lhs = flattened[: n_frames - lag]
        rhs = flattened[lag:]
        correlations[lag] = np.mean(np.einsum("ij,ij->i", lhs, rhs))

    if normalize:
        zero_lag = correlations[0]
        if zero_lag != 0:
            correlations /= zero_lag
        else:
            correlations[:] = 0.0

    return correlations


def time_cross_correlation(
    data_a: np.ndarray,
    data_b: np.ndarray,
    max_lag: int | None = None,
    *,
    normalize: bool = True,
    demean: bool = True,
) -> np.ndarray:
    """Compute the time-lagged cross correlation between two trajectories.

    The implementation mirrors :func:`time_autocorrelation` but accepts two
    different signals.  The function is particularly useful when analysing the
    coupling between different order parameters in polymer simulations.

    Parameters
    ----------
    data_a, data_b:
        Arrays containing the two time series.  Both must share the same shape
        and number of frames.
    max_lag:
        Maximum lag to evaluate.  The default ``None`` means ``n_frames - 1``.
    normalize:
        If ``True`` (default) the cross correlation is normalised by the square
        root of the two zero-lag values so that ``C(0)`` falls within
        ``[-1, 1]``.
    demean:
        If ``True`` (default) the mean of each component is subtracted before
        evaluating the correlation.

    Returns
    -------
    numpy.ndarray
        Cross correlation values for increasing lags.
    """

    series_a = _validate_time_axis(data_a)
    series_b = _validate_time_axis(data_b)
    if series_a.shape != series_b.shape:
        raise ValueError("data_a and data_b must share the same shape")

    n_frames = series_a.shape[0]
    if max_lag is None:
        max_lag = n_frames - 1
    if max_lag < 0 or max_lag >= n_frames:
        raise ValueError("max_lag must satisfy 0 <= max_lag < n_frames")

    a_flat = series_a.reshape(n_frames, -1)
    b_flat = series_b.reshape(n_frames, -1)
    if demean:
        a_flat = a_flat - a_flat.mean(axis=0, keepdims=True)
        b_flat = b_flat - b_flat.mean(axis=0, keepdims=True)

    correlations = np.empty(max_lag + 1, dtype=float)
    for lag in range(max_lag + 1):
        lhs = a_flat[: n_frames - lag]
        rhs = b_flat[lag:]
        correlations[lag] = np.mean(np.einsum("ij,ij->i", lhs, rhs))

    if normalize:
        zero_norm = np.sqrt(
            np.mean(np.einsum("ij,ij->i", a_flat, a_flat))
            * np.mean(np.einsum("ij,ij->i", b_flat, b_flat))
        )
        if zero_norm != 0:
            correlations /= zero_norm

    return correlations


__all__ = ["time_autocorrelation", "time_cross_correlation"]
