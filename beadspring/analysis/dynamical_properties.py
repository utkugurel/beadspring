"""Routines for analysing dynamical observables of bead-spring simulations.

The functions in this module operate on trajectories represented as NumPy
arrays with shape ``(n_frames, n_particles, 3)``.  All public functions
perform strict input validation and provide detailed docstrings so that they
can be used directly by end-users and automatically documented by tools such
as Sphinx.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import ArrayLike
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.special import gammainc

from ._validation import ensure_trajectory_array, ensure_wave_vectors

__all__ = [
    "VanHoveResult",
    "compute_msd",
    "compute_ngp",
    "compute_debye_waller_factor",
    "compute_van_hove_correlation",
    "get_k_vectors",
    "compute_fskt",
    "compute_fskt_batched",
    "chi_squared",
    "oneparam_fit",
    "fit_msd_with_quality_control",
    "fit_line_with_fixed_slope",
    "compute_vacf",
    "compute_time_autocorrelation",
]


@dataclass(frozen=True)
class VanHoveResult:
    """Container holding the self part of the Van Hove correlation function.

    Attributes
    ----------
    radii:
        Mid-points of the radial histogram bins in Cartesian units.
    g_r_t:
        Histogram values normalised as a probability density.  The array has the
        shape ``(n_lags, n_bins)`` where ``n_lags`` equals ``n_frames - 1``.
    lag_times:
        Optional array of lag times associated with the correlation function.
        The array is empty when no time information is provided.
    """

    radii: np.ndarray
    g_r_t: np.ndarray
    lag_times: np.ndarray


def _iterate_displacements(trajectory: np.ndarray) -> Iterable[Tuple[int, np.ndarray]]:
    """Yield displacements for each time lag.

    Parameters
    ----------
    trajectory:
        Three dimensional trajectory array returned by
        :func:`ensure_trajectory_array`.

    Yields
    ------
    tuple
        Pairs of ``(lag, displacements)`` where ``lag`` is the integer time lag
        in frames and ``displacements`` is an array with shape
        ``(n_frames - lag, n_particles, 3)`` containing the coordinate
        differences for all possible time origins.
    """

    n_frames = trajectory.shape[0]
    for lag in range(1, n_frames):
        yield lag, trajectory[lag:] - trajectory[:-lag]


def compute_msd(
    positions: ArrayLike,
    *,
    per_particle: bool = False,
    average_time_origins: bool = True,
) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
    """Compute the mean-squared displacement for a trajectory.

    Parameters
    ----------
    positions:
        Array-like trajectory with shape ``(n_frames, n_particles, 3)`` containing
        unwrapped Cartesian coordinates.
    per_particle:
        When ``True`` the MSD per particle is returned in addition to the
        ensemble average.
    average_time_origins:
        If ``True`` (default) the displacement for each time lag is averaged over
        all possible time origins.  Setting this argument to ``False`` yields the
        displacement with respect to the first frame only, mimicking the
        historical behaviour of the project.

    Returns
    -------
    numpy.ndarray or tuple of numpy.ndarray
        The mean squared displacement for each lag time.  When
        ``per_particle=True`` the function returns ``(msd, msd_per_particle)``
        where ``msd_per_particle`` has shape ``(n_lags, n_particles)``.

    Raises
    ------
    ValueError
        If the input trajectory does not contain at least two frames or has an
        invalid shape.
    """

    trajectory = ensure_trajectory_array(positions, min_frames=2)
    n_frames, n_particles, _ = trajectory.shape

    msd_per_particle = np.zeros((n_frames - 1, n_particles), dtype=float)

    for lag, displacement in _iterate_displacements(trajectory):
        squared = np.sum(displacement**2, axis=-1)
        if average_time_origins:
            squared = squared.mean(axis=0)
        else:
            squared = squared[0]
        msd_per_particle[lag - 1] = squared

    msd = msd_per_particle.mean(axis=1)
    if per_particle:
        return msd, msd_per_particle
    return msd


def compute_ngp(positions: ArrayLike) -> np.ndarray:
    r"""Compute the non-Gaussian parameter (NGP).

    The NGP provides a measure of how strongly particle displacements deviate
    from a Gaussian distribution.  The calculation follows the conventional
    definition :math:`\alpha_2 = 3\langle r^4 \rangle / (5\langle r^2 \rangle^2) - 1`.

    Parameters
    ----------
    positions:
        Array-like trajectory with shape ``(n_frames, n_particles, 3)``.

    Returns
    -------
    numpy.ndarray
        The non-Gaussian parameter for each lag time (excluding the zero lag).
    """

    trajectory = ensure_trajectory_array(positions, min_frames=2)

    ngp = np.zeros(trajectory.shape[0] - 1, dtype=float)
    for lag, displacement in _iterate_displacements(trajectory):
        squared = np.sum(displacement**2, axis=-1)
        mean_r2 = squared.mean()
        mean_r4 = (squared**2).mean()
        if np.isclose(mean_r2, 0.0):
            ngp[lag - 1] = 0.0
        else:
            ngp[lag - 1] = 3.0 * mean_r4 / (5.0 * mean_r2**2) - 1.0
    return ngp


def compute_debye_waller_factor(
    time_log: ArrayLike,
    msd: ArrayLike,
    *,
    tau_p: float = 3.0,
    kind: str = "linear",
) -> float:
    r"""Evaluate the Debye-Waller factor at a specific time scale.

    Parameters
    ----------
    time_log:
        One-dimensional array containing the sampling times of the MSD data.
        The array must be strictly increasing.
    msd:
        Mean-squared displacement values recorded at ``time_log``.
    tau_p:
        The time point at which the Debye-Waller factor is evaluated.  The
        default value of ``3.0`` corresponds to the conventional choice of
        :math:`3\,\tau`.
    kind:
        Interpolation method passed to :func:`scipy.interpolate.interp1d`.

    Returns
    -------
    float
        The interpolated MSD value at ``tau_p``.

    Raises
    ------
    ValueError
        If ``tau_p`` lies outside the provided time range or if ``time_log`` is
        not strictly increasing.
    """

    times = np.asarray(time_log, dtype=float)
    msd = np.asarray(msd, dtype=float)

    if times.ndim != 1 or msd.ndim != 1:
        raise ValueError("time_log and msd must be one-dimensional arrays.")
    if times.shape[0] != msd.shape[0]:
        raise ValueError("time_log and msd must have the same length.")
    if not np.all(np.diff(times) > 0):
        raise ValueError("time_log must be strictly increasing.")
    if tau_p < times[0] or tau_p > times[-1]:
        raise ValueError("tau_p must lie within the range of time_log.")

    interpolator = interp1d(times, msd, kind=kind)
    return float(interpolator(tau_p))


def compute_van_hove_correlation(
    positions: ArrayLike,
    time_log: Optional[ArrayLike] = None,
    *,
    bins: int = 100,
    rmax: float = 8.0,
    density: bool = True,
) -> VanHoveResult:
    """Compute the self part of the Van Hove correlation function.

    Parameters
    ----------
    positions:
        Array-like trajectory with shape ``(n_frames, n_particles, 3)``.
    time_log:
        Optional sequence with the physical times corresponding to each frame.
        When provided, its length must match the number of frames in
        ``positions``.
    bins:
        Number of radial bins used for the histogram.
    rmax:
        Maximum radius considered in the histogram.
    density:
        Passed directly to :func:`numpy.histogram`.  When ``True`` the histogram
        is normalised to a probability density.

    Returns
    -------
    VanHoveResult
        Dataclass containing the bin centres, the ``G_s(r, t)`` values and the
        optional lag-time information.
    """

    trajectory = ensure_trajectory_array(positions, min_frames=2)
    if time_log is not None:
        lag_times = np.asarray(time_log, dtype=float)
        if lag_times.ndim != 1:
            raise ValueError("time_log must be one-dimensional when provided.")
        if lag_times.shape[0] != trajectory.shape[0]:
            raise ValueError("time_log must contain one value per trajectory frame.")
        lag_times = lag_times[1:] - lag_times[0]
    else:
        lag_times = np.empty(0, dtype=float)

    bin_edges = np.linspace(0.0, rmax, bins + 1)
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    g_r_t = np.zeros((trajectory.shape[0] - 1, bins), dtype=float)

    for lag, displacement in _iterate_displacements(trajectory):
        distances = np.linalg.norm(displacement.reshape(-1, displacement.shape[-1]), axis=1)
        histogram, _ = np.histogram(distances, bins=bin_edges, density=density)
        g_r_t[lag - 1] = histogram

    return VanHoveResult(radii=bin_centres, g_r_t=g_r_t, lag_times=lag_times)


def get_k_vectors(
    ktarget: float,
    box_length: float,
    *,
    max_points: int = 1000,
    tolerance: float = 0.1,
    seed: Optional[int] = None,
) -> np.ndarray:
    r"""Generate reciprocal lattice vectors close to a target magnitude.

    Parameters
    ----------
    ktarget:
        Target magnitude of the wave-vector.
    box_length:
        Edge length of the (cubic) simulation box.
    max_points:
        Maximum number of vectors returned.  When more matches are found, a
        deterministic random subset is chosen.
    tolerance:
        Acceptable deviation from the target magnitude measured in units of the
        discrete lattice spacing :math:`2\pi/L`.
    seed:
        Optional seed passed to :func:`numpy.random.default_rng` when sampling.

    Returns
    -------
    numpy.ndarray
        Array with shape ``(n_vectors, 3)`` containing the selected k-vectors.

    Raises
    ------
    ValueError
        If no vectors satisfying the tolerance are found.
    """

    if box_length <= 0:
        raise ValueError("box_length must be positive.")
    if ktarget < 0:
        raise ValueError("ktarget must be non-negative.")

    k_step = 2.0 * np.pi / box_length
    k_discrete = ktarget / k_step

    upper_bound = int(np.ceil(k_discrete + tolerance))
    index_range = np.arange(-upper_bound, upper_bound + 1)
    indices = np.array(list(product(index_range, repeat=3)), dtype=int)

    magnitudes = np.linalg.norm(indices, axis=1)
    mask = np.abs(magnitudes - k_discrete) <= tolerance
    vectors = indices[mask] * k_step

    if vectors.size == 0:
        raise ValueError("No k-vectors found within the specified tolerance.")

    # Remove duplicates caused by rounding errors
    vectors = np.unique(vectors, axis=0)

    if vectors.shape[0] > max_points:
        rng = np.random.default_rng(seed)
        selection = rng.choice(vectors.shape[0], size=max_points, replace=False)
        vectors = vectors[selection]

    return vectors


def _fskt_from_displacements(
    displacements: Sequence[np.ndarray],
    k_vectors: np.ndarray,
) -> np.ndarray:
    """Compute :math:`F_s(k, t)` from a sequence of displacement arrays."""

    values = []
    for disp in displacements:
        phases = np.einsum("tnc,kc->tnk", disp, k_vectors, optimize=True)
        values.append(np.cos(phases).mean(axis=(0, 1)))
    return np.vstack(values)


def compute_fskt(
    positions: ArrayLike,
    k_vectors: ArrayLike,
    *,
    average_over_k: bool = True,
    average_time_origins: bool = True,
) -> np.ndarray:
    """Compute the self-intermediate scattering function :math:`F_s(k, t)`.

    Parameters
    ----------
    positions:
        Array-like trajectory with shape ``(n_frames, n_particles, 3)``.
    k_vectors:
        Array of wave-vectors with shape ``(n_k, 3)``.
    average_over_k:
        When ``True`` (default) the values are averaged over the supplied
        wave-vectors and the result has shape ``(n_lags,)``.  Otherwise the shape
        is ``(n_lags, n_k)``.
    average_time_origins:
        Control whether the displacement is averaged over all possible time
        origins.  When ``False`` only the first frame is used as the origin,
        reproducing the legacy behaviour of the project.

    Returns
    -------
    numpy.ndarray
        The self-intermediate scattering function.
    """

    trajectory = ensure_trajectory_array(positions, min_frames=2)
    k_vectors = ensure_wave_vectors(k_vectors)

    displacements = []
    for _, disp in _iterate_displacements(trajectory):
        if not average_time_origins:
            disp = disp[:1]
        displacements.append(disp)

    fskt = _fskt_from_displacements(displacements, k_vectors)
    if average_over_k:
        return fskt.mean(axis=1)
    return fskt


def compute_fskt_batched(
    positions: ArrayLike,
    k_vectors: ArrayLike,
    *,
    batch_size: int = 100,
    average_over_k: bool = True,
    average_time_origins: bool = True,
) -> np.ndarray:
    """Batched version of :func:`compute_fskt` for memory efficiency.

    Parameters
    ----------
    positions, k_vectors:
        Same meaning as in :func:`compute_fskt`.
    batch_size:
        Number of wave-vectors processed per batch.
    average_over_k, average_time_origins:
        Forwarded to :func:`compute_fskt`.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")

    trajectory = ensure_trajectory_array(positions, min_frames=2)
    k_vectors = ensure_wave_vectors(k_vectors)

    displacements = []
    for _, disp in _iterate_displacements(trajectory):
        if not average_time_origins:
            disp = disp[:1]
        displacements.append(disp)

    n_k = k_vectors.shape[0]
    result = np.zeros((len(displacements), n_k), dtype=float)

    start = 0
    while start < n_k:
        stop = min(start + batch_size, n_k)
        batch = k_vectors[start:stop]
        result[:, start:stop] = _fskt_from_displacements(displacements, batch)
        start = stop

    if average_over_k:
        return result.mean(axis=1)
    return result


def chi_squared(observed: ArrayLike, expected: ArrayLike, scaling: ArrayLike) -> float:
    """Compute the chi-squared value between observed and expected data."""

    observed = np.asarray(observed, dtype=float)
    expected = np.asarray(expected, dtype=float)
    scaling = np.asarray(scaling, dtype=float)

    if observed.shape != expected.shape:
        raise ValueError("observed and expected must have identical shapes.")
    if not np.broadcast_to(scaling, observed.shape).shape == observed.shape:
        raise ValueError("scaling must be broadcastable to the data shape.")

    return float(np.sum((observed - expected) ** 2 / scaling))


def oneparam_fit(function, x: ArrayLike, y: ArrayLike) -> Tuple[float, float]:
    """Fit a one-parameter model to data and compute the quality factor.

    Parameters
    ----------
    function:
        Callable ``f(x, p)`` describing the model.
    x, y:
        Observed data points.

    Returns
    -------
    tuple
        ``(p_opt, q)`` where ``p_opt`` is the fitted parameter and ``q`` is the
        quality factor derived from the chi-squared statistic.
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    popt, _ = curve_fit(function, x, y)
    parameter = float(popt[0])

    y_expected = function(x, parameter)
    scaling = np.where(np.isclose(y_expected, 0.0), 1.0, np.abs(y_expected))
    chi2 = chi_squared(y, y_expected, scaling)

    dof = max(len(x) - 1, 1)
    if chi2 <= 0:
        quality = 1.0
    else:
        quality = 1.0 - gammainc(dof / 2.0, chi2 / 2.0)

    return parameter, float(quality)


def fit_msd_with_quality_control(
    t: ArrayLike,
    msd: ArrayLike,
    msd_std: ArrayLike,
) -> Tuple[float, float]:
    """Estimate the diffusion coefficient from MSD data.

    The function progressively discards early-time data until the fit quality
    factor exceeds :math:`1/2`, thereby mimicking the protocol often used in
    glassy dynamics studies.

    Parameters
    ----------
    t:
        One-dimensional array of time points.
    msd:
        Mean-squared displacement corresponding to ``t``.
    msd_std:
        Standard deviation (or standard error) of ``msd``.

    Returns
    -------
    tuple
        Diffusion coefficient ``D`` and its uncertainty ``sigma_D``.
    """

    t = np.asarray(t, dtype=float)
    msd = np.asarray(msd, dtype=float)
    msd_std = np.asarray(msd_std, dtype=float)

    if np.any(t <= 0) or np.any(msd <= 0):
        raise ValueError("t and msd must contain strictly positive values.")

    def diffusion_model(time, diffusion):
        return 6.0 * diffusion * time

    log_t = np.log10(t)
    log_msd = np.log10(msd)

    msd_min = msd - msd_std
    msd_max = msd + msd_std

    start_index = 0
    quality = 0.0
    while quality < 0.5 and start_index < len(t) - 1:
        parameter, quality = oneparam_fit(lambda x, b: x + b, log_t[start_index:], log_msd[start_index:])
        if quality < 0.5:
            start_index += 1

    diffusion, _ = oneparam_fit(diffusion_model, t[start_index:], msd[start_index:])
    diff_min, _ = oneparam_fit(diffusion_model, t[start_index:], msd_min[start_index:])
    diff_max, _ = oneparam_fit(diffusion_model, t[start_index:], msd_max[start_index:])

    sigma = (diff_max - diff_min) / 2.0
    uncertainty = sigma / np.sqrt(max(len(log_msd) - start_index - 1, 1))

    return diffusion, float(uncertainty)


def fit_line_with_fixed_slope(x: ArrayLike, y: ArrayLike) -> float:
    """Fit a straight line with unit slope and return the intercept."""

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.shape != y.shape:
        raise ValueError("x and y must have identical shapes.")

    return float(np.mean(y - x))


def compute_vacf(velocities: ArrayLike) -> np.ndarray:
    """Compute the velocity autocorrelation function (VACF).

    Parameters
    ----------
    velocities:
        Array-like object with shape ``(n_frames, n_particles, 3)`` containing
        the velocities of each particle.

    Returns
    -------
    numpy.ndarray
        VACF values for each lag time.
    """

    trajectory = ensure_trajectory_array(velocities, min_frames=2)

    reference = trajectory[0]
    norm = np.sum(reference * reference)
    if np.isclose(norm, 0.0):
        raise ValueError("The initial velocity norm must be non-zero.")

    dot_products = np.einsum("nc,tnc->t", reference, trajectory[1:], optimize=True)
    vacf = dot_products / norm

    return vacf


def compute_time_autocorrelation(
    values: ArrayLike,
    *,
    normalise: bool = True,
) -> np.ndarray:
    """Compute the autocorrelation of a one-dimensional time series.

    Parameters
    ----------
    values:
        Sequence of scalars describing an observable along a trajectory.
    normalise:
        When ``True`` the correlation is normalised by the zero-lag value.

    Returns
    -------
    numpy.ndarray
        Autocorrelation values for lags ``0`` to ``n-1``.
    """

    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError("values must be a one-dimensional array.")

    array = array - array.mean()
    correlation = np.correlate(array, array, mode="full")
    correlation = correlation[array.size - 1 :]

    if normalise:
        zero_lag = correlation[0]
        if np.isclose(zero_lag, 0.0):
            raise ValueError("Cannot normalise correlation with zero variance.")
        correlation = correlation / zero_lag

    return correlation
