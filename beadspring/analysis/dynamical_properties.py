"""Dynamical observables computed from bead-spring polymer trajectories."""

from __future__ import annotations

from itertools import product
from typing import Callable, Tuple

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.special import gammainc

from .time_correlation import time_autocorrelation

ArrayLike = np.ndarray


def _validate_positions(positions: ArrayLike) -> ArrayLike:
    """Return a float copy of ``positions`` and validate its shape."""

    coords = np.asarray(positions, dtype=float)
    if coords.ndim != 3 or coords.shape[-1] != 3 or coords.shape[0] < 2:
        raise ValueError("positions must be shaped (n_frames, n_particles, 3)")
    return coords


def compute_msd(positions: ArrayLike, *, per_particle: bool = False):
    """Compute the mean-squared displacement (MSD).

    Parameters
    ----------
    positions:
        Unwrapped particle coordinates with shape ``(n_frames, n_particles, 3)``.
    per_particle:
        If ``True`` return the per-particle MSD alongside the ensemble average.

    Returns
    -------
    numpy.ndarray or tuple[numpy.ndarray, numpy.ndarray]
        The ensemble-averaged MSD.  When ``per_particle`` is ``True`` the second
        array contains the MSD of every particle with shape
        ``(n_frames - 1, n_particles)``.
    """

    coords = _validate_positions(positions)
    displacements = coords[1:] - coords[0]
    squared_displacements = np.sum(displacements**2, axis=2)
    msd = np.mean(squared_displacements, axis=1)

    if per_particle:
        return msd, squared_displacements
    return msd


def compute_ngp(positions: ArrayLike) -> np.ndarray:
    """Compute the non-Gaussian parameter (NGP).

    The NGP highlights deviations from a purely diffusive process.  Values above
    zero indicate heterogeneous motion among the particles.
    """

    coords = _validate_positions(positions)
    _, squared_displacements = compute_msd(coords, per_particle=True)
    fourth_moment = np.sum((coords[1:] - coords[0]) ** 4, axis=2)

    msd = np.mean(squared_displacements, axis=1)
    fourth_moment_mean = np.mean(fourth_moment, axis=1)

    denominator = 5.0 * msd**2
    with np.errstate(divide="ignore", invalid="ignore"):
        ngp = 3.0 * np.divide(
            fourth_moment_mean,
            denominator,
            out=np.zeros_like(msd),
            where=denominator != 0,
        ) - 1.0
    return ngp


def compute_debye_waller_factor(time_log: ArrayLike, msd: ArrayLike, tau_p: float = 3.0) -> float:
    """Evaluate the Debye–Waller factor at a specific relaxation time.

    Linear interpolation is used when ``tau_p`` does not exactly match a time in
    ``time_log``.  Extrapolation is avoided to prevent misleading values.
    """

    times = np.asarray(time_log, dtype=float)
    msd = np.asarray(msd, dtype=float)
    if times.ndim != 1 or msd.ndim != 1 or times.size != msd.size:
        raise ValueError("time_log and msd must be one-dimensional arrays of equal length")
    if tau_p < times.min() or tau_p > times.max():
        raise ValueError("tau_p must lie within the bounds of time_log")

    interpolator = interp1d(times, msd, kind="linear")
    return float(interpolator(tau_p))


def compute_van_hove_correlation(positions: ArrayLike, bins: int = 100, rmax: float = 8.0) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the self part of the Van Hove correlation function.

    Returns one histogram per time lag describing the displacement distribution
    relative to the initial configuration.
    """

    coords = _validate_positions(positions)
    displacements = coords[1:] - coords[0]
    distances = np.linalg.norm(displacements, axis=2)

    histograms = np.empty((coords.shape[0] - 1, bins), dtype=float)
    bin_edges = None
    for index, frame_distances in enumerate(distances):
        hist, edges = np.histogram(frame_distances, bins=bins, range=(0.0, rmax), density=True)
        histograms[index] = hist
        if bin_edges is None:
            bin_edges = edges

    radial_grid = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return radial_grid, histograms


def get_k_vectors(
    ktarget: float,
    box_length: float,
    max_points: int = 1000,
    *,
    save_vectors: bool = False,
) -> np.ndarray:
    """Generate reciprocal lattice vectors close to ``ktarget``.

    Parameters
    ----------
    ktarget:
        Target magnitude of the wave vector.
    box_length:
        Edge length of the cubic simulation box.
    max_points:
        Maximum number of returned vectors.  When more candidates are available
        a deterministic subsample is chosen.
    save_vectors:
        If ``True`` the k-vectors are written to ``k_vectors.npy`` for quick
        reuse.
    """

    k_step = 2.0 * np.pi / box_length
    k_discrete = ktarget / k_step
    k_max = int(np.ceil(k_discrete))

    n_values = np.arange(-k_max, k_max + 1)
    k_indices = np.array(list(product(n_values, repeat=3)))

    k_magnitudes = np.linalg.norm(k_indices, axis=1)
    mask = np.abs(k_magnitudes - k_discrete) < 0.1
    k_vectors = k_indices[mask] * k_step

    if k_vectors.shape[0] > max_points:
        rng = np.random.default_rng(1)
        k_vectors = k_vectors[rng.choice(k_vectors.shape[0], max_points, replace=False)]

    if save_vectors:
        np.save("k_vectors.npy", k_vectors)

    return k_vectors


def compute_fskt(positions: ArrayLike, k_vectors: ArrayLike) -> np.ndarray:
    """Compute the self intermediate scattering function :math:`F_s(k, t)`.

    The returned array has length ``n_frames - 1`` to mirror the MSD definition
    used throughout the module.
    """

    coords = _validate_positions(positions)
    k_vectors = np.asarray(k_vectors, dtype=float)
    if k_vectors.ndim != 2 or k_vectors.shape[1] != 3:
        raise ValueError("k_vectors must have shape (n_vectors, 3)")

    displacements = coords[1:] - coords[0]
    displacement_dot_k = np.einsum("tnd,kd->tnk", displacements, k_vectors)
    return np.mean(np.cos(displacement_dot_k), axis=(1, 2))


def compute_fskt_batched(positions: ArrayLike, k_vectors: ArrayLike, batch_size: int = 100) -> np.ndarray:
    """Compute :math:`F_s(k, t)` using batches of k-vectors to reduce memory use."""

    coords = _validate_positions(positions)
    k_vectors = np.asarray(k_vectors, dtype=float)
    if k_vectors.ndim != 2 or k_vectors.shape[1] != 3:
        raise ValueError("k_vectors must have shape (n_vectors, 3)")

    n_vectors = k_vectors.shape[0]
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    displacements = coords[1:] - coords[0]
    result = np.zeros(displacements.shape[0], dtype=float)

    for start in range(0, n_vectors, batch_size):
        batch = k_vectors[start : start + batch_size]
        weights = batch.shape[0]
        displacement_dot_k = np.einsum("tnd,kd->tnk", displacements, batch)
        result += np.mean(np.cos(displacement_dot_k), axis=(1, 2)) * weights

    return result / n_vectors


def chi_squared(observed: ArrayLike, expected: ArrayLike, scaling: ArrayLike) -> float:
    r"""Return the :math:`\chi^2` distance between two arrays."""

    observed = np.asarray(observed, dtype=float)
    expected = np.asarray(expected, dtype=float)
    scaling = np.asarray(scaling, dtype=float)
    return float(np.sum((observed - expected) ** 2 / scaling))


def oneparam_fit(function: Callable[[ArrayLike, float], ArrayLike], x: ArrayLike, y: ArrayLike) -> Tuple[float, float]:
    """Fit a model with a single free parameter and compute the quality factor."""

    popt, _ = curve_fit(function, x, y)
    parameter = popt[0]

    y_expected = function(x, parameter)
    chi2 = chi_squared(y, y_expected, y_expected)
    dof = len(x) - 1
    if chi2 <= 0:
        quality = 1.0
    else:
        quality = 1 - gammainc(dof / 2.0, chi2 / 2.0)

    return parameter, quality


def fit_msd_with_quality_control(
    t: ArrayLike,
    msd: ArrayLike,
    msd_std: ArrayLike,
    *,
    plot: bool = False,
    title: str = "MSD",
) -> Tuple[float, float]:
    """Fit the long-time MSD regime to extract a diffusion coefficient."""

    del plot, title  # Plotting hooks are intentionally unused in tests.

    def diffusion(time, diffusion_coeff):
        return 6.0 * diffusion_coeff * time

    times = np.asarray(t, dtype=float)
    msd = np.asarray(msd, dtype=float)
    msd_std = np.asarray(msd_std, dtype=float)

    msd_min = msd - msd_std
    msd_max = msd + msd_std

    start_index = 0
    quality = 0.0
    while quality < 0.5 and start_index < times.size - 1:
        diffusion_coeff, quality = oneparam_fit(diffusion, times[start_index:], msd[start_index:])
        start_index += 1

    diffusion_min, _ = oneparam_fit(diffusion, times[start_index - 1 :], msd_min[start_index - 1 :])
    diffusion_max, _ = oneparam_fit(diffusion, times[start_index - 1 :], msd_max[start_index - 1 :])

    diffusion_sigma = (diffusion_max - diffusion_min) / 2.0
    diffusion_uncertainty = diffusion_sigma / np.sqrt(len(msd) - 1)

    return diffusion_coeff, diffusion_uncertainty


def fit_line_with_fixed_slope(x: ArrayLike, y: ArrayLike) -> float:
    """Return the intercept of a best-fit line with slope fixed to one."""

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return float(np.mean(y - x))


def compute_vacf(velocities: ArrayLike) -> np.ndarray:
    """Compute the velocity autocorrelation function (VACF).

    Parameters
    ----------
    velocities:
        Particle velocities with shape ``(n_frames, n_particles, 3)``.

    Returns
    -------
    numpy.ndarray
        Normalised VACF excluding the zero-lag value.
    """

    vel = _validate_positions(velocities)
    correlation = time_autocorrelation(vel, normalize=True, demean=True)
    return correlation[1:]


__all__ = [
    "compute_debye_waller_factor",
    "compute_fskt",
    "compute_fskt_batched",
    "compute_msd",
    "compute_ngp",
    "compute_vacf",
    "compute_van_hove_correlation",
    "fit_line_with_fixed_slope",
    "fit_msd_with_quality_control",
    "get_k_vectors",
    "chi_squared",
    "oneparam_fit",
]
