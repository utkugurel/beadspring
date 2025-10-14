"""Unit tests for :mod:`beadspring.analysis.dynamical_properties`."""

from __future__ import annotations

import math

import numpy as np
import numpy.testing as npt
import pytest

from beadspring.analysis.dynamical_properties import (
    VanHoveResult,
    chi_squared,
    compute_debye_waller_factor,
    compute_fskt,
    compute_fskt_batched,
    compute_msd,
    compute_ngp,
    compute_time_autocorrelation,
    compute_vacf,
    compute_van_hove_correlation,
    fit_line_with_fixed_slope,
    fit_msd_with_quality_control,
    get_k_vectors,
    oneparam_fit,
)


@pytest.fixture
def straight_line_positions() -> np.ndarray:
    """Create a simple deterministic trajectory with constant velocity."""

    frames = 4
    particles = 2
    positions = np.zeros((frames, particles, 3), dtype=float)
    for frame in range(frames):
        positions[frame, 0, 0] = frame
        positions[frame, 1, 0] = frame + 1
    return positions


def test_compute_msd_matches_manual_result(straight_line_positions):
    msd, per_particle = compute_msd(straight_line_positions, per_particle=True)

    expected_per_particle = np.array([[1.0, 1.0], [4.0, 4.0], [9.0, 9.0]])
    expected_msd = expected_per_particle.mean(axis=1)

    npt.assert_allclose(msd, expected_msd)
    npt.assert_allclose(per_particle, expected_per_particle)


def test_compute_msd_without_time_origin_averaging(straight_line_positions):
    msd, per_particle = compute_msd(
        straight_line_positions,
        per_particle=True,
        average_time_origins=False,
    )

    expected = np.array([[1.0, 1.0], [4.0, 4.0], [9.0, 9.0]])
    npt.assert_allclose(per_particle, expected)
    npt.assert_allclose(msd, expected.mean(axis=1))


def test_compute_ngp_for_deterministic_motion(straight_line_positions):
    ngp = compute_ngp(straight_line_positions)
    expected = np.full(3, 3.0 / 5.0 - 1.0)
    npt.assert_allclose(ngp, expected)


def test_compute_debye_waller_factor_interpolates():
    times = np.array([1.0, 2.0, 4.0, 8.0])
    msd = times**2
    dwf = compute_debye_waller_factor(times, msd, tau_p=3.0)
    assert pytest.approx(dwf, rel=1e-7) == 10.0


def test_compute_debye_waller_factor_requires_monotonic_times():
    with pytest.raises(ValueError):
        compute_debye_waller_factor([0.0, 0.0, 1.0], [0.0, 1.0, 2.0])


def test_compute_van_hove_correlation_normalises_density(straight_line_positions):
    result = compute_van_hove_correlation(straight_line_positions, bins=20, rmax=5.0)
    assert isinstance(result, VanHoveResult)
    bin_width = result.radii[1] - result.radii[0]
    normalisation = np.sum(result.g_r_t * bin_width, axis=1)
    npt.assert_allclose(normalisation, np.ones_like(normalisation), rtol=1e-2)


def test_compute_van_hove_correlation_checks_time_length(straight_line_positions):
    with pytest.raises(ValueError):
        compute_van_hove_correlation(
            straight_line_positions,
            time_log=[0.0, 1.0],
        )


def test_get_k_vectors_returns_expected_magnitudes():
    box_length = 10.0
    ktarget = 2.0 * math.pi / box_length
    k_vectors = get_k_vectors(ktarget, box_length, max_points=50, seed=1)
    magnitudes = np.linalg.norm(k_vectors, axis=1)
    npt.assert_allclose(magnitudes, np.full_like(magnitudes, ktarget), atol=1e-12)


def test_compute_fskt_matches_batched_version(straight_line_positions):
    k_vectors = get_k_vectors(2.0 * math.pi / 10.0, 10.0, max_points=20, seed=0)
    fskt_full = compute_fskt(straight_line_positions, k_vectors, average_over_k=False)
    fskt_batched = compute_fskt_batched(
        straight_line_positions,
        k_vectors,
        batch_size=5,
        average_over_k=False,
    )
    npt.assert_allclose(fskt_full, fskt_batched)


def test_chi_squared_handles_broadcasting():
    observed = np.array([1.0, 2.0, 3.0])
    expected = np.array([1.0, 2.5, 2.5])
    scaling = 0.5
    value = chi_squared(observed, expected, scaling)
    assert pytest.approx(value) == ((observed - expected) ** 2 / scaling).sum()


def test_oneparam_fit_recovers_parameter():
    x = np.linspace(0.0, 1.0, 10)
    y = 2.5 * x
    parameter, quality = oneparam_fit(lambda arr, a: a * arr, x, y)
    assert pytest.approx(parameter, rel=1e-7) == 2.5
    assert 0.0 <= quality <= 1.0


def test_fit_msd_with_quality_control_returns_diffusion():
    t = np.linspace(1.0, 4.0, 4)
    diffusion = 0.5
    msd = 6.0 * diffusion * t
    msd_std = np.full_like(msd, 0.1)
    fitted, uncertainty = fit_msd_with_quality_control(t, msd, msd_std)
    assert pytest.approx(fitted, rel=1e-6) == diffusion
    assert uncertainty >= 0


def test_fit_line_with_fixed_slope():
    x = np.array([0.0, 1.0, 2.0])
    y = x + 2.0
    intercept = fit_line_with_fixed_slope(x, y)
    assert intercept == pytest.approx(2.0)


def test_compute_vacf_requires_non_zero_reference():
    velocities = np.zeros((3, 2, 3))
    with pytest.raises(ValueError):
        compute_vacf(velocities)


def test_compute_vacf_returns_expected_values(straight_line_positions):
    velocities = np.diff(straight_line_positions, axis=0, prepend=straight_line_positions[:1])
    velocities[0] = velocities[1]
    vacf = compute_vacf(velocities)
    expected = np.ones(velocities.shape[0] - 1)
    npt.assert_allclose(vacf, expected)


def test_compute_time_autocorrelation_matches_manual():
    values = np.array([1.0, 2.0, 3.0, 4.0])
    correlation = compute_time_autocorrelation(values)
    manual = np.correlate(values - values.mean(), values - values.mean(), mode="full")
    manual = manual[manual.size - values.size :] / manual.max()
    npt.assert_allclose(correlation, manual)


def test_compute_time_autocorrelation_requires_variance():
    with pytest.raises(ValueError):
        compute_time_autocorrelation(np.ones(4))
