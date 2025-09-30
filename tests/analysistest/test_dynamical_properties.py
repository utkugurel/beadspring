"""Unit tests for :mod:`beadspring.analysis.dynamical_properties`."""

import numpy as np
import numpy.testing as npt
import pytest

from beadspring.analysis import dynamical_properties as dyn


@pytest.fixture
def sample_positions():
    """Simple two-particle trajectory with linear motion."""

    return np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            [[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
        ]
    )


def test_compute_msd(sample_positions):
    msd, per_particle = dyn.compute_msd(sample_positions, per_particle=True)
    npt.assert_allclose(msd, np.array([1.0, 4.0]))
    expected_pp = np.array([[1.0, 1.0], [4.0, 4.0]])
    npt.assert_allclose(per_particle, expected_pp)


def test_compute_ngp(sample_positions):
    ngp = dyn.compute_ngp(sample_positions)
    npt.assert_allclose(ngp, np.full(2, -0.4))


def test_compute_debye_waller_factor():
    times = np.array([1.0, 2.0, 3.0])
    msd = np.array([0.5, 1.0, 1.5])
    value = dyn.compute_debye_waller_factor(times, msd, tau_p=2.5)
    assert pytest.approx(1.25) == value

    with pytest.raises(ValueError):
        dyn.compute_debye_waller_factor(times, msd, tau_p=0.0)


def test_compute_van_hove_correlation(sample_positions):
    radial_grid, histograms = dyn.compute_van_hove_correlation(sample_positions, bins=5, rmax=4.0)
    assert radial_grid.shape == (5,)
    assert histograms.shape == (2, 5)

    for histogram in histograms:
        integral = np.trapz(histogram, radial_grid)
        assert pytest.approx(1.0, rel=1e-3) == integral


def test_get_k_vectors_reproducible():
    k_vectors = dyn.get_k_vectors(ktarget=4.0, box_length=10.0, max_points=20)
    assert k_vectors.shape[1] == 3
    magnitudes = np.linalg.norm(k_vectors, axis=1)
    assert np.allclose(magnitudes, magnitudes.mean(), atol=0.5)


def test_compute_fskt_matches_batched(sample_positions):
    k_vectors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    direct = dyn.compute_fskt(sample_positions, k_vectors)
    batched = dyn.compute_fskt_batched(sample_positions, k_vectors, batch_size=1)
    npt.assert_allclose(direct, batched)


def test_chi_squared():
    observed = np.array([1.0, 2.0, 3.0])
    expected = np.array([1.0, 2.5, 2.5])
    scaling = np.ones(3)
    assert pytest.approx(0.5) == dyn.chi_squared(observed, expected, scaling)


def test_oneparam_fit_linear():
    def model(x, a):
        return a * x

    x = np.linspace(0.1, 1.0, 10)
    y = 2.0 * x
    parameter, quality = dyn.oneparam_fit(model, x, y)
    assert pytest.approx(2.0, rel=1e-6) == parameter
    assert 0.0 <= quality <= 1.0


def test_fit_line_with_fixed_slope():
    x = np.array([0.0, 1.0, 2.0])
    y = x + 2.5
    intercept = dyn.fit_line_with_fixed_slope(x, y)
    assert pytest.approx(2.5) == intercept


def test_compute_vacf():
    velocities = np.array(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.5, 0.0, 0.0], [0.0, 0.5, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ]
    )
    vacf = dyn.compute_vacf(velocities)
    assert vacf.shape == (2,)
    assert vacf[0] < 1.0
    assert vacf[1] <= vacf[0]
