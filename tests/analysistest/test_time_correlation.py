"""Tests for :mod:`beadspring.analysis.time_correlation`."""

import numpy as np
import numpy.testing as npt
import pytest

from beadspring.analysis import time_correlation as tc


def test_time_autocorrelation_normalisation():
    data = np.array(
        [
            [1.0, 0.0],
            [0.5, 0.5],
            [0.0, 1.0],
        ]
    )
    correlation = tc.time_autocorrelation(data)
    assert correlation.shape == (3,)
    assert pytest.approx(1.0) == correlation[0]
    assert np.all(correlation[1:] <= 1.0)


def test_time_autocorrelation_zero_variance():
    data = np.ones((4, 3))
    correlation = tc.time_autocorrelation(data)
    npt.assert_array_equal(correlation, np.zeros(4))


def test_time_cross_correlation_symmetry():
    data_a = np.linspace(0.0, 1.0, 5)[:, None]
    data_b = np.linspace(1.0, 0.0, 5)[:, None]
    corr_ab = tc.time_cross_correlation(data_a, data_b)
    corr_ba = tc.time_cross_correlation(data_b, data_a)
    npt.assert_allclose(corr_ab, corr_ba)
    assert pytest.approx(-1.0) == corr_ab[0]


def test_time_cross_correlation_shape_mismatch():
    with pytest.raises(ValueError):
        tc.time_cross_correlation(np.ones((3, 2)), np.ones((4, 2)))
