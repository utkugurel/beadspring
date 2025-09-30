"""Tests for validation helper utilities."""

from __future__ import annotations

import numpy as np
import pytest

from beadspring.analysis._validation import ensure_trajectory_array, ensure_wave_vectors


def test_ensure_trajectory_array_validates_shape():
    positions = np.zeros((5, 2, 3))
    validated = ensure_trajectory_array(positions)
    assert validated.shape == (5, 2, 3)
    assert validated.dtype == float


def test_ensure_trajectory_array_requires_three_dimensions():
    with pytest.raises(ValueError):
        ensure_trajectory_array(np.zeros((3, 3)))


@pytest.mark.parametrize("frames", [0, 1])
def test_ensure_trajectory_array_requires_minimum_frames(frames):
    with pytest.raises(ValueError):
        ensure_trajectory_array(np.zeros((frames, 2, 3)))


def test_ensure_wave_vectors_validates_shape():
    vectors = np.array([[0.0, 1.0, 2.0], [2.0, 0.0, 1.0]])
    validated = ensure_wave_vectors(vectors)
    assert validated.shape == (2, 3)


def test_ensure_wave_vectors_rejects_incorrect_shape():
    with pytest.raises(ValueError):
        ensure_wave_vectors(np.zeros((3, 3, 3)))
