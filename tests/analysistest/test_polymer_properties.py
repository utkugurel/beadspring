"""Tests for the beadspring.analysis.polymer_properties module."""

import MDAnalysis as mda
import numpy as np
import numpy.testing as npt
import pytest
from pyprojroot.here import here
from unittest.mock import MagicMock


from tests.testing_utils import setup_universe

universe = setup_universe()


@pytest.fixture
def mock_universe():
    """Fixture to create a mock MDAnalysis universe."""
    mock_universe = MagicMock()

    # Mock bond positions
    mock_universe.atoms.bonds.atom1.positions = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
    ])
    mock_universe.atoms.bonds.atom2.positions = np.array([
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
    ])
    return mock_universe


def select_linear_chain(u):
    """Select a single polymer as an atom group"""
    chain = u.select_atoms("type 4").residues[0].atoms
    return chain


def test_compute_gyration_tensor(u=universe):
    """Test the values of gyration tensor"""
    from beadspring.analysis.polymer_properties import compute_gyration_tensor

    chain = select_linear_chain(u)
    positions = chain.positions
    expected_tensor = np.array(
        [
            [7.023423, 1.217949, 1.8205513],
            [1.217949, 7.399779, -4.4355016],
            [1.8205513, -4.4355016, 5.156402],
        ],
        dtype=np.float32,
    )

    expected_evals = np.array([0.92054224, 7.7983627, 10.860699], dtype=np.float32)
    tensor, evals = compute_gyration_tensor(positions)
    npt.assert_allclose(tensor, expected_tensor, rtol=1e-5)
    npt.assert_allclose(evals, expected_evals, rtol=1e-5)


def test_calculate_asphericity():
    from beadspring.analysis.polymer_properties import calculate_asphericity

    lmin, lmid, lmax = np.array([0.92054224, 7.7983627, 10.860699], dtype=np.float32)

    expected_asphericity = lmax - 0.5 * (lmin + lmid)
    asphericity = calculate_asphericity(lmin, lmid, lmax)
    npt.assert_allclose(asphericity, expected_asphericity, rtol=1e-5)


def test_calculate_acylindricity():
    from beadspring.analysis.polymer_properties import calculate_acylindricity

    lmin, lmid, lmax = np.array([0.92054224, 7.7983627, 10.860699], dtype=np.float32)

    expected_acylindricity = lmid - lmin
    acylindricity = calculate_acylindricity(lmin, lmid, lmax)
    npt.assert_allclose(acylindricity, expected_acylindricity, rtol=1e-5)


def test_calculate_rg2():
    from beadspring.analysis.polymer_properties import calculate_rg2

    lmin, lmid, lmax = np.array([0.92054224, 7.7983627, 10.860699], dtype=np.float32)

    expected_rg2 = lmin + lmid + lmax
    rg2 = calculate_rg2(lmin, lmid, lmax)
    npt.assert_allclose(rg2, expected_rg2, rtol=1e-5)


def test_calculate_shape_anisotropy():
    from beadspring.analysis.polymer_properties import (
        calculate_acylindricity, calculate_asphericity, calculate_rg2,
        calculate_shape_anisotropy)

    lmin, lmid, lmax = np.array([0.92054224, 7.7983627, 10.860699], dtype=np.float32)

    num = (
        calculate_asphericity(lmin, lmid, lmax) ** 2
        + 0.75 * calculate_acylindricity(lmin, lmid, lmax=0.0) ** 2
    )
    denum = calculate_rg2(lmin, lmid, lmax) ** 2

    expected_shape_anisotropy = num / denum
    shape_anisotropy = calculate_shape_anisotropy(lmin, lmid, lmax)
    npt.assert_allclose(shape_anisotropy, expected_shape_anisotropy, rtol=1e-5)


def test_calculate_prolateness():
    from beadspring.analysis.polymer_properties import calculate_prolateness

    lmin, lmid, lmax = np.array([0.92054224, 7.7983627, 10.860699], dtype=np.float32)

    n1 = 2 * np.sqrt(lmin) - np.sqrt(lmid) - np.sqrt(lmax)
    n2 = 2 * np.sqrt(lmid) - np.sqrt(lmin) - np.sqrt(lmax)
    n3 = 2 * np.sqrt(lmax) - np.sqrt(lmin) - np.sqrt(lmid)

    d1 = 2 * (lmin + lmid + lmax)
    d2 = 2 * np.sqrt(lmin) * np.sqrt(lmid)
    d3 = 2 * np.sqrt(lmid) * np.sqrt(lmax)
    d4 = 2 * np.sqrt(lmin) * np.sqrt(lmax)

    expected_prolateness = (n1 * n2 * n3) / (d1 - d2 - d3 - d4)
    rg2 = calculate_prolateness(lmin, lmid, lmax)
    npt.assert_allclose(rg2, expected_prolateness, rtol=1e-5)


def test_identify_end_to_end_vector():
    from beadspring.analysis.polymer_properties import \
        identify_end_to_end_vector

    chain = select_linear_chain(universe)
    positions = chain.positions
    expected_end_to_end_vector = positions[-1] - positions[0]
    expected_end_to_end_vector = expected_end_to_end_vector.reshape(1, 3)
    end_to_end_vector = identify_end_to_end_vector([chain])
    npt.assert_allclose(end_to_end_vector, expected_end_to_end_vector, rtol=1e-5)


def test_calculate_end_to_end_correlation():
    from beadspring.analysis.polymer_properties import \
        calculate_end_to_end_correlation

    N_FRAMES = universe.trajectory.n_frames
    N_CHAINS = universe.select_atoms("type 4").n_residues
    end_to_end_vector = np.zeros((N_FRAMES, N_CHAINS, 3))
    expected_correlation = np.zeros(N_FRAMES)
    linear_chains = universe.select_atoms("type 4").residues
    for i, ts in enumerate(universe.trajectory):
        for j, elem in enumerate(linear_chains):
            positions = elem.atoms.positions
            end_to_end_vector[i, j] = positions[-1] - positions[0]

    for i in range(N_FRAMES):
        denominator = np.array(
            [
                np.inner(end_to_end_vector[0, j], end_to_end_vector[0, j])
                for j in range(N_CHAINS)
            ]
        )
        inner_products = np.array(
            [
                np.dot(end_to_end_vector[0, j], end_to_end_vector[i, j])
                for j in range(N_CHAINS)
            ]
        )
        expected_correlation[i] = np.mean(inner_products / denominator)

    # expected_correlation = np.dot(end_to_end_vector.T, end_to_end_vector)
    correlation = calculate_end_to_end_correlation(end_to_end_vector)
    npt.assert_allclose(correlation, expected_correlation, rtol=1e-5)
    #TODO: This testing is based on another implementation of the same function.
    # We should test the other function too.


def test_compute_p2_from_vectors():
    """Test the compute_p2_from_vectors function."""
    from beadspring.analysis.polymer_properties import compute_p2_from_vectors

    # Case 1: Perfect alignment with reference axis
    bond_vectors = np.array([
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ])
    reference_axis = np.array([1, 0, 0])
    expected_p2 = 1.0  # Perfect alignment
    p2 = compute_p2_from_vectors(bond_vectors, reference_axis)
    npt.assert_allclose(p2, expected_p2, rtol=1e-5)

    # Case 2: Orthogonal to reference axis
    bond_vectors = np.array([
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    expected_p2 = -0.5  # Perfect orthogonal
    p2 = compute_p2_from_vectors(bond_vectors, reference_axis)
    npt.assert_allclose(p2, expected_p2, rtol=1e-5)

    # Case 3: Random bond orientations
    bond_vectors = np.random.rand(10, 3)
    p2 = compute_p2_from_vectors(bond_vectors, reference_axis)
    assert -0.5 <= p2 <= 1, "P2 should be in the range [-0.5, 1]"

    # Case 4: Single bond
    bond_vectors = np.array([
        [1.0, 1.0, 1.0],
    ])
    p2 = compute_p2_from_vectors(bond_vectors, reference_axis)
    assert -0.5 <= p2 <= 1, "P2 for a single bond should also be in the range [-0.5, 1]"


def test_compute_p2(mock_universe):
    """Test the compute_p2 function using a mock universe."""
    from beadspring.analysis.polymer_properties import compute_p2

    # Mock bond positions
    mock_universe.atoms.bonds.atom1.positions = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
    ])
    mock_universe.atoms.bonds.atom2.positions = np.array([
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
    ])

    # Compute P2 for perfect alignment
    reference_axis = np.array([1, 0, 0])
    expected_p2 = 1.0
    p2 = compute_p2(mock_universe, reference_axis=reference_axis)
    npt.assert_allclose(p2, expected_p2, rtol=1e-5)
