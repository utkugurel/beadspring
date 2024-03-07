"""Tests for the beadspring.analysis.polymer_properties module."""
import pytest
import MDAnalysis as mda
import numpy as np
import numpy.testing as npt
from pyprojroot.here import here
from tests.testing_utils import setup_universe


universe = setup_universe()


def select_single_polymer_chain(u):
    """Select a single polymer as an atom group"""
    chain = u.select_atoms("type 1").residues[0].atoms
    return chain


def test_compute_gyration_tensor(u=universe):
    """Test the values of gyration tensor
    """
    from beadspring.analysis.polymer_properties import compute_gyration_tensor
    chain = select_single_polymer_chain(u)
    positions = chain.positions
    expected_tensor = np.array(
        [[37.116905,  2.8693573,  2.8826792],
       [2.8693573, 49.027607, -3.3597746],
       [2.8826792, -3.3597746, 35.663506]], dtype=np.float32)

    expected_evals = np.array(
        [32.255394, 39.36003 , 50.19259 ], dtype=np.float32)
    tensor, evals = compute_gyration_tensor(positions)
    npt.assert_allclose(tensor, expected_tensor, rtol=1e-5)
    npt.assert_allclose(evals, expected_evals, rtol=1e-5)


def test_calculate_asphericity():
    from beadspring.analysis.polymer_properties import calculate_asphericity

    lmin, lmid, lmax = np.array(
        [32.255394, 39.36003 , 50.19259 ], dtype=np.float32)
    
    expected_asphericity = lmax - 0.5*(lmin + lmid)
    asphericity = calculate_asphericity(lmin, lmid, lmax)
    npt.assert_allclose(asphericity, expected_asphericity, rtol=1e-5)


def test_calculate_acylindricity():
    from beadspring.analysis.polymer_properties import calculate_acylindricity

    lmin, lmid, lmax = np.array(
        [32.255394, 39.36003 , 50.19259 ], dtype=np.float32)
    
    expected_acylindricity = lmid - lmin
    acylindricity = calculate_acylindricity(lmin, lmid, lmax)
    npt.assert_allclose(acylindricity, expected_acylindricity, rtol=1e-5)


def test_calculate_rg2():
    from beadspring.analysis.polymer_properties import calculate_rg2

    lmin, lmid, lmax = np.array(
        [32.255394, 39.36003 , 50.19259 ], dtype=np.float32)
    
    expected_rg2 = lmin + lmid + lmax
    rg2 = calculate_rg2(lmin, lmid, lmax)
    npt.assert_allclose(rg2, expected_rg2, rtol=1e-5)


def test_calculate_shape_anisotropy():
    from beadspring.analysis.polymer_properties import calculate_acylindricity
    from beadspring.analysis.polymer_properties import calculate_asphericity
    from beadspring.analysis.polymer_properties import calculate_rg2
    from beadspring.analysis.polymer_properties import calculate_shape_anisotropy

    lmin, lmid, lmax = np.array(
        [32.255394, 39.36003 , 50.19259 ], dtype=np.float32)
    
    num = calculate_asphericity(lmin, lmid, lmax)**2 + 0.75*calculate_acylindricity(lmin, lmid, lmax=0.)**2
    denum = calculate_rg2(lmin, lmid, lmax)**2

    expected_shape_anisotropy = num / denum
    shape_anisotropy = calculate_shape_anisotropy(lmin, lmid, lmax)
    npt.assert_allclose(shape_anisotropy, expected_shape_anisotropy, rtol=1e-5)


def test_calculate_prolateness():
    from beadspring.analysis.polymer_properties import calculate_prolateness

    lmin, lmid, lmax = np.array(
        [32.255394, 39.36003 , 50.19259 ], dtype=np.float32)
    
    n1 = 2*np.sqrt(lmin) - np.sqrt(lmid) - np.sqrt(lmax)
    n2 = 2*np.sqrt(lmid) - np.sqrt(lmin) - np.sqrt(lmax)
    n3 = 2*np.sqrt(lmax) - np.sqrt(lmin) - np.sqrt(lmid)

    d1 = 2*(lmin + lmid + lmax) 
    d2 = 2*np.sqrt(lmin) * np.sqrt(lmid)
    d3 = 2*np.sqrt(lmid) * np.sqrt(lmax)
    d4 = 2*np.sqrt(lmin) * np.sqrt(lmax)

    expected_prolateness = (n1 * n2 * n3) / (d1 - d2 - d3 - d4)
    rg2 = calculate_prolateness(lmin, lmid, lmax)
    npt.assert_allclose(rg2, expected_prolateness, rtol=1e-5)


def test_identify_end_to_end_vector():
    from beadspring.analysis.polymer_properties import identify_end_to_end_vector

    chain = select_single_polymer_chain(universe)
    positions = chain.positions
    expected_end_to_end_vector = positions[-1] - positions[0]
    expected_end_to_end_vector = expected_end_to_end_vector.reshape(1, 3)
    end_to_end_vector = identify_end_to_end_vector([chain])
    npt.assert_allclose(end_to_end_vector, expected_end_to_end_vector, rtol=1e-5)

#TODO: Add test for calculate_end_to_end_correlation
#def test_calculate_end_to_end_correlation():
#    from beadspring.analysis.polymer_properties import calculate_end_to_end_correlation

#TODO: Add test for calculate_end_to_end_correlation_optimised
#def test_calculate_end_to_end_correlation_optimised():
#    from beadspring.analysis.polymer_properties import calculate_end_to_end_correlation_optimised    


