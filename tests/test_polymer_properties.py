import numpy as np
import pytest
from beadspring.analysis import polymer_properties as pp

def test_gyration_tensor_sphere():
    # Create points on a sphere
    # For a perfect sphere distribution, eigenvalues should be equal
    # Let's use a simple symmetric set of points
    positions = np.array([
        [1, 0, 0], [-1, 0, 0],
        [0, 1, 0], [0, -1, 0],
        [0, 0, 1], [0, 0, -1]
    ], dtype=float)
    
    tensor, eigenvalues = pp.compute_gyration_tensor(positions)
    
    # Check eigenvalues are equal
    assert np.allclose(eigenvalues, eigenvalues[0])
    
    # Check shape descriptors
    b = pp.calculate_asphericity(*eigenvalues)
    c = pp.calculate_acylindricity(*eigenvalues)
    k2 = pp.calculate_shape_anisotropy(*eigenvalues)
    
    assert np.isclose(b, 0.0)
    assert np.isclose(c, 0.0)
    assert np.isclose(k2, 0.0)

def test_gyration_tensor_linear():
    # Linear chain along x
    N = 10
    positions = np.zeros((N, 3))
    positions[:, 0] = np.arange(N)
    
    tensor, eigenvalues = pp.compute_gyration_tensor(positions)
    lmin, lmid, lmax = eigenvalues
    
    # Should be 0 for y and z (numerical noise aside)
    assert np.isclose(lmin, 0.0)
    assert np.isclose(lmid, 0.0)
    assert lmax > 0.0
    
    # Check shape descriptors for rod
    # k2 should be 1 for ideal linear chain (infinite thin rod)
    k2 = pp.calculate_shape_anisotropy(*eigenvalues)
    assert np.isclose(k2, 1.0)

def test_persistence_length_straight():
    # Straight chain
    bond_vectors = np.zeros((10, 3))
    bond_vectors[:, 0] = 1.0
    
    # Should be infinite, but our simple linear fit might give weird results if slope is 0
    # The code does -1/slope. If slope is 0 (log(1)=0), lp -> inf
    # Let's see how it handles it.
    # Actually, log(1) is 0, so y_fit is all 0. slope is 0. lp is inf.
    
    with pytest.warns(RuntimeWarning, match="divide by zero"):
        lp = pp.calculate_persistence_length(bond_vectors)
    
    assert np.isinf(lp)

def test_persistence_length_random():
    # Random chain
    rng = np.random.default_rng(42)
    bond_vectors = rng.standard_normal((100, 3))
    
    lp = pp.calculate_persistence_length(bond_vectors)
    # Should be small, close to 0 or 1 order of magnitude
    assert lp < 100.0
    assert lp > 0.0

def test_rouse_modes_single_frame():
    # Single frame
    positions = np.zeros((10, 3))
    positions[:, 0] = np.arange(10)
    
    modes = pp.compute_rouse_modes(positions)
    assert modes.shape == (10, 3)
    
    # Mode 0 should be related to CM
    # X_0 = sqrt(2/N) * sum(R_n) = sqrt(2/N) * N * R_cm = sqrt(2N) * R_cm
    # R_cm = (4.5, 0, 0)
    # X_0 = sqrt(20) * 4.5 approx 4.47 * 4.5 = 20.1
    
    expected_x0_x = np.sqrt(20) * 4.5
    assert np.isclose(modes[0, 0], expected_x0_x)

def test_rouse_modes_trajectory():
    # Trajectory
    n_frames = 5
    n_monomers = 10
    positions = np.zeros((n_frames, n_monomers, 3))
    
    modes = pp.compute_rouse_modes(positions)
    assert modes.shape == (n_frames, n_monomers, 3)
    assert np.allclose(modes, 0.0)

def test_bond_lengths():
    # Mock AtomGroup
    class MockAtomGroup:
        class Bonds:
            class Atom:
                def __init__(self, pos):
                    self.positions = pos
            def __init__(self, p1, p2):
                self.atom1 = self.Atom(p1)
                self.atom2 = self.Atom(p2)
        def __init__(self, p1, p2):
            self.bonds = self.Bonds(p1, p2)
            
    p1 = np.array([[0, 0, 0], [1, 0, 0]])
    p2 = np.array([[1, 0, 0], [1, 1, 0]])
    ag = MockAtomGroup(p1, p2)
    
    lengths = pp.compute_bond_lengths(ag)
    assert np.allclose(lengths, 1.0)
