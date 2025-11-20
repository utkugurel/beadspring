import numpy as np
import pytest
import sys
from beadspring.analysis import structural_properties as sp

class MockBox:
    def __init__(self, Lx, Ly, Lz):
        self.Lx = Lx
        self.Ly = Ly
        self.Lz = Lz
    
    def wrap(self, positions):
        return positions - np.floor(positions / self.L + 0.5) * self.L
    
    # Add freud-like properties if needed
    @property
    def L(self):
        return np.array([self.Lx, self.Ly, self.Lz])

def test_bounding_sphere():
    # Points on unit sphere
    positions = np.array([
        [1, 0, 0], [-1, 0, 0],
        [0, 1, 0], [0, -1, 0],
        [0, 0, 1], [0, 0, -1]
    ], dtype=float)
    
    radius, center = sp.bounding_sphere(positions)
    
    assert np.isclose(radius, 1.0)
    assert np.allclose(center, 0.0)

def test_compute_rdf_simple():
    # Two particles at distance 2.0
    positions = np.array([[0, 0, 0], [2, 0, 0]], dtype=float)
    
    if 'freud' in sys.modules:
        import freud
        box = freud.box.Box.cube(10)
    else:
        box = MockBox(10, 10, 10)
    
    # compute_rdf returns (bin_centers, rdf, r_min, r_peak)
    bin_centers, rdf_values, r_min, r_peak = sp.compute_rdf(positions, box, r_max=3.0, bins=10)
    
    assert len(bin_centers) == 10
    assert len(rdf_values) == 10
    
    peak_idx = np.argmax(rdf_values)
    peak_r = bin_centers[peak_idx]
    assert np.isclose(peak_r, 2.0, atol=0.3)

def test_compute_structure_factor_fourier():
    # Random positions
    N = 100
    positions = np.random.rand(N, 3) * 10
    box_size = 10.0
    
    k_min = 0.1
    k_max = 2.0
    num_bins = 10
    
    # The error was "ValueError: too many values to unpack (expected 2)"
    # Returns 5 values: k_bin_centers, S_total_binned, S_AA_binned, S_AB_binned, S_BB_binned
    k_centers, s_total, s_aa, s_ab, s_bb = sp.compute_structure_factor_fourier(
        positions, positions, box_size, k_min, k_max, num_bins
    )
    
    assert len(k_centers) == num_bins
    assert len(s_total) == num_bins
    assert np.all(s_total >= 0)

def test_contacts_within_cutoff():
    # Two groups of particles
    pos1 = np.array([[0, 0, 0]])
    pos2 = np.array([[1.0, 0, 0], [5.0, 0, 0]])
    
    # Same issue with box.wrap likely
    try:
        import freud
        box = freud.box.Box.cube(10)
    except ImportError:
        box = MockBox(10, 10, 10)
        pytest.skip("freud not installed")
        
    n_contacts = sp.contacts_within_cutoff(pos1, pos2, box, radius=2.0)
    assert n_contacts == 1
