"""Utility functions for testing"""

import freud
import MDAnalysis as mda
import numpy as np
from pyprojroot.here import here


def setup_universe():
    """Set up mdanalysis universe before testing"""
    topology_file = here("tests/data/topo.data")
    trajectory_file = here("tests/data/traj.dat")
    u = mda.Universe(topology_file, trajectory_file, format="LAMMPSDUMP", dt=0.005)
    return u


def setup_freud():
    universe = setup_universe()
    box = freud.Box.cube(universe.dimensions[0])
    return box
