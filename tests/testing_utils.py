"""Utility helpers shared across the test suite."""

from pathlib import Path

import freud
import MDAnalysis as mda


DATA_DIR = Path(__file__).resolve().parent / "data"


def setup_universe():
    """Return an :class:`MDAnalysis.Universe` configured for the test data."""

    topology_file = DATA_DIR / "topo.data"
    trajectory_file = DATA_DIR / "traj.dat"
    return mda.Universe(topology_file, trajectory_file, format="LAMMPSDUMP", dt=0.005)


def setup_freud():
    """Construct a cubic :class:`freud.box.Box` based on the test universe."""

    universe = setup_universe()
    return freud.Box.cube(universe.dimensions[0])
