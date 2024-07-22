import freud
import numpy as np
from MDAnalysis.analysis.rdf import InterRDF


def compute_rdf_pair(ag1, ag2, r_max=6.0, nbins=75):
    """
    Computes the averaged radial distribution function (RDF)
    between two groups of atoms. From MDAnalysis documentation:
    The RDF effectively counts the average number of ag2
    neighbours in a shell at distance r around a ag1 particle
    and represents it as a density.

    Parameters
    ----------
    ag1 : First AtomGroup 1, can be from u.select_atoms()
            ag1 = u.select_atoms('type 1')

    ag2 : Second AtomGroup, similar to ag1

    nbins : Number of bins in the histogram


    r_max : float
        Maximum distance to compute the RDF

    Returns
    -------
    rdf.bin_centers : np.ndarray
        bins of radial distribution function (# of bins is 200)
    rdf.rdf : np.ndarray
        radial distribution function

    r_min : float
        value of r for which g(r) attains its minimum
    """

    rdf_pair = InterRDF(ag1, ag2, range=(0, r_max), nbins=nbins)
    rdf_pair.run()

    rdf_vals = rdf_pair.results.rdf
    rdf_bin_centers = rdf_pair.results.bins
    max_ind = np.where(rdf_vals == max(rdf_vals))[0][0]
    index = np.where(rdf_vals == min(rdf_vals[max_ind:]))[0][0]

    r_min = rdf_bin_centers[index]
    r_peak = rdf_bin_centers[max_ind]

    return rdf_bin_centers, rdf_vals, r_min, r_peak


def compute_structure_factor_freud(positions, box, bins=100, kmax=8.0, kmin=0.1):
    """
    Computes the static structure factor for a given trajectory
    using the freud library (much faster than the previous implementation)

    Parameters
    ----------
    box : list
        freud box item ([Lx, Ly, Lz])
    positions : np.ndarray
        trajectory array with the shape (traj_length, N, 3)
    kmin : float, optional
        Starting point of S(k) grid. The default is 0.0.
    kmax : float, optional
        End point of S(k) grid. The default is 8.0.

    Returns
    -------
    np.ndarray
        Grid points of the static structure factor.
    np.ndarray
        The static structure factor.

    """

    wrapped_positions = box.wrap(positions)
    sf = freud.diffraction.StaticStructureFactorDirect(
        bins=bins, k_max=kmax, k_min=kmin
    )
    sf.compute((box, wrapped_positions))

    return sf.bin_centers, sf.S_k
