import freud
import numpy as np
from MDAnalysis.analysis.rdf import InterRDF
from MDAnalysis.analysis import contacts
from scipy.spatial import ConvexHull



def compute_rdf(positions, box, r_max=6.0, bins=50):
    """
    Computes the radial distribution function (RDF)
    at a given frame

    Parameters
    ----------
    positions : np.ndarray
        Positions of particles with the shape (N, 3)
    box : freud.box.Box
        freud box item ([Lx, Ly, Lz])
    r_max : float
        Maximum distance to compute the RDF
    Returns
    -------
    rdf.bin_centers : np.ndarray
        bins of radial distribution function (# of bins is 50)
    rdf.rdf : np.ndarray
        radial distribution function

    r_min : float
        value of r for which g(r) attains its minimum
    """
    system = (box, box.wrap(positions))
    rdf = freud.density.RDF(bins=bins, r_max=r_max)
    rdf.compute(system)
    max_ind = np.where(rdf.rdf == max(rdf.rdf))[0][0]
    index = np.where(rdf.rdf == min(rdf.rdf[max_ind:]))[0][0]
    r_min = rdf.bin_centers[index]
    r_peak = rdf.bin_centers[max_ind]
    
    return rdf.bin_centers, rdf.rdf, r_min, r_peak


def compute_rdf_pair_mda(ag1, ag2, r_max=6.0, nbins=75):
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


def compute_rdf_pair_freud(type1_positions, type2_positions, box, r_max=6.0, nbins=50):
    """
    
    Computes the radial distribution function (RDF) between two groups of particles
    using the freud library (much faster than the previous implementation)

    Parameters
    ----------
    type1_positions : np.ndarray
        Positions of group A particles with the shape (N, 3)
    type2_positions : np.ndarray
        Positions of group B particles with the shape (M, 3)
    box : freud.box.Box
        freud box item ([Lx, Ly, Lz])
    r_max : float, optional
        Maximum distance to compute the RDF. The default is 6.0.
    bins : int, optional
        Number of bins for the histogram. The default is 50.

    Returns
    -------
    rdf_bin_centres : np.ndarray
        bins of radial distribution function
    rdf_ab_vals : np.ndarray
        radial distribution function

    """ 
    # Wrap the positions within the box
    type1_wrapped = box.wrap(type1_positions)
    type2_wrapped = box.wrap(type2_positions)

    # Define the query objects
    query_type1 = freud.locality.AABBQuery(box, type1_wrapped)

    # Initialize the RDF calculator
    rdf = freud.density.RDF(bins=nbins, r_max=r_max)
    
    # Compute the RDF
    rdf.compute(system=query_type1, query_points=type2_wrapped, reset=False)

    # Retrieve the RDF data
    rdf_bin_centres = rdf.bin_centers
    rdf_12_vals = rdf.rdf
    
    max_ind = np.where(rdf_12_vals == max(rdf_12_vals))[0][0]
    index = np.where(rdf_12_vals == min(rdf_12_vals[max_ind:]))[0][0]
    r_min = rdf_bin_centres[index]
    r_peak = rdf_bin_centres[max_ind]

    return rdf_bin_centres, rdf_12_vals, r_min, r_peak


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


def contacts_within_cutoff(type1_positions, type2_positions, box, radius=2.5):
    '''
    Computes the number of contacts between two groups of particles
    within a given cutoff distance using the freud library
    
    Parameters
    ----------
    type1_positions : np.ndarray
        Positions of group A particles with the shape (N, 3)
    type2_positions : np.ndarray
        Positions of group B particles with the shape (M, 3)
    box : freud.box.Box
        freud box item ([Lx, Ly, Lz])
    radius : float, optional
        Cutoff distance for the contact calculation. The default is 2.5.

    Returns
    -------
    n_contacts : int
        Number of contacts between the two groups of particles
    '''

    type1_wrapped = box.wrap(type1_positions) 
    type2_wrapped = box.wrap(type2_positions)

    dist = contacts.distance_array(type1_wrapped, type2_wrapped)
    # determine which distances <= radius
    n_contacts = contacts.contact_matrix(dist, radius).sum()
    return n_contacts


def bounding_sphere(positions):
    """
    Computes the bounding sphere of a set of points in 3D space

    Parameters
    ----------
    positions : np.ndarray
        (N, 3) shaped array containing the positions of particles
    Returns
    -------
    radius : float
        Radius of the bounding sphere
    centre : np.ndarray
        (3,) shaped array containing the center of the bounding sphere
    """
    
    # Convert the list of tuples to a NumPy array
    positions = np.array(positions)
    
    # Compute the convex hull
    hull = ConvexHull(positions)
    
    # Get the vertices of the convex hull
    hull_points = positions[hull.vertices]
    
    # Calculate the centroid of the convex hull vertices
    centre = np.mean(hull_points, axis=0)
    
    # Calculate the radius as the maximum distance from the center to any convex hull vertex
    radius = np.max(np.linalg.norm(hull_points - centre, axis=1))
    
    return radius, centre