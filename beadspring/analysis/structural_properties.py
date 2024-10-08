import freud
import numpy as np
from MDAnalysis.analysis.rdf import InterRDF
from MDAnalysis.analysis import contacts
from scipy.spatial import ConvexHull
from numpy.fft import fftn, fftshift


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
    using the freud library (much faster than the MDA implementation)

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


def compute_structure_factor_fourier(positions_type1, positions_type2, box_size, k_min, k_max, num_bins, grid_size=None, desired_dx=None):
    """
    Compute the static structure factor S(k) and partial structure factors using FFT methods.
    Performs binning and returns binned k-values and S(k) values suitable for plotting.

    Parameters:
    positions_type1: numpy array of shape (N1, 3) - positions of particles of type 1
    positions_type2: numpy array of shape (N2, 3) - positions of particles of type 2
    box_size: float - size of the simulation box (assumed cubic)
    k_min: float - minimum k magnitude to compute
    k_max: float - maximum k magnitude to compute
    num_bins: int - number of bins for k
    grid_size: int (optional) - number of grid points along each axis (overrides desired_dx if provided)
    desired_dx: float (optional) - desired grid spacing in real space (used to calculate grid_size if grid_size not provided)

    Returns:
    k_bin_centers: numpy array - bin centers for k
    S_total_binned: numpy array - binned total static structure factor values
    S_AA_binned: numpy array - binned partial structure factor S_AA(k)
    S_AB_binned: numpy array - binned partial structure factor S_AB(k)
    S_BB_binned: numpy array - binned partial structure factor S_BB(k)
    """
    # Determine grid_size
    if grid_size is not None:
        grid_size = int(grid_size)
    elif desired_dx is not None:
        grid_size = int(np.ceil(box_size / desired_dx))
        # Ensure grid_size is a power of 2 for efficient FFT
        grid_size = 2 ** int(np.ceil(np.log2(grid_size)))
    else:
        # Default grid_size based on box_size
        default_grid_points = 128  # Default value
        grid_size = default_grid_points

    # Ensure grid_size is at least 16 to avoid too small grids
    grid_size = max(grid_size, 16)

    # Number of particles in each type
    N1 = positions_type1.shape[0]
    N2 = positions_type2.shape[0]
    N = N1 + N2  # Total number of particles

    # Initialize density fields for each particle type
    rho1 = np.zeros((grid_size, grid_size, grid_size), dtype=complex)
    rho2 = np.zeros_like(rho1)

    # Map positions to grid indices for type 1 particles
    grid_indices1 = np.mod(positions_type1 / box_size * grid_size, grid_size).astype(int)
    for idx in grid_indices1:
        rho1[idx[0], idx[1], idx[2]] += 1.0

    # Map positions to grid indices for type 2 particles
    grid_indices2 = np.mod(positions_type2 / box_size * grid_size, grid_size).astype(int)
    for idx in grid_indices2:
        rho2[idx[0], idx[1], idx[2]] += 1.0

    # Compute Fourier transforms of the density fields
    rho1_k = fftshift(fftn(rho1))
    rho2_k = fftshift(fftn(rho2))

    # Compute partial structure factors
    S_AA = (np.abs(rho1_k) ** 2) / N1
    S_BB = (np.abs(rho2_k) ** 2) / N2
    S_AB = (rho1_k * np.conj(rho2_k)) / np.sqrt(N1 * N2)
    S_AB = np.real(S_AB)  # Take the real part

    # Compute the total structure factor
    S_total = (N1 / N) * S_AA + (N2 / N) * S_BB + (2 / N) * S_AB

    # Generate k-vector components
    dk = 2 * np.pi / box_size
    k_values = dk * np.fft.fftfreq(grid_size, d=1.0 / grid_size)
    kx = fftshift(k_values)
    ky = fftshift(k_values)
    kz = fftshift(k_values)

    # Create a grid of k-vector magnitudes
    kx_grid, ky_grid, kz_grid = np.meshgrid(kx, ky, kz, indexing='ij')
    k_magnitude = np.sqrt(kx_grid**2 + ky_grid**2 + kz_grid**2)

    # Flatten the arrays
    k_values_flat = k_magnitude.flatten()
    S_total_flat = S_total.flatten()
    S_AA_flat = S_AA.flatten()
    S_AB_flat = S_AB.flatten()
    S_BB_flat = S_BB.flatten()

    # Filter k-values within k_min and k_max
    k_filter = (k_values_flat >= k_min) & (k_values_flat <= k_max)
    k_values_filtered = k_values_flat[k_filter]
    S_total_filtered = S_total_flat[k_filter]
    S_AA_filtered = S_AA_flat[k_filter]
    S_AB_filtered = S_AB_flat[k_filter]
    S_BB_filtered = S_BB_flat[k_filter]

    # Bin the data
    bins = np.linspace(k_min, k_max, num_bins + 1)
    bin_indices = np.digitize(k_values_filtered, bins) - 1  # Adjust indices

    # Initialize arrays for binned data
    S_total_binned = np.zeros(num_bins)
    S_AA_binned = np.zeros(num_bins)
    S_AB_binned = np.zeros(num_bins)
    S_BB_binned = np.zeros(num_bins)
    counts = np.zeros(num_bins)

    for i in range(num_bins):
        mask = bin_indices == i
        counts[i] = np.sum(mask)
        if counts[i] > 0:
            S_total_binned[i] = np.mean(S_total_filtered[mask])
            S_AA_binned[i] = np.mean(S_AA_filtered[mask])
            S_AB_binned[i] = np.mean(S_AB_filtered[mask])
            S_BB_binned[i] = np.mean(S_BB_filtered[mask])

    # Compute bin centers
    k_bin_centers = 0.5 * (bins[:-1] + bins[1:])

    return k_bin_centers, S_total_binned, S_AA_binned, S_AB_binned, S_BB_binned


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