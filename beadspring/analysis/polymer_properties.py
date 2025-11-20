"""Polymer specific structural descriptors used throughout the project."""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist, pdist

__all__ = [
    "compute_gyration_tensor",
    "calculate_asphericity",
    "calculate_acylindricity",
    "calculate_rg2",
    "calculate_hydrodynamic_radius",
    "calculate_shape_anisotropy",
    "calculate_prolateness",
    "identify_end_to_end_vector",
    "calculate_end_to_end_correlation",
    "calculate_end_to_end_correlation_optimised",
    "compute_bond_lengths",
    "compute_p2_from_vectors",
    "compute_p2",
    "calculate_persistence_length",
    "compute_rouse_modes",
]


def compute_gyration_tensor(positions):
    """
    Calculates the gyration tensor for a given set of positions
    in a single time frame

    Parameters
    ----------
    positions : np.ndarray
        (N, 3) shaped array containing the positions of particles

    Returns
    -------
    gyration tensor :  np.ndarray
        (3, 3) shaped array of gyration tensor

    eigenvalues : np.ndarray
        array of length 3 sorted in the ascending order

    """

    diff = positions - positions.mean(axis=0)[np.newaxis, :]
    gyration_tensor = np.einsum("ij,ik->jk", diff, diff) / len(positions)
    eigenvalues = np.sort(np.linalg.eigvals(gyration_tensor))

    return gyration_tensor, eigenvalues


def calculate_asphericity(lmin, lmid, lmax):
    """
    Parameters
    ----------
    lmin, lmid, lmax : float
        Eigenvalues of the gyration tensor
    Returns
    -------
    b : float
        Asphericity parameter for polymer chains. b=0
        corresponds to a perfect sphere
    """
    b = lmax - 0.5 * (lmin + lmid)
    return b


def calculate_acylindricity(lmin, lmid, lmax=0.0):
    """
    Parameters
    ----------
    lmin, lmid, lmax : float
        Eigenvalues of the gyration tensor
    Returns
    -------
    c : float
        Acylindricity parameter for polymer chains. c=0
        corresponds to a perfect cylinder.
    """
    c = lmid - lmin
    return c


def calculate_rg2(lmin, lmid, lmax):
    """
    Parameters
    ----------
    lmin, lmid, lmax : float
        Eigenvalues of the gyration tensor
    Returns
    -------
    rg2 : float
        Radius of gyration squared for polymer chains.
    """

    rg2 = lmin + lmid + lmax
    return rg2


def calculate_hydrodynamic_radius(positions):
    """
    Parameters
    ----------
    positions : np.ndarray
        (N, 3) shaped array containing the positions of particles
    Returns
    -------
    hydrodynamic_radius : float
        Hydrodynamic radius of the polymer chain
    """
    N = len(pdist(positions))
    inv_dist = (1 / pdist(positions)).sum()
    hydrodynamic_radius = 1 / (inv_dist / N)

    return hydrodynamic_radius


def calculate_shape_anisotropy(lmin, lmid, lmax):
    """
    Parameters
    ----------
    lmin, lmid, lmax : float
        Eigenvalues of the gyration tensor
    Returns
    -------
    k2 : float
        Relative shape anisotropy parameter for polymer chains.
        k2=1 for an ideal linear chain, k2=0 for highly symmetric
        conformations. 0 < k2 < 1
    """

    num = (
        calculate_asphericity(lmin, lmid, lmax) ** 2
        + 0.75 * calculate_acylindricity(lmin, lmid, lmax=0.0) ** 2
    )
    denum = calculate_rg2(lmin, lmid, lmax) ** 2

    k2 = num / denum

    return k2


def calculate_prolateness(lmin, lmid, lmax):
    """
    Parameters
    ----------
    lmin, lmid, lmax : float
        Eigenvalues of the gyration tensor
    Returns
    -------
    p : float
        p=-1 for perfectly oblate shape and p=1 for perfectly prolate shape

    """
    n1 = 2 * np.sqrt(lmin) - np.sqrt(lmid) - np.sqrt(lmax)
    n2 = 2 * np.sqrt(lmid) - np.sqrt(lmin) - np.sqrt(lmax)
    n3 = 2 * np.sqrt(lmax) - np.sqrt(lmin) - np.sqrt(lmid)

    d1 = 2 * (lmin + lmid + lmax)
    d2 = 2 * np.sqrt(lmin) * np.sqrt(lmid)
    d3 = 2 * np.sqrt(lmid) * np.sqrt(lmax)
    d4 = 2 * np.sqrt(lmin) * np.sqrt(lmax)

    p = (n1 * n2 * n3) / (d1 - d2 - d3 - d4)

    return p


def identify_end_to_end_vector(atom_groups_list):
    """
    Parameters
    ----------
    atoms_group_list : list
        List of <AtomGroup> objects. It can be all backbones in a given system,
        or all chains over which we want to compute the end to end vector
    Returns
    -------
    end_to_end_vector : np.ndarray
        (N, 3) shaped array containing the end-to-end vector for each polymer chain
    """
    end_to_end_vector = np.array(
        [elem.positions[-1] - elem.positions[0] for elem in atom_groups_list]
    )
    return end_to_end_vector


def calculate_end_to_end_correlation(end_to_end_vector):
    """
    This function computes the auto correlation of the end to end vector

    Parameters
    ----------
    Ree : np.ndarray
        end to end distance vector with shape (len(frames), N_chains, 3)

    Returns
    -------
    correlations :  np.ndarray

    """
    correlation = np.zeros(len(end_to_end_vector))
    for i in range(len(end_to_end_vector)):
        tmp = 0.0
        for j in range(len(end_to_end_vector[i])):
            tmp += np.inner(
                end_to_end_vector[0][j], end_to_end_vector[i][j]
            ) / np.inner(end_to_end_vector[0][j], end_to_end_vector[0][j])
        correlation[i] = tmp / len(end_to_end_vector[i])

    return correlation


def calculate_end_to_end_correlation_optimised(end_to_end_vector):
    """
    Optimized function to compute the auto correlation of the end to end vector using NumPy vectorisation.

    Parameters
    ----------
    end_to_end_vector : np.ndarray
        End to end distance vector with shape (len(frames), N_chains, 3)

    Returns
    -------
    correlations : np.ndarray
    """
    # Pre-compute the lengths and the denominator for each chain
    num_frames = len(end_to_end_vector)
    num_chains = len(end_to_end_vector[0])

    # Compute the denominator for the correlation calculation (normalization factor for each chain)
    denominator = np.array(
        [
            np.inner(end_to_end_vector[0][j], end_to_end_vector[0][j])
            for j in range(num_chains)
        ]
    )

    # Initialize the correlation array
    correlation = np.zeros(num_frames)

    # Compute correlation using vectorized operations
    for i in range(num_frames):
        # Vectorized computation of inner products for the current frame with the first frame
        inner_products = np.array(
            [
                np.inner(end_to_end_vector[0][j], end_to_end_vector[i][j])
                for j in range(num_chains)
            ]
        )

        # Compute the correlation for the current frame
        correlation[i] = np.mean(inner_products / denominator)

    return correlation


def compute_bond_lengths(atom_group):
    """
    Parameters
    ----------
    atom_group : <AtomGroup> object
        AtomGroup object containing the bond information
    Returns
    -------
    bond_length : np.ndarray
        Array containing the bond lengths for all bonds in the system
    """
    atom1_positions = atom_group.bonds.atom1.positions
    atom2_positions = atom_group.bonds.atom2.positions
    bond_vectors = atom2_positions - atom1_positions
    bond_length = np.linalg.norm(bond_vectors, axis=1)

    return bond_length


def compute_p2_from_vectors(bond_vectors, reference_axis=np.array([1, 0, 0])):
    """
    Compute the P2 parameter for a polymer chain from bond vectors.

    Parameters:
        bond_vectors (numpy.ndarray): Array of bond vectors (N, 3).
        reference_axis (numpy.ndarray): The reference direction vector (default is x-axis).

    Returns:
        float: The P2 parameter of the bonds.
    """
    # Normalize the reference axis
    reference_axis = reference_axis / np.linalg.norm(reference_axis)

    # Normalize bond vectors
    bond_vectors = bond_vectors / np.linalg.norm(bond_vectors, axis=1)[:, np.newaxis]

    # Compute cosine of angles between bond vectors and the reference axis
    cos_theta = np.dot(bond_vectors, reference_axis)

    # Compute P2 values for each bond
    p2_values = 0.5 * (3 * cos_theta**2 - 1)

    # Compute the average P2 value
    return np.mean(p2_values)


def compute_p2(universe, reference_axis=np.array([1, 0, 0])):
    """
    Compute the P2 parameter for a polymer chain using bond orientations from an MDAnalysis Universe.
    
    Parameters:
        universe (MDAnalysis.Universe): The MDAnalysis Universe containing the polymer chain.
        reference_axis (numpy.ndarray): The reference direction vector (default is x-axis).
        
    Returns:
        float: The P2 parameter of the bonds.
    """
    # Get the positions of the two atoms in the bond
    pos1 = universe.atoms.bonds.atom1.positions
    pos2 = universe.atoms.bonds.atom2.positions

    # Compute bond vectors
    bond_vectors = pos2 - pos1

    # Delegate P2 computation to the helper function
    return compute_p2_from_vectors(bond_vectors, reference_axis=reference_axis)


def calculate_persistence_length(bond_vectors):
    """
    Calculate the persistence length of a polymer chain.

    The persistence length is computed by fitting the decay of bond correlations:
    <cos(theta)> = exp(-s / lp)
    where s is the contour distance along the chain.

    Parameters
    ----------
    bond_vectors : np.ndarray
        Array of bond vectors with shape (N_bonds, 3) or (N_frames, N_bonds, 3).

    Returns
    -------
    lp : float
        The persistence length in units of bond length.
    """
    bond_vectors = np.asarray(bond_vectors)
    
    # Handle trajectory data by reshaping or averaging if needed, 
    # but for now let's assume a single chain or average over frames first.
    # If input is (N_frames, N_bonds, 3), we average correlations over frames.
    
    if bond_vectors.ndim == 3:
        # (Frames, Bonds, 3)
        # Normalize vectors
        norms = np.linalg.norm(bond_vectors, axis=2, keepdims=True)
        u = bond_vectors / norms
        
        n_frames, n_bonds, _ = u.shape
        correlations = np.zeros(n_bonds)
        
        # Compute <u_i . u_{i+s}> averaged over i and frames
        # This is equivalent to the autocorrelation of the bond vectors along the chain index
        
        # We can use a simpler approach: average cos theta for separation s
        for s in range(n_bonds):
            # Vectorized dot product for all i: u[:, i, :] . u[:, i+s, :]
            # We have n_bonds - s pairs
            if s == 0:
                correlations[s] = 1.0
                continue
                
            # u[:, :-s, :] and u[:, s:, :]
            # dot product along last axis (2)
            dots = np.sum(u[:, :-s, :] * u[:, s:, :], axis=2)
            correlations[s] = np.mean(dots)
            
    elif bond_vectors.ndim == 2:
        # (Bonds, 3) - Single conformation
        norms = np.linalg.norm(bond_vectors, axis=1, keepdims=True)
        u = bond_vectors / norms
        n_bonds = len(u)
        correlations = np.zeros(n_bonds)
        
        for s in range(n_bonds):
            if s == 0:
                correlations[s] = 1.0
                continue
            dots = np.sum(u[:-s] * u[s:], axis=1)
            correlations[s] = np.mean(dots)
    else:
        raise ValueError("bond_vectors must be 2D or 3D array.")

    # Fit exponential decay: y = exp(-x/lp) -> ln(y) = -x/lp
    # We fit only the first part where correlation is positive to avoid log(negative)
    # and usually only for s where correlation is significant
    
    x = np.arange(len(correlations))
    y = correlations
    
    # Filter for valid log values
    mask = y > 0
    x_fit = x[mask]
    y_fit = np.log(y[mask])
    
    # Linear fit through origin is not strictly required but standard model is exp(-s/lp)
    # so slope is -1/lp.
    # We can use simple least squares for -x/lp
    
    # slope = sum(x*y) / sum(x^2) for y = slope * x
    slope = np.sum(x_fit * y_fit) / np.sum(x_fit**2)
    
    lp = -1.0 / slope
    return lp


def compute_rouse_modes(positions, p_modes=None):
    """
    Compute the Rouse modes for polymer chains.
    
    X_p = sqrt(2/N) * sum_{n=1}^N R_n * cos(p * pi * (n - 0.5) / N)
    
    Parameters
    ----------
    positions : np.ndarray
        Array of particle positions with shape (N_frames, N_monomers, 3) or (N_monomers, 3).
    p_modes : list or int, optional
        The mode indices 'p' to compute. If None, computes all modes p=0 to N-1.
        If int, computes modes 0 to p_modes-1.
        
    Returns
    -------
    modes : np.ndarray
        The Rouse modes coordinates. 
        Shape (N_frames, len(p_modes), 3) or (len(p_modes), 3).
    """
    positions = np.asarray(positions)
    
    if positions.ndim == 2:
        # (N_monomers, 3)
        positions = positions[np.newaxis, :, :]
        single_frame = True
    else:
        single_frame = False
        
    n_frames, n_monomers, dim = positions.shape
    
    if p_modes is None:
        p_indices = np.arange(n_monomers)
    elif isinstance(p_modes, int):
        p_indices = np.arange(p_modes)
    else:
        p_indices = np.asarray(p_modes)
        
    n_modes = len(p_indices)
    
    # Precompute cosine matrix: (N_modes, N_monomers)
    n = np.arange(1, n_monomers + 1)
    # cos(p * pi * (n - 0.5) / N)
    # We need p as column, n as row for broadcasting or dot product
    # Let's make a matrix M of shape (N_monomers, N_modes) to do pos . M
    
    # Argument: p * pi * (n - 0.5) / N
    # shape (N_monomers, N_modes)
    args = np.outer(n - 0.5, p_indices) * np.pi / n_monomers
    cos_matrix = np.cos(args) # (N_monomers, N_modes)
    
    # X_p = sqrt(2/N) * sum(R_n * cos(...))
    # We want to sum over monomers (axis 1 of positions)
    # positions: (Frames, Monomers, 3)
    # We can use einsum: f m d, m p -> f p d
    
    modes = np.einsum('fmd,mp->fpd', positions, cos_matrix)
    
    # Multiply by normalization sqrt(2/N)
    # Note: Mode 0 is usually defined as center of mass * sqrt(N) or similar, 
    # The formula sqrt(2/N) is standard for p >= 1. 
    # For p=0, cos(0) = 1, sum is sum(R_n) = N * R_cm. 
    # With factor sqrt(2/N), X_0 = sqrt(2/N) * N * R_cm = sqrt(2N) * R_cm.
    # Sometimes X_0 is defined as sqrt(1/N) * sum(R_n) = sqrt(N) * R_cm.
    # We will stick to the sqrt(2/N) factor for all modes as requested by "functional form" usually,
    # but strictly speaking X_0 normalization might differ in literature.
    # Let's use sqrt(2/N) for all for consistency with the formula provided in docstring.
    
    norm_factor = np.sqrt(2.0 / n_monomers)
    modes *= norm_factor
    
    # Special case for p=0 if we want to match standard definition X_0 = sqrt(N) R_cm
    # The current formula gives X_0 = sqrt(2N) R_cm. 
    # This is a factor of sqrt(2) difference. 
    # I will leave it as the raw formula implies unless specified otherwise.
    
    if single_frame:
        return modes[0]
    return modes
