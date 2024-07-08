"""Documentation about the beadspring module."""
import numpy as np
from scipy.spatial.distance import pdist
from scipy.spatial.distance import cdist

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
    gyration_tensor = np.einsum('ij,ik->jk', diff, diff) / len(positions)
    eigenvalues = np.sort(np.linalg.eigvals(gyration_tensor))

    return gyration_tensor, eigenvalues


def calculate_asphericity(lmin, lmid, lmax):
    '''
    Parameters
    ----------
    lmin, lmid, lmax : float
        Eigenvalues of the gyration tensor
    Returns
    -------
    b : float
        Asphericity parameter for polymer chains. b=0 
        corresponds to a perfect sphere
    '''
    b = lmax - 0.5*(lmin + lmid)
    return b 

def calculate_acylindricity(lmin, lmid, lmax=0.):
    '''
    Parameters
    ----------
    lmin, lmid, lmax : float
        Eigenvalues of the gyration tensor
    Returns
    -------
    c : float
        Acylindricity parameter for polymer chains. c=0 
        corresponds to a perfect cylinder.
    '''
    c = lmid - lmin
    return c

def calculate_rg2(lmin, lmid, lmax):
    '''
    Parameters
    ----------
    lmin, lmid, lmax : float
        Eigenvalues of the gyration tensor
    Returns
    -------
    rg2 : float
        Radius of gyration squared for polymer chains.
    '''

    rg2 = lmin + lmid + lmax
    return rg2

def calculate_hydrodynamic_radius(positions):
    '''
    Parameters
    ----------
    positions : np.ndarray
        (N, 3) shaped array containing the positions of particles
    Returns
    -------
    hydrodynamic_radius : float
        Hydrodynamic radius of the polymer chain
    '''
    N = len(pdist(positions))
    inv_dist = (1/pdist(positions)).sum()
    hydrodynamic_radius = 1/(inv_dist/N)
    
    return hydrodynamic_radius    

def calculate_shape_anisotropy(lmin, lmid, lmax):
    '''
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
    '''

    num = calculate_asphericity(lmin, lmid, lmax)**2 + 0.75*calculate_acylindricity(lmin, lmid, lmax=0.)**2
    denum = calculate_rg2(lmin, lmid, lmax)**2

    k2 = num / denum

    return k2

def calculate_prolateness(lmin, lmid, lmax):
    '''
    Parameters
    ----------
    lmin, lmid, lmax : float
        Eigenvalues of the gyration tensor
    Returns
    -------
    p : float
        p=-1 for perfectly oblate shape and p=1 for perfectly prolate shape
        
    '''
    n1 = 2*np.sqrt(lmin) - np.sqrt(lmid) - np.sqrt(lmax)
    n2 = 2*np.sqrt(lmid) - np.sqrt(lmin) - np.sqrt(lmax)
    n3 = 2*np.sqrt(lmax) - np.sqrt(lmin) - np.sqrt(lmid)

    d1 = 2*(lmin + lmid + lmax) 
    d2 = 2*np.sqrt(lmin) * np.sqrt(lmid)
    d3 = 2*np.sqrt(lmid) * np.sqrt(lmax)
    d4 = 2*np.sqrt(lmin) * np.sqrt(lmax)

    p = (n1 * n2 * n3) / (d1 - d2 - d3 - d4)

    return p

def identify_end_to_end_vector(atom_groups_list):
    '''
    Parameters
    ----------
    atoms_group_list : list
        List of <AtomGroup> objects. It can be all backbones in a given system,
        or all chains over which we want to compute the end to end vector
    Returns
    -------
    end_to_end_vector : np.ndarray
        (N, 3) shaped array containing the end-to-end vector for each polymer chain
    '''
    end_to_end_vector = np.array([elem.positions[-1] - elem.positions[0] for elem in atom_groups_list])
    return end_to_end_vector

def calculate_end_to_end_correlation(end_to_end_vector):
    '''
    This function computes the auto correlation of the end to end vector

    Parameters
    ----------
    Ree : np.ndarray 
        end to end distance vector with shape (len(frames), N_chains, 3) 

    Returns
    -------
    correlations :  np.ndarray

    '''
    correlation = np.zeros(len(end_to_end_vector))
    for i in range(len(end_to_end_vector)):
        tmp = 0.0
        for j in range(len(end_to_end_vector[i])):
            tmp += np.inner(end_to_end_vector[0][j], end_to_end_vector[i][j]) / np.inner(end_to_end_vector[0][j], end_to_end_vector[0][j])
        correlation[i] = tmp / len(end_to_end_vector[i])

    return correlation


def calculate_end_to_end_correlation_optimised(end_to_end_vector):
    '''
    Optimized function to compute the auto correlation of the end to end vector using NumPy vectorization.

    Parameters
    ----------
    end_to_end_vector : np.ndarray 
        End to end distance vector with shape (len(frames), N_chains, 3) 

    Returns
    -------
    correlations : np.ndarray
    '''
    # Pre-compute the lengths and the denominator for each chain
    num_frames = len(end_to_end_vector)
    num_chains = len(end_to_end_vector[0])
    
    # Compute the denominator for the correlation calculation (normalization factor for each chain)
    denominator = np.array([np.inner(end_to_end_vector[0][j], end_to_end_vector[0][j]) for j in range(num_chains)])
    
    # Initialize the correlation array
    correlation = np.zeros(num_frames)
    
    # Compute correlation using vectorized operations
    for i in range(num_frames):
        # Vectorized computation of inner products for the current frame with the first frame
        inner_products = np.array([np.inner(end_to_end_vector[0][j], end_to_end_vector[i][j]) for j in range(num_chains)])
        
        # Compute the correlation for the current frame
        correlation[i] = np.mean(inner_products / denominator)
    
    return correlation


def compute_bond_lengths(atom_group):
    '''
    Parameters
    ----------
    atom_group : <AtomGroup> object
        AtomGroup object containing the bond information
    Returns
    -------
    bond_length : np.ndarray
        Array containing the bond lengths for all bonds in the system
    '''
    atom1_positions = atom_group.bonds.atom1.positions
    atom2_positions = atom_group.bonds.atom2.positions
    bond_vectors = atom2_positions - atom1_positions
    bond_length = np.linalg.norm(bond_vectors, axis=1)

    return bond_length

