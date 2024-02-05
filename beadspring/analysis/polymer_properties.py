"""Documentation about the beadspring module."""
import numpy as np


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
