import numpy as np
import freud

def compute_structure_factor_freud(positions, box, bins=100, kmax=8.0, kmin=0.1):
    '''
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

    '''

    wrapped_positions = box.wrap(positions)
    sf = freud.diffraction.StaticStructureFactorDirect(
        bins=bins, k_max=kmax, k_min=kmin
    )
    sf.compute((box, wrapped_positions))

    return sf.bin_centers, sf.S_k