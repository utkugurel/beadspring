def compute_fskt(positions, k_vectors):
    """
Compute the self-intermediate scattering function F_s(k, t).

Parameters
----------
positions : np.ndarray
    Particle positions over time with shape (T, N, 3), where T is number of time frames.
k_vectors : np.ndarray
    Array of k-vectors with shape (K, 3).

Returns
-------
np.ndarray
    F_s(k, t) averaged over k-vectors, array of shape (T - 1,).

Examples
--------
>>> self_intermediate_sf = compute_fskt(positions, k_vectors)
>>> print(self_intermediate_sf.shape)
(99,)
"""
    dr = positions[1:] - positions[0]
    displacement_dot_k = np.dot(dr, k_vectors.T)
    self_intermediate_sf = np.mean(np.cos(displacement_dot_k), axis=(1, 2))
    return self_intermediate_sf

def compute_fskt_batched(positions, k_vectors, batch_size=100):
    """
Compute the self-intermediate scattering function F_s(k, t) using batched k-vectors.

Parameters
----------
positions : np.ndarray
    Particle positions over time with shape (T, N, 3).
k_vectors : np.ndarray
    Array of k-vectors with shape (K, 3).
batch_size : int, optional
    Number of k-vectors per batch to reduce memory usage (default is 100).

Returns
-------
np.ndarray
    F_s(k, t) averaged over all k-vectors, array of shape (T - 1,).

Examples
--------
>>> self_intermediate_sf = compute_fskt_batched(positions, k_vectors, batch_size=50)
>>> print(self_intermediate_sf.shape)
(99,)
"""
    num_batches = int(np.ceil(len(k_vectors) / batch_size))
    self_intermediate_sf = np.zeros(positions.shape[0] - 1)
    for i in range(num_batches):
        k_batch = k_vectors[i * batch_size : (i + 1) * batch_size]
        displacement_dot_k = np.dot(positions[1:] - positions[0], k_batch.T)
        fskt_batch = np.mean(np.cos(displacement_dot_k), axis=(1, 2))
        self_intermediate_sf += fskt_batch * len(k_batch)
    self_intermediate_sf /= len(k_vectors)
    return self_intermediate_sf

def compute_vacf(velocities):
    """
    Computes the velocity autocorrelation function for a given trajectory.

    Parameters
    ----------
    velocities : np.ndarray
        trajectory array with the shape (traj_length, N, 3)

    Returns
    -------
    vacf : np.ndarray
        Velocity autocorrelation function -> len (traj_length - 1)
    """
    v0 = velocities[0]
    v0_dot = np.sum(v0 * v0)

    dot_products = np.einsum('ij,tij->t', v0, velocities)

    vacf = dot_products / v0_dot

    return vacf