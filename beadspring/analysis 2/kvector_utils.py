def get_k_vectors(
    ktarget, box_length, max_points=1000, save_vectors=False,
):
    """
Generate k-vectors in a periodic box that are close in magnitude to a target wave number.

Parameters
----------
ktarget : float
    The target wave number (magnitude of k-vector).
box_length : float
    Length of the cubic simulation box.
max_points : int, optional
    Maximum number of k-vectors to return. If more are found, a random subset is used (default is 1000).
save_vectors : bool, optional
    If True, saves the resulting k-vectors to 'k_vectors.npy' (default is False).

Returns
-------
np.ndarray
    Array of shape (N, 3) containing the filtered k-vectors.

Examples
--------
>>> k_vectors = get_k_vectors(ktarget=5.0, box_length=10.0, max_points=100)
>>> print(k_vectors.shape)
(100, 3)

>>> get_k_vectors(ktarget=3.14, box_length=8.0, save_vectors=True)
# Saves 'k_vectors.npy' to current directory
"""
    k_step = 2 * np.pi / box_length
    k_discrete = ktarget / k_step
    k_max = int(np.ceil(k_discrete))

    # Generate all possible k-indices within the range
    n_values = np.arange(-k_max, k_max + 1)
    k_indices = np.array(list(product(n_values, repeat=3)))

    # Compute magnitudes and filter
    k_magnitudes = np.linalg.norm(k_indices, axis=1)
    close_to_k_discrete = np.abs(k_magnitudes - k_discrete) < 0.1
    valid_indices = k_indices[close_to_k_discrete]

    # Compute actual k-vectors
    k_vectors = valid_indices * k_step

    # Sample if necessary
    if len(k_vectors) > max_points:
        np.random.seed(1)
        k_vectors = k_vectors[np.random.choice(len(k_vectors), max_points, replace=False)]

    if save_vectors:
        np.save("k_vectors.npy", k_vectors)

    return k_vectors