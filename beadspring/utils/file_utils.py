import os

import numpy as np


def find_latest_file(directory, search_string):
    """
    Find the latest file in a directory where all fiels start with the
    specified string and return the one with the largest suffix.
    This function might be useful to find the latest binary/trajectory
    file in a directory.

    Args:
    - directory: The directory to search in
    - search_string: The string to search for
    Returns:
    - latest_file: The file with the largest value after the search string
    """
    # Get all files in the directory
    all_files = os.listdir(directory)

    # Filter files starting with "Conf_"
    search_files = [f for f in all_files if f.startswith(search_string)]

    # If no such files are found, return None
    if not search_files:
        print(f"No files found in {directory} starting with {search_string}")
        return None

    # Extract values from filenames and get the file with the largest value
    latest_file = max(search_files, key=lambda x: int(x.split("_")[1].split(".")[0]))

    return latest_file


def generate_lin_log_timesteps(start_lin_log_power, final_step, save_file=False):
    """
    Generate a sequence of linearly and logarithmically spaced timesteps for LAMMPS simulations.

    Parameters:
    - start_lin_log_power (int): The starting power of 10 after which the logarithmnic spacing starts over.
    - final_step (int): The final simulation step.
    - save_file (bool): Whether to save the generated timesteps to a text file.

    Returns:
    - np.ndarray: An array of timesteps for LAMMPS simulations.

    Example:
    >>> generate_lin_log_timesteps(7, 5*10**8, save_file=True)
    Creates a logarithmically spaced time steps upto 10^7 and
    then starts over from 10^7+1 upto 5*10^8 restarting the log
    save every 10^7 steps.

    """
    # Initial range of powers and base multipliers
    powers = np.arange(1, start_lin_log_power)
    multipliers = np.arange(1, 11)

    # Generate all combinations of 10**i * j where i ranges from 1 to 8 and j from 1 to 10
    all_combinations = (
        10 ** powers[:, None] * multipliers
    )  # Broadcasting to create a 2D array of combinations

    # Flatten the array and sort it (flattening turns the 2D array into a 1D array)
    steps = np.unique(all_combinations.ravel())

    # Ensure that the initial steps from 1 to 9 are included
    initial_steps = np.arange(1, 10) * 10**0
    log_part = np.unique(np.concatenate((initial_steps, steps)))
    log_part = log_part.astype(np.int64)

    # Initialize final steps array
    linlog_part = log_part.copy()
    max_value = np.int64(final_step)  # The last simulation step

    # Iteratively build the sequence until the max value is reached or exceeded
    while linlog_part[-1] < max_value:
        # Generate the next set of steps by adding the spacing to the last element
        next_steps = linlog_part[-1] + log_part

        # Keep only new steps that are less than or equal to max_value
        next_steps = next_steps[next_steps <= max_value]

        # Concatenate with the existing steps and eliminate duplicates
        linlog_part = np.unique(np.concatenate((linlog_part, next_steps)))

    # Append max_value+1 to final_steps to prevent LAMMPS errors
    linlog_part = np.append(linlog_part, np.int64(max_value + 1))

    # Optionally save the final_steps into a text file if an argument is provided
    if save_file:
        np.savetxt("timesteps.txt", linlog_part, fmt="%d")

    return log_part, linlog_part
