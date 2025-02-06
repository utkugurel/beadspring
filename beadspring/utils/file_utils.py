import os

import numpy as np
import MDAnalysis as mda
import freud

def setup_universe(topology_file, trajectory_file, dt=0.005):
    """
    Set up an MDAnalysis universe by loading the topology and trajectory files.

    Parameters
    ----------
    topology_file : str
        The path to the topology file. Usually a .data file.
    trajectory_file : str
        The path to the trajectory file. Usually a .dat file.
    dt : float (optional)
        The time step of the simulation. Default is 0.005.

    Returns
    -------
    universe : MDAnalysis.Universe
        The MDAnalysis universe

    Example
    -------
    >>> topology_file = 'topo.data'
    >>> trajectory_file = 'traj.dat'
    >>> u = setup_universe(topology_file, trajectory_file, dt=0.005)
    """
    universe = mda.Universe(topology_file, trajectory_file, format='LAMMPSDUMP', dt=dt)
    return universe


def setup_freud_box(lbox, dimensions=3):
    """
    Set up a freud box from the box lengths.

    Parameters
    ----------
    lbox : float
        The box length.
    dimensions : int (optional)
        The number of dimensions. Default is 3.

    Returns
    -------
    box : freud.box.Box
        The freud box object.

    Example
    -------
    >>> lbox = 10.0
    >>> box = setup_freud_box(lbox, dimensions=3)
    """
    if dimensions == 3:
        box = freud.box.Box(lbox, lbox, lbox, is2D=False)
    elif dimensions == 2:
        box = freud.box.Box(lbox, lbox, is2D=True)
    else:
        raise ValueError("Only 2D and 3D boxes are supported.")
    return box


def wrap_coordinates(positions, box):
    """
    Wrap the coordinates using the freud box object.

    Parameters
    ----------
    positions : np.ndarray
        The positions of the atoms/molecules.
    box : freud.box.Box
        The freud box object.

    Returns
    -------
    wrapped_positions : np.ndarray
        The wrapped positions of the universe.

    Example
    -------
    >>> u = setup_universe('topo.data', 'traj.dat')
    >>> positions = u.atoms.positions()
    >>> box = freud.box.Box.cube(10.0)
    >>> wrapped_positions = wrap_coordinates(positions, box)
    """
    unwrapped_coordinates = positions.copy()
    wrapped_coordinates = box.wrap(unwrapped_coordinates)
    return wrapped_coordinates


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
    - np.ndarray: An array of timesteps with logarithmic spacing (as in LAMMPS).
    - np.ndarray: An array of special timesteps with a combination of linear and logarithmic spacing.

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

def convert_data_to_molecule(data_file, output_file):
    """
    Convert a LAMMPS data file into a molecule file format.

    The function reads atom and bond data from the input file and writes them
    in the molecule format. This is mainly useful when using the fix bond/react 
    command.

    The molecule file format follows the LAMMPS specifications:
    https://docs.lammps.org/molecule.html

    Parameters
    ----------
    data_file : str
        Name of the input LAMMPS data file.
    output_file : str
        Name of the output molecule file.

    Returns
    -------
    None
    """
    # Check if the input file exists
    if not os.path.isfile(data_file):
        print(f"Error: The file '{data_file}' does not exist.")
        return

    # Read the input file line by line
    with open(data_file, "r") as f:
        lines = f.readlines()

    atoms = []  # List to store atom data
    bonds = []  # List to store bond data
    reading_atoms = False  # Flag to track atom section
    reading_bonds = False  # Flag to track bond section

    # Process each line in the file
    for line in lines:
        if "Atoms" in line:  # Start reading atoms
            reading_atoms = True
            continue
        if "Bonds" in line:  # Start reading bonds
            reading_atoms = False
            reading_bonds = True
            continue
        if reading_atoms and line.strip() and not line.startswith("#"):
            # Extract atom data
            parts = line.split()
            atom_id = int(parts[0])
            atom_type = int(parts[2])
            x, y, z = float(parts[4]), float(parts[5]), float(parts[6])
            atoms.append((atom_id, atom_type, x, y, z))
        if reading_bonds and line.strip() and not line.startswith("#"):
            # Extract bond data
            parts = line.split()
            bond_id = int(parts[0])
            bond_type = int(parts[1])
            atom1, atom2 = int(parts[2]), int(parts[3])
            bonds.append((bond_id, bond_type, atom1, atom2))

    # Write the output molecule file
    with open(output_file, "w") as f:
        # Write file header with input filename
        f.write(f"molecule created from {data_file}\n\n")
        f.write(f"{len(atoms)} atoms\n")
        f.write(f"{len(bonds)} bonds\n")
        f.write("0 angles\n0 dihedrals\n\n")

        # Write atom types
        f.write("Types\n\n")
        for atom_id, atom_type, _, _, _ in atoms:
            f.write(f"{atom_id} {atom_type}\n")

        # Write atom coordinates
        f.write("\nCoords\n\n")
        for atom_id, _, x, y, z in atoms:
            f.write(f"{atom_id:4d} {x:12.6f} {y:12.6f} {z:12.6f}\n")

        # Write bond information
        f.write("\nBonds\n\n")
        for bond_id, bond_type, atom1, atom2 in bonds:
            f.write(f"{bond_id:4d} {bond_type:4d} {atom1:4d} {atom2:4d}\n")

