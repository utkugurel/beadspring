"""Utility helpers for setting up simulations and managing files."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
from numpy.typing import ArrayLike

__all__ = [
    "setup_universe",
    "setup_freud_box",
    "wrap_coordinates",
    "select_atoms_by_types",
    "find_latest_file",
    "generate_lin_log_timesteps",
    "convert_data_to_molecule",
]


def _import_mdanalysis():  # pragma: no cover - optional dependency
    try:
        import MDAnalysis as mda  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("MDAnalysis is required for this functionality.") from exc
    return mda


def _import_freud():  # pragma: no cover - optional dependency
    try:
        import freud  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("freud must be installed to use this functionality.") from exc
    return freud


def setup_universe(topology_file: str | Path, trajectory_file: str | Path, *, dt: float = 0.005):
    """Create an :class:`MDAnalysis.Universe` from topology and trajectory files."""

    mda = _import_mdanalysis()
    return mda.Universe(str(topology_file), str(trajectory_file), format="LAMMPSDUMP", dt=dt)


def setup_freud_box(lbox: float, *, dimensions: int = 3):
    """Construct a cubic ``freud`` simulation box."""

    freud = _import_freud()

    if dimensions == 3:
        return freud.box.Box(lbox, lbox, lbox, is2D=False)
    if dimensions == 2:
        return freud.box.Box(lbox, lbox, is2D=True)
    raise ValueError("Only 2D and 3D boxes are supported.")


def wrap_coordinates(positions: ArrayLike, box) -> np.ndarray:
    """Wrap coordinates according to the provided ``freud`` box."""

    freud = _import_freud()
    _ = freud  # Silence linter about unused import for documentation builds.
    positions = np.asarray(positions, dtype=float)
    return box.wrap(positions)


def select_atoms_by_types(universe, atom_type_list: Sequence[int | str], *, updating_atom_group: bool = False):
    """Select atoms from an MDAnalysis universe matching the provided types."""

    _import_mdanalysis()
    if not hasattr(universe, "select_atoms"):
        raise ValueError("The provided universe does not expose a 'select_atoms' method.")

    if not isinstance(atom_type_list, Sequence):
        raise ValueError("atom_type_list must be a sequence of atom types.")
    if not atom_type_list:
        raise ValueError("atom_type_list cannot be empty.")

    selection_query = " or ".join(f"type {atom_type}" for atom_type in atom_type_list)

    try:
        return universe.select_atoms(selection_query, updating=updating_atom_group)
    except Exception as exc:  # pragma: no cover - thin wrapper
        raise RuntimeError(f"Error selecting atoms with query '{selection_query}': {exc}") from exc


def _parse_suffix(file_name: str, prefix: str) -> int | None:
    remainder = file_name[len(prefix) :]
    digits = []
    for char in remainder:
        if char.isdigit():
            digits.append(char)
        else:
            break
    if not digits:
        return None
    return int("".join(digits))


def find_latest_file(directory: str | Path, search_string: str) -> str | None:
    """Return the newest file starting with ``search_string`` ordered by numeric suffix."""

    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory '{directory}' does not exist.")

    candidates: list[Tuple[int, str]] = []
    for entry in directory.iterdir():
        if entry.is_file() and entry.name.startswith(search_string):
            suffix = _parse_suffix(entry.name, search_string)
            if suffix is not None:
                candidates.append((suffix, entry.name))

    if not candidates:
        return None

    _, name = max(candidates, key=lambda item: item[0])
    return name


def generate_lin_log_timesteps(
    start_lin_log_power: int,
    final_step: int,
    *,
    save_file: bool = False,
    output_file: str | Path = "timesteps.txt",
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate combined linear and logarithmic time steps for LAMMPS runs."""

    if start_lin_log_power < 1:
        raise ValueError("start_lin_log_power must be at least 1.")
    if final_step < 1:
        raise ValueError("final_step must be positive.")

    powers = np.arange(0, start_lin_log_power)
    multipliers = np.arange(1, 11)

    log_part = np.unique((10**powers[:, None] * multipliers).astype(np.int64).ravel())
    log_part = log_part[log_part <= final_step]

    linlog_part = log_part.copy()
    while linlog_part[-1] < final_step:
        next_steps = linlog_part[-1] + log_part
        linlog_part = np.unique(np.concatenate((linlog_part, next_steps[next_steps <= final_step])))

    linlog_part = np.append(linlog_part, np.int64(final_step + 1))

    if save_file:
        np.savetxt(Path(output_file), linlog_part, fmt="%d")

    return log_part, linlog_part


def convert_data_to_molecule(data_file: str | Path, output_file: str | Path) -> None:
    """Convert a LAMMPS data file into the ``molecule`` input format."""

    data_file = Path(data_file)
    output_file = Path(output_file)

    if not data_file.exists():  # pragma: no cover - thin wrapper
        raise FileNotFoundError(f"The file '{data_file}' does not exist.")

    atoms = []
    bonds = []
    reading_atoms = False
    reading_bonds = False

    with data_file.open("r") as handle:
        for line in handle:
            if "Atoms" in line:
                reading_atoms = True
                continue
            if "Bonds" in line:
                reading_atoms = False
                reading_bonds = True
                continue
            if reading_atoms and line.strip() and not line.startswith("#"):
                parts = line.split()
                atom_id = int(parts[0])
                atom_type = int(parts[2])
                x, y, z = map(float, parts[4:7])
                atoms.append((atom_id, atom_type, x, y, z))
            if reading_bonds and line.strip() and not line.startswith("#"):
                parts = line.split()
                bond_id = int(parts[0])
                bond_type = int(parts[1])
                atom1, atom2 = int(parts[2]), int(parts[3])
                bonds.append((bond_id, bond_type, atom1, atom2))

    with output_file.open("w") as handle:
        handle.write(f"molecule created from {data_file.name}\n\n")
        handle.write(f"{len(atoms)} atoms\n")
        handle.write(f"{len(bonds)} bonds\n")
        handle.write("0 angles\n0 dihedrals\n\n")

        handle.write("Types\n\n")
        for atom_id, atom_type, *_ in atoms:
            handle.write(f"{atom_id} {atom_type}\n")

        handle.write("\nCoords\n\n")
        for atom_id, _, x, y, z in atoms:
            handle.write(f"{atom_id:4d} {x:12.6f} {y:12.6f} {z:12.6f}\n")

        handle.write("\nBonds\n\n")
        for bond_id, bond_type, atom1, atom2 in bonds:
            handle.write(f"{bond_id:4d} {bond_type:4d} {atom1:4d} {atom2:4d}\n")
