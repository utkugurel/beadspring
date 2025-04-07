---
# Bead Spring Analytics

Created by [Utku Gürel](https://github.com/utkugurel)



## Introduction
Bead-Spring Analytics is a Python package for analysing coarse-grained molecular dynamics simulations. The name is inspired by the very well known bead-spring model [[1]](#1). It provides tools for computing structural and dynamical properties of polymers using LAMMPS-generated trajectory data. It is also capable of generating some coarse-grained polymer architectures that are used in these simulations. It heavily relies on other python packages and does not claim to be complete. We employ functional programming, so we avoid writing classes. Feel free to include your own functions keeping this desing choice in mind.

[![Powered by MDAnalysis](https://img.shields.io/badge/powered%20by-MDAnalysis-orange.svg?logoWidth=16&logo=data:image/x-icon;base64,AAABAAEAEBAAAAEAIAAoBAAAFgAAACgAAAAQAAAAIAAAAAEAIAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAJD+XwCY/fEAkf3uAJf97wGT/a+HfHaoiIWE7n9/f+6Hh4fvgICAjwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACT/yYAlP//AJ///wCg//8JjvOchXly1oaGhv+Ghob/j4+P/39/f3IAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAJH8aQCY/8wAkv2kfY+elJ6al/yVlZX7iIiI8H9/f7h/f38UAAAAAAAAAAAAAAAAAAAAAAAAAAB/f38egYF/noqAebF8gYaagnx3oFpUUtZpaWr/WFhY8zo6OmT///8BAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgICAn46Ojv+Hh4b/jouJ/4iGhfcAAADnAAAA/wAAAP8AAADIAAAAAwCj/zIAnf2VAJD/PAAAAAAAAAAAAAAAAICAgNGHh4f/gICA/4SEhP+Xl5f/AwMD/wAAAP8AAAD/AAAA/wAAAB8Aov9/ALr//wCS/Z0AAAAAAAAAAAAAAACBgYGOjo6O/4mJif+Pj4//iYmJ/wAAAOAAAAD+AAAA/wAAAP8AAABhAP7+FgCi/38Axf4fAAAAAAAAAAAAAAAAiIiID4GBgYKCgoKogoB+fYSEgZhgYGDZXl5e/m9vb/9ISEjpEBAQxw8AAFQAAAAAAAAANQAAADcAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAjo6Mb5iYmP+cnJz/jY2N95CQkO4pKSn/AAAA7gAAAP0AAAD7AAAAhgAAAAEAAAAAAAAAAACL/gsAkv2uAJX/QQAAAAB9fX3egoKC/4CAgP+NjY3/c3Nz+wAAAP8AAAD/AAAA/wAAAPUAAAAcAAAAAAAAAAAAnP4NAJL9rgCR/0YAAAAAfX19w4ODg/98fHz/i4uL/4qKivwAAAD/AAAA/wAAAP8AAAD1AAAAGwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAALGxsVyqqqr/mpqa/6mpqf9KSUn/AAAA5QAAAPkAAAD5AAAAhQAAAAEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADkUFBSuZ2dn/3V1df8uLi7bAAAATgBGfyQAAAA2AAAAMwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAB0AAADoAAAA/wAAAP8AAAD/AAAAWgC3/2AAnv3eAJ/+dgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA9AAAA/wAAAP8AAAD/AAAA/wAKDzEAnP3WAKn//wCS/OgAf/8MAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIQAAANwAAADtAAAA7QAAAMAAABUMAJn9gwCe/e0Aj/2LAP//AQAAAAAAAAAA)](https://www.mdanalysis.org)



## Installation

#### Prerequisites

Ensure you have: Python 3.11+

Required Python libraries: NumPy, MDAnalysis, Freud, scipy, 

The `environment.yml` fiel contains more packages that are useful and will be needed if you are part of our research group. We keep them as requirements for convenience.



We suggest the use of virtual environments. Follow the steps below to create the `bsa` environment.

##### Clone to repository
```bash
git clone git@github.com:utkugurel/beadspring.git
```
##### Install the dependencies in a conda environment and activate `bsa`

```bash
conda env create -f environment.yml
conda activate bsa
```
Sometimes the installation takes a long time. In that case, follow these steps:
First, create the environment with the specified Python version

```bash
conda create -n bsa python=3.11.6 -c conda-forge
```
Then, activate the environment and install the remaining packages
```bash
conda activate bsa
conda env update --name bsa --file environment.yml --prune
```
If this also takes a long time, try to install the packages individually.

##### Install `beadspring` into `bsa` environment

```bash
pip install -e .
```

##### Check the installation
```bash
$ python -m beadspring
#or
$ beadspring
```
###### Optional: Run the tests under `beadspring` directory

```bash
pytest .
```

## Usage

Since we employ functional programming, the example below can be generalised into all submodules. Any function can be called from a python interpreter (e.g. iPython) as well as within python scripts.

### Example analysis script

Let's assume w.l.o.g. there is only a single polymer chain in the trajectory.

```python
import numpy as np
import beadspring as bsa

def main():
    # Define the topology and trajectory files
    topology = 'topo.data'
    trajectory = 'traj.dat'
    u = bsa.setup_universe(topology, trajectory)

    N_FRAMES = u.trajectory.n_frames
    N_ATOMS = u.atoms.n_atoms

    # Initialise the position and time arrays
    positions = np.zeros((N_FRAMES, N_ATOMS, 3))
    time = np.zeros(N_FRAMES)

    # Loop over the trajectory and load the positions
    for i,traj in enumerate(u.trajectory):                          
        positions[i] = u.atoms..positions   
        time[i] = u.trajectory.ts.data['time']

    # Now use bsa to calculate the hydrodynamics radius
    rhydro = np.zeros(N_FRAMES)
    for i, frame in enumerate(positions):
        rhydro[i] = bsa.calculate_hydrodynamic_radius(frame)
    # you can save the time average with np.mean(rhydro)

    # Now let's compute a dynamical property: mean squared displacement
    msd = bsa.compute_msd(positions)
    # you can plot this as a function of time array using your favourite plotting package

if __name__ == '__main__':
    main()
```


## Notes
- We suggest looping over the trajectory only once and load needed quantities to numpy arrays.
- You can use atom selection commands of mdanalysis to select which molecules to act on.
- It is wise to call different functions within the same loop (whenever possible) to optimise the runtime.
- We use `Numpy` based operations to optimise the speed in almost all cases. If a specific function creates a bottleneck in the runtime, you can accelerate it with `@njit` decorator form `Numba`.

## Additional Resources
- [LAMMPS Documentation](https://docs.lammps.org/Manual.html)
- [NumPy](https://numpy.org/)
- [MDAnalysis](https://www.mdanalysis.org/)
- [freud](https://freud.readthedocs.io/en/latest/index.html)
- [SciPy](https://scipy.org/)
- [Numba: A High Performance Python Compiler](https://numba.pydata.org/)
- [signac - simple data management](https://signac.io)

For more information, contact the package maintainers or create an issue.

## References
<a id="1">[1]</a> Kremer, K.; Grest, G. S. Dynamics of entangled linear polymer melts: A molecular-dynamics simulation. The Journal of Chemical Physics 1990, 92, 5057–5086.