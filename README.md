---
# Bead Spring Analytics

beadspring created by utkugurel

## Install locally

Clone to repository
```bash
git clone git@github.com:utkugurel/beadspring.git
```
Install the dependencies in a conda environment and activate bsa

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

Install beadspring into bsa environment

```bash
pip install -e .
```

Check the installation
```bash
$ python -m beadspring
#or
$ beadspring
```
Run the tests with pytest

```bash
pytest .
```

## Usage

```py
import beadspring as bsa
```

