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

