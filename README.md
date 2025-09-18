# `unicirc` - Unitary synthesis with optimal brick wall circuits

Here we present a small Python package defining quantum circuits with a brick wall structure,
providing numerical tools to check necessary conditions for their universality, and implementing
variational optimization workflows to compile typical unitaries to them.

The code used to produce the data and figures in the preprint "Unitary synthesis with optimal
brick wall circuits" is enclosed as well.

The repository is structured as follows:

- `manuscript/` contains all scripts, data and figures for the preprint. The subdirs are arranged by sections and figures
- `pyproject.toml` defines the Python package `unicirc`
- `unicirc/` contains the source code for the Python package `unicirc`
- `tests/` contains tests for the `unicirc` package

## Installation of `unicirc`

The package is set up via the `pyproject.toml` file for installation via setuptools.
It can be installed locally including its requirements via `pip install .` while in the top level
directory of the repository.

```
git clone git@github.com:XanaduAI/unicirc.git
cd unicirc
pip install .
```

## Script execution and data management

The Jupyter notebooks in `manuscript/` mostly run quite quickly and therefore do not have any
management of storing data to disk. Other numerics in `manuscript/` are split into data
generation scripts (`gen_data.py`) that store data in the respective `data/` subdirectory,
and plotting scripts (`gen_fig_X.py`) retrieving from the respective `data/` and storing
figures directly in the respective section's directory. Note that there is no
test for the pre-existence of data files, numerics are just recomputed blindly.
Some of the Python scripts take command line arguments. See the documentation of each respective
file for details.
