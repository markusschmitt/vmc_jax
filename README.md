[![Documentation Status](https://readthedocs.org/projects/jvmc/badge/?version=latest)](https://jvmc.readthedocs.io/en/latest/?badge=latest)

# jVMC
This is an impementation of Variational Monte Carlo (VMC) for quantum many-body dynamics using the [JAX library](https://jax.readthedocs.io "JAX library") (and [Flax](https://flax.readthedocs.io "FLAX library") on top) to exploit the blessings of automatic differentiation for easy model composition and just-in-time compilation for execution on accelerators.

## Required packages

- `jax` and `jaxlib`
- `flax`
- `mpi4py`
- `h5py`

## Installation

[How to compile JAX on a supercomputing cluster](documentation/readme/compile_jax_on_cluster.md)
