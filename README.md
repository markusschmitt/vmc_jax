[![Documentation Status](https://readthedocs.org/projects/jvmc/badge/?version=latest)](https://jvmc.readthedocs.io/en/latest/?badge=latest)

# jVMC
This is an impementation of Variational Monte Carlo (VMC) for quantum many-body dynamics using the [JAX library](https://jax.readthedocs.io "JAX library") (and [Flax](https://flax.readthedocs.io "FLAX library") on top) to exploit the blessings of automatic differentiation for easy model composition and just-in-time compilation for execution on accelerators.

## Documentation

Documentation is available [here](https://jvmc.readthedocs.io/en/latest/ "Documentation").

## Installation

### Option 1: ``pip``-install

1. We recommend you create a new conda environment to work with jVMC:

```
    conda create -n jvmc python=3.8
    conda activate jvmc
```

2. ``pip``-install the package

```
    pip install jVMC
```

### Option 2: Clone and ``pip``-install

1. Clone the jVMC repository and check out the development branch:

```
    git clone https://github.com/markusschmitt/vmc_jax.git
    cd vmc_jax
```

2. We recommend you create a new conda environment to work with jVMC:

```
    conda create -n jvmc python=3.8
    conda activate jvmc
```

3. Create a wheel and ``pip``-install the package
```
    python setup.py bdist_wheel
    python -m pip install dist/*.whl
```
Test that everything worked, e.g. run 'python -c "import jVMC"' from a place different than ``vmc_jax``.

### Option 3: Manually install dependencies

If you want to work on the jVMC code you might prefer to [install dependencies and set up jVMC](documentation/readme/installation_instructions.md) without ``pip``-install.

### Compiling JAX

[How to compile JAX on a supercomputing cluster](documentation/readme/compile_jax_on_cluster.md)


## Online example

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/markusschmitt/vmc_jax/blob/dev_0.1.0/examples/ex0_ground_state_search.ipynb)

Click on the badge above to open a notebook that implements an exemplary ground state search in Google Colab.
