[![Documentation Status](https://readthedocs.org/projects/jvmc/badge/?version=latest)](https://jvmc.readthedocs.io/en/latest/?badge=latest)
[![PyPi version](https://badgen.net/pypi/v/jVMC/)](https://pypi.org/project/jVMC/)
[![arXiv](https://img.shields.io/badge/arXiv-2108.03409-b31b1b.svg)](https://arxiv.org/abs/2108.03409)

# jVMC
This is an impementation of Variational Monte Carlo (VMC) for quantum many-body dynamics using the [JAX library](https://jax.readthedocs.io "JAX library") (and [Flax](https://flax.readthedocs.io "FLAX library") on top) to exploit the blessings of automatic differentiation for easy model composition and just-in-time compilation for execution on accelerators.

1. [Documentation](#documentation)
2. [Installation](#installation)
3. [Online example](#online-example)
4. [Important gotchas](#important-gotchas)
5. [Citing jVMC](#citing-jvmc)

Please report bugs as well as other issues or suggestions on our [issues page](https://github.com/markusschmitt/vmc_jax/issues).

## Documentation

Documentation is available [here](https://jvmc.readthedocs.io/en/latest/ "Documentation").

## Installation

### Option 1: ``pip``-install

        pip install jVMC


### Option 2: Clone and ``pip``-install for development

1. Clone the jVMC repository and check out the development branch:

        git clone https://github.com/markusschmitt/vmc_jax.git
        cd vmc_jax

2. ``pip``-install the package  

        pip install -e ".[dev]"

Test that everything worked, e.g. run 'python -c "import jVMC"' from a place different than ``vmc_jax``.


## Online example

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/markusschmitt/vmc_jax/blob/master/examples/ex0_ground_state_search.ipynb)

Click on the badge above to open a notebook that implements an exemplary ground state search in Google Colab.

## Important gotchas
### Out-of-memory issues and batching
Memory requirements grow with increasing network sizes. To avoid out-of-memory issues, the ``batchSize`` parameter of the ``NQS`` class has to be adjusted. The ``batchSize`` indicates on how many input configurations the network is evaluated concurrently. Out-of-memory issues are usually resolved by reducing this number. The ``numChains`` parameter of the ``Sampler`` class for Markov Chain Monte Carlo sampling plays a similar role, but its optimal values in terms of computational speed are typically not memory critical.

## Citing jVMC

If you use the jVMC package for your research, please cite our reference paper [SciPost Phys. Codebases 2 (2022)](https://scipost.org/10.21468/SciPostPhysCodeb.2)

        @Article{jVMC,
                title={{jVMC: Versatile and performant variational Monte Carlo leveraging automated differentiation and GPU acceleration}},
                author={Markus Schmitt and Moritz Reh},
                journal={SciPost Phys. Codebases},
                pages={2},
                year={2022},
                publisher={SciPost},
                doi={10.21468/SciPostPhysCodeb.2},
                url={https://scipost.org/10.21468/SciPostPhysCodeb.2},
        }
