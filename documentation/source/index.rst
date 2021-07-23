.. jVMC documentation master file, created by
   sphinx-quickstart on Sun Sep 27 09:17:47 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. toctree::
   :hidden:
   
   self

.. toctree::
    :hidden:
    :maxdepth: 1
    :caption: Design choices

    parallelism

.. toctree::
   :hidden:
   :glob:
   :maxdepth: 2
   :caption: API documentation

   vqs
   operator
   sampler
   nets
   mpi
   util

.. toctree::
    :hidden:
    :maxdepth: 1
    :caption: Examples

    examples

.. toctree::
    :hidden:
    :maxdepth: 1
    :caption: Installation

    installation

`jVMC`: Versatile and performant variational Monte Carlo
========================================================

This package, `available on GitHub <https://github.com/markusschmitt/vmc_jax>`_, 
provides a versatile and efficient implementation of variational
quantum Monte Carlo in Python. It utilizes Google's 
`JAX <https://github.com/google/jax>`_ library to exploit the
blessings of automatic differentiation and just-in-time compilation for
available computing resources. The package is devised to serve as transparent
implementation of the core computational tasks that guarantees efficiency, 
while at the same time providing large flexibility.

In particular, `jVMC` provides a framework that allows to work with arbitrary
variational wave functions and quantum operators. The code was
written mainly with neural quantum states (NQS) as variational wave functions
in mind, but it is not restricted to that; the ansatz wave functions can be
arbitrary parametrized programs. Nonetheless, throughout the documentation
we will refer to the variational ansätze as "networks".

Design choices
##############

Variational wave functions
--------------------------
A core part of this codebase is the ``NQS`` class, an abstract wrapper class 
for variational wave functions, which proves an interface that other parts 
of the code rely on. At initialization the specific variational wave function 
is passed to ``NQS`` in the form of a `Flax <https://github.com/google/flax>`_
module. `Flax <https://github.com/google/flax>`_ is a
library that supplements `JAX` with a class structure to enable simple
implementation of neural networks (and more) based on modules as it is
known also from `Pytorch`.

Parallelism
-----------
The performance of the code relies on a few design choices, which enable
efficient computation for typical use cases on a desktop device as well as
on distributed multi GPU clusters. An important manifestation of these
choices are required array dimensions when interfacing `jVMC`: All data
that is related to network evaluations will have two leading dimensions, 
namely the `device dimension` and the `batch dimension`. Distributed
computing is enabled using the MPI through the ``mpi4py`` package. 
See :ref:`Parallelism` for details.


Example
#######
The core task in Variational Monte Carlo is sampling from the Born
distribution :math:`|\psi_\theta(s)|^2` of a variational wave function
:math:`\psi_\theta(s)` and computing the mean of local estimators

    :math:`O_{loc}(s)=\sum_{s'}O_{s,s'}\frac{\psi_\theta(s')}{\psi_\theta(s)}` ,

where :math:`O_{s,s'}=\langle s|\hat O|s'\rangle` are the matrix elements 
of some quantum operator :math:`\hat O`.

Assume that ``op`` is an ``Operator`` object (see :ref:`Operator`)
corresponding to :math:`\hat O` and ``psi`` is an ``NQS`` object implementing
the variational wave function. Moreover, assume that ``sampler`` is a suited
``Sampler`` object (see :ref:`Sampler`). Then, estimating its expectation value
using the `jVMC` framework boils down to::

    s, logPsi, _ = sampler.sample()                 # Get samples (parallelized)
    sPrime, _ = op.get_s_primes(sampleConfigs)      # Get s', where O_{s,s'}!=0
    logPsiOffd = psi(sPrime)                        # Evaluate wave function on s'
    Oloc = get_O_loc(logPsi, logPsiOffd)            # Compute local estimator
    Omean = jVMC.mpi_wrapper.get_global_mean(Oloc)  # Compute mean of all processes

Also computing the variational derivatives :math:`\partial_{\theta_k}\log\psi_\theta(s)`
is straightforward when using the ``NQS`` class::

    grad_psi = psi.gradients(s)
 
See the :ref:`Examples` section for a number of more elaborate example applications.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
