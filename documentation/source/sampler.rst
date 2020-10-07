..

Sampler module
==============

The sampler module provides two possibilities to sample variational
quantum states: Markov chain Monte Carlo (MCMC) and direct sampling.
For direct sampling the variational ansatz needs to provide a 
``sample()`` member function. 
If the variational wave function provides such a function,
direct sampling is used. Otherwise, MCMC is employed.

.. automodule:: jVMC.sampler
    :members:
