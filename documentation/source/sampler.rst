..

Sampler module
==============

The sampler module provides two possibilities to sample variational
quantum states: Markov Chain Monte Carlo (MCMC) and direct sampling.
For direct sampling the variational ansatz needs to provide a 
``sample()`` member function. 
If the variational wave function provides such a function,
direct sampling is used. Otherwise, MCMC is employed.
Additionally, the ``ExactSampler`` class exists, which works with all 
basis states instead of samples, which can be helpful for quick
troubleshooting.

.. automodule:: jVMC.sampler
    :members:
