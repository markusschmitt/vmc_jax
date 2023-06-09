.. _stats:

Sample statistics module
========================

The ``SampledObs`` class provides funcitonality to conveniently compute sample statistics.

**Example:**

    Assuming that ``sampler`` is an instance of a sampler class and ``psi`` is a variational quantum state,
    the quantum geometric tensor 
    :math:`S_{k,k'}=\langle(\partial_{\theta_k}\log\psi_\theta)^*\partial_{\theta_{k'}}\log\psi_\theta\rangle_c` 
    can be computed as

    >>> s, logPsi, p = sampler.sample()
    >>> grads = SampledObs( psi.gradients(s), p)
    >>> S = grads.covar()

.. automodule:: jVMC.stats
    :members:
    :special-members: __call__
