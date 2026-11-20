.. _Sampler:

Sampler module
==============

The sampler module provides the possibilities to sample variational quantum states:

* :class:`~jVMC.sampler.mc_sampler.MCSampler` samples discrete computational basis
  configurations. If the variational wave function provides a ``sample()`` member function,
  direct sampling is used. Otherwise, Markov Chain Monte Carlo (MCMC) is employed.
* :class:`~jVMC.sampler.mc_sampler.MCSamplerCont` does the same for continuous particle
  configurations.
* :class:`~jVMC.sampler.mc_sampler.CutoffSampler` is an MCMC sampler that clips small
  probabilities from below, and corrects for this in the weights of the samples.
* :class:`~jVMC.sampler.exact_sampler.ExactSampler` works with all basis states instead of
  samples, which can be helpful for quick troubleshooting.

The MCMC updates come from the proposers of the ``jVMC.sampler.propose`` module, which is also
available as ``jVMC.propose``.

Usage
^^^^^

A sampler is bound to a variational state. Calling ``sample()`` returns the configurations, their
logarithmic wave function coefficients and their normalized weights, and stores them in the
``samples``, ``log_psi`` and ``weights`` properties. Calling the sampler evaluates an observable on
these samples::

    psi = jVMC.vqs.NQS(net, L, batch_size=2**10, seed=1234)
    sampler = jVMC.sampler.MCSampler(
        psi, jVMC.propose.SpinFlip(), key=123, num_chains=2**6, num_samples=2**12
    )

    s, log_psi, p = sampler.sample()
    energy = sampler(hamiltonian)
    energy = sampler(hamiltonian, resample=True)    # after changing psi.parameters

Stored samples are not refreshed when the parameters of the variational state change, and neither
are their coefficients and weights. Draw new samples after updating the parameters.

The number of chains and the number of samples are rounded up to fit the available devices, and all
chains contribute the same number of samples. A call can therefore return more samples than
requested.

Samplers
^^^^^^^^

.. autoclass:: jVMC.sampler.base.AbstractSampler
    :members:
    :special-members: __call__

.. autoclass:: jVMC.sampler.base.AbstractMCSampler
    :members:
    :special-members: __call__
    :show-inheritance:

.. autoclass:: jVMC.sampler.mc_sampler.MCSampler
    :show-inheritance:

.. autoclass:: jVMC.sampler.mc_sampler.MCSamplerCont
    :show-inheritance:

.. autoclass:: jVMC.sampler.mc_sampler.CutoffSampler
    :members:
    :show-inheritance:

.. autoclass:: jVMC.sampler.exact_sampler.ExactSampler
    :members:
    :special-members: __call__
    :show-inheritance:

Proposers
^^^^^^^^^

.. autoclass:: jVMC.sampler.propose.AbstractProposer
    :members:
    :special-members: __call__
    :private-members: _custom_therm_fun

.. autoclass:: jVMC.sampler.propose.AbstractProposeCont
    :members:
    :show-inheritance:

.. autoclass:: jVMC.sampler.propose.SpinFlip
    :show-inheritance:

.. autoclass:: jVMC.sampler.propose.SpinFlipZ2
    :show-inheritance:

.. autoclass:: jVMC.sampler.propose.SpinFlipZeroMag
    :show-inheritance:

.. autoclass:: jVMC.sampler.propose.POVMOutcome
    :show-inheritance:

.. autoclass:: jVMC.sampler.propose.RWM
    :members: adapt_rate, target_rate
    :special-members: __call__
    :show-inheritance:

.. autoclass:: jVMC.sampler.propose.MALA
    :members: apply_fun, mu, adapt_rate, target_rate
    :special-members: __call__
    :show-inheritance:
