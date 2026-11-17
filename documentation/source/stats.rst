.. _Stats:

Sample statistics module
========================

The ``jVMC_exp.stats`` module computes statistics of samples. ``SampledObs`` holds the values
:math:`O_n` of an observable on :math:`N` samples, together with their normalized weights
:math:`w_n`, and estimates means, variances, covariances and their statistical errors:

* mean: :math:`\langle O\rangle=\sum_n w_n O_n`,
* variance: :math:`\mathrm{Var}(O)=\sum_n w_n|O_n-\langle O\rangle|^2`,
* covariance: :math:`\mathrm{Cov}(O,P)=\sum_n w_n(O_n-\langle O\rangle)^*(P_n-\langle P\rangle)`,
* error of the mean: :math:`\sqrt{\mathrm{Var}(O)\sum_n w_n^2}`.

Calling a sampler on an operator returns the local estimator as a ``SampledObs`` (see
:ref:`Sampler`). With the gradients :math:`\partial_{\theta_k}\ln\psi(s_n)` as observations, the
covariances give the quantum geometric tensor
:math:`S_{k,k'}=\langle(\partial_{\theta_k}\ln\psi)^*\partial_{\theta_{k'}}\ln\psi\rangle_c` and the
force :math:`F_k=\langle(\partial_{\theta_k}\ln\psi)^*E_{\rm loc}\rangle_c`::

    s, logPsi, p = sampler.sample()
    energy = sampler(hamiltonian)           # SampledObs of the local energy
    print(energy)                           # mean ± error of the mean (Var = variance)

    grads = jVMC_exp.stats.SampledObs(psi.gradients(s), p)
    S = grads.get_covar()                   # Shape (P, P)
    F = grads.get_covar(energy)             # Shape (P,)

The error of the mean treats the samples as independent. For samples of Markov chains, the
integrated autocorrelation time :math:`\tau` makes the error larger by a factor of about
:math:`\sqrt{\tau}`, and the Gelman-Rubin diagnostic :math:`\hat R` compares the chains::

    tau = energy.get_autocorrelation_time(sampler.numChains)
    r_hat = energy.get_R_hat(sampler.numChains)

The observations are sharded across the devices along the sample axis. If the number of samples is
not a multiple of the number of devices, they are padded with rows of zeros of weight zero, which
do not contribute to any estimate.

``LazySampledObs`` computes the same estimates from observations that are computed one batch at a
time, so that they are never held in memory at once. Objective functions use it for the gradients
when they are created with ``batched_jacobian=True``.

.. autoclass:: jVMC_exp.stats.SampledObs
    :members:

.. autoclass:: jVMC_exp.stats.LazySampledObs
    :members:
