.. _VQS:

Variational quantum state module
================================

The ``NQS`` class wraps a network into a variational quantum state. It evaluates the wave function
coefficients and their gradients in batches, distributed across the available devices, and holds
the variational parameters::

    psi = jVMC_exp.vqs.NQS(jVMC_exp.nets.CpxRBM(numHidden=4), L, batchSize=2**10, seed=1234)

    logPsi = psi(s)                         # Shape (N,), for s of shape (N, L)
    gradients = psi.gradients(s)            # Shape (N, P)
    psi.parameters = psi.parameters_flat    # Parameters can be assigned as a flat array

.. autoclass:: jVMC_exp.vqs.NQS
    :members:
    :special-members: __call__
