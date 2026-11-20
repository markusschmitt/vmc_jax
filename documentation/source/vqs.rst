.. _VQS:

Variational quantum state module
================================

The ``NQS`` class wraps a network into a variational quantum state. It evaluates the wave function
coefficients and their gradients in batches, distributed across the available devices, and holds
the variational parameters::

    psi = jVMC.vqs.NQS(jVMC.nets.CpxRBM(num_hidden=4), L, batch_size=2**10, seed=1234)

    log_psi = psi(s)                         # Shape (N,), for s of shape (N, L)
    gradients = psi.gradients(s)            # Shape (N, P)
    psi.parameters = psi.parameters_flat    # Parameters can be assigned as a flat array

.. autoclass:: jVMC.vqs.NQS
    :members:
    :special-members: __call__
