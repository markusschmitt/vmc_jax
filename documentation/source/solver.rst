.. _Solver:

Solver module
=============

A solver solves the linear equation :math:`Ax=b` of an optimizer (see :ref:`Optimizer`): ``SR``
and ``TDVP`` solve :math:`S\dot\theta=cF` with the quantum geometric tensor :math:`S`, and
``MinSR`` the equation of its tangent kernel. The solvers are

* ``PinvSNR``: pseudo-inverse from the eigendecomposition of :math:`A`, with a smooth cutoff of
  the small eigenvalues and, optionally, of the modes that the samples do not resolve. It is the
  default solver of ``SR`` and ``TDVP``.
* ``Pinv``: pseudo-inverse from the eigendecomposition of :math:`A`, with a sharp cutoff of the
  small eigenvalues. It is the default solver of ``MinSR``.
* ``CG``: conjugate gradient. It is matrix-free: the optimizer passes the function
  :math:`v\mapsto Av` instead of the matrix.

Usage
^^^^^

::

    H = sum(-op.SigmaZ(l) * op.SigmaZ((l + 1) % L) - g * op.SigmaX(l) for l in range(L))
    energy = jVMC_exp.objective_function.Observable(H)

    # Real-time evolution from Monte Carlo samples: modes with a signal-to-noise ratio
    # below about 2 are suppressed
    solver = jVMC_exp.solver.PinvSNR(snr_tol=2, pinv_cutoff=1e-8)
    opt = jVMC_exp.optimizer.TDVP(sampler, psi, make_real=False, solver=solver)
    new_parameters, dt = opt.step(0., jVMC_exp.stepper.Euler(1e-3), energy)
    opt.meta_data["pinv_cutoff"], opt.meta_data["residual"], opt.meta_data["snr"]

    # Ground state search with the matrix-free conjugate gradient
    opt = jVMC_exp.optimizer.SR(sampler, psi, diagonalShift=1e-3, solver=jVMC_exp.solver.CG(tol=1e-8))

    # A solver can also be called directly
    x, info = jVMC_exp.solver.Pinv(pinv_cutoff=1e-10)(A, b)

The optimizer keeps the dictionary ``info`` of the last solve as its ``meta_data``. Its entries
are listed in the description of each solver; ``CG`` returns an empty dictionary.

The dense solvers get :math:`A` sharded across the devices along its rows, with rows and columns
of zero padding at the end (see :ref:`Optimizer`). They diagonalize it with
``jVMC_exp.solver.util.diagonalize()``, by default with ``jax.numpy.linalg.eigh`` on the devices
of :math:`A`. On several GPUs, ``diagonalization_mode="distributed"`` diagonalizes the matrix
sharded in blocks with the ``jaxmg`` package, which needs one process per GPU
(``JVMC_USE_DISTRIBUTED=true``, see :ref:`GlobalDefs`). The eigenvectors keep the padding and the
sharding of the diagonalization.

Solvers
^^^^^^^

.. autoclass:: jVMC_exp.solver.PinvSNR
    :members:
    :special-members: __call__
    :show-inheritance:

.. autoclass:: jVMC_exp.solver.Pinv
    :members:
    :special-members: __call__
    :show-inheritance:

.. autoclass:: jVMC_exp.solver.CG
    :members:
    :special-members: __call__
    :show-inheritance:

Base class
^^^^^^^^^^

.. autoclass:: jVMC_exp.solver.base.AbstractSolver
    :members:
    :private-members: _needs_dense_matrix
    :special-members: __call__

Diagonalization
^^^^^^^^^^^^^^^

.. autofunction:: jVMC_exp.solver.util.diagonalize

.. autofunction:: jVMC_exp.solver.util.smooth_cutoff_fn
