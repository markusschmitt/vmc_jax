.. _Optimizer:

Optimizer module
================

An optimizer changes the parameters of a variational state with the force of an objective
function (see :ref:`ObjectiveFunction`):

* ``SR``: stochastic reconfiguration, i.e. imaginary time evolution, for ground state searches,
* ``MinSR``: the same update as ``SR``, computed with a matrix over the samples instead of the
  parameters,
* ``Adam``: the Adam optimizer on the gradient of the objective function,
* ``TDVP``: real-time evolution with the time-dependent variational principle.

``SR`` and ``TDVP`` solve the equation :math:`S\dot\theta=cF` with the quantum geometric tensor
:math:`S` and the force :math:`F`, with a solver of ``jVMC_exp.solver``. A stepper of
``jVMC_exp.stepper`` (see :ref:`Stepper`) integrates the update :math:`\dot\theta`: its time step is the learning rate
of a ground state search, and the time step of a time evolution.

Usage
^^^^^

::

    H = sum(-op.SigmaZ(l) * op.SigmaZ((l + 1) % L) - g * op.SigmaX(l) for l in range(L))
    energy = jVMC_exp.objective_function.Observable(H)

    # Ground state search
    opt = jVMC_exp.optimizer.SR(sampler, psi, diagonalShift=1e-3)
    out = opt.ground_state_search(500, energy, stepper=jVMC_exp.stepper.Euler(5e-2))
    out["energy"]["mean"]                   # Mean of the local energies at every step

    # Real-time evolution with an adaptive time step, measuring an observable
    opt = jVMC_exp.optimizer.TDVP(sampler, psi, make_real=False, diagonalShift=0.)
    out = opt.time_evolution(
        t_max, jVMC_exp.objective_function.Observable(H_quench),
        jVMC_exp.stepper.AdaptiveHeun(timeStep=1e-3, rtol=1e-5, atol=1e-5),
        observables={"ZZ": ZZ}
    )
    out["times"], out["ZZ"]["mean"]

``ground_state_search()`` and ``time_evolution()`` return the data recorded by the
``OutputManager`` of the optimizer: the times (the step numbers in a ground state search), the
mean, variance and statistical error of the local losses in ``"energy"``, and the same for every
observable. At every step the samples are drawn once and used for the update and the
measurements. The time evolution of a time-dependent Hamiltonian gets the time ``t`` as a keyword
argument of the operators.

Optimizers
^^^^^^^^^^

.. autoclass:: jVMC_exp.optimizer.SR
    :show-inheritance:

.. autoclass:: jVMC_exp.optimizer.TDVP
    :members: cross_validation
    :show-inheritance:

.. autoclass:: jVMC_exp.optimizer.MinSR
    :members:
    :show-inheritance:

.. autoclass:: jVMC_exp.optimizer.Adam
    :members:
    :show-inheritance:

Base classes
^^^^^^^^^^^^

.. autoclass:: jVMC_exp.optimizer.base.AbstractOptimizer
    :members:
    :special-members: __call__

.. autoclass:: jVMC_exp.optimizer.base.Evolution
    :members:
    :show-inheritance:
