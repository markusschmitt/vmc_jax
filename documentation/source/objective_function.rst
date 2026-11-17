.. _ObjectiveFunction:

Objective function module
=========================

An objective function is a loss :math:`L(\theta)` of the variational state, estimated from the
samples of a sampler as the mean of a local loss,
:math:`L(\theta)=\langle\ell_\theta\rangle_{p_\theta}` with
:math:`p_\theta(s)\propto|\psi_\theta(s)|^2`. Its ``value_and_grad()`` method returns an
``ObjectiveFunctionOutput`` with

* the local losses ``o_loc``, a ``SampledObs`` (see :ref:`Stats`), whose ``mean`` is the loss,
* the logarithmic derivatives ``grad_log_psi``,
  :math:`\Gamma_k(s)=\partial_{\theta_k}\ln\psi_\theta(s)`,
* the force ``grad``, :math:`F_k`, with :math:`\partial_{\theta_k}L=2\,\mathrm{Re}\,F_k`.

The optimizers take an objective function and minimize it::

    H = sum(-op.SigmaZ(l) * op.SigmaZ((l + 1) % L) - g * op.SigmaX(l) for l in range(L))
    energy = jVMC_exp.objective_function.Observable(H)

    out = energy.value_and_grad(sampler)
    out.o_loc.mean                          # Energy
    out.grad                                # Force, one entry per real parameter

    optimizer = jVMC_exp.optimizer.SR(sampler, psi)
    optimizer.ground_state_search(500, energy, stepper=jVMC_exp.stepper.Euler(1e-2))

The local losses of

* ``Observable`` are the local estimator of an operator,
* ``Estimator`` are a real function of the configuration and of the parameters,
* ``ParametricObservable`` are the local estimator of an operator whose matrix elements depend on
  the parameters,
* ``Infidelity`` give the infidelity with a reference state transformed by an operator, e.g. to fit
  a state to :math:`\hat U|\chi\rangle` for a gate :math:`\hat U`::

    U, U_dag = jVMC_exp.util.util.matrix_to_jvmc_operator(gate, (0, 1), return_conjugate=True)
    loss = jVMC_exp.objective_function.Infidelity(chi_sampler, U, U_dag, control_variate="adaptive")
    optimizer.ground_state_search(steps, loss, stepper=jVMC_exp.stepper.Euler(5e-2))

All of them use the samples and weights stored in the sampler, which the optimizers refresh at
every step. With ``batched_jacobian=True`` the logarithmic derivatives are computed one batch at a
time, for networks whose Jacobian does not fit in memory.

Objective functions
^^^^^^^^^^^^^^^^^^^

.. autoclass:: jVMC_exp.objective_function.base.Observable
    :members:
    :special-members: __call__

.. autoclass:: jVMC_exp.objective_function.base.Estimator
    :members:
    :special-members: __call__

.. autoclass:: jVMC_exp.objective_function.base.ParametricObservable
    :members:
    :special-members: __call__

.. autoclass:: jVMC_exp.objective_function.infidelity.Infidelity
    :members:
    :special-members: __call__

Interface
^^^^^^^^^

New objective functions derive from ``AbstractObjectiveFunction`` and implement ``__call__()``
and ``value_and_grad()``.

.. autoclass:: jVMC_exp.objective_function.base.AbstractObjectiveFunction
    :members:
    :special-members: __call__

.. autoclass:: jVMC_exp.objective_function.base.ObjectiveFunctionOutput

.. autoclass:: jVMC_exp.objective_function.infidelity.MovingAverage
    :members:
