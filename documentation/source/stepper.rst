.. _Stepper:

Stepper module
==============

.. automodule:: jVMC_exp.stepper

With an optimizer, the stepper is passed to ``ground_state_search()`` or ``time_evolution()``
(see :ref:`Optimizer`), which make the steps and call ``update_dt()`` where needed::

    opt = jVMC_exp.optimizer.TDVP(sampler, psi, make_real=False)
    out = opt.time_evolution(
        t_max, jVMC_exp.objective_function.Observable(H),
        jVMC_exp.stepper.RK23(timeStep=1e-3, rtol=1e-5, atol=1e-5)
    )

    # Ground state search with a learning rate that decays from 5e-2 to 5e-3
    opt = jVMC_exp.optimizer.SR(sampler, psi, diagonalShift=1e-3)
    schedule = optax.exponential_decay(5e-2, transition_steps=500, decay_rate=0.1)
    out = opt.ground_state_search(500, energy, jVMC_exp.stepper.Euler(schedule))

The keyword arguments of these methods go to ``stepper.step()``: the adaptive steppers take
``normFunction`` and pass the others on to the right-hand side.

Steppers with a fixed time step
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: jVMC_exp.stepper.Euler
    :members:
    :show-inheritance:

.. autoclass:: jVMC_exp.stepper.Heun
    :members:
    :show-inheritance:

Steppers with an adaptive time step
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: jVMC_exp.stepper.AdaptiveHeun
    :members:
    :show-inheritance:

.. autoclass:: jVMC_exp.stepper.RK23
    :members:
    :show-inheritance:

.. autoclass:: jVMC_exp.stepper.RK45
    :members:
    :show-inheritance:

Base classes
^^^^^^^^^^^^

.. autoclass:: jVMC_exp.stepper.AbstractStepper
    :members:

.. autoclass:: jVMC_exp.stepper.AbstractAdaptiveStepper
    :members:
    :show-inheritance:
