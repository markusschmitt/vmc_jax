.. _Stepper:

Stepper module
==============

.. automodule:: jVMC.stepper

With an optimizer, the stepper is passed to ``ground_state_search()`` or ``time_evolution()``
(see :ref:`Optimizer`), which make the steps and call ``update_dt()`` where needed::

    opt = jVMC.optimizer.TDVP(sampler, psi, make_real=False)
    out = opt.time_evolution(
        t_max, jVMC.objective_function.Observable(H),
        jVMC.stepper.RK23(time_step=1e-3, rtol=1e-5, atol=1e-5)
    )

    # Ground state search with a learning rate that decays from 5e-2 to 5e-3
    opt = jVMC.optimizer.SR(sampler, psi, diagonal_shift=1e-3)
    schedule = optax.exponential_decay(5e-2, transition_steps=500, decay_rate=0.1)
    out = opt.ground_state_search(500, energy, jVMC.stepper.Euler(schedule))

The keyword arguments of these methods go to ``stepper.step()``: the adaptive steppers take
``norm_function`` and pass the others on to the right-hand side.

Steppers with a fixed time step
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: jVMC.stepper.Euler
    :members:
    :show-inheritance:

.. autoclass:: jVMC.stepper.Heun
    :members:
    :show-inheritance:

Steppers with an adaptive time step
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: jVMC.stepper.AdaptiveHeun
    :members:
    :show-inheritance:

.. autoclass:: jVMC.stepper.RK23
    :members:
    :show-inheritance:

.. autoclass:: jVMC.stepper.RK45
    :members:
    :show-inheritance:

Base classes
^^^^^^^^^^^^

.. autoclass:: jVMC.stepper.AbstractStepper
    :members:

.. autoclass:: jVMC.stepper.AbstractAdaptiveStepper
    :members:
    :show-inheritance:
