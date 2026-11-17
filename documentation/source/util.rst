.. _Util:

Utilities
=========

The ``jVMC_exp.util`` package collects helpers. Everything on this page can be imported from
``jVMC_exp.util``.

Measurements
^^^^^^^^^^^^

``measure()`` evaluates a dictionary of observables with a sampler. The optimizers use it for the
``observables`` of ``ground_state_search()`` and ``time_evolution()`` (see :ref:`Optimizer`)::

    result = jVMC_exp.util.measure(
        {"energy": H, "X0": op.SigmaX(0), "H(t)": (H_t, {"t": 0.5})}, sampler
    )
    result["energy"]["mean"], result["energy"]["MC_error"]

.. autofunction:: jVMC_exp.util.util.measure

.. autodata:: jVMC_exp.util.util.ObservableEntry
    :annotation:

.. autodata:: jVMC_exp.util.util.OperatorWithKwargs
    :annotation:

Output and input
^^^^^^^^^^^^^^^^

Every optimizer records the measurements of its steps in an ``OutputManager``. With a path, they
are written to an HDF5 file at every step, and an ``InputManager`` reads them back::

    outp = jVMC_exp.util.OutputManager("results.h5", append=False)
    opt = jVMC_exp.optimizer.SR(sampler, psi, diagonalShift=1e-3, output_manager=outp)
    opt.ground_state_search(500, energy, jVMC_exp.stepper.Euler(5e-2), observables={"X0": op.SigmaX(0)})
    outp.write_parameters(500, psi.parameters)

    inp = jVMC_exp.util.InputManager("results.h5")
    inp.data["observables"]["X0"]["mean"]               # One entry per step
    psi.parameters = inp.load_parameters(psi.parameters)

.. autoclass:: jVMC_exp.util.output_manager.OutputManager
    :members:

.. autoclass:: jVMC_exp.util.input_manager.InputManager
    :members:

Operators from matrices
^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: jVMC_exp.util.util.matrix_to_jvmc_operator

Norm for adaptive steppers
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: jVMC_exp.util.util.s_norm_fn

Internal helpers
^^^^^^^^^^^^^^^^

These functions are used by ``NQS``, the samplers and the optimizers.

Parameter layouts
-----------------

.. autofunction:: jVMC_exp.util.util.make_real_array

.. autofunction:: jVMC_exp.util.util.make_cmplx_array

.. autofunction:: jVMC_exp.util.util.remove_double

.. autofunction:: jVMC_exp.util.util.has_callable_attr

.. autofunction:: jVMC_exp.util.util.get_iterable

Gradients
---------

.. automodule:: jVMC_exp.util.grads
    :members:

PRNG keys
---------

.. autofunction:: jVMC_exp.util.key_gen.format_key

.. autofunction:: jVMC_exp.util.key_gen.generate_seed
