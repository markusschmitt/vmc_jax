`jVMC`: Versatile and performant variational Monte Carlo
========================================================

`jVMC <https://github.com/computational-quantum-science/jvmc-2.0>`_ is a Python library for
variational Monte Carlo with neural quantum states, built on
`JAX <https://docs.jax.dev>`_ and `Flax <https://flax.readthedocs.io>`_. It uses automatic
differentiation and just-in-time compilation, and runs on one or many GPUs without changes to
the user code.

A calculation is a chain of independent components: a network, wrapped into a variational state
(:ref:`VQS`), a sampler (:ref:`Sampler`), an operator (:ref:`Operator`), an objective function,
and an optimizer together with a solver and a stepper::

    import jVMC_exp
    import jVMC_exp.operator.discrete as op

    L = 10
    H = sum(-op.SigmaZ(l) * op.SigmaZ((l + 1) % L) - 0.7 * op.SigmaX(l) for l in range(L))

    psi = jVMC_exp.vqs.NQS(jVMC_exp.nets.CpxRBM(num_hidden=8), L, batch_size=2**11, seed=1)
    sampler = jVMC_exp.sampler.MCSampler(psi, jVMC_exp.propose.SpinFlip(), key=1,
                                         num_chains=2**9, num_samples=2**11)
    opt = jVMC_exp.optimizer.SR(sampler, psi, diagonal_shift=1e-3)
    out = opt.ground_state_search(100, jVMC_exp.objective_function.Observable(H),
                                  jVMC_exp.stepper.Euler(5e-2))

See :ref:`Examples` for complete applications, and :doc:`migration` if you come from jVMC 1.x.

.. toctree::
    :maxdepth: 1
    :caption: Getting started

    installation
    examples
    migration

.. toctree::
    :maxdepth: 1
    :caption: Design choices

    parallelism

.. toctree::
    :maxdepth: 1
    :caption: API documentation

    vqs
    nets
    symmetry
    operator
    geometry
    sampler
    stats
    objective_function
    optimizer
    solver
    stepper
    util
    global_defs
    sharding_config
