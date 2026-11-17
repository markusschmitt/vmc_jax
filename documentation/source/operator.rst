.. _Operator:

Operator module
===============

The operator module provides the operators whose expectation values are estimated with
variational Monte Carlo. For an operator :math:`\hat O` and a variational state :math:`\psi`, the
key quantity is the local estimator

.. math:: O_{loc}(s) = \sum_{s'} O_{s,s'}\frac{\psi(s')}{\psi(s)},

whose average over :math:`|\psi(s)|^2` is the expectation value :math:`\langle\hat O\rangle`. Every
operator implements it as ``get_O_loc(s, psi)``, and samplers evaluate operators through it, e.g.,
as ``sampler(H)``.

The module has two parts: ``jVMC_exp.operator.discrete`` for computational basis configurations of
lattice models, and ``jVMC_exp.operator.continuous`` for particles in continuous space.

.. autoclass:: jVMC_exp.operator.base.AbstractOperator
    :members:

Discrete operators
^^^^^^^^^^^^^^^^^^

Operators on lattices are built from branch-free elementary operators, which act on a single site
and map each local state to a single local state. They are combined with ``+``, ``-``, ``*`` and
``/``, and multiplied by numbers. For example, the Hamiltonian of the transverse-field Ising chain
of length ``L`` with periodic boundary conditions,

.. math:: \hat H=-\sum_{l=0}^{L-1}\hat\sigma_l^z\hat\sigma_{l+1}^z-g\sum_{l=0}^{L-1}\hat\sigma_l^x,

is::

    import jVMC_exp.operator.discrete as op

    H = sum(-op.SigmaZ(l) * op.SigmaZ((l + 1) % L) - g * op.SigmaX(l) for l in range(L))

    O_loc = H.get_O_loc(s, psi)

Sites are indexed in the flattened configuration, so that site ``l`` of a two-dimensional lattice
of shape ``(L, L)`` is ``s.ravel()[l]``.

Prefactors can also be callables that accept arbitrary keyword arguments. They are evaluated with
the keyword arguments given when the operator is used, which allows, e.g., time-dependent
Hamiltonians::

    h = lambda **kwargs: jnp.cos(kwargs["t"])
    H = sum(-op.SigmaZ(l) * op.SigmaZ((l + 1) % L) + h * op.SigmaX(l) for l in range(L))

    O_loc = H.get_O_loc(s, psi, t=0.5)

Fermionic operators include the sign of a Jordan-Wigner string. Spinless fermions occupy sites with
local dimension 2, and spinful fermions sites with local dimension 4, whose local states 0, 1, 2
and 3 stand for an empty site, a spin up, a spin down and a doubly occupied site. For example, the
Hubbard model on a ring is::

    H = sum(
        -t * (op.Creation(i, spin) * op.Annihilation(j, spin)
              + op.Creation(j, spin) * op.Annihilation(i, spin))
        for i, j in ((l, (l + 1) % L) for l in range(L))
        for spin in ("up", "down")
    ) + U * sum(op.Number(l, "up") * op.Number(l, "down") for l in range(L))

For small systems, ``H.to_sparse(basis)`` and ``H.to_dense(basis)`` return the matrix of the
operator in an ordered basis, e.g., to compare with exact diagonalization.

.. autoclass:: jVMC_exp.operator.discrete.base.Operator
    :members:

.. autoclass:: jVMC_exp.operator.discrete.branch_free.Operator
    :members: idx, fermionic, spin, diag, mat_els, map
    :show-inheritance:

Elementary operators
--------------------

.. autoclass:: jVMC_exp.operator.discrete.branch_free.SigmaX
.. autoclass:: jVMC_exp.operator.discrete.branch_free.SigmaY
.. autoclass:: jVMC_exp.operator.discrete.branch_free.SigmaZ
.. autoclass:: jVMC_exp.operator.discrete.branch_free.SigmaPlus
.. autoclass:: jVMC_exp.operator.discrete.branch_free.SigmaMinus
.. autofunction:: jVMC_exp.operator.discrete.branch_free.Creation
.. autofunction:: jVMC_exp.operator.discrete.branch_free.Annihilation
.. autofunction:: jVMC_exp.operator.discrete.branch_free.Number
.. autoclass:: jVMC_exp.operator.discrete.branch_free.IdentityOperator

Combined operators
------------------

These classes are created by the arithmetic operations on operators.

.. autoclass:: jVMC_exp.operator.discrete.branch_free.CompositeOperator
.. autoclass:: jVMC_exp.operator.discrete.branch_free.ScaledOperator

Continuous operators
^^^^^^^^^^^^^^^^^^^^

Operators on particles in continuous space act on configurations that hold the positions of all
the particles, with shape ``(n_particles * n_dim,)``, and are defined on a geometry from
``jVMC_exp.geometry``. They are combined in the same way as discrete operators, and numbers can be
added to them. For example, interacting electrons in a two-dimensional harmonic trap are
described by::

    import jVMC_exp.operator.continuous as op

    geometry = jVMC_exp.geometry.HyperRectangle(N, 2, PBC=False, extent=(6., 6.))
    H = (
        op.TotalKineticOperator(geometry)
        + op.PotentialOperator(geometry, lambda x: 0.5 * omega ** 2 * jnp.sum(x ** 2))
        + op.CoulombInteraction(geometry)
    )

.. autoclass:: jVMC_exp.operator.continuous.base.Operator
    :members:

.. autoclass:: jVMC_exp.operator.continuous.non_diag_operator.TotalKineticOperator
    :members:
    :show-inheritance:

.. autoclass:: jVMC_exp.operator.continuous.diag_operator.PotentialOperator
    :members:
    :show-inheritance:

.. autoclass:: jVMC_exp.operator.continuous.diag_operator.CoulombInteraction
    :show-inheritance:

.. autoclass:: jVMC_exp.operator.continuous.diag_operator.ParticleDensity
    :show-inheritance:

.. autoclass:: jVMC_exp.operator.continuous.diag_operator.Monopole
    :show-inheritance:

.. autoclass:: jVMC_exp.operator.continuous.base.IdentityOperator
.. autoclass:: jVMC_exp.operator.continuous.base.SumOperator
.. autoclass:: jVMC_exp.operator.continuous.base.ScaledOperator
.. autoclass:: jVMC_exp.operator.continuous.base.MulOperator
