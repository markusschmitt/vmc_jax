.. _Symmetry:

Symmetry module
===============

A symmetry projector restricts a variational wave function to a symmetry sector. It holds the
elements :math:`g` of a symmetry group, which map a configuration :math:`s` to :math:`g\cdot s`,
together with their characters :math:`\chi_g`, and the projected wave function is

.. math::

    \psi_P(s)=\frac{1}{n_\text{symm}}\sum_g \chi_g\,\sigma_g(s)\,\psi(g\cdot s),

where the sign :math:`\sigma_g(s)` is one for spins and bosons, and contains the fermionic signs
of the transformation for fermions, in the mode order of the fermionic operators of
:ref:`Operator`.

The functions of ``jVMC_exp.symmetry.lattice_symetries`` build the projectors of translations
with a given momentum, of the point groups, of the spin flip and of the particle-hole
transformation. Projectors are composed with ``*``, and passed to ``NQS`` as ``orbit``::

    from jVMC_exp.symmetry import lattice_symetries as ls

    symm = (
        ls.square_translation_symmetry(L, L, "spin")
        * ls.square_d4_symmetry(L, L, "spin")
        * ls.spin_flip_symmetry(L, L)
    )
    psi = jVMC_exp.vqs.NQS(net, L * L, batchSize=2**10, seed=1234, orbit=symm)

``symm * net`` returns the projected network, a ``ProjectedOrbitNet``, which evaluates the network
on all the ``symm.nsymm`` transformed configurations. The cost of an evaluation and its memory
therefore grow with the number of elements, and ``NQS`` collects the gradients of the projected
network in blocks of ``batchSize * nsymm`` rows (see :ref:`VQS`).

By default the coefficients are combined as above (``symmetry_average="exp"``). The other averages
of ``ProjectedOrbitNet`` average the logarithmic coefficients, or the probabilities and phases
separately. The latter, ``"sep"``, keeps a normalized network normalized, so that networks that
sample directly, such as autoregressive networks, can be projected on sectors whose weights are
all one.

Projectors
^^^^^^^^^^

.. autoclass:: jVMC_exp.symmetry.projector.SymmetryProjector
    :members:
    :special-members: __mul__

.. autofunction:: jVMC_exp.symmetry.projector.compose_projectors

Lattice symmetries
^^^^^^^^^^^^^^^^^^

.. automodule:: jVMC_exp.symmetry.lattice_symetries
    :members:

Projected networks
^^^^^^^^^^^^^^^^^^

.. autoclass:: jVMC_exp.symmetry.projector.ProjectedOrbitNet
    :members: __call__, sample

.. autofunction:: jVMC_exp.symmetry.average_fun.average_coefficients

.. autofunction:: jVMC_exp.symmetry.average_fun.avg_coefficients_exp

.. autofunction:: jVMC_exp.symmetry.average_fun.avg_coefficients_log

.. autofunction:: jVMC_exp.symmetry.average_fun.avg_coefficients_sep

Helpers
^^^^^^^

.. autofunction:: jVMC_exp.symmetry.projector.split_spinful_site_state

.. autofunction:: jVMC_exp.symmetry.util.deduplicate_symmetry_arrays

.. autofunction:: jVMC_exp.symmetry.util.normalize_particle_type
