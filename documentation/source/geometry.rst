.. _Geometry:

Geometry module
===============

Particles in continuous space live in a geometry, which sets the number of particles, the number of
spatial dimensions, the extent of the system and its boundary conditions. Continuous operators
(see :ref:`Operator`) and the proposers of ``MCSamplerCont`` (see :ref:`Sampler`) take the geometry
as their first argument.

A configuration holds the positions of all the particles in a flat array of shape
``(n_particles * n_dim,)``, with the coordinates of each particle next to each other. For example,
with two particles in two dimensions, a configuration is ``[x_0, y_0, x_1, y_1]``.

For example, ``N`` electrons in a two-dimensional harmonic trap are described with open boundary
conditions, and the extent sets the box where the Markov chains start::

    import jVMC_exp.operator.continuous as op

    geometry = jVMC_exp.geometry.HyperRectangle(N, 2, PBC=False, extent=(6., 6.))
    H = (
        op.TotalKineticOperator(geometry)
        + op.PotentialOperator(geometry, lambda x: 0.5 * omega ** 2 * jnp.sum(x ** 2))
        + op.CoulombInteraction(geometry)
    )
    psi = jVMC_exp.vqs.NQS(net, sampleShape=2 * N, batchSize=1000, seed=0)
    sampler = jVMC_exp.sampler.MCSamplerCont(psi, jVMC_exp.propose.RWM(geometry))

Geometries
^^^^^^^^^^

.. autoclass:: jVMC_exp.geometry.HyperRectangle
    :members:
    :show-inheritance:

Interface
^^^^^^^^^

New geometries derive from ``AbstractGeometry`` and implement ``domain``, ``apply_PBC()``,
``uniform_populate()`` and ``get_distance()``.

.. autoclass:: jVMC_exp.geometry.AbstractGeometry
    :members:
