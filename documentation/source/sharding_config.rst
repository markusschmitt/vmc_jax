.. _ShardingConfig:

Sharding module
===============

.. automodule:: jVMC.sharding_config

See :ref:`Parallelism` for an overview.

Meshes and shardings
^^^^^^^^^^^^^^^^^^^^

.. autodata:: jVMC.sharding_config.MESH
    :no-value:
.. autodata:: jVMC.sharding_config.DEVICE_SHARDING
    :no-value:
.. autodata:: jVMC.sharding_config.REPLICATED_SHARDING
    :no-value:
.. autodata:: jVMC.sharding_config.DEVICE_SPEC
    :no-value:
.. autodata:: jVMC.sharding_config.REPLICATED_SPEC
    :no-value:
.. autodata:: jVMC.sharding_config.MESH_2D
    :no-value:
.. autodata:: jVMC.sharding_config.DEVICE_SHARDING_2D
    :no-value:
.. autodata:: jVMC.sharding_config.DEVICE_SPEC_2D
    :no-value:
.. autofunction:: jVMC.sharding_config.make_2d_mesh

Sharded methods
^^^^^^^^^^^^^^^

.. autoclass:: jVMC.sharding_config.sharded
.. autoclass:: jVMC.sharding_config.sharded_python
.. autoclass:: jVMC.sharding_config.sharded_compiled

Batches
^^^^^^^

.. autoclass:: jVMC.sharding_config.BatchLayout
    :members:
.. autoclass:: jVMC.sharding_config.SizedIterable

Helper functions
^^^^^^^^^^^^^^^^

.. autofunction:: jVMC.sharding_config.pad_to_devices
.. autofunction:: jVMC.sharding_config.distribute
.. autofunction:: jVMC.sharding_config.broadcast_split_key
