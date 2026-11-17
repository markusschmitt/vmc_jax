.. _ShardingConfig:

Sharding module
===============

.. automodule:: jVMC_exp.sharding_config

See :ref:`Parallelism` for an overview.

Meshes and shardings
^^^^^^^^^^^^^^^^^^^^

.. autodata:: jVMC_exp.sharding_config.MESH
    :no-value:
.. autodata:: jVMC_exp.sharding_config.DEVICE_SHARDING
    :no-value:
.. autodata:: jVMC_exp.sharding_config.REPLICATED_SHARDING
    :no-value:
.. autodata:: jVMC_exp.sharding_config.DEVICE_SPEC
    :no-value:
.. autodata:: jVMC_exp.sharding_config.REPLICATED_SPEC
    :no-value:
.. autodata:: jVMC_exp.sharding_config.MESH_2D
    :no-value:
.. autodata:: jVMC_exp.sharding_config.DEVICE_SHARDING_2D
    :no-value:
.. autodata:: jVMC_exp.sharding_config.DEVICE_SPEC_2D
    :no-value:
.. autofunction:: jVMC_exp.sharding_config.make_2d_mesh

Sharded methods
^^^^^^^^^^^^^^^

.. autoclass:: jVMC_exp.sharding_config.sharded
.. autoclass:: jVMC_exp.sharding_config.sharded_python
.. autoclass:: jVMC_exp.sharding_config.sharded_compiled

Batches
^^^^^^^

.. autoclass:: jVMC_exp.sharding_config.BatchLayout
    :members:
.. autoclass:: jVMC_exp.sharding_config.SizedIterable

Helper functions
^^^^^^^^^^^^^^^^

.. autofunction:: jVMC_exp.sharding_config.pad_to_devices
.. autofunction:: jVMC_exp.sharding_config.distribute
.. autofunction:: jVMC_exp.sharding_config.broadcast_split_key
