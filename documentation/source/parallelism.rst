.. _Parallelism:

Parallelism
-----------
`jVMC` distributes its computations with JAX sharding: the same program runs on all the available
devices, on one or several processes, and the arrays are split across the devices. The arrays
passed to or obtained from the `jVMC` API have the same shape whatever the number of devices:
configurations have the shape ``(N, *sampleShape)``, wave function coefficients ``(N,)`` and
gradients ``(N, P)``, with ``N`` the number of samples. The tools used for this are in
``jVMC_exp.sharding_config`` (see :ref:`ShardingConfig`).

Devices
^^^^^^^
All the devices form a one-dimensional mesh, with the axis ``"devices"``. Arrays with one entry
per sample are sharded along their first axis, so that every device holds a block of
``N / n_devices`` consecutive samples, while the network parameters are replicated on every
device. Samplers therefore round the number of chains up to a multiple of the number of devices,
and the number of samples up to a multiple of the number of chains, with a message.

**Example:**

    With 4 devices, ``numChains=50`` and ``numSamples=1000``, the sampler uses 52 chains and
    returns 1040 configurations of a system of size :math:`L=7`, sharded along the first axis:

    >>> configs, logPsi, weights = sampler.sample()
    >>> configs.shape, configs.sharding.spec
    ((1040, 7), PartitionSpec('devices',))
    >>> psi.gradients(configs).shape
    (1040, 28)

    Several devices can be simulated on a CPU to try this out:

    .. code-block:: bash

        XLA_FLAGS=--xla_force_host_platform_device_count=4 JAX_PLATFORMS=cpu python script.py

Batching
^^^^^^^^
Wave function coefficients, gradients and local estimators are computed in batches of
``batchSize`` samples, an argument of ``NQS``, which bounds the memory used at once and fixes the
shapes of the compiled functions whatever the number of samples. A batch is split evenly between
the devices, so ``batchSize`` has to be a multiple of the number of devices. To avoid moving data
between devices, a batch is made of a block of the samples of every device: on more than one
device, it does not consist of consecutive samples (see ``BatchLayout``).

Several processes
^^^^^^^^^^^^^^^^^
To run on several processes, e.g., one per GPU or one per node of a cluster, set the environment
variable ``JVMC_USE_DISTRIBUTED=true`` or call ``jax.distributed.initialize()`` before importing
the library. Without arguments, ``jax.distributed.initialize()`` works on the clusters that JAX
detects, e.g., with SLURM; otherwise it has to be called with the address of the coordinator, the
number of processes and the index of the process.

The same script then runs on every process, and all the devices of all the processes form a
single mesh: the samples are distributed across the devices of all the processes. Random keys are
split on process 0 and sent to the other processes, so that all of them work with the same keys.
An array with one entry per sample is spread over the processes, and
``jax.experimental.multihost_utils.process_allgather`` collects it on every process.
