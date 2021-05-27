..

MPI wrapper module
==================

The ``jVMC.mpi_wrapper`` module wraps typically required MPI communications, for which the `mpi4py` package is used.
This means especially the statistical evaluation of
Monte Carlo samples. For this purpose, it can interact with the sampler classes from the ``jVMC.sampler`` module as follows:

**Example:**

    Assuming that ``sampler`` is an instance of a sampler class, ``psi`` is a variational quantum state,
    and ``op`` is an instance of a class derived from the 
    ``Operator`` class, associated with a quantum operator :math:`\hat O`. Then we can get samples
    and the corresponding :math:`O_{loc}(\mathbf s)` via
    
    >>> s, logPsi, _ = sampler.sample()
    >>> sPrime, _ = op.get_s_primes(sampleConfigs)
    >>> logPsiOffd = psi(sPrime)
    >>> Oloc = get_O_loc(logPsi, logPsiOffd)

    Now, on each MPI process ``Oloc`` is a two-dimensional array of size (number of devices) :math:`\times` (number of samples
    per device). To get, for example, the Monte Carlo estimate of the expectation value of :math:`\hat O`,

        :math:`\langle\hat O\rangle\approx\frac{1}{N_S}\sum_{j=1}^{N_S}O_{loc}(\mathbf s_j)`

    we can use the ``jVMC.mpi_wrapper.get_global_mean`` function

    >>> Omean = jVMC.mpi_wrapper.get_global_mean(Oloc)

    Thereby, we obtain the mean computed across all MPI processes and local devices.

.. automodule:: jVMC.mpi_wrapper
    :members:
