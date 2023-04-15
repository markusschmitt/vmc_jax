import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
from jax import vmap, jit

import jVMC.mpi_wrapper as mpi

from functools import partial

import time

import jVMC.global_defs as global_defs


def propose_spin_flip(key, s, info):
    idx = random.randint(key, (1,), 0, s.size)[0]
    idx = jnp.unravel_index(idx, s.shape)
    update = (s[idx] + 1) % 2
    return s.at[idx].set(update)


def propose_POVM_outcome(key, s, info):
    key, subkey = random.split(key)
    idx = random.randint(subkey, (1,), 0, s.size)[0]
    idx = jnp.unravel_index(idx, s.shape)
    update = (s[idx] + random.randint(key, (1,), 0, 3) % 4)
    return s.at[idx].set(update)


def propose_spin_flip_Z2(key, s, info):
    idxKey, flipKey = jax.random.split(key)
    idx = random.randint(idxKey, (1,), 0, s.size)[0]
    idx = jnp.unravel_index(idx, s.shape)
    update = (s[idx] + 1) % 2
    s = s.at[idx].set(update)
    # On average, do a global spin flip every 30 updates to
    # reflect Z_2 symmetry
    doFlip = random.randint(flipKey, (1,), 0, 5)[0]
    return jax.lax.cond(doFlip == 0, lambda x: 1 - x, lambda x: x, s)


def propose_spin_flip_zeroMag(key, s, info):
    # propose spin flips that stay in the zero magnetization sector

    idxKeyUp, idxKeyDown, flipKey = jax.random.split(key, num=3)

    # can't use jnp.where because then it is not jit-compilable
    # find indices based on cumsum
    bound_up = jax.random.randint(idxKeyUp, (1,), 1, s.shape[0] * s.shape[1] / 2 + 1)
    bound_down = jax.random.randint(idxKeyDown, (1,), 1, s.shape[0] * s.shape[1] / 2 + 1)

    id_up = jnp.searchsorted(jnp.cumsum(s), bound_up)
    id_down = jnp.searchsorted(jnp.cumsum(1 - s), bound_down)

    idx_up = jnp.unravel_index(id_up, s.shape)
    idx_down = jnp.unravel_index(id_down, s.shape)

    s = s.at[idx_up[0], idx_up[1]].set(0)
    s = s.at[idx_down[0], idx_down[1]].set(1)

    # On average, do a global spin flip every 30 updates to
    # reflect Z_2 symmetry
    doFlip = random.randint(flipKey, (1,), 0, 5)[0]
    return jax.lax.cond(doFlip == 0, lambda x: 1 - x, lambda x: x, s)


class MCSampler:
    """A sampler class.

    This class provides functionality to sample computational basis states from \
    the probability distribution induced by the variational wave function, \
    :math:`|\\psi(s)|^2`.

    Sampling is automatically distributed accross MPI processes and locally available \
    devices.

    Initializer arguments:
        * ``net``: Network defining the probability distribution.
        * ``sampleShape``: Shape of computational basis configurations.
        * ``key``: An instance of ``jax.random.PRNGKey``. Alternatively, an ``int`` that will be used \
                   as seed to initialize a ``PRNGKey``.
        * ``updateProposer``: A function to propose updates for the MCMC algorithm. \
        It is called as ``updateProposer(key, config, **kwargs)``, where ``key`` is an instance of \
        ``jax.random.PRNGKey``, ``config`` is a computational basis configuration, and ``**kwargs`` \
        are optional additional arguments.
        * ``numChains``: Number of Markov chains, which are run in parallel.
        * ``updateProposerArg``: An optional argument that will be passed to the ``updateProposer`` \
        as ``kwargs``.
        * ``numSamples``: Default number of samples to be returned by the ``sample()`` member function.
        * ``thermalizationSweeps``: Number of sweeps to perform for thermalization of the Markov chain.
        * ``sweepSteps``: Number of proposed updates per sweep.
        * ``mu``: exponent, giving the probability of a configuration. \
        The usual choice is :math: `|\psi(s)|^2`, i.e. mu=2.
    """

    def __init__(self, net, sampleShape, key, updateProposer=None, numChains=1, updateProposerArg=None,
                 numSamples=100, thermalizationSweeps=10, sweepSteps=10, initState=None, mu=2):
        """Initializes the MCSampler class.

        """

        self.sampleShape = sampleShape

        self.net = net
        if (not net.is_generator) and (updateProposer is None):
            raise RuntimeError("Instantiation of MCSampler: `updateProposer` is `None` and cannot be used for MCMC sampling.")

        stateShape = (global_defs.device_count(), numChains) + sampleShape
        if initState is None:
            initState = jnp.zeros(sampleShape, dtype=np.int32)
        self.states = jnp.stack([initState] * (global_defs.device_count() * numChains), axis=0).reshape(stateShape)

        # Make sure that net is initialized
        self.net(self.states)

        self.mu = mu
        if mu < 0 or mu > 2:
            raise ValueError("mu must be in the range [0, 2]")
        self.updateProposer = updateProposer
        self.updateProposerArg = updateProposerArg

        if isinstance(key, jax.lib.xla_extension.DeviceArray):
            self.key = key
        else:
            self.key = jax.random.PRNGKey(key)
        self.key = jax.random.split(self.key, mpi.commSize)[mpi.rank]
        self.key = jax.random.split(self.key, global_defs.device_count())
        self.thermalizationSweeps = thermalizationSweeps
        self.sweepSteps = sweepSteps
        self.numSamples = numSamples

        shape = (global_defs.device_count(),) + (1,)
        self.numProposed = jnp.zeros(shape, dtype=np.int64)
        self.numAccepted = jnp.zeros(shape, dtype=np.int64)

        self.numChains = numChains

        # jit'd member functions
        self._get_samples_jitd = {}  # will hold a jit'd function for each number of samples
        self._randomize_samples_jitd = {}  # will hold a jit'd function for each number of samples

    def set_number_of_samples(self, N):
        """Set default number of samples.

        Arguments:
            * ``N``: Number of samples.
        """

        self.numSamples = N

    def set_random_key(self, key):
        """Set key for pseudo random number generator.

        Args:
            * ``key``: Key (jax.random.PRNGKey)
        """

        self.key = jax.random.split(key, global_defs.device_count())

    def get_last_number_of_samples(self):
        """Return last number of samples.

        This function is required, because the actual number of samples might \
        exceed the requested number of samples when sampling is distributed \
        accross multiple processes or devices.

        Returns:
            Number of samples generated by last call to ``sample()`` member function.
        """
        return mpi.globNumSamples

    def sample(self, parameters=None, numSamples=None, multipleOf=1):
        """Generate random samples from wave function.

        If supported by ``net``, direct sampling is peformed. Otherwise, MCMC is run \
        to generate the desired number of samples. For direct sampling the real part \
        of ``net`` needs to provide a ``sample()`` member function that generates \
        samples from :math:`|\\psi(s)|^2`.

        Sampling is automatically distributed accross MPI processes and available \
        devices. In that case the number of samples returned might exceed ``numSamples``.

        Arguments:
            * ``parameters``: Network parameters to use for sampling.
            * ``numSamples``: Number of samples to generate. When running multiple processes \
            or on multiple devices per process, the number of samples returned is \
            ``numSamples`` or more. If ``None``, the default number of samples is returned \
            (see ``set_number_of_samples()`` member function).
            * ``multipleOf``: This argument allows to choose the number of samples returned to \
            be the smallest multiple of ``multipleOf`` larger than ``numSamples``. This feature \
            is useful to distribute a total number of samples across multiple processors in such \
            a way that the number of samples per processor is identical for each processor.

        Returns:
            A sample of computational basis configurations drawn from :math:`|\\psi(s)|^2`.
        """

        if numSamples is None:
            numSamples = self.numSamples

        if self.net.is_generator:
            configs = self._get_samples_gen(parameters, numSamples, multipleOf)
            return configs, self.net(configs), jnp.ones(configs.shape[:2]) / jnp.prod(jnp.asarray(configs.shape[:2]))

        configs, logPsi = self._get_samples_mcmc(parameters, numSamples, multipleOf)
        p = jnp.exp((2 - self.mu) * jnp.real(logPsi))
        return configs, logPsi, p / mpi.global_sum(p)

    def _randomize_samples(self, samples, key, orbit):
        """ For a given set of samples apply a random symmetry transformation to each sample
        """
        orbit_indices = random.choice(key, orbit.shape[0], shape=(samples.shape[0],))
        samples = samples * 2 - 1
        return jax.vmap(lambda o, idx, s: (o[idx].dot(s.ravel()).reshape(s.shape) + 1) // 2, in_axes=(None, 0, 0))(orbit, orbit_indices, samples)

    def _get_samples_gen(self, params, numSamples, multipleOf=1):

        numSamples = mpi.distribute_sampling(numSamples, localDevices=global_defs.device_count(), numChainsPerDevice=multipleOf)

        tmpKeys = random.split(self.key[0], 3 * global_defs.device_count())
        self.key = tmpKeys[:global_defs.device_count()]
        tmpKey = tmpKeys[global_defs.device_count():2 * global_defs.device_count()]
        tmpKey2 = tmpKeys[2 * global_defs.device_count():]

        samples = self.net.sample(numSamples, tmpKey, parameters=params)

        if not str(numSamples) in self._randomize_samples_jitd:
            self._randomize_samples_jitd[str(numSamples)] = global_defs.pmap_for_my_devices(self._randomize_samples, static_broadcasted_argnums=(), in_axes=(0, 0, None))

        return self._randomize_samples_jitd[str(numSamples)](samples, tmpKey2, self.net.net.orbit.orbit)

    def _get_samples_mcmc(self, params, numSamples, multipleOf=1):

        net, netParams = self.net.get_sampler_net()

        if params is None:
            params = netParams

        # Initialize sampling stuff
        self._mc_init(netParams)

        numSamples = mpi.distribute_sampling(numSamples, localDevices=global_defs.device_count(), numChainsPerDevice=np.lcm(self.numChains, multipleOf))
        numSamplesStr = str(numSamples)

        # check whether _get_samples is already compiled for given number of samples
        if not numSamplesStr in self._get_samples_jitd:
            self._get_samples_jitd[numSamplesStr] = global_defs.pmap_for_my_devices(partial(self._get_samples, sweepFunction=partial(self._sweep, net=net)),
                                                                                    static_broadcasted_argnums=(1, 2, 3, 9, 11),
                                                                                    in_axes=(None, None, None, None, 0, 0, 0, 0, 0, None, None, None))

        (self.states, self.log_accProb, self.key, self.numProposed, self.numAccepted), configs =\
            self._get_samples_jitd[numSamplesStr](params, numSamples, self.thermalizationSweeps, self.sweepSteps,
                                                  self.states, self.log_accProb, self.key, self.numProposed, self.numAccepted,
                                                  self.updateProposer, self.updateProposerArg, self.sampleShape)

        # return configs, None
        return configs, self.net(configs)

    def _get_samples(self, params, numSamples,
                     thermSweeps, sweepSteps,
                     states, log_accProb, key,
                     numProposed, numAccepted,
                     updateProposer, updateProposerArg,
                     sampleShape, sweepFunction=None):

        # Thermalize
        states, log_accProb, key, numProposed, numAccepted =\
            sweepFunction(states, log_accProb, key, numProposed, numAccepted, params, thermSweeps * sweepSteps, updateProposer, updateProposerArg)

        # Collect samples
        def scan_fun(c, x):

            states, log_accProb, key, numProposed, numAccepted =\
                sweepFunction(c[0], c[1], c[2], c[3], c[4], params, sweepSteps, updateProposer, updateProposerArg)

            return (states, log_accProb, key, numProposed, numAccepted), states

        meta, configs = jax.lax.scan(scan_fun, (states, log_accProb, key, numProposed, numAccepted), None, length=numSamples)

        # return meta, configs.reshape((configs.shape[0]*configs.shape[1], -1))
        return meta, configs.reshape((configs.shape[0] * configs.shape[1],) + sampleShape)

    def _sweep(self, states, log_accProb, key, numProposed, numAccepted, params, numSteps, updateProposer, updateProposerArg, net=None):

        def perform_mc_update(i, carry):

            # Generate update proposals
            newKeys = random.split(carry[2], carry[0].shape[0] + 1)
            carryKey = newKeys[-1]
            newStates = vmap(updateProposer, in_axes=(0, 0, None))(newKeys[:len(carry[0])], carry[0], updateProposerArg)

            # Compute acceptance probabilities
            new_log_accProb = jax.vmap(lambda y: self.mu * jnp.real(net(params, y)), in_axes=(0,))(newStates)
            P = jnp.exp(new_log_accProb - carry[1])

            # Roll dice
            newKey, carryKey = random.split(carryKey,)
            accepted = random.bernoulli(newKey, P).reshape((-1,))

            # Bookkeeping
            numProposed = carry[3] + len(newStates)
            numAccepted = carry[4] + jnp.sum(accepted)

            # Perform accepted updates
            def update(acc, old, new):
                return jax.lax.cond(acc, lambda x: x[1], lambda x: x[0], (old, new))
            carryStates = vmap(update, in_axes=(0, 0, 0))(accepted, carry[0], newStates)

            carryLog_accProb = jnp.where(accepted == True, new_log_accProb, carry[1])

            return (carryStates, carryLog_accProb, carryKey, numProposed, numAccepted)

        (states, log_accProb, key, numProposed, numAccepted) =\
            jax.lax.fori_loop(0, numSteps, perform_mc_update, (states, log_accProb, key, numProposed, numAccepted))

        return states, log_accProb, key, numProposed, numAccepted

    def _mc_init(self, netParams):

        # Initialize log_accProb
        net, _ = self.net.get_sampler_net()
        self.log_accProb = global_defs.pmap_for_my_devices(
            lambda x: jax.vmap(lambda y: self.mu * jnp.real(net(netParams, y)), in_axes=(0,))(x)
        )(self.states)

        shape = (global_defs.device_count(),) + (1,)

        self.numProposed = jnp.zeros(shape, dtype=np.int64)
        self.numAccepted = jnp.zeros(shape, dtype=np.int64)

    def acceptance_ratio(self):
        """Get acceptance ratio.

        Returns:
            Acceptance ratio observed in the last call to ``sample()``.
        """

        numProp = mpi.global_sum(self.numProposed)
        if numProp > 0:
            return mpi.global_sum(self.numAccepted) / numProp

        return jnp.array([0.])

# ** end class Sampler


class ExactSampler:
    """Class for full enumeration of basis states.

    This class generates a full basis of the many-body Hilbert space. Thereby, it \
    allows to exactly perform sums over the full Hilbert space instead of stochastic \
    sampling.

    Initialization arguments:
        * ``net``: Network defining the probability distribution.
        * ``sampleShape``: Shape of computational basis states.
        * ``lDim``: Local Hilbert space dimension.
    """

    def __init__(self, net, sampleShape, lDim=2, logProbFactor=0.5):

        self.psi = net
        self.N = jnp.prod(jnp.asarray(sampleShape))
        self.sampleShape = sampleShape
        self.lDim = lDim
        self.logProbFactor = logProbFactor

        # pmap'd member functions
        self._get_basis_ldim2_pmapd = global_defs.pmap_for_my_devices(self._get_basis_ldim2, in_axes=(0, 0, None), static_broadcasted_argnums=2)
        self._get_basis_pmapd = global_defs.pmap_for_my_devices(self._get_basis, in_axes=(0, 0, None, None), static_broadcasted_argnums=(2, 3))
        self._compute_probabilities_pmapd = global_defs.pmap_for_my_devices(self._compute_probabilities, in_axes=(0, None, 0))
        self._normalize_pmapd = global_defs.pmap_for_my_devices(self._normalize, in_axes=(0, None))

        self.get_basis()

        # Make sure that net params are initialized
        self.psi(self.basis)

        self.lastNorm = 0.

    def get_basis(self):

        myNumStates = mpi.distribute_sampling(self.lDim**self.N)
        myFirstState = mpi.first_sample_id()

        deviceCount = global_defs.device_count()

        self.numStatesPerDevice = [(myNumStates + deviceCount - 1) // deviceCount] * deviceCount
        self.numStatesPerDevice[-1] += myNumStates - deviceCount * self.numStatesPerDevice[0]
        self.numStatesPerDevice = jnp.array(self.numStatesPerDevice)

        totalNumStates = deviceCount * self.numStatesPerDevice[0]

        intReps = jnp.arange(myFirstState, myFirstState + totalNumStates)
        intReps = intReps.reshape((global_defs.device_count(), -1))
        self.basis = jnp.zeros(intReps.shape + (self.N,), dtype=np.int32)
        if self.lDim == 2:
            self.basis = self._get_basis_ldim2_pmapd(self.basis, intReps, self.sampleShape)
        else:
            self.basis = self._get_basis_pmapd(self.basis, intReps, self.lDim, self.sampleShape)

    def _get_basis_ldim2(self, states, intReps, sampleShape):

        def make_state(state, intRep):

            def for_fun(i, x):
                return (jax.lax.cond(x[1] >> i & 1, lambda x: x[0].at[x[1]].set(1), lambda x: x[0], (x[0], i)), x[1])

            (state, _) = jax.lax.fori_loop(0, state.shape[0], for_fun, (state, intRep))

            return state.reshape(sampleShape)

        basis = jax.vmap(make_state, in_axes=(0, 0))(states, intReps)

        return basis

    def _get_basis(self, states, intReps, lDim, sampleShape):

        def make_state(state, intRep):

            def scan_fun(c, x):
                locState = c % lDim
                c = (c - locState) // lDim
                return c, locState

            _, state = jax.lax.scan(scan_fun, intRep, state)

            return state[::-1].reshape(sampleShape)

        basis = jax.vmap(make_state, in_axes=(0, 0))(states, intReps)

        return basis

    def _compute_probabilities(self, logPsi, lastNorm, numStates):

        # p = jnp.exp(2. * jnp.real(logPsi - lastNorm))
        p = jnp.exp(jnp.real(logPsi - lastNorm) / self.logProbFactor)

        def scan_fun(c, x):
            out = jax.lax.cond(c[1] < c[0], lambda x: x[0], lambda x: x[1], (x, 0.))
            newC = c[1] + 1
            return (c[0], newC), out

        _, p = jax.lax.scan(scan_fun, (numStates, 0), p)

        return p

    def _normalize(self, p, nrm):

        return p / nrm

    def sample(self, parameters=None, numSamples=None, multipleOf=None):
        """Return all computational basis states.

        Sampling is automatically distributed accross MPI processes and available \
        devices.

        Arguments:
            * ``parameters``: Dummy argument to provide identical interface as the \
            ``MCSampler`` class.
            * ``numSamples``: Dummy argument to provide identical interface as the \
            ``MCSampler`` class.
            * ``multipleOf``: Dummy argument to provide identical interface as the \
            ``MCSampler`` class.

        Returns:
            ``configs, logPsi, p``: All computational basis configurations, \
            corresponding wave function coefficients, and probabilities \
            :math:`|\psi(s)|^2` (normalized).
        """

        logPsi = self.psi(self.basis)

        p = self._compute_probabilities_pmapd(logPsi, self.lastNorm, self.numStatesPerDevice)

        nrm = mpi.global_sum(p)
        p = self._normalize_pmapd(p, nrm)

        self.lastNorm += self.logProbFactor * jnp.log(nrm)

        return self.basis, logPsi, p

    def set_number_of_samples(self, N):
        pass

    def get_last_number_of_samples(self):
        return jnp.inf

# ** end class ExactSampler
