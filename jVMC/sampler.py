import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
from jax import vmap, jit

import sys
# Find jVMC package
sys.path.append(sys.path[0]+"/..")

import jVMC.mpi_wrapper as mpi

from functools import partial

import time

import jVMC.global_defs as global_defs


def propose_spin_flip(key, s, info):
    idx = random.randint(key,(1,),0,s.size)[0]
    idx = jnp.unravel_index(idx, s.shape)
    update = (s[idx] + 1 ) % 2
    return jax.ops.index_update( s, jax.ops.index[idx], update )


def propose_spin_flip_Z2(key, s, info):
    idxKey, flipKey = jax.random.split(key)
    idx = random.randint(idxKey,(1,),0,s.size)[0]
    idx = jnp.unravel_index(idx, s.shape)
    update = (s[idx] + 1 ) % 2
    s = jax.ops.index_update( s, jax.ops.index[idx], update )
    # On average, do a global spin flip every 30 updates to
    # reflect Z_2 symmetry
    doFlip = random.randint(flipKey,(1,),0,5)[0]
    return jax.lax.cond(doFlip==0, lambda x: 1-x, lambda x: x, s)


class MCMCSampler:
    """A sampler class.

    This class provides functionality to sample computational basis states from \
    the probability distribution induced by the variational wave function, \
    :math:`|\\psi(s)|^2`.

    Sampling is automatically distributed accross MPI processes and locally available \
    devices.
    
    Initializer arguments:
    
    * ``key``: An instance of ``jax.random.PRNGKey``.
    * ``updateProposer``: A function to propose updates for the MCMC algorithm. \
    It is called as ``updateProposer(key, config, **kwargs)``, where ``key`` is an instance of \
    ``jax.random.PRNGKey``, ``config`` is a computational basis configuration, and ``**kwargs`` \
    are optional additional arguments.
    * ``sampleShape``: Shape of computational basis configurations.
    * ``numChains``: Number of Markov chains, which are run in parallel.
    * ``updateProposerArg``: An optional argument that will be passed to the ``updateProposer`` \
    as ``kwargs``.
    * ``numSamples``: Default number of samples to be returned by the ``sample()`` member function.
    * ``thermalizationSweeps``: Number of sweeps to perform for thermalization of the Markov chain.
    * ``sweepSteps``: Number of proposed updates per sweep.
    """

    def __init__(self, key, updateProposer, sampleShape, numChains=1, updateProposerArg=None,
                    numSamples=100, thermalizationSweeps=10, sweepSteps=10):
        """Initializes the MCMCSampler class.

        Args:
        
        * ``key``: An instance of ``jax.random.PRNGKey``.
        * ``updateProposer``: A function to propose updates for the MCMC algorithm. \
        It is called as ``updateProposer(key, config, **kwargs)``, where ``key`` is an instance of \
        ``jax.random.PRNGKey``, ``config`` is a computational basis configuration, and ``**kwargs`` \
        are optional additional arguments. The function is supposed to return a computational basis \
        state that is used as update proposal in the MCMC algorithm.
        * ``sampleShape``: Shape of computational basis configurations.
        * ``numChains``: Number of Markov chains, which are run in parallel.
        * ``updateProposerArg``: An optional argument that will be passed to the ``updateProposer`` \
        as ``kwargs``.
        * ``numSamples``: Default number of samples to be returned by the ``sample()`` member function.
        * ``thermalizationSweeps``: Number of sweeps to perform for thermalization of the Markov chain.
        * ``sweepSteps``: Number of proposed updates per sweep.
        """

        self.sampleShape = sampleShape

        stateShape = (numChains,) + sampleShape
        if global_defs.usePmap:
            stateShape = (global_defs.device_count(),) + stateShape
        self.states=jnp.zeros(stateShape, dtype=np.int32)

        self.updateProposer = updateProposer
        self.updateProposerArg = updateProposerArg

        self.key = jax.random.split(key, mpi.commSize)[mpi.rank]
        if global_defs.usePmap:
            self.key = jax.random.split(self.key, global_defs.device_count())
        self.thermalizationSweeps = thermalizationSweeps
        self.sweepSteps = sweepSteps
        self.numSamples = numSamples
        
        self.numChains = numChains

        # jit'd member functions
        self._get_samples_jitd = {} # will hold a jit'd function for each number of samples
        self._get_samples_gen_jitd = {} # will hold a jit'd function for each number of samples


    def set_number_of_samples(self, N):
        """Set default number of samples.

        Args:
        
        * ``N``: Number of samples.
        """

        self.numSamples = N


    def set_random_key(self, key):
        """Set key for pseudo random number generator.

        Args:
        
        * ``key``: Key (jax.random.PRNGKey)
        """
        
        self.key = key
        if global_defs.usePmap:
            self.key = jax.random.split(self.key, global_defs.device_count())


    def get_last_number_of_samples(self):
        """Return last number of samples.

        This function is required, because the actual number of samples might \
        exceed the requested number of samples when sampling is distributed \
        accross multiple processes or devices.

        Returns:
            Number of samples generated by last call to ``sample()`` member function.
        """
        return mpi.globNumSamples


    def sample(self, net, numSamples=None, multipleOf=1):
        """Generate random samples from wave function.

        If supported by ``net``, direct sampling is peformed. Otherwise, MCMC is run \
        to generate the desired number of samples. For direct sampling the real part \
        of ``net`` needs to provide a ``sample()`` member function that generates \
        samples from :math:`|\\psi(s)|^2`.

        Sampling is automatically distributed accross MPI processes and available \
        devices. In that case the number of samples returned might exceed ``numSamples``.

        Args:

        * ``net``: Variational wave function, instance of ``jVMC.NQS`` class.
        * ``numSamples``: Number of samples to generate. When running multiple processes \
        or on multiple devices per process, the number of samples returned is \
        ``numSamples`` or more. If ``None``, the default number of samples is returned \
        (see ``set_number_of_samples()`` member function).

        Returns:
            A sample of computational basis configurations drawn from :math:`|\\psi(s)|^2`.
        """

        if numSamples is None:
            numSamples = self.numSamples


        if net.is_generator:

            configs = self._get_samples_gen(net, numSamples, multipleOf)

            return configs, net(configs), None


        configs, logPsi = self._get_samples_mcmc(net, numSamples, multipleOf)

        return configs, logPsi, None


    def _get_samples_gen(self, net, numSamples, multipleOf=1):
        
        def dc():
            if global_defs.usePmap:
                return global_defs.device_count()
            return 1
        
        numSamples = mpi.distribute_sampling(numSamples, localDevices=dc(), numChainsPerDevice=multipleOf)
        numSamplesStr = str(numSamples)

        # check whether _get_samples is already compiled for given number of samples
        if not numSamplesStr in self._get_samples_gen_jitd:
            if global_defs.usePmap:
                self._get_samples_gen_jitd[numSamplesStr] = global_defs.pmap_for_my_devices(lambda x,y,z: x.sample(y,z), static_broadcasted_argnums=(1,), in_axes=(None, None, 0))
            else:
                self._get_samples_gen_jitd[numSamplesStr] = global_defs.jit_for_my_device(lambda x,y,z: x.sample(y,z), static_argnums=(1,))

        tmpKey = None
        if global_defs.usePmap:
            tmpKey = random.split(self.key[0], 2*global_defs.device_count())
            self.key = tmpKey[:global_defs.device_count()]
            tmpKey = tmpKey[global_defs.device_count():]
        else:
            tmpKey, self.key = random.split(self.key)

        return self._get_samples_gen_jitd[numSamplesStr](net.get_sampler_net(), numSamples, tmpKey)


    def _get_samples_mcmc(self, net, numSamples, multipleOf=1):

        # Initialize sampling stuff
        self._mc_init(net)
        
        def dc():
            if global_defs.usePmap:
                return global_defs.device_count()
            return 1

        numSamples = mpi.distribute_sampling(numSamples, localDevices=dc(), numChainsPerDevice=np.lcm(self.numChains,multipleOf))
        numSamplesStr = str(numSamples)

        # check whether _get_samples is already compiled for given number of samples
        if not numSamplesStr in self._get_samples_jitd:
            if global_defs.usePmap:
                self._get_samples_jitd[numSamplesStr] = global_defs.pmap_for_my_devices(partial(self._get_samples, sweepFunction=self._sweep),
                                                                    static_broadcasted_argnums=(1,2,3,9,11),
                                                                    in_axes=(None, None, None, None, 0, 0, 0, 0, 0, None, None, None))
            else:
                self._get_samples_jitd[numSamplesStr] = global_defs.jit_for_my_device(partial(self._get_samples, sweepFunction=self._sweep),
                                                                    static_argnums=(1,2,3,9,11))

        (self.states, self.logPsiSq, self.key, self.numProposed, self.numAccepted), configs =\
            self._get_samples_jitd[numSamplesStr](net.get_sampler_net(), numSamples, self.thermalizationSweeps, self.sweepSteps,
                                                    self.states, self.logPsiSq, self.key, self.numProposed, self.numAccepted,
                                                    self.updateProposer, self.updateProposerArg, self.sampleShape)

        #return configs, None
        return configs, net(configs)


    def _get_samples(self, net, numSamples,
                            thermSweeps, sweepSteps,
                            states, logPsiSq, key, 
                            numProposed, numAccepted,
                            updateProposer, updateProposerArg,
                            sampleShape, sweepFunction=None):

        # Thermalize
        states, logPsiSq, key, numProposed, numAccepted =\
            sweepFunction(states, logPsiSq, key, numProposed, numAccepted, net, thermSweeps*sweepSteps, updateProposer, updateProposerArg)

        # Collect samples
        def scan_fun(c, x):

            states, logPsiSq, key, numProposed, numAccepted =\
                sweepFunction(c[0], c[1], c[2], c[3], c[4], net, sweepSteps, updateProposer, updateProposerArg)

            return (states, logPsiSq, key, numProposed, numAccepted), states

        meta, configs = jax.lax.scan(scan_fun, (states, logPsiSq, key, numProposed, numAccepted), None, length=numSamples)

        #return meta, configs.reshape((configs.shape[0]*configs.shape[1], -1))
        return meta, configs.reshape((configs.shape[0]*configs.shape[1],) + sampleShape)

    def _sweep(self, states, logPsiSq, key, numProposed, numAccepted, net, numSteps, updateProposer, updateProposerArg):
        
        def perform_mc_update(i, carry):
            
            # Generate update proposals
            newKeys = random.split(carry[2],carry[0].shape[0]+1)
            carryKey = newKeys[-1]
            newStates = vmap(updateProposer, in_axes=(0, 0, None))(newKeys[:len(carry[0])], carry[0], updateProposerArg)
            #newStates = carry[0] 

            # Compute acceptance probabilities
            newLogPsiSq = jax.vmap(lambda x,y: 2.*jnp.real(x(y)), in_axes=(None,0))(net,newStates)
            P = jnp.exp( newLogPsiSq - carry[1] )

            # Roll dice
            newKey, carryKey = random.split(carryKey,)
            accepted = random.bernoulli(newKey, P).reshape((-1,))

            # Bookkeeping
            numProposed = carry[3] + len(newStates)
            numAccepted = carry[4] + jnp.sum(accepted)

            # Perform accepted updates
            def update(acc, old, new):
                return jax.lax.cond(acc, lambda x: x[1], lambda x: x[0], (old,new))
            carryStates = vmap(update, in_axes=(0,0,0))(accepted, carry[0], newStates)

            carryLogPsiSq = jnp.where(accepted==True, newLogPsiSq, carry[1])

            return (carryStates, carryLogPsiSq, carryKey, numProposed, numAccepted)

        (states, logPsiSq, key, numProposed, numAccepted) =\
            jax.lax.fori_loop(0, numSteps, perform_mc_update, (states, logPsiSq, key, numProposed, numAccepted))

        return states, logPsiSq, key, numProposed, numAccepted


    def _mc_init(self, net):
        
        # Initialize logPsiSq
        self.logPsiSq = 2. * net.real_coefficients(self.states)

        shape = (1,)
        if global_defs.usePmap:
            shape = (global_defs.device_count(),) + shape
 
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

        return 0.

# ** end class Sampler


class ExactSampler:
    """Class for full enumeration of basis states.

    This class generates a full basis of the many-body Hilbert space. Thereby, it \
    allows to exactly perform sums over the full Hilbert space instead of stochastic \
    sampling.

    Initialization arguments:
    
    * sampleShape: Shape of computational basis states.
    * lDim: Local Hilbert space dimension.
    """

    def __init__(self, sampleShape, lDim=2):

        self.N = jnp.prod(jnp.asarray(sampleShape))
        self.sampleShape = sampleShape
        self.lDim = lDim

        # jit'd member functions
        if global_defs.usePmap:
            self._get_basis_ldim2_pmapd = global_defs.pmap_for_my_devices(self._get_basis_ldim2, in_axes=(0, 0,None), static_broadcasted_argnums=2)
            self._get_basis_pmapd = global_defs.pmap_for_my_devices(self._get_basis, in_axes=(0, 0, None,None), static_broadcasted_argnums=(2,3))
            self._compute_probabilities_pmapd = global_defs.pmap_for_my_devices(self._compute_probabilities, in_axes=(0, None, 0))
            self._normalize_pmapd = global_defs.pmap_for_my_devices(self._normalize, in_axes=(0, None))
        else:
            self._get_basis_ldim2_pmapd = global_defs.jit_for_my_device(self._get_basis_ldim2)
            self._get_basis_pmapd = global_defs.jit_for_my_device(self._get_basis)
            self._compute_probabilities_pmapd = global_defs.jit_for_my_device(self._compute_probabilities)
            self._normalize_pmapd = global_defs.jit_for_my_device(self._normalize)

        self.get_basis()

        self.lastNorm = 0.


    def get_basis(self):

        myNumStates = mpi.distribute_sampling(self.lDim**self.N)
        myFirstState = mpi.first_sample_id()

        deviceCount = global_defs.device_count()
        if not global_defs.usePmap:
            deviceCount = 1

        self.numStatesPerDevice = [(myNumStates + deviceCount - 1) // deviceCount] * deviceCount
        self.numStatesPerDevice[-1] += myNumStates - deviceCount * self.numStatesPerDevice[0]
        self.numStatesPerDevice = jnp.array(self.numStatesPerDevice)

        totalNumStates = deviceCount * self.numStatesPerDevice[0]
        
        if not global_defs.usePmap:
            self.numStatesPerDevice = self.numStatesPerDevice[0]

        intReps = jnp.arange(myFirstState, myFirstState + totalNumStates)
        if global_defs.usePmap:
            intReps = intReps.reshape((global_defs.device_count(), -1))
        self.basis = jnp.zeros(intReps.shape + (self.N,), dtype=np.int32)
        if self.lDim == 2:
            self.basis = self._get_basis_ldim2_pmapd(self.basis, intReps, self.sampleShape)
        else:
            self.basis = self._get_basis_pmapd(self.basis, intReps, self.lDim, self.sampleShape)


    def _get_basis_ldim2(self, states, intReps, sampleShape):

        def make_state(state, intRep):

            def for_fun(i, x):
                return (jax.lax.cond(x[1]>>i & 1, lambda x: jax.ops.index_update(x[0], jax.ops.index[x[1]], 1), lambda x : x[0], (x[0],i)), x[1])

            (state, _)=jax.lax.fori_loop(0,state.shape[0],for_fun,(state, intRep))

            return state.reshape(sampleShape)

        basis = jax.vmap(make_state, in_axes=(0,0))(states, intReps)

        return basis


    def _get_basis(self, states, intReps, lDim, sampleShape):

        def make_state(state, intRep):

            def scan_fun(c, x):
                locState = c % lDim
                c = (c - locState) // lDim
                return c, locState

            _, state = jax.lax.scan(scan_fun,intRep,state)

            return state[::-1].reshape(sampleShape)

        basis = jax.vmap(make_state, in_axes=(0,0))(states,intReps)

        return basis


    def _compute_probabilities(self, logPsi, lastNorm, numStates):

        p = jnp.exp(2. * jnp.real( logPsi - lastNorm ))

        def scan_fun(c, x):
            out = jax.lax.cond(c[1]<c[0], lambda x: x[0], lambda x: x[1], (x, 0.))
            newC = c[1] + 1
            return (c[0], newC), out

        _, p = jax.lax.scan(scan_fun, (numStates, 0), p)

        return p


    def _normalize(self, p, nrm):

            return p/nrm


    def sample(self, net, numSamples=None, multipleOf=None):
        """Return all computational basis states.

        Sampling is automatically distributed accross MPI processes and available \
        devices.

        Args:

        * ``net``: Variational wave function, instance of ``jVMC.NQS`` class.
        * ``numSamples``: Dummy argument to provde identical interface as the \
        ``MCMCSampler`` class.

        Returns:
            ``configs, logPsi, p``: All computational basis configurations, \
            corresponding wave function coefficients, and probabilities \
            :math:`|\psi(s)|^2` (normalized).
        """

        logPsi = net(self.basis)

        p = self._compute_probabilities_pmapd(logPsi, self.lastNorm, self.numStatesPerDevice)

        nrm = mpi.global_sum(p)
        p = self._normalize_pmapd(p,nrm)
        
        self.lastNorm += 0.5 * jnp.log(nrm)
 
        return self.basis, logPsi, p


    def set_number_of_samples(self, N):
        pass

# ** end class ExactSampler


if __name__ == "__main__":

    import nets
    from vqs import NQS
    from flax import nn

    L=128
    sampler = MCMCSampler(random.PRNGKey(123), propose_spin_flip, (L,), numChains=500, sweepSteps=128)
    #sampler = ExactSampler((L,))

    rbm = nets.CpxRBM.partial(numHidden=128,bias=True)
    _,params = rbm.init_by_shape(random.PRNGKey(0),[(L,)])
    rbmModel = nn.Model(rbm,params)
    psiC = NQS(rbmModel)
    tic=time.perf_counter()
    configs, logspi, p = sampler.sample(psiC, numSamples=100000)
    #configs, logpsi, p = sampler.sample(psiC, numSamples=10)

    print(configs.shape)
    exit()

    configs.block_until_ready()
    print("total time:", time.perf_counter()-tic)
    
    tic=time.perf_counter()
    configs, logspi, p = sampler.sample(psiC, numSamples=100000)
    configs.block_until_ready()
    print("total time:", time.perf_counter()-tic)


    # Set up variational wave function
    L=64
    rnn = nets.RNN.partial( L=L, units=[50] )
    _, params = rnn.init_by_shape( random.PRNGKey(0), [(L,)] )
    rnnModel = nn.Model(rnn,params)
    rbm = nets.RBM.partial(numHidden=2,bias=False)
    _, params = rbm.init_by_shape(random.PRNGKey(0),[(L,)])
    rbmModel = nn.Model(rbm,params)
    
    psi = NQS(rnnModel, rbmModel)
    
    tic=time.perf_counter()
    configs, logpsi, p = sampler.sample(psi, numSamples=500000)
    configs.block_until_ready()
    print("total time:", time.perf_counter()-tic)
    tic=time.perf_counter()
    configs, logpsi, p = sampler.sample(psi, numSamples=500000)
    configs.block_until_ready()
    print("total time:", time.perf_counter()-tic)
