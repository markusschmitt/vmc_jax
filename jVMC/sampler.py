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

def propose_spin_flip(key, s, info):
    idx = random.randint(key,(1,),0,s.size)[0]
    idx = jnp.unravel_index(idx, s.shape)
    update = (s[idx] + 1 ) % 2
    return jax.ops.index_update( s, jax.ops.index[idx], update )


class MCMCSampler:

    def __init__(self, key, updateProposer, sampleShape, numChains=1, updateProposerArg=None,
                    numSamples=100, thermalizationSweeps=10, sweepSteps=10):
        stateShape = [jax.device_count(), numChains]
        for s in sampleShape:
            stateShape.append(s)
        self.states=jnp.zeros(stateShape, dtype=np.int32)

        self.updateProposer = updateProposer
        self.updateProposerArg = updateProposerArg

        self.key = jax.random.split(key, jax.device_count())
        self.thermalizationSweeps = thermalizationSweeps
        self.sweepSteps = sweepSteps
        self.numSamples = numSamples
        
        self.numChains = numChains

        # pmap'd member functions
        self._get_samples_pmapd = {} # will hold a pmap'd function for each number of samples
        self._get_samples_gen_pmapd = {} # will hold a pmap'd function for each number of samples


    def set_number_of_samples(self, N):

        self.numSamples = N


    def get_last_number_of_samples(self):

        return mpi.globNumSamples


    def sample(self, net, numSamples=None):

        if numSamples is None:
            numSamples = self.numSamples


        if net.is_generator:

            configs, logPsi = self._get_samples_gen(net, numSamples)

            return configs, logPsi, None


        configs, logPsi = self._get_samples_mcmc(net, numSamples)

        return configs, logPsi, None


    def _get_samples_gen(self, net, numSamples):
        
        numSamples = mpi.distribute_sampling(numSamples, localDevices=jax.device_count())
        numSamplesStr = str(numSamples)

        # check whether _get_samples is already compiled for given number of samples
        if not numSamplesStr in self._get_samples_gen_pmapd:
            self._get_samples_gen_pmapd[numSamplesStr] = jax.pmap(lambda x,y,z: x.sample(y,z), static_broadcasted_argnums=(1,), in_axes=(None, None, 0))

        tmpKey = random.split(self.key[0], 2*jax.device_count())
        self.key = tmpKey[:jax.device_count()]

        return self._get_samples_gen_pmapd[numSamplesStr](net.get_sampler_net(), numSamples, tmpKey[jax.device_count():])


    def _get_samples_mcmc(self, net, numSamples):

        # Initialize sampling stuff
        self._mc_init(net)
        
        numSamples = mpi.distribute_sampling(numSamples, localDevices=jax.device_count(), numChainsPerDevice=self.numChains)
        numSamplesStr = str(numSamples)

        # Determine output shape
        outShape = [s for s in self.states.shape]
        outShape[1] = numSamples * self.numChains

        # check whether _get_samples is already compiled for given number of samples
        if not numSamplesStr in self._get_samples_pmapd:
            self._get_samples_pmapd[numSamplesStr] = jax.pmap(partial(self._get_samples, sweepFunction=self._sweep),
                                                                static_broadcasted_argnums=(1,9),
                                                                in_axes=(None, None, None, None, 0, 0, 0, 0, 0, None, None))

        (self.states, self.logPsiSq, self.key, self.numProposed, self.numAccepted), configs =\
            self._get_samples_pmapd[numSamplesStr](net.get_sampler_net(), numSamples, self.thermalizationSweeps, self.sweepSteps,
                                                    self.states, self.logPsiSq, self.key, self.numProposed, self.numAccepted,
                                                    self.updateProposer, self.updateProposerArg)

        return configs, net(configs)


    def _get_samples(self, net, numSamples, thermSweeps, sweepSteps, states, logPsiSq, key, numProposed, numAccepted, updateProposer, updateProposerArg, sweepFunction=None):

        # Thermalize
        states, logPsiSq, key, numProposed, numAccepted =\
            sweepFunction(states, logPsiSq, key, numProposed, numAccepted, net, thermSweeps*sweepSteps, updateProposer, updateProposerArg)

        # Collect samples
        def scan_fun(c, x):

            states, logPsiSq, key, numProposed, numAccepted =\
                sweepFunction(c[0], c[1], c[2], c[3], c[4], net, sweepSteps, updateProposer, updateProposerArg)

            return (states, logPsiSq, key, numProposed, numAccepted), states

        meta, configs = jax.lax.scan(scan_fun, (states, logPsiSq, key, numProposed, numAccepted), None, length=numSamples)

        return meta, configs.reshape((configs.shape[0]*configs.shape[1], -1))

    def _sweep(self, states, logPsiSq, key, numProposed, numAccepted, net, numSteps, updateProposer, updateProposerArg):
        
        def perform_mc_update(i, carry):
            
            # Generate update proposals
            newKeys = random.split(carry[2],carry[0].shape[0]+1)
            carryKey = newKeys[-1]
            newStates = vmap(updateProposer, in_axes=(0, 0, None))(newKeys[:len(carry[0])], carry[0], updateProposerArg)

            # Compute acceptance probabilities
            def eval(net, s):
                return net(s)
            newLogPsiSq = 2.*jnp.real(jax.vmap(eval, in_axes=(None,0))(net,newStates))
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

        tmpInt = numSteps.astype(int)
        (states, logPsiSq, key, numProposed, numAccepted) =\
            jax.lax.fori_loop(0, numSteps, perform_mc_update, (states, logPsiSq, key, numProposed, numAccepted))

        return states, logPsiSq, key, numProposed, numAccepted


    def _mc_init(self, net):
        
        # Initialize logPsiSq
        self.logPsiSq = 2. * net.real_coefficients(self.states)

        self.numProposed = jnp.zeros((jax.device_count(),1), dtype=np.int64)
        self.numAccepted = jnp.zeros((jax.device_count(),1), dtype=np.int64)


    def acceptance_ratio(self):

        numProp = mpi.global_sum(self.numProposed)
        if numProp > 0:
            return mpi.global_sum(self.numAccepted) / numProp

        return 0.

# ** end class Sampler


class ExactSampler:

    def __init__(self, sampleShape):

        self.N = jnp.prod(jnp.asarray(sampleShape))
        self.sampleShape = sampleShape

        # pmap'd member functions
        self._get_basis_pmapd = jax.pmap(self._get_basis, in_axes=(0, 0))
        self._compute_probabilities_pmapd = jax.pmap(self._compute_probabilities, in_axes=(0, None, 0))
        self._normalize_pmapd = jax.pmap(self._normalize, in_axes=(0, None))

        self.get_basis()

        self.lastNorm = 0.


    def get_basis(self):

        myNumStates = mpi.distribute_sampling(2**self.N)
        myFirstState = mpi.first_sample_id()

        self.numStatesPerDevice = [(myNumStates + jax.device_count() - 1) // jax.device_count()] * jax.device_count()
        self.numStatesPerDevice[-1] += myNumStates - jax.device_count() * self.numStatesPerDevice[0]
        self.numStatesPerDevice = jnp.array(self.numStatesPerDevice)

        totalNumStates = jax.device_count() * self.numStatesPerDevice[0]

        intReps = jnp.arange(myFirstState, myFirstState + totalNumStates).reshape((jax.device_count(), -1))
        self.basis = jnp.zeros((intReps.shape[0], intReps.shape[1], self.N), dtype=np.int32)
        self.basis = self._get_basis_pmapd(self.basis, intReps)


    def _get_basis(self, states, intReps):

        def make_state(state, intRep):

            def for_fun(i, x):
                return (jax.lax.cond(x[1]>>i & 1, lambda x: jax.ops.index_update(x[0], jax.ops.index[x[1]], 1), lambda x : x[0], (x[0],i)), x[1])

            (state, _)=jax.lax.fori_loop(0,state.shape[0],for_fun,(state, intRep))

            return state

        basis = jax.vmap(make_state, in_axes=(0,0))(states, intReps)

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


    def sample(self, net, numSamples=0):

        logPsi = net(self.basis)

        p = self._compute_probabilities_pmapd(logPsi, self.lastNorm, self.numStatesPerDevice)

        nrm = mpi.global_sum(p)
        p = self._normalize_pmapd(p,nrm)
        
        self.lastNorm += 0.5 * jnp.log(nrm)
 
        return self.basis, logPsi, p

# ** end class ExactSampler


if __name__ == "__main__":
    

    import nets
    from vqs import NQS
    from flax import nn

    L=64
    sampler = MCMCSampler(random.PRNGKey(123), propose_spin_flip, [L], numChains=1000)
    #sampler = ExactSampler((L,))

    rbm = nets.CpxRBM.partial(numHidden=20,bias=True)
    _,params = rbm.init_by_shape(random.PRNGKey(0),[(L,)])
    rbmModel = nn.Model(rbm,params)
    psiC = NQS(rbmModel)
    tic=time.perf_counter()
    configs, logspi, p = sampler.sample(psiC, numSamples=100000)
    #configs, logpsi, p = sampler.sample(psiC, numSamples=10)

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
