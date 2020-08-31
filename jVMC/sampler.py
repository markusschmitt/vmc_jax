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
                    numSamples=100, thermalizationSteps=10, sweepSteps=10):
        stateShape = [numChains]
        for s in sampleShape:
            stateShape.append(s)
        self.states=jnp.zeros(stateShape, dtype=np.int32)

        self.updateProposer = updateProposer
        self.updateProposerArg = updateProposerArg

        self.key = key
        self.thermalizationSteps = thermalizationSteps
        self.sweepSteps = sweepSteps
        self.numSamples = numSamples
        
        self.numChains = numChains


    def set_number_of_samples(self, N):

        self.numSamples = N


    def get_last_number_of_samples(self):

        return self.lastNumSamples


    def sample(self, net, numSamples=None):

        if numSamples is None:
            numSamples = self.numSamples

        self.lastNumSamples = numSamples
        numSamples = mpi.distribute_sampling(numSamples)

        if net.is_generator:
            tmpKey, self.key = random.split(self.key)
            configs, logP = net.sample(numSamples, self.key)
            return configs, logP, None

        # Prepare for output
        outShape = [s for s in self.states.shape]
        outShape[0] = numSamples
        configs = jnp.empty(outShape, dtype=np.int32)

        # Initialize sampling stuff
        self._mc_init(net)
        # Thermalize
        self.sweep(net, self.thermalizationSteps)

        numMissing = numSamples
        numAdd = min(self.numChains, numMissing)
        configs = jax.ops.index_update(configs, jax.ops.index[numSamples-numMissing:numSamples-numMissing+numAdd], self.states[:numAdd])
        numMissing -= numAdd
        while numMissing > 0:
            self.sweep(net, self.sweepSteps) 
            numAdd = min(self.numChains, numMissing)
            configs = jax.ops.index_update(configs, jax.ops.index[numSamples-numMissing:numSamples-numMissing+numAdd], self.states[:numAdd])
            numMissing -= numAdd

        return configs, net(configs), None


    def sweep(self, net, numSteps):

        self.states, self.logPsiSq, self.key, self.numProposed, self.numAccepted =\
            self._sweep(self.states, self.logPsiSq, self.key, self.numProposed, self.numAccepted,
                        net, numSteps, self.updateProposer, self.updateProposerArg)


    @partial(jax.jit, static_argnums=(0,8))
    def _sweep(self, states, logPsiSq, key, numProposed, numAccepted, net, numSteps, updateProposer, updateProposerArg):
        
        def perform_mc_update(i, carry):
            
            # Generate update proposals
            newKeys = random.split(carry[2],carry[0].shape[0]+1)
            carryKey = newKeys[-1]
            newStates = jit(vmap(updateProposer, in_axes=(0, 0, None)))(newKeys[:len(carry[0])], carry[0], updateProposerArg)

            # Compute acceptance probabilities
            newLogPsiSq = 2.*net.real_coefficients(newStates)
            P = jnp.exp( newLogPsiSq - carry[1] )

            # Roll dice
            newKey, carryKey = random.split(carryKey,)
            accepted = random.bernoulli(newKey, P).reshape((-1,))

            # Bookkeeping
            numProposed = carry[3] + len(newStates)
            numAccepted = carry[4] + jnp.sum(accepted)

            # Perform accepted updates
            def update(carry, x):
                newState,_ = jax.lax.cond(x[0], lambda x: (x[1],x[0]), lambda x: (x[0],x[1]), (x[1],x[2]))
                return carry, newState
            _, carryStates = jax.lax.scan(update, [None], (accepted, carry[0], newStates))

            carryLogPsiSq = jnp.where(accepted==True, newLogPsiSq, carry[1])

            return (carryStates, carryLogPsiSq, carryKey, numProposed, numAccepted)

        tmpInt = numSteps.astype(int)
        (states, logPsiSq, key, numProposed, numAccepted) =\
            jax.lax.fori_loop(0, numSteps, perform_mc_update, (states, logPsiSq, key, numProposed, numAccepted))

        return states, logPsiSq, key, numProposed, numAccepted

    def _mc_init(self, net):
        
        # Initialize logPsiSq
        self.logPsiSq = 2. * net.real_coefficients(self.states)

        self.numProposed = 0
        self.numAccepted = 0


    def acceptance_ratio(self):

        if self.numProposed > 0:
            return self.numAccepted / self.numProposed

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

        print(self.numStatesPerDevice)

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

    L=4
    #sampler = Sampler(random.PRNGKey(123), propose_spin_flip, [L], numChains=5)
    sampler = ExactSampler((L,))

    rbm = nets.CpxRBM.partial(numHidden=2,bias=True)
    _,params = rbm.init_by_shape(random.PRNGKey(0),[(L,)])
    rbmModel = nn.Model(rbm,params)
    psiC = NQS(rbmModel)
    configs, logpsi, p = sampler.sample(psiC, 10)

    print(configs[1].device_buffer.device())
    print(p[1].device_buffer.device())
    print(logpsi[1].device_buffer.device())
    print(jnp.sum(p))
#    print(sampler.acceptance_ratio())
