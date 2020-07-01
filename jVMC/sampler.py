import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
from jax import vmap, jit

from functools import partial

import time

def propose_spin_flip(key, s, info):
    idx = random.randint(key,(1,),0,s.size)[0]
    idx = jnp.unravel_index(idx, s.shape)
    update = (s[idx] + 1 ) % 2
    return jax.ops.index_update( s, jax.ops.index[idx], update )

class MCMCSampler:

    def __init__(self, key, updateProposer, sampleShape, numChains=1, updateProposerArg=None,
                    thermalizationSteps=10, sweepSteps=10):
        stateShape = [numChains]
        for s in sampleShape:
            stateShape.append(s)
        self.states=jnp.zeros(stateShape, dtype=np.int32)

        self.updateProposer = updateProposer
        self.updateProposerArg = updateProposerArg

        self.key = key
        self.thermalizationSteps = thermalizationSteps
        self.sweepSteps = sweepSteps

        self.numChains = numChains


    def sample(self, net, numSamples):

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
            newLogPsiSq = net.real_coefficients(newStates)
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
        self.logPsiSq = net.real_coefficients(self.states)

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

        self.get_basis()

        self.lastNorm = 0.


    def get_basis(self):

        intReps = jnp.arange(2**self.N)
        self.basis = jnp.zeros((2**self.N, self.N), dtype=np.int32)
        self.basis = self._get_basis(self.basis, intReps)


    @partial(jax.jit, static_argnums=(0,))
    def _get_basis(self, states, intReps):

        def make_state(state, intRep):

            def for_fun(i, x):
                return (jax.lax.cond(x[1]>>i & 1, lambda x: jax.ops.index_update(x[0], jax.ops.index[x[1]], 1), lambda x : x[0], (x[0],i)), x[1])

            (state, _)=jax.lax.fori_loop(0,state.shape[0],for_fun,(state, intRep))

            return state

        basis = jax.vmap(make_state, in_axes=(0,0))(states, intReps)

        return basis
    

    def sample(self, net, numSamples=0):

        logPsi = net(self.basis)
        nrm = jnp.linalg.norm( jnp.exp( logPsi - self.lastNorm ) )
        self.lastNorm += jnp.log(nrm)
        p = jnp.exp(2 * jnp.real( logPsi - self.lastNorm ))
        
        return self.basis, logPsi, p

# ** end class ExactSampler


if __name__ == "__main__":
    import nets
    from vqs import NQS
    from flax import nn

    L=8
    sampler = Sampler(random.PRNGKey(123), propose_spin_flip, [L], numChains=5)

    rbm = nets.CpxRBM.partial(L=L,numHidden=2,bias=True)
    _,params = rbm.init_by_shape(random.PRNGKey(0),[(1,L)])
    rbmModel = nn.Model(rbm,params)
    psiC = NQS(rbmModel)
    configs, _ = sampler.sample(psiC, 10)

    print(configs)
    print(sampler.acceptance_ratio())
