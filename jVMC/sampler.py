import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
from jax import vmap, jit

import time

def propose_spin_flip(key, s, info):
    idx = random.randint(key,(1,),0,s.size)[0]
    idx = jnp.unravel_index(idx, s.shape)
    update = (s[idx] + 1 ) % 2
    return jax.ops.index_update( s, jax.ops.index[idx], update )

class Sampler:

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
        t0=time.perf_counter()
        # Thermalize
        self.sweep(net, self.thermalizationSteps)
        print("Thermalization took ", time.perf_counter()-t0)

        numMissing = numSamples
        numAdd = min(self.numChains, numMissing)
        configs = jax.ops.index_update(configs, jax.ops.index[numSamples-numMissing:numSamples-numMissing+numAdd], self.states[:numAdd])
        numMissing -= numAdd
        while numMissing > 0:
            t0=time.perf_counter()
            self.sweep(net, self.sweepSteps) 
            print("Sweep took ", time.perf_counter()-t0)
            numAdd = min(self.numChains, numMissing)
            t0=time.perf_counter()
            configs = jax.ops.index_update(configs, jax.ops.index[numSamples-numMissing:numSamples-numMissing+numAdd], self.states[:numAdd])
            print("Saving configs took ", time.perf_counter()-t0)
            numMissing -= numAdd

        return configs, net(configs)

    
    def sweep(self, net, numSteps):
        
        for n in range(numSteps):
            self._mc_update(net)


    def _mc_init(self, net):
        
        # Initialize logPsiSq
        self.logPsiSq = net.real_coefficients(self.states)

        self.numProposed = 0
        self.numAccepted = 0


    def _mc_update(self, net):
        
        # Generate update proposals
        newKeys = random.split(self.key,len(self.states)+1)
        self.key = newKeys[-1]
        newStates = jit(vmap(self.updateProposer, in_axes=(0, 0, None)))(newKeys[:len(self.states)], self.states, self.updateProposerArg)

        # Compute acceptance probabilities
        newLogPsiSq = net.real_coefficients(newStates)
        P = jnp.exp( newLogPsiSq - self.logPsiSq )

        # Roll dice
        newKey, self.key = random.split(self.key,)
        accepted = random.bernoulli(newKey, P)
        update = jnp.where(accepted)

        # Bookkeeping
        self.numProposed += len(newStates)
        self.numAccepted += update[0].shape[0]

        # Perform accepted updates
        self.states = jax.ops.index_update(self.states, jax.ops.index[update], newStates[update])
        self.logPsiSq = jax.ops.index_update(self.logPsiSq, jax.ops.index[update], newLogPsiSq[update])


    def acceptance_ratio(self):

        if self.numProposed > 0:
            return self.numAccepted / self.numProposed

        return 0.

# ** end class Sampler

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
