import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
from jax import vmap

def propose_spin_flip(key, s, info):
    idx = random.randint(key,(1,),0,s.size)[0]
    idx = jnp.unravel_index(idx, s.shape)
    update = (s[idx] + 1 ) % 2
    return jax.ops.index_update( s, jax.ops.index[idx], update )

class Sampler:

    def __init__(self, key, updateProposer, sampleShape, numChains=1, updateProposerArg=None):
        stateShape = [numChains]
        for s in sampleShape:
            stateShape.append(s)
        self.states=jnp.zeros(stateShape, dtype=np.int32)

        self.updateProposer = updateProposer
        self.updateProposerArg = updateProposerArg

        self.key = key

    def sample(self, numSamples):
        newKeys = random.split(self.key,len(self.states))
        self.key = newKeys[len(self.states)]
        newStates = vmap(self.updateProposer, in_axes=(0, 0, None))(newKeys[:len(self.states)], self.states, self.updateProposerArg)
        print(self.states)
        print(newStates)
        return 0


sampler = Sampler(random.PRNGKey(0), propose_spin_flip, [4,4], numChains=3)

sampler.sample(1)
