import jax
from jax.config import config
config.update("jax_enable_x64", True)
import flax
from flax import nn
import numpy as np
import jax.numpy as jnp

import jVMC.global_defs as global_defs

from functools import partial

class LSTMCell(nn.Module):

    def apply(self, carry, x, inputDim=2, actFun=nn.elu):

        newCarry, out = nn.LSTMCell(carry, x,
                                    kernel_init=partial(flax.nn.linear.default_kernel_init, dtype=global_defs.tReal),
                                    recurrent_kernel_init=partial(flax.nn.initializers.orthogonal(), dtype=global_defs.tReal),
                                    bias_init=partial(flax.nn.initializers.zeros, dtype=global_defs.tReal))

        out = actFun( nn.Dense(out, features=inputDim) )

        return newCarry, out.reshape((-1))


class LSTM(nn.Module):
    """Long short-term memory (LSTM).
    """

    def apply(self, x, L=10, hiddenSize=10, inputDim=2, actFun=nn.elu, logProbFactor=0.5):
        
        lstmCell = LSTMCell.shared(name="myCell", inputDim=inputDim, actFun=actFun)

        state = nn.LSTMCell.initialize_carry(jax.random.PRNGKey(0), (1,), hiddenSize)

        def lstm_cell(carry, x):
            newCarry, out = lstmCell(carry[0],carry[1])
            prob = nn.softmax(out)
            prob = jnp.log( jnp.sum( prob * x, axis=-1 ) )
            return (newCarry, x), prob
      
        _, probs = jax.lax.scan(lstm_cell, (state, jnp.zeros(inputDim)), jax.nn.one_hot(x,inputDim))

        return logProbFactor * jnp.sum(probs, axis=0)


    @nn.module_method
    def sample(self,batchSize,key,L,hiddenSize=10,inputDim=2,actFun=nn.elu, logProbFactor=0.5):
        """sampler
        """
        
        lstmCell = LSTMCell.shared(name="myCell", inputDim=inputDim, actFun=actFun)

        outputs = jnp.asarray(np.zeros((batchSize,L,L)))
        
        state = nn.LSTMCell.initialize_carry(jax.random.PRNGKey(0), (batchSize,), hiddenSize)

        def lstm_cell(carry, x):
            newCarry, logits = jax.vmap(lstmCell)(carry[0],carry[1])
            sampleOut = jax.random.categorical( x, logits )
            sample = jax.nn.one_hot( sampleOut, inputDim )
            logProb = jnp.log( jnp.sum( nn.softmax(logits) * sample, axis=1 ) )
            return (newCarry, sample), (logProb, sampleOut)
 
        keys = jax.random.split(key,L)
        _, res = jax.lax.scan( lstm_cell, (state, jnp.zeros((batchSize,inputDim))), keys )

        return jnp.transpose( res[1] )#, 0.5 * jnp.sum(res[0], axis=0)

# ** end class RNN


class LSTMsym(nn.Module):
    """LSTM with symmetries.
    """

    def apply(self, x, L=10, hiddenSize=10, inputDim=2, actFun=nn.elu, logProbFactor=0.5, orbit=None):

        lstm = LSTM.shared(L=L, hiddenSize=hiddenSize, inputDim=inputDim, actFun=actFun, name='myLSTM')
        
        x = jax.vmap(lambda o,s: jnp.dot(o,s), in_axes=(0,None))(orbit, x)

        def evaluate(x):
            return lstm(x)

        logProbs = logProbFactor * jnp.log( jnp.mean(jnp.exp((1./logProbFactor)*jax.vmap(evaluate)(x)), axis=0) )

        return logProbs

    @nn.module_method
    def sample(self, batchSize, key, L, hiddenSize=10, inputDim=2, actFun=nn.elu, logProbFactor=0.5, orbit=None):
        
        lstm = LSTM.shared(L=L, hiddenSize=hiddenSize, inputDim=inputDim, actFun=actFun, name='myLSTM')

        key1, key2 = jax.random.split(key)

        configs = lstm.sample(batchSize, key1)

        orbitIdx = jax.random.choice(key2, orbit.shape[0], shape=(batchSize,))

        configs = jax.vmap(lambda k,o,s: jnp.dot(o[k], s), in_axes=(0,None,0))(orbitIdx, orbit, configs)

        return configs

# ** end class RNNsym
