import jax
from jax.config import config
config.update("jax_enable_x64", True)
import flax
from flax import nn
import numpy as np
import jax.numpy as jnp

import jVMC.global_defs as global_defs

from functools import partial

class RNNCell2D(nn.Module):

    def apply(self, carryH, carryV, state, hiddenSize=10, outDim=2, actFun=nn.elu, initScale=1.0):
        
        initFunctionCell = partial(jax.nn.initializers.variance_scaling(scale=1.0, mode="fan_avg", distribution="uniform"),
                                    dtype=global_defs.tReal)
        initFunctionOut = partial(jax.nn.initializers.variance_scaling(scale=initScale, mode="fan_in", distribution="uniform"),
                                    dtype=global_defs.tReal)

        cellIn = nn.Dense.partial(features=hiddenSize,
                                    bias=False, dtype=global_defs.tReal,
                                    kernel_init=initFunctionCell)
        cellCarryH = nn.Dense.partial(features=hiddenSize,
                                    bias=False,
                                    kernel_init=initFunctionCell, dtype=global_defs.tReal,
                                    bias_init=partial(jax.nn.initializers.zeros,dtype=global_defs.tReal))
        cellCarryV = nn.Dense.partial(features=hiddenSize,
                                    bias=True,
                                    kernel_init=initFunctionCell, dtype=global_defs.tReal,
                                    bias_init=partial(jax.nn.initializers.zeros,dtype=global_defs.tReal))

        outputDense = nn.Dense.partial(features=outDim,
                                      kernel_init=initFunctionOut, dtype=global_defs.tReal,
                                      bias_init=jax.nn.initializers.normal(stddev=0.1,dtype=global_defs.tReal))
        
        newCarry = actFun(cellCarryH(carryH) + cellCarryV(carryV) + cellIn(state))
        out = outputDense(newCarry)
        return newCarry, out

# ** end class RNNCell


class RNNCellStack2D(nn.Module):

    def apply(self, carryH, carryV, stateH, stateV, hiddenSize=10, outDim=2, passDim=None, actFun=nn.elu, initScale=1.0):

        if passDim is None:
            passDim = outDim

        outDims = [passDim] * carryH.shape[0]
        outDims[-1] = outDim

        newCarry = [None] * carryH.shape[0]
        
        newR = jnp.concatenate((stateH,stateV), axis=0)
        # Can't use scan for this, because then flax doesn't realize that each cell has different parameters
        for j in range(carryH.shape[0]):
            newCarry[j], newR = RNNCell2D(carryH[j], carryV[j], newR, hiddenSize=hiddenSize, outDim=outDims[j], actFun=actFun, initScale=initScale)

        return jnp.array(newCarry), newR

# ** end class RNNCellStack


class RNN2D(nn.Module):
    """Recurrent neural network for two-dimensional input.
    """

    def reverse_line(self, line, b):
        return jax.lax.cond(b==1, lambda z : z, lambda z : jnp.flip(z,0), line)

    def apply(self, x, L=10, hiddenSize=10, depth=1, inputDim=2, passDim=None, actFun=nn.elu, initScale=1.0, logProbFactor=0.5):
        
        if passDim is None:
            passDim = inputDim

        rnnCell = RNNCellStack2D.shared(hiddenSize=hiddenSize, outDim=inputDim, passDim=passDim, actFun=actFun, initScale=initScale, name="myCell")

        # Scan directions for zigzag path
        direction = np.ones(L,dtype=np.int32)
        direction[1::2] = -1
        direction = jnp.asarray(direction)

        def rnn_cell_H(carry, x):
            newCarry, out = rnnCell(carry,x[0],x[1],x[2])
            return newCarry, (newCarry, nn.log_softmax(out))
      
        def rnn_cell_V(carry, x):
            _, out = jax.lax.scan(rnn_cell_H, jnp.zeros((depth, hiddenSize)),
                                    (self.reverse_line(carry[0], x[1]),
                                     jnp.concatenate((jnp.zeros((1,inputDim)),self.reverse_line(x[0],x[1])), axis=0)[:-1],
                                     self.reverse_line(carry[1], x[1]))
                                 )
            
            logProb = jnp.sum( self.reverse_line(out[1], x[1]) * x[0], axis=-1 )
            return (self.reverse_line(out[0], x[1]), x[0]), jnp.nan_to_num(logProb, nan=-35)

        _, probs = jax.lax.scan(rnn_cell_V,
                                (jnp.zeros((L,depth, hiddenSize)), jnp.zeros((L,inputDim))),
                                (jax.nn.one_hot(x,inputDim), direction)
                               )

        return logProbFactor * jnp.sum(probs)


    @nn.module_method
    def sample(self,batchSize,key,L,hiddenSize=10,depth=1,inputDim=2,passDim=None,actFun=nn.elu, initScale=1.0, logProbFactor=0.5):
        """sampler
        """
        
        if passDim is None:
            passDim = inputDim
        
        rnnCell = RNNCellStack2D.shared(hiddenSize=hiddenSize, outDim=inputDim, passDim=passDim, actFun=actFun, initScale=initScale, name="myCell")
        
        # Scan directions for zigzag path
        direction = np.ones(L,dtype=np.int32)
        direction[1::2] = -1
        direction = jnp.asarray(direction)

        def rnn_cell_H(carry, x):
            newCarry, out = rnnCell(carry[0],x[0],carry[1],x[1])
            sampleOut=jax.random.categorical(x[2],out)
            return (newCarry, jax.nn.one_hot(sampleOut,inputDim)), (newCarry, sampleOut)
      
        def rnn_cell_V(carry, x):
            keys = jax.random.split(x[0],L)
            _, out = jax.lax.scan(rnn_cell_H, (jnp.zeros((depth,hiddenSize)), jnp.zeros(inputDim)),
                                    (self.reverse_line(carry[0], x[1]),
                                     self.reverse_line(carry[1], x[1]),
                                     keys)
                                 )

            configLine = self.reverse_line(out[1],x[1])
            
            return (self.reverse_line(out[0], x[1]), jax.nn.one_hot(configLine, inputDim)), configLine

        def generate_sample(key):
            myKeys = jax.random.split(key,L)
            _, sample = jax.lax.scan(rnn_cell_V,
                                     (jnp.zeros((L,depth,hiddenSize)), jnp.zeros((L,inputDim))),
                                     (myKeys, direction)
                                    )
            return sample

        keys = jax.random.split(key,batchSize)

        return jax.vmap(generate_sample)(keys)

# ** end class RNN2D


#class RNNsym(nn.Module):
#    """Recurrent neural network with symmetries.
#    """
#
#    def apply(self, x, L=10, hiddenSize=10, depth=1, inputDim=2, passDim=None, actFun=nn.elu, initScale=1.0, logProbFactor=0.5, orbit=None, z2sym=False):
#
#        self.rnn = RNN.shared(L=L, hiddenSize=hiddenSize, depth=depth,\
#                                inputDim=inputDim, passDim=passDim,\
#                                actFun=actFun, initScale=initScale,\
#                                logProbFactor=logProbFactor, name='myRNN')
#
#        self.orbit = orbit
#        
#        x = jax.vmap(lambda o,s: jnp.dot(o,s), in_axes=(0,None))(self.orbit, x)
#
#        def evaluate(x):
#            return self.rnn(x)
#
#        res = jnp.mean(jnp.exp((1./logProbFactor)*jax.vmap(evaluate)(x)), axis=0)
#
#        if z2sym:
#            res = 0.5 * (res + jnp.mean(jnp.exp((1./logProbFactor)*jax.vmap(evaluate)(1-x)), axis=0))
#
#        logProbs = logProbFactor * jnp.log( res )
#
#        return logProbs
#
#    @nn.module_method
#    def sample(self,batchSize,key,L,hiddenSize=10,depth=1,inputDim=2,passDim=None,actFun=nn.elu, initScale=1.0, logProbFactor=0.5, orbit=None, z2sym=False):
#        
#        rnn = RNN.shared(L=L, hiddenSize=hiddenSize, depth=depth,\
#                                inputDim=inputDim, passDim=passDim,\
#                                actFun=actFun, initScale=initScale,\
#                                logProbFactor=logProbFactor, name='myRNN')
#
#        key1, key2 = jax.random.split(key)
#
#        configs = rnn.sample(batchSize, key1)
#
#        orbitIdx = jax.random.choice(key2, orbit.shape[0], shape=(batchSize,))
#
#        configs = jax.vmap(lambda k,o,s: jnp.dot(o[k], s), in_axes=(0,None,0))(orbitIdx, orbit, configs)
#
#        if z2sym:
#            key3, _ = jax.random.split(key2)
#            flipChoice = jax.random.choice(key3, 2, shape=(batchSize,))
#            configs = jax.vmap(lambda b,c: jax.lax.cond(b==1, lambda x: 1-x, lambda x: x, c), in_axes=(0,0))(flipChoice, configs)
#
#        return configs
#
## ** end class RNNsym
#
#
#
#class PhaseRNN(nn.Module):
#    """Recurrent neural network.
#    """
#
#    def apply(self, x, L=10, hiddenSize=10, depth=1, inputDim=2, actFun=nn.elu, initScale=1.0):
#        
#        if passDim is None:
#            passDim = inputDim
#        
#        rnnCell = RNNCellStack.partial(hiddenSize=hiddenSize, outDim=inputDim, passDim=passDim, actFun=actFun, initScale=initScale)
#
#        state = jnp.zeros((depth, hiddenSize))
#
#        def rnn_cell(carry, x):
#            newCarry, out = rnnCell(carry[0],carry[1])
#            return (newCarry, x), out
#      
#        _, res = jax.lax.scan(rnn_cell, (state, jnp.zeros(inputDim)), jax.nn.one_hot(x,inputDim))
#
#        res = nn.Dense(res.ravel(), features=8, dtype=global_defs.tReal,
#                        kernel_init=jax.nn.initializers.lecun_normal(dtype=global_defs.tReal), 
#                        bias_init=partial(jax.nn.initializers.zeros, dtype=global_defs.tReal))
#
#        return jnp.mean(actFun(res))
#
## ** end class PhaseRNN
#
#
#class PhaseRNNsym(nn.Module):
#    """Recurrent neural network with symmetries.
#    """
#
#    def apply(self, x, L=10, hiddenSize=10, depth=1, inputDim=2, passDim=None, actFun=nn.elu, initScale=1.0, orbit=None, z2sym=False):
#
#        self.rnn = PhaseRNN.shared(L=L, hiddenSize=hiddenSize, depth=depth, inputDim=inputDim, passDim=passDim, actFun=actFun, initScale=initScale, name='myRNN')
#        
#        x = jax.vmap(lambda o,s: jnp.dot(o,s), in_axes=(0,None))(orbit, x)
#
#        def evaluate(x):
#            return self.rnn(x)
#
#        res = jnp.mean(jax.vmap(evaluate)(x), axis=0)
#
#        if z2sym:
#            res = 0.5 * (res + jnp.mean(jax.vmap(evaluate)(1-x), axis=0))
#
#        return res
#
## ** end class PhaseRNNsym
#
#
#class CpxRNN(nn.Module):
#    """Recurrent neural network.
#    """
#
#    def apply(self, x, L=10, hiddenSize=10, inputDim=2, actFun=nn.elu, initScale=1.0, logProbFactor=0.5):
#        
#        rnnCell = RNNCell.shared(hiddenSize=hiddenSize, outDim=hiddenSize, actFun=actFun, initScale=initScale, name="myCell")
#
#        probDense = nn.Dense.shared(features=inputDim, name="probDense", dtype=global_defs.tReal,
#                                kernel_init=jax.nn.initializers.lecun_normal(dtype=global_defs.tReal), 
#                                bias_init=partial(jax.nn.initializers.zeros, dtype=global_defs.tReal))
#
#        state = jnp.zeros((hiddenSize,))
#
#        def rnn_cell(carry, x):
#            newCarry, out = rnnCell(carry[0],carry[1])
#            logProb = nn.log_softmax(actFun(probDense(out)))
#            logProb = jnp.sum( logProb * x, axis=-1 )
#            return (newCarry, x), (jnp.nan_to_num(logProb, nan=-35), out)
#      
#        _, (probs, phaseOut) = jax.lax.scan(rnn_cell, (state, jnp.zeros(inputDim)), jax.nn.one_hot(x,inputDim))
#
#        phase = nn.Dense(phaseOut, features=6, dtype=global_defs.tReal,
#                            kernel_init=jax.nn.initializers.lecun_normal(dtype=global_defs.tReal), 
#                            bias_init=partial(jax.nn.initializers.zeros, dtype=global_defs.tReal))
#        phase = actFun(phase)
#        phase = nn.Dense(phaseOut, features=4, dtype=global_defs.tReal,
#                            kernel_init=jax.nn.initializers.lecun_normal(dtype=global_defs.tReal), 
#                            bias_init=partial(jax.nn.initializers.zeros, dtype=global_defs.tReal))
#
#        return logProbFactor * jnp.sum(probs, axis=0) + 1.j * jnp.mean(actFun(phase))
#
#
#    @nn.module_method
#    def sample(self,batchSize,key,L,hiddenSize=10,inputDim=2,actFun=nn.elu, initScale=1.0, logProbFactor=0.5):
#        """sampler
#        """
#        
#        rnnCell = RNNCell.shared(hiddenSize=hiddenSize, outDim=inputDim, actFun=actFun, initScale=initScale, name="myCell")
#        
#        probDense = nn.Dense.shared(features=inputDim, name="probDense", dtype=global_defs.tReal)
#
#        outputs = jnp.asarray(np.zeros((batchSize,L,L)))
#        
#        state = jnp.zeros((batchSize, hiddenSize))
#            
#        def eval_cell(x,y):
#            newCarry, out = rnnCell(x,y)
#            return newCarry, actFun(probDense(out))
#
#        def rnn_cell(carry, x):
#            newCarry, logits = jax.vmap(eval_cell)(carry[0],carry[1])
#            sampleOut = jax.random.categorical( x, logits )
#            sample = jax.nn.one_hot( sampleOut, inputDim )
#            logProb = jnp.sum( nn.log_softmax(logits) * sample, axis=1 )
#            return (newCarry, sample), (jnp.nan_to_num(logProb, nan=-35), sampleOut)
# 
#        keys = jax.random.split(key,L)
#        _, res = jax.lax.scan( rnn_cell, (state, jnp.zeros((batchSize,inputDim))), keys )
#
#        return jnp.transpose( res[1] )#, 0.5 * jnp.sum(res[0], axis=0)
