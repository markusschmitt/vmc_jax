import jax
from jax.config import config
config.update("jax_enable_x64", True)
import flax
from flax import nn
import numpy as np
import jax.numpy as jnp

import jVMC.global_defs as global_defs
import jVMC.activation_functions as act_funs

from functools import partial

def cplx_init(rng, shape):
    rng1,rng2 = jax.random.split(rng)
    unif=jax.nn.initializers.uniform()
    return unif(rng1,shape,dtype=global_defs.tReal)+1.j*unif(rng2,shape,dtype=global_defs.tReal)

# ! ! !
# Nets have to be defined to act on a single configuration (not a batch)

class CpxRBM(nn.Module):

    def apply(self, s, numHidden=2, bias=False):
        """Restricted Boltzmann machine with complex parameters.

        Args:

            * ``s``: Computational basis configuration.
            * ``numHidden``: Number of hidden units.
            * ``bias``: ``Boolean`` indicating whether to use bias.
        """

        layer = nn.Dense.shared(features=numHidden, name='rbm_layer', bias=bias, dtype=global_defs.tCpx,
                                kernel_init=cplx_init, 
                                bias_init=partial(jax.nn.initializers.zeros, dtype=global_defs.tCpx))

        return jnp.sum(jnp.log(jnp.cosh(layer(2*s-1))))


# ** end class CpxRBM


class RBM(nn.Module):

    def apply(self, s, numHidden=2, bias=False):
        """Restricted Boltzmann machine with real parameters.

        Args:

            * ``s``: Computational basis configuration.
            * ``numHidden``: Number of hidden units.
            * ``bias``: ``Boolean`` indicating whether to use bias.
        """

        layer = nn.Dense.shared(features=numHidden, name='rbm_layer', bias=bias, dtype=global_defs.tReal, 
                                kernel_init=jax.nn.initializers.lecun_normal(dtype=global_defs.tReal), 
                                bias_init=partial(jax.nn.initializers.zeros, dtype=global_defs.tReal))

        return jnp.sum(jnp.log(jnp.cosh(layer(2*s-1))))

# ** end class RBM


class FFN(nn.Module):
    """Feed-forward network.
    """

    def apply(self, s, layers=[10], bias=False, actFun=[jax.nn.elu,]):
        
        for l in range(len(actFun),len(layers)+1):
            actFun.append(actFun[-1])

        s = 2*s-1
        for l,fun in zip(layers,actFun[:-1]):
            s = fun(
                    nn.Dense(s, features=l, bias=bias, dtype=global_defs.tReal, 
                                kernel_init=jax.nn.initializers.lecun_normal(dtype=global_defs.tReal), 
                                bias_init=partial(jax.nn.initializers.zeros, dtype=global_defs.tReal))
                )

        return jnp.sum(actFun[-1]( nn.Dense(s, features=1, bias=bias, dtype=global_defs.tReal,
                                kernel_init=jax.nn.initializers.lecun_normal(dtype=global_defs.tReal), 
                                bias_init=partial(jax.nn.initializers.zeros, dtype=global_defs.tReal))
                     ))

# ** end class FFN


class CNN(nn.Module):
    """Convolutional neural network.
    """

    def apply(self, x, F=[8,], channels=[10], strides=[1], actFun=[nn.elu], bias=True, firstLayerBias=False):
      
        initFunction = partial(jax.nn.initializers.variance_scaling(scale=1.0, mode="fan_avg", distribution="uniform"), 
                               dtype=global_defs.tReal)
       
        # Set up padding for periodic boundary conditions 
        # Padding size must be 1 - filter diameter
        pads=[(0,0)]
        for f in F:
            pads.append((0,f-1))
        pads.append((0,0))

        bias=[bias]*len(channels)
        bias[0] = firstLayerBias

        for l in range(len(actFun),len(channels)):
            actFun.append(actFun[-1])

        # List of axes that will be summed for symmetrization
        reduceDims=tuple([-i-1 for i in range(len(strides)+2)])

        # Add feature dimension
        #x = jnp.expand_dims(2*x-1, axis=-1)
        x = jnp.expand_dims(jnp.expand_dims(2*x-1, axis=0), axis=-1)
        for c,fun,b in zip(channels,actFun,bias):
            x = jnp.pad(x, pads, 'wrap')
            x = fun( nn.Conv(x, features=c, kernel_size=tuple(F),
                             strides=strides, padding=[(0,0)]*len(strides),
                             bias=b, dtype=global_defs.tReal,
                             kernel_init=initFunction) )

        nrm = jnp.sqrt( jnp.prod(jnp.array(x.shape[reduceDims[-1]:])) )
        
        return jnp.sum(x, axis=reduceDims) / nrm

# ** end class CNN


class CpxCNN(nn.Module):
    """Convolutional neural network with complex parameters.
    """

    def apply(self, x, F=[8], channels=[10], strides=[1], actFun=[act_funs.poly6,], bias=True):
      
        initFunction = jax.nn.initializers.variance_scaling(scale=1.0, mode="fan_avg", distribution="uniform")
 
        # Set up padding for periodic boundary conditions 
        # Padding size must be 1 - filter diameter
        pads=[(0,0)]
        for f in F:
            pads.append((0,f-1))
        pads.append((0,0))

        for l in range(len(actFun),len(channels)):
            actFun.append(actFun[-1])

        # List of axes that will be summed for symmetrization
        reduceDims=tuple([-i-1 for i in range(len(strides)+2)])

        # Add feature dimension
        #x = jnp.expand_dims(2*x-1, axis=-1)
        x = jnp.expand_dims(jnp.expand_dims(2*x-1, axis=0), axis=-1)
        for c,f in zip(channels, actFun):
            x = jnp.pad(x, pads, 'wrap')
            x = f( nn.Conv(x, features=c, kernel_size=tuple(F),
                           strides=strides, padding=[(0,0)]*len(strides),
                           bias=bias, dtype=global_defs.tCpx,
                           kernel_init=initFunction) )

        nrm = jnp.sqrt( jnp.prod(jnp.array(x.shape[reduceDims[-1]:])) )
        
        return jnp.sum(x, axis=reduceDims) / nrm

# ** end class CNN


class RNNCell(nn.Module):

    def apply(self, carry, x, hiddenSize=10, outDim=2, actFun=nn.elu, initScale=1.0):
        
        initFunctionCell = partial(jax.nn.initializers.variance_scaling(scale=1.0, mode="fan_avg", distribution="uniform"),
                                    dtype=global_defs.tReal)
        initFunctionOut = partial(jax.nn.initializers.variance_scaling(scale=initScale, mode="fan_in", distribution="uniform"),
                                    dtype=global_defs.tReal)

        cellIn = nn.Dense.partial(features=hiddenSize,
                                    bias=False, dtype=global_defs.tReal,
                                    kernel_init=initFunctionCell)
        cellCarry = nn.Dense.partial(features=hiddenSize,
                                    bias=True,
                                    kernel_init=initFunctionCell, dtype=global_defs.tReal,
                                    bias_init=partial(jax.nn.initializers.zeros,dtype=global_defs.tReal))

        outputDense = nn.Dense.partial(features=outDim,
                                      kernel_init=initFunctionOut, dtype=global_defs.tReal,
                                      bias_init=jax.nn.initializers.normal(stddev=0.1,dtype=global_defs.tReal))
        
        newCarry = actFun(cellCarry(carry) + cellIn(x))
        out = outputDense(newCarry)
        return newCarry, out

# ** end class RNNCell


class RNNCellStack(nn.Module):

    def apply(self, carry, x, hiddenSize=10, outDim=2, actFun=nn.elu, initScale=1.0):

        newCarry = [None] * carry.shape[0]
        
        newR = x
        # Can't use scan for this, because then flax doesn't realize that each cell has different parameters
        for j,c in enumerate(carry):
            newCarry[j], newR = RNNCell(c, newR, hiddenSize=hiddenSize, outDim=outDim, actFun=actFun, initScale=initScale)

        return jnp.array(newCarry), newR

# ** end class RNNCellStack


class RNN(nn.Module):
    """Recurrent neural network.
    """

    def apply(self, x, L=10, hiddenSize=10, depth=1, inputDim=2, actFun=nn.elu, initScale=1.0, logProbFactor=0.5):
        
        rnnCell = RNNCellStack.shared(hiddenSize=hiddenSize, outDim=inputDim, actFun=actFun, initScale=initScale, name="myCell")

        state = jnp.zeros((depth, hiddenSize))

        def rnn_cell(carry, x):
            newCarry, out = rnnCell(carry[0],carry[1])
            logProb = nn.log_softmax(out)
            logProb = jnp.sum( logProb * x, axis=-1 )
            return (newCarry, x), jnp.nan_to_num(logProb, nan=-35)
      
        _, probs = jax.lax.scan(rnn_cell, (state, jnp.zeros(inputDim)), jax.nn.one_hot(x,inputDim))

        return logProbFactor * jnp.sum(probs, axis=0)


    @nn.module_method
    def sample(self,batchSize,key,L,hiddenSize=10,depth=1,inputDim=2,actFun=nn.elu, initScale=1.0, logProbFactor=0.5):
        """sampler
        """
        
        rnnCell = RNNCellStack.shared(hiddenSize=hiddenSize, outDim=inputDim, actFun=actFun, initScale=initScale, name="myCell")

        outputs = jnp.asarray(np.zeros((batchSize,L,L)))
        
        state = jnp.zeros((batchSize, depth, hiddenSize))

        def rnn_cell(carry, x):
            newCarry, logits = jax.vmap(rnnCell)(carry[0],carry[1])
            sampleOut = jax.random.categorical( x, logits )
            sample = jax.nn.one_hot( sampleOut, inputDim )
            logProb = jnp.sum( nn.log_softmax(logits) * sample, axis=1 )
            return (newCarry, sample), (jnp.nan_to_num(logProb, nan=-35), sampleOut)
 
        keys = jax.random.split(key,L)
        _, res = jax.lax.scan( rnn_cell, (state, jnp.zeros((batchSize,inputDim))), keys )

        return jnp.transpose( res[1] )#, 0.5 * jnp.sum(res[0], axis=0)

# ** end class RNN


class RNNsym(nn.Module):
    """Recurrent neural network with symmetries.
    """

    def apply(self, x, L=10, hiddenSize=10, depth=1, inputDim=2, actFun=nn.elu, initScale=1.0, logProbFactor=0.5, orbit=None):

        self.rnn = RNN.shared(L=L, hiddenSize=hiddenSize, depth=depth, inputDim=inputDim, actFun=actFun, initScale=initScale, logProbFactor=logProbFactor, name='myRNN')

        self.orbit = orbit
        
        x = jax.vmap(lambda o,s: jnp.dot(o,s), in_axes=(0,None))(self.orbit, x)

        def evaluate(x):
            return self.rnn(x)

        logProbs = logProbFactor * jnp.log( jnp.mean(jnp.exp((1./logProbFactor)*jax.vmap(evaluate)(x)), axis=0) )

        return logProbs

    @nn.module_method
    def sample(self,batchSize,key,L,hiddenSize=10,depth=1,inputDim=2,actFun=nn.elu, initScale=1.0, logProbFactor=0.5, orbit=None):
        
        rnn = RNN.shared(L=L, hiddenSize=hiddenSize, depth=depth, inputDim=inputDim, actFun=actFun, initScale=initScale, logProbFactor=logProbFactor, name='myRNN')

        key1, key2 = jax.random.split(key)

        configs = rnn.sample(batchSize, key1)

        orbitIdx = jax.random.choice(key2, orbit.shape[0], shape=(batchSize,))

        configs = jax.vmap(lambda k,o,s: jnp.dot(o[k], s), in_axes=(0,None,0))(orbitIdx, orbit, configs)

        return configs

# ** end class RNNsym


class LSTMCell(nn.Module):

    def apply(self, carry, x, inputDim=2, actFun=nn.elu):

        newCarry, out = nn.LSTMCell(carry, x)

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


class PhaseRNN(nn.Module):
    """Recurrent neural network.
    """

    def apply(self, x, L=10, hiddenSize=10, depth=1, inputDim=2, actFun=nn.elu, initScale=1.0):
        
        rnnCell = RNNCellStack.partial(hiddenSize=hiddenSize, outDim=inputDim, actFun=actFun, initScale=initScale)

        state = jnp.zeros((depth, hiddenSize))

        def rnn_cell(carry, x):
            newCarry, out = rnnCell(carry[0],carry[1])
            return (newCarry, x), out
      
        _, res = jax.lax.scan(rnn_cell, (state, jnp.zeros(inputDim)), jax.nn.one_hot(x,inputDim))

        res = nn.Dense(res.ravel(), features=8, dtype=global_defs.tReal,
                        kernel_init=jax.nn.initializers.lecun_normal(dtype=global_defs.tReal), 
                        bias_init=partial(jax.nn.initializers.zeros, dtype=global_defs.tReal))

        return jnp.mean(actFun(res))

# ** end class PhaseRNN


class PhaseRNNsym(nn.Module):
    """Recurrent neural network with symmetries.
    """

    def apply(self, x, L=10, hiddenSize=10, depth=1, inputDim=2, actFun=nn.elu, initScale=1.0, orbit=None):

        self.rnn = PhaseRNN.shared(L=L, hiddenSize=hiddenSize, depth=depth, inputDim=inputDim, actFun=actFun, initScale=initScale, name='myRNN')
        
        x = jax.vmap(lambda o,s: jnp.dot(o,s), in_axes=(0,None))(orbit, x)

        def evaluate(x):
            return self.rnn(x)

        res = jnp.mean(jax.vmap(evaluate)(x), axis=0)

        return res

# ** end class PhaseRNNsym


class CpxRNN(nn.Module):
    """Recurrent neural network.
    """

    def apply(self, x, L=10, hiddenSize=10, inputDim=2, actFun=nn.elu, initScale=1.0, logProbFactor=0.5):
        
        rnnCell = RNNCell.shared(hiddenSize=hiddenSize, outDim=hiddenSize, actFun=actFun, initScale=initScale, name="myCell")

        probDense = nn.Dense.shared(features=inputDim, name="probDense", dtype=global_defs.tReal,
                                kernel_init=jax.nn.initializers.lecun_normal(dtype=global_defs.tReal), 
                                bias_init=partial(jax.nn.initializers.zeros, dtype=global_defs.tReal))

        state = jnp.zeros((hiddenSize,))

        def rnn_cell(carry, x):
            newCarry, out = rnnCell(carry[0],carry[1])
            logProb = nn.log_softmax(actFun(probDense(out)))
            logProb = jnp.sum( logProb * x, axis=-1 )
            return (newCarry, x), (jnp.nan_to_num(logProb, nan=-35), out)
      
        _, (probs, phaseOut) = jax.lax.scan(rnn_cell, (state, jnp.zeros(inputDim)), jax.nn.one_hot(x,inputDim))

        phase = nn.Dense(phaseOut, features=6, dtype=global_defs.tReal,
                            kernel_init=jax.nn.initializers.lecun_normal(dtype=global_defs.tReal), 
                            bias_init=partial(jax.nn.initializers.zeros, dtype=global_defs.tReal))
        phase = actFun(phase)
        phase = nn.Dense(phaseOut, features=4, dtype=global_defs.tReal,
                            kernel_init=jax.nn.initializers.lecun_normal(dtype=global_defs.tReal), 
                            bias_init=partial(jax.nn.initializers.zeros, dtype=global_defs.tReal))

        return logProbFactor * jnp.sum(probs, axis=0) + 1.j * jnp.mean(actFun(phase))


    @nn.module_method
    def sample(self,batchSize,key,L,hiddenSize=10,inputDim=2,actFun=nn.elu, initScale=1.0, logProbFactor=0.5):
        """sampler
        """
        
        rnnCell = RNNCell.shared(hiddenSize=hiddenSize, outDim=inputDim, actFun=actFun, initScale=initScale, name="myCell")
        
        probDense = nn.Dense.shared(features=inputDim, name="probDense", dtype=global_defs.tReal)

        outputs = jnp.asarray(np.zeros((batchSize,L,L)))
        
        state = jnp.zeros((batchSize, hiddenSize))
            
        def eval_cell(x,y):
            newCarry, out = rnnCell(x,y)
            return newCarry, actFun(probDense(out))

        def rnn_cell(carry, x):
            newCarry, logits = jax.vmap(eval_cell)(carry[0],carry[1])
            sampleOut = jax.random.categorical( x, logits )
            sample = jax.nn.one_hot( sampleOut, inputDim )
            logProb = jnp.sum( nn.log_softmax(logits) * sample, axis=1 )
            return (newCarry, sample), (jnp.nan_to_num(logProb, nan=-35), sampleOut)
 
        keys = jax.random.split(key,L)
        _, res = jax.lax.scan( rnn_cell, (state, jnp.zeros((batchSize,inputDim))), keys )

        return jnp.transpose( res[1] )#, 0.5 * jnp.sum(res[0], axis=0)

# ** end class CpxRNN
