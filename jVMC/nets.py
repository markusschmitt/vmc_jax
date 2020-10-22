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


class RNN(nn.Module):
    """Recurrent neural network.
    """

    def apply(self, x, L=10, units=[10], inputDim=2, actFun=[nn.elu,], initScale=1.0):

        initFunctionCell = partial(jax.nn.initializers.variance_scaling(scale=1.0, mode="fan_avg", distribution="uniform"),
                                    dtype=global_defs.tReal)
        initFunctionOut = partial(jax.nn.initializers.variance_scaling(scale=initScale, mode="fan_in", distribution="uniform"),
                                    dtype=global_defs.tReal)

        cellIn = nn.Dense.shared(features=units[0],
                                    name='rnn_cell_in',
                                    bias=False, dtype=global_defs.tReal,
                                    kernel_init=initFunctionCell)
        cellCarry = nn.Dense.shared(features=units[0],
                                    name='rnn_cell_carry',
                                    bias=True,
                                    kernel_init=initFunctionCell, dtype=global_defs.tReal,
                                    bias_init=partial(jax.nn.initializers.zeros, dtype=global_defs.tReal))

        outputDense = nn.Dense.shared(features=inputDim,
                                      name='rnn_output_dense',
                                      kernel_init=initFunctionOut, dtype=global_defs.tReal,
                                      bias_init=partial(jax.nn.initializers.zeros, dtype=global_defs.tReal))

        state = jnp.zeros((units[0]))

        def rnn_cell(carry, x):
            newCarry = actFun[0](cellCarry(carry[0]) + cellIn(carry[1]))
            prob = nn.softmax(outputDense(newCarry))
            prob = jnp.log( jnp.sum( prob * x, axis=-1 ) )
            return (newCarry, x), prob
      
        _, probs = jax.lax.scan(rnn_cell, (state, jnp.zeros(inputDim)), jax.nn.one_hot(x,inputDim))

        return 0.5 * jnp.sum(probs, axis=0)


    @nn.module_method
    def sample(self,batchSize,key,L,units,inputDim=2,actFun=[nn.elu,], initScale=1.0):
        """sampler
        """

        initFunctionCell = partial(jax.nn.initializers.variance_scaling(scale=1.0, mode="fan_avg", distribution="uniform"),
                                    dtype=global_defs.tReal)
        initFunctionOut = partial(jax.nn.initializers.variance_scaling(scale=initScale, mode="fan_in", distribution="uniform"),
                                    dtype=global_defs.tReal)

        cellIn = nn.Dense.shared(features=units[0],
                                    name='rnn_cell_in',
                                    bias=False, dtype=global_defs.tReal,
                                    kernel_init=initFunctionCell)
        cellCarry = nn.Dense.shared(features=units[0],
                                    name='rnn_cell_carry',
                                    bias=True,
                                    kernel_init=initFunctionCell, dtype=global_defs.tReal,
                                    bias_init=partial(jax.nn.initializers.zeros, dtype=global_defs.tReal))

        outputDense = nn.Dense.shared(features=inputDim,
                                      name='rnn_output_dense',
                                      kernel_init=initFunctionOut, dtype=global_defs.tReal,
                                      bias_init=partial(jax.nn.initializers.zeros, dtype=global_defs.tReal))

        outputs = jnp.asarray(np.zeros((batchSize,L,L)))
        
        state = jnp.zeros((batchSize, units[0]))

        def rnn_cell(carry, x):
            newCarry = actFun[0]( cellCarry(carry[0]) + cellIn(carry[1]) )
            logits = outputDense(newCarry)
            sampleOut = jax.random.categorical( x, logits )
            sample = jax.nn.one_hot( sampleOut, inputDim )
            logProb = jnp.log( jnp.sum( nn.softmax(logits) * sample, axis=1 ) )
            return (newCarry, sample), (logProb, sampleOut)
 
        keys = jax.random.split(key,L)
        _, res = jax.lax.scan( rnn_cell, (state, jnp.zeros((batchSize,inputDim))), keys )

        return jnp.transpose( res[1] )#, 0.5 * jnp.sum(res[0], axis=0)

# ** end class RNN
