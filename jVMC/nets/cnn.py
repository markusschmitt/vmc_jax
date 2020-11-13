import jax
from jax.config import config
config.update("jax_enable_x64", True)
import flax
from flax import nn
import jax.numpy as jnp

import jVMC.global_defs as global_defs
import jVMC.activation_functions as act_funs

from functools import partial

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
