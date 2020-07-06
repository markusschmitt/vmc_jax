import jax
import flax
from flax import nn
import numpy as np
import jax.numpy as jnp

import jVMC.global_defs as global_defs

from functools import partial

def cplx_init(rng, shape):
    rng1,rng2 = jax.random.split(rng)
    unif=jax.nn.initializers.uniform()
    return unif(rng1,shape)+1.j*unif(rng2,shape)

class CpxRBM(nn.Module):

    def apply(self, s, numHidden=2, bias=False):

        layer = nn.Dense.shared(features=numHidden, name='rbm_layer', bias=bias, dtype=global_defs.tCpx,
                                kernel_init=cplx_init, 
                                bias_init=partial(jax.nn.initializers.zeros, dtype=global_defs.tCpx))

        return jnp.sum(jnp.log(jnp.cosh(layer(2*s-1))))

# ** end class CpxRBM

class RBM(nn.Module):

    def apply(self, s, numHidden=2, bias=False):

        layer = nn.Dense.shared(features=numHidden, name='rbm_layer', bias=bias, dtype=global_defs.tReal, 
                                kernel_init=jax.nn.initializers.lecun_normal(dtype=global_defs.tReal), 
                                bias_init=partial(jax.nn.initializers.zeros, dtype=global_defs.tReal))

        return jnp.sum(jnp.log(jnp.cosh(layer(2*s-1))))

# ** end class RBM


class FFN(nn.Module):

    def apply(self, s, layers=[10], bias=False, actFun=jax.nn.elu):

        s = 2*s-1
        for l in layers:
            s = actFun(
                    nn.Dense(s, features=l, bias=bias, dtype=global_defs.tReal, 
                                kernel_init=jax.nn.initializers.lecun_normal(dtype=global_defs.tReal), 
                                bias_init=partial(jax.nn.initializers.zeros, dtype=global_defs.tReal))
                )

        return jnp.sum(actFun( nn.Dense(s, features=1, bias=bias, dtype=global_defs.tReal,
                                kernel_init=jax.nn.initializers.lecun_normal(dtype=global_defs.tReal), 
                                bias_init=partial(jax.nn.initializers.zeros, dtype=global_defs.tReal))
                     ))

# ** end class FFN


class CNN(nn.Module):

    def apply(self, x, F=(8,), channels=[10], strides=[1], actFun=nn.elu, bias=True):
       
        # Set up padding for periodic boundary conditions 
        pads=[(0,0)]
        for f in F:
            pads.append((0,f-1))
        pads.append((0,0))

        # List of axes that will be summed for symmetrization
        reduceDims=tuple([-i-1 for i in range(len(strides)+1)])

        # Add feature dimension
        x = jnp.expand_dims(2*x-1, axis=-1)
        for c in channels:
            x = jnp.pad(x, pads, 'wrap')
            x = actFun( nn.Conv(x, features=c, kernel_size=F, strides=strides, padding=[(0,0)]*len(strides), bias=bias, dtype=global_defs.tReal) )

        nrm = jnp.sqrt( jnp.prod(x.shape[reduceDims[-1]:]) )
        return jnp.sum(x, axis=reduceDims) / nrm

# ** end class CNN
