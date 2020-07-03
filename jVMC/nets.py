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

# ** end class RBM
