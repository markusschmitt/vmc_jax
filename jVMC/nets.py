import jax
import flax
from flax import nn
import numpy as np
import jax.numpy as jnp

import jVMC.global_defs as global_defs

def cplx_init(rng, shape):
    rng1,rng2 = jax.random.split(rng)
    unif=jax.nn.initializers.uniform()
    return unif(rng1,shape)+1.j*unif(rng2,shape)

def cplx_zeros(rng, shape):
    return jnp.zeros(shape, dtype=global_defs.tCpx)

class CpxRBM(nn.Module):
    def apply(self, s, L=4, numHidden=2, bias=False):
        layer = nn.Dense.shared(features=numHidden, name='rbm_layer', bias=bias, dtype=global_defs.tCpx,
                                kernel_init=cplx_init, bias_init=cplx_zeros)

        return jnp.sum(jnp.log(jnp.cosh(layer(2*s-1))))

class RBM(nn.Module):
    def apply(self, s, L=4, numHidden=2, bias=False):
        layer = nn.Dense.shared(features=numHidden, name='rbm_layer', bias=bias, dtype=global_defs.tReal)

        return jnp.sum(jnp.log(jnp.cosh(layer(s))))

