import jax
import flax
from flax import nn
import numpy as np
import jax.numpy as jnp

def cplx_init(rng, shape):
    rng1,rng2 = jax.random.split(rng)
    unif=jax.nn.initializers.uniform()
    return unif(rng1,shape)+1.j*unif(rng2,shape)

class CpxRBM(nn.Module):
    def apply(self, s, L=4, numHidden=2, bias=False):
        layer = nn.Dense.shared(features=numHidden, name='rbm_layer', bias=bias, dtype=np.complex64,
                                kernel_init=cplx_init, bias_init=cplx_init)

        return jnp.sum(jnp.log(jnp.cosh(layer(s))), axis=0)

class RBM(nn.Module):
    def apply(self, s, L=4, numHidden=2, bias=False):
        layer = nn.Dense.shared(features=numHidden, name='rbm_layer', bias=bias, dtype=np.float32)

        return jnp.sum(jnp.log(jnp.cosh(layer(s))))

