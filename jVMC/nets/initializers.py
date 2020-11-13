import jax
from jax.config import config
config.update("jax_enable_x64", True)
import flax
from flax import nn

import jVMC.global_defs as global_defs

def cplx_init(rng, shape):
    rng1,rng2 = jax.random.split(rng)
    unif=jax.nn.initializers.uniform()
    return unif(rng1,shape,dtype=global_defs.tReal)+1.j*unif(rng2,shape,dtype=global_defs.tReal)
