import jax
from jax.config import config
config.update("jax_enable_x64", True)
import flax.linen as nn

import jVMC.global_defs as global_defs


def cplx_init(rng, shape):
    rng1, rng2 = jax.random.split(rng)
    unif = jax.nn.initializers.uniform()
    return unif(rng1, shape, dtype=global_defs.tReal) + 1.j * unif(rng2, shape, dtype=global_defs.tReal)


def cplx_variance_scaling(rng, shape):
    rng1, rng2 = jax.random.split(rng)
    unif = jax.nn.initializers.uniform(scale=1.)
    elems = 1
    for k in shape[:-2]:
        elems *= k
    w = jax.numpy.sqrt((shape[-1] + shape[-2]) * elems)
    return (1. / w) * unif(rng1, shape, dtype=global_defs.tReal) * jax.numpy.exp(1.j * 3.141593 * unif(rng2, shape, dtype=global_defs.tReal))
