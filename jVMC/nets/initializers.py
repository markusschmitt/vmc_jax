import jax
jax.config.update("jax_enable_x64", True)
import flax
import flax.linen as nn

import jVMC.global_defs as global_defs
from functools import partial

# Workaround for backwards compatibility with flax0.3.6
# def init_fn_args(dtype=None, **kwargs):
def init_fn_args(**kwargs):
    if "dtype" in kwargs.keys():
        kwargs["param_dtype"] = kwargs["dtype"]
    return kwargs


def cplx_init(rng, shape, dtype):
    rng1, rng2 = jax.random.split(rng)
    unif = jax.nn.initializers.uniform()
    return unif(rng1, shape, dtype=global_defs.tReal) + 1.j * unif(rng2, shape, dtype=global_defs.tReal)


def cplx_variance_scaling(rng, shape, dtype):
    rng1, rng2 = jax.random.split(rng)
    unif = jax.nn.initializers.uniform(scale=1.)
    elems = 1
    for k in shape[:-2]:
        elems *= k
    w = jax.numpy.sqrt((shape[-1] + shape[-2]) * elems)
    return (1. / w) * unif(rng1, shape, dtype=global_defs.tReal) * jax.numpy.exp(1.j * 3.141593 * unif(rng2, shape, dtype=global_defs.tReal))
