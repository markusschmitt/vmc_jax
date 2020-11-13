import jax
from jax.config import config
config.update("jax_enable_x64", True)
import flax
from flax import nn
import jax.numpy as jnp

import jVMC.global_defs as global_defs

from functools import partial

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
