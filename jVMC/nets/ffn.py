import jax
import flax
import flax.linen as nn
import jax.numpy as jnp

import jVMC.global_defs as global_defs

from functools import partial
from typing import Sequence

from jVMC.nets.initializers import init_fn_args

class FFN(nn.Module):
    """Feed forward network with real parameters.

    Initialization arguments:
        * ``layers``: Computational basis configuration.
        * ``bias``: ``Boolean`` indicating whether to use bias.
        * ``actFun``: Non-linear activation function.

    """
    layers: Sequence[int] = (10,)
    bias: bool = False
    actFun: Sequence[callable] = (jax.nn.elu,)

    @nn.compact
    def __call__(self, s):

        activationFunctions = [f for f in self.actFun]
        for l in range(len(activationFunctions), len(self.layers) + 1):
            activationFunctions.append(self.actFun[-1])
        
        init_args = init_fn_args(dtype=global_defs.tReal, 
                                    kernel_init=jax.nn.initializers.lecun_normal(), 
                                    bias_init=jax.nn.initializers.zeros)

        s = 2 * s.ravel() - 1
        for l, fun in zip(self.layers, activationFunctions[:-1]):
            s = fun(
                nn.Dense(features=l, use_bias=self.bias, **init_args)(s)
            )

        return jnp.sum(activationFunctions[-1](
                        nn.Dense(features=1, use_bias=self.bias, **init_args)(s)
                      ))

# ** end class FFN
