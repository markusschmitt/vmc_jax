from collections.abc import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp

from jVMC import global_defs
from jVMC.nets.initializers import init_fn_args


class FFN(nn.Module):
    """Feed forward network with real parameters.

    Initialization arguments:
        * ``layers``: Computational basis configuration.
        * ``bias``: ``Boolean`` indicating whether to use bias.
        * ``act_fun``: Non-linear activation function.

    """

    layers: Sequence[int] = (10,)
    bias: bool = False
    act_fun: Sequence[callable] = (jax.nn.elu,)

    @nn.compact
    def __call__(self, s):

        activation_functions = [f for f in self.act_fun]
        for l in range(len(activation_functions), len(self.layers) + 1):
            activation_functions.append(self.act_fun[-1])

        init_args = init_fn_args(
            param_dtype=global_defs.DT_PARAMS_REAL,
            kernel_init=jax.nn.initializers.lecun_normal(),
            bias_init=jax.nn.initializers.zeros,
        )

        s = 2 * s.ravel() - 1
        for l, fun in zip(self.layers, activation_functions[:-1]):
            s = fun(nn.Dense(features=l, use_bias=self.bias, **init_args)(s))

        return jnp.sum(
            activation_functions[-1](
                nn.Dense(features=1, use_bias=self.bias, **init_args)(s)
            )
        )
