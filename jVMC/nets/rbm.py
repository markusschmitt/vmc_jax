import flax.linen as nn
import jax
import jax.numpy as jnp

import jVMC.nets.activation_functions as act_funs
import jVMC.nets.initializers
from jVMC import global_defs
from jVMC.nets.initializers import init_fn_args


class CpxRBM(nn.Module):
    """Restricted Boltzmann machine with complex parameters.

    Initialization arguments:
        * ``s``: Computational basis configuration.
        * ``num_hidden``: Number of hidden units.
        * ``bias``: ``Boolean`` indicating whether to use bias.

    """

    num_hidden: int = 2
    bias: bool = False

    @nn.compact
    def __call__(self, s):
        layer = nn.Dense(
            self.num_hidden,
            use_bias=self.bias,
            **init_fn_args(
                kernel_init=jVMC.nets.initializers.cplx_init,
                bias_init=jax.nn.initializers.zeros,
                param_dtype=global_defs.DT_PARAMS_CPX,
            ),
        )

        x = act_funs.log_cosh(layer(2 * s.ravel() - 1))

        return jnp.sum(x, dtype=x.dtype)


class CpxRBMNoSpinFlip(nn.Module):
    """Restricted Boltzmann machine with complex parameters.

    Initialization arguments:
        * ``s``: Computational basis configuration.
        * ``num_hidden``: Number of hidden units.
        * ``bias``: ``Boolean`` indicating whether to use bias.

    """

    num_hidden: int = 2
    bias: bool = False

    @nn.compact
    def __call__(self, s):
        layer = nn.Dense(
            self.num_hidden,
            use_bias=self.bias,
            **init_fn_args(
                kernel_init=jVMC.nets.initializers.cplx_init,
                bias_init=jax.nn.initializers.zeros,
                param_dtype=global_defs.DT_PARAMS_CPX,
            ),
        )

        x = act_funs.log_cosh(layer(s.ravel()))

        return jnp.sum(x, dtype=x.dtype)


class RBM(nn.Module):
    """Restricted Boltzmann machine with real parameters.

    Initialization arguments:
        * ``s``: Computational basis configuration.
        * ``num_hidden``: Number of hidden units.
        * ``bias``: ``Boolean`` indicating whether to use bias.

    """

    num_hidden: int = 2
    bias: bool = False

    @nn.compact
    def __call__(self, s):
        layer = nn.Dense(
            self.num_hidden,
            use_bias=self.bias,
            **init_fn_args(
                kernel_init=jax.nn.initializers.lecun_normal(
                    dtype=global_defs.DT_PARAMS_REAL
                ),
                bias_init=jax.nn.initializers.zeros,
                param_dtype=global_defs.DT_PARAMS_REAL,
            ),
        )

        x = act_funs.log_cosh(layer(2 * s.ravel() - 1))

        return jnp.sum(x, dtype=x.dtype)


# TODO: isn't this class completely useless?
class CpxRBMRatio(nn.Module):
    """Restricted Boltzmann machine with complex parameters.

    Initialization arguments:
        * ``s``: Computational basis configuration.
        * ``num_hidden``: Number of hidden units.
        * ``bias``: ``Boolean`` indicating whether to use bias.

    """

    num_hidden: int = 2
    bias: bool = False

    @nn.compact
    def __call__(self, s):
        layer = nn.Dense(
            self.num_hidden,
            use_bias=self.bias,
            **init_fn_args(
                kernel_init=jVMC.nets.initializers.cplx_init,
                bias_init=jax.nn.initializers.zeros,
                param_dtype=global_defs.DT_PARAMS_CPX,
            ),
        )

        x = act_funs.log_cosh(layer(2 * s.ravel() - 1))

        return jnp.sum(x, dtype=x.dtype)

    def eval_ratio(self, s, sp):
        return jnp.exp(self(sp) - self(s))
