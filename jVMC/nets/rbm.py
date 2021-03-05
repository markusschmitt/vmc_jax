import jax
from jax.config import config
config.update("jax_enable_x64", True)
import flax
#from flax import nn
import flax.linen as nn
import jax.numpy as jnp

import jVMC.global_defs as global_defs
import jVMC.activation_functions as act_funs

from functools import partial

import jVMC.nets.initializers

class CpxRBM(nn.Module):
    """Restricted Boltzmann machine with complex parameters.

    Args:

        * ``s``: Computational basis configuration.
        * ``numHidden``: Number of hidden units.
        * ``bias``: ``Boolean`` indicating whether to use bias.
    """
    numHidden: int = 2
    bias: bool = False

    @nn.compact
    def __call__(self, s):

        layer = nn.Dense(self.numHidden, name='rbm_layer', use_bias=self.bias, dtype=global_defs.tCpx,
                                kernel_init=jVMC.nets.initializers.cplx_init, 
                                bias_init=partial(jax.nn.initializers.zeros, dtype=global_defs.tCpx))

        return jnp.sum(jnp.log(jnp.cosh(layer(2*s-1))))

# ** end class CpxRBM


class RBM(nn.Module):
    """Restricted Boltzmann machine with real parameters.

    Args:

        * ``s``: Computational basis configuration.
        * ``numHidden``: Number of hidden units.
        * ``bias``: ``Boolean`` indicating whether to use bias.
    """
    numHidden: int = 2
    bias: bool = False

    @nn.compact
    def __call__(self, s):

        layer = nn.Dense(self.numHidden, name='rbm_layer', use_bias=self.bias, dtype=global_defs.tReal, 
                                kernel_init=jax.nn.initializers.lecun_normal(dtype=global_defs.tReal), 
                                bias_init=partial(jax.nn.initializers.zeros, dtype=global_defs.tReal))

        return jnp.sum(jnp.log(jnp.cosh(layer(2*s-1))))

# ** end class RBM
