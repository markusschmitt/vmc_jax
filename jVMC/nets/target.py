##############################################
# J Rigo
# rigojonas@gmail.com
# Regensburg 20/11/2023
##############################################

import jax
from jax.config import config
config.update("jax_enable_x64", True)
import flax
#from flax import nn
import flax.linen as nn
import jax.numpy as jnp

import jVMC.global_defs as global_defs
import jVMC.nets.activation_functions as act_funs
from jVMC.nets.initializers import init_fn_args

from functools import partial

import jVMC.nets.initializers


class Target(nn.Module):
  """Target wave function, returns a vector with the same dimension as the Hilbert space

    Initialization arguments:
        * ``L``: System size
        * ``d``: local Hilbert space dimension

    """
  L: int
  d: float = 2.00

  @nn.compact
  def __call__(self, s):
    kernel = self.param('kernel',
                        nn.initializers.constant(1),
                        (int(self.d**self.L)))
    # return amplitude for state s
    return jnp.log(kernel[((self.d**jnp.arange(self.L)).dot(s)).astype(int)])
