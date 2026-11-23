import flax.linen as nn
import jax.numpy as jnp

from jVMC import global_defs


class FullConfigurationState(nn.Module):
    """
    Full configuration wave function, returns a vector with the same dimension as the Hilbert space

    Initialization arguments:
            * ``L``: System size
            * ``d``: local Hilbert space dimension
            * ``delta``: small number to avoid log(0)

    """

    L: int
    d: float = 2.00
    delta: float = 1e-15

    @property
    def num_parameters(self):
        return int(self.d**self.L)

    @nn.compact
    def __call__(self, s):
        kernel = self.param(
            "kernel",
            nn.initializers.constant(1, dtype=global_defs.DT_PARAMS_CPX),
            (self.num_parameters),
        )

        idx = ((self.d ** jnp.arange(self.L - 1, -1, -1)).dot(s)).astype(int)

        return jnp.log(kernel[idx] + self.delta)
