import jax
import jax.numpy as jnp
import flax.linen as nn

from typing import Sequence


class TwoNets(nn.Module):
    net: Sequence[callable]

    def __call__(self, s):
        return self.net[0](s) + 1j * self.net[1](s)

    def sample(self, *args):
        # Will produce exact samples if net[0] contains a sample function.
        # Won't be called if net[0] does not have a sample method.
        return self.net[0].sample(*args)

    def eval_real(self, s):
        return self.net[0](s)
