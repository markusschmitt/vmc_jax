import jax
import jax.numpy as jnp
import flax.linen as nn

from typing import Sequence


class TwoNets(nn.Module):
    net1: callable
    net2: callable

    def __call__(self, s):
        return self.net1(s) + 1j * self.net2(s)

    def sample(self, *args):
        # Will produce exact samples if net[0] contains a sample function.
        # Won't be called if net[0] does not have a sample method.
        return self.net1.sample(*args)

    def eval_real(self, s):
        return self.net1(s)
