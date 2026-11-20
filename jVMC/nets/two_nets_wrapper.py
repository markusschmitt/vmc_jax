from collections.abc import Sequence

import flax.linen as nn


class TwoNets(nn.Module):
    nets: Sequence[callable]

    def __post_init__(self):
        if "sample" in dir(self.nets[0]):
            self.sample = self._sample_fun

        super().__post_init__()

    def __call__(self, s):
        return self.nets[0](s) + 1j * self.nets[1](s)

    def eval_real(self, s):
        return self.nets[0](s)

    def _sample_fun(self, *args):
        return self.nets[0].sample(*args)
