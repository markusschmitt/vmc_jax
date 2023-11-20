"""Simple GPT model for autoregressive encoding of quantum states."""

from functools import partial
from typing import Tuple

from flax.linen import (
    Dense,
    Embed,
    LayerNorm,
    Module,
    MultiHeadDotProductAttention,
    Sequential,
    compact,
    gelu,
    log_softmax,
    make_causal_mask,
    scan,
)
from jax import Array, vmap
from jax.config import config  # type: ignore
from jax.numpy import arange, expand_dims, full, int64, take_along_axis, zeros
from jax.random import KeyArray, categorical, split
from jVMC.global_defs import tReal

config.update("jax_enable_x64", True)


class _TransformerBlock(Module):
    """The transformer decoder block."""

    embeddingDim: int
    nHeads: int
    paramDType: type = tReal

    @compact
    def __call__(self, x: Array) -> Array:
        x = x + MultiHeadDotProductAttention(
            self.nHeads, param_dtype=self.paramDType
        )(
            x,
            x,
            mask=make_causal_mask(
                zeros((x.shape[-2]), self.paramDType), dtype=self.paramDType
            ),
        )
        x = LayerNorm(param_dtype=self.paramDType)(x)
        x = x + Sequential(
            [
                Dense(self.embeddingDim * 4, param_dtype=self.paramDType),
                gelu,
                Dense(self.embeddingDim, param_dtype=self.paramDType),
            ]
        )(x)
        x = LayerNorm(param_dtype=self.paramDType)(x)
        return x


class GPT(Module):
    """GPT model for autoregressive decoding of neural quantum states.

    This model outputs the log amplitude of a wave function which in turn is
    a log probability density. It contains a ``sample`` method that peforms
    autorgressive sampling.

    Initialization arguments:
        * ``L``: Length of the spin chain.
        * ``embeddingDim``: Embedding dimension.
        * ``depth``: Number of transformer blocks.
        * ``nHeads``: Number of attention heads.
        * ``logProbFactor``: Factor defining how output and associated sample
                probability are related. 0.5 for pure states and 1.0 for POVMs
                (default: 0.5).
        * ``paramDType``: Data type of the model parameters
                (default: ``jVMC.global_defs.tReal``).
        * ``spinDType``: Data type of the spin configurations
                (default: ``jax.numpy.int64``).
    """

    L: int
    embeddingDim: int
    depth: int
    nHeads: int
    logProbFactor: float = 0.5
    paramDType: type = tReal
    spinDType: type = int64

    @compact
    def __call__(self, s: Array, returnLogAmp: bool = True) -> Array:
        """Forward pass of the model.

        Args:
            * ``s``: A spin configuration.
            * ``returnLogAmp``: Whether to return the log amplitude of the spin
                configuration (default: True).

        Returns:
            The log amplitude of the wave function.
        """
        if not self.embeddingDim % self.nHeads == 0:
            raise AttributeError(
                "The embedding dimension should be divisible by the number of"
                " heads."
            )
        if not s.shape[-1] == self.L:
            raise ValueError(
                "Input length should be equal to context length, L."
            )

        y = Embed(2, self.embeddingDim, param_dtype=self.paramDType)(s)
        p = self.variable(
            "params",
            "positional_embeddings",
            zeros,
            (self.L, self.embeddingDim),
            self.paramDType,
        ).value
        y = y + p
        y = Sequential(
            [
                _TransformerBlock(
                    self.embeddingDim, self.nHeads, self.paramDType
                )
                for _ in range(self.depth)
            ]
        )(y)
        y = Dense(2, param_dtype=self.paramDType)(y)
        if returnLogAmp:
            return (
                take_along_axis(log_softmax(y), expand_dims(s, -1), axis=-1)
                .sum(axis=-2)
                .squeeze(-1)
                * self.logProbFactor
            )
        return y

    def sample(self, numSamples: int, key: KeyArray) -> Array:
        """Autoregressively sample a spin configuration.

        Args:
            * ``numSamples``: The number of configurations to generate.
            * ``key``: JAX random key.

        Returns:
            A batch of spin configurations.
        """

        def generate_sample(key):
            keys = split(key, self.L)
            s = full((self.L,), -1, self.spinDType)
            s, _ = self._scanning_fn(s, (keys, arange(self.L)))
            return s

        keys = split(key, numSamples)
        return vmap(generate_sample)(keys)

    @partial(scan, variable_broadcast="params", split_rngs={"params": False})
    def _scanning_fn(
        self, s: Array, x: Tuple[KeyArray, Array]
    ) -> Tuple[Array, None]:
        logits = self(s, False)
        choice = categorical(x[0], logits[x[1]])
        return s.at[x[1]].set(choice), None
