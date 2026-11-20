import unittest

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax import nnx

import jVMC
import jVMC.nets.activation_functions as act_funs
import jVMC.nets.initializers
from jVMC import global_defs
from jVMC.nets.initializers import init_fn_args


class RBMNN(nn.Module):
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
                dtype=global_defs.DT_PARAMS_CPX,
            ),
        )
        return jnp.sum(act_funs.log_cosh(layer(2 * s.ravel() - 1)))


class RBMNNX(nnx.Module):
    def __init__(self, in_features, num_hidden, bias=False, *, rngs: nnx.Rngs):
        self.layer = nnx.Linear(
            in_features,
            out_features=num_hidden,
            use_bias=bias,
            param_dtype=global_defs.DT_OPERATORS_CPX,
            bias_init=jax.nn.initializers.zeros,
            kernel_init=jVMC.nets.initializers.cplx_init,
            rngs=rngs,
        )
        self.activation = act_funs.log_cosh

    def __call__(self, x):
        return jnp.sum(self.activation(self.layer(2 * x.ravel() - 1)))


L = 10
SEED = 123
NNX_INIT = dict(in_features=L, num_hidden=2, bias=True)


def _make_nqs_nnx(**kwargs):
    return jVMC.vqs.NQS(
        RBMNNX,
        L,
        batch_size=2**5,
        seed=SEED,
        nnx_init=NNX_INIT,
        **kwargs,
    )


def _make_nqs_nn(**kwargs):
    return jVMC.vqs.NQS(
        RBMNN(num_hidden=2, bias=True),
        L,
        batch_size=2**5,
        seed=SEED,
        **kwargs,
    )


class TestNNXWrapper(unittest.TestCase):
    def test_nqs_accepts_nnx_class(self):
        """NQS should construct without error when given an NNX class + nnx_init."""
        psi = _make_nqs_nnx()
        self.assertIsNotNone(psi)

    def test_nqs_accepts_linen_instance(self):
        """NQS should still work with a plain Linen model (regression)."""
        psi = _make_nqs_nn()
        self.assertIsNotNone(psi)

    def test_nqs_rejects_nnx_instance(self):
        """Passing an NNX *instance* (not a class) should raise a clear ValueError."""
        instance = RBMNNX(in_features=L, num_hidden=2, bias=True, rngs=nnx.Rngs(SEED))
        with self.assertRaises((ValueError, TypeError)):
            jVMC.vqs.NQS(instance, L, batch_size=2**5, seed=SEED)

    def test_nnx_forward_pass(self):
        """NQS built from an NNX class should produce a scalar output."""
        psi = _make_nqs_nnx()
        sample = jnp.ones(L)
        out = psi(sample[None])
        self.assertEqual(out.ndim, 1)

    def test_linen_forward_pass(self):
        """NQS built from a Linen model should also produce a scalar output."""
        psi = _make_nqs_nn()
        sample = jnp.ones(L)
        out = psi(sample[None])
        self.assertEqual(out.ndim, 1)

    def test_nnx_sampler_runs(self):
        """Full sampling pipeline should complete without error for NNX model."""
        psi = _make_nqs_nnx()
        proposer = jVMC.propose.SpinFlip()
        sampler = jVMC.sampler.MCSampler(
            psi, proposer, num_chains=2**5, num_samples=2**7
        )
        sampler.sample()
        self.assertIsNotNone(sampler.samples)

    def test_linen_sampler_runs(self):
        """Full sampling pipeline should complete without error for Linen model (regression)."""
        psi = _make_nqs_nn()
        proposer = jVMC.propose.SpinFlip()
        sampler = jVMC.sampler.MCSampler(
            psi, proposer, num_chains=2**5, num_samples=2**7
        )
        sampler.sample()
        self.assertIsNotNone(sampler.samples)

    def test_nnx_output_is_finite(self):
        """Log-amplitudes from NNX model should be finite (no NaN / Inf)."""
        psi = _make_nqs_nnx()
        proposer = jVMC.propose.SpinFlip()
        sampler = jVMC.sampler.MCSampler(
            psi, proposer, num_chains=2**5, num_samples=2**7
        )
        sampler.sample()
        log_amps = psi(sampler.samples)
        self.assertTrue(jnp.all(jnp.isfinite(log_amps)))

    def test_nnx_and_linen_output_dtype_match(self):
        """NNX and Linen models should return the same output dtype."""
        psi_nnx = _make_nqs_nnx()
        psi_nn = _make_nqs_nn()
        sample = jnp.ones((1, L))
        self.assertEqual(psi_nnx(sample).dtype, psi_nn(sample).dtype)


if __name__ == "__main__":
    unittest.main()
