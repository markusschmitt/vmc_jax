import unittest

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

import jVMC_exp
import jVMC_exp.nets
import jVMC_exp.operator.discrete as op
import jVMC_exp.sampler
import jVMC_exp.vqs
from jVMC_exp import global_defs
from jVMC_exp.objective_function import (
    Estimator,
    ObjectiveFunctionOutput,
    Observable,
    ParametricObservable,
)
from jVMC_exp.stats import LazySampledObs, SampledObs

L = 4
N_SAMPLES = 16
# 3 samples per device and batch: the last batch over N_SAMPLES is never full
LAZY_BATCH_SIZE = 3 * jax.device_count()
# All configurations
BASIS = jnp.array((np.arange(2**L)[:, None] >> np.arange(L - 1, -1, -1)) & 1)


class ModulusPhaseNet(nn.Module):
    """Real parameters and complex output"""

    @nn.compact
    def __call__(self, s):
        x = 2 * s.ravel() - 1.0
        dense = lambda: nn.Dense(3, param_dtype=global_defs.DT_PARAMS_REAL)

        return jnp.sum(jnp.tanh(dense()(x))) + 1j * jnp.sum(jnp.sin(dense()(x)))


class NonHolomorphicRBM(nn.Module):
    """Complex parameters, non-holomorphic output"""

    @nn.compact
    def __call__(self, s):
        theta = nn.Dense(
            2,
            param_dtype=global_defs.DT_PARAMS_CPX,
            kernel_init=jVMC_exp.nets.initializers.cplx_init,
        )(2 * s.ravel() - 1.0)

        return jnp.sum(jnp.log(jnp.cosh(theta))) + 0.1 * jnp.sum(jnp.abs(theta) ** 2)


NETS = {
    "real_parameters": ModulusPhaseNet,
    "holomorphic": lambda: jVMC_exp.nets.CpxRBM(2, True),
    "non_holomorphic": NonHolomorphicRBM,
}


def build_sampler(batch_size=N_SAMPLES):
    model = jVMC_exp.nets.CpxRBM(1, True)
    psi = jVMC_exp.vqs.NQS(model, L, batch_size=batch_size)
    sampler = jVMC_exp.sampler.ExactSampler(psi, 2)
    return sampler, psi


def build_hamiltonian():
    J, h = -1, 1
    H = 0
    for i in range(L):
        H += J * op.SigmaZ(i) * op.SigmaZ((i + 1) % L) + h * op.SigmaX(i)
    return H


def param_norm(parameters):
    return sum(jnp.sum(jnp.abs(p) ** 2) for p in jax.tree_util.tree_leaves(parameters))


def dummy_estimator_fn(parameters, samples):
    """
    Real-valued differentiable estimator: depends on both the spin configuration and
    the network parameters
    """
    return jnp.mean(samples.astype(jnp.float32), axis=-1) + param_norm(parameters)


def sigma_z_fn(parameters, samples):
    """Local estimator of O(theta) = |theta|^2 sigma^z_0, with sigma^z = +1 on s = 0"""
    return param_norm(parameters) * (1 - 2 * samples[..., 0])


def make_sigma_x_fn(psi):
    """
    Local estimator of O(theta) = |theta|^2 sigma^x_0, |theta|^2 psi(s') / psi(s) with s' the
    configuration flipped on site 0. The wave-function ratio is held constant, so that the
    derivative is (d_k O)_loc
    """

    def fn(parameters, samples):
        frozen = jax.lax.stop_gradient(parameters)
        log_psi = jnp.vectorize(lambda s: psi.apply_fun(frozen, s), signature="(l)->()")
        flipped = samples.at[..., 0].set(1 - samples[..., 0])

        return param_norm(parameters) * jnp.exp(log_psi(flipped) - log_psi(samples))

    return fn


def random_nqs(net, seed, batch_size=2**L):
    """NQS with O(1) random parameters"""
    psi = jVMC_exp.vqs.NQS(net, L, batch_size=batch_size, seed=seed)
    noise = jax.random.normal(jax.random.PRNGKey(seed), psi.parameters_flat.shape)
    psi.parameters = psi.parameters_flat + 0.5 * noise

    return psi


def exact_loss(psi, ell):
    """sum_s p(s) ell(s; theta) over all configurations, with p(s) = |psi(s)|^2 / <psi|psi>"""
    p = np.exp(2 * np.asarray(psi(BASIS)).ravel().real)

    return np.sum(p * np.asarray(ell(psi.parameters, BASIS))).real / np.sum(p)


def finite_differences(psi, ell, h=1e-5):
    """Central finite differences of the exact loss with respect to the flat parameters"""
    theta = psi.parameters_flat
    result = []
    for k in range(len(theta)):
        shift = jnp.zeros_like(theta).at[k].set(h)
        losses = []
        for sign in (1, -1):
            psi.parameters = theta + sign * shift
            losses.append(exact_loss(psi, ell))
        result.append((losses[0] - losses[1]) / (2 * h))
    psi.parameters = theta

    return np.array(result)


class TestEstimator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sampler, cls.psi = build_sampler()
        cls.obj = Estimator(dummy_estimator_fn)

    def test_call_returns_sampled_obs(self):
        result = self.obj(self.sampler)
        self.assertIsInstance(result, SampledObs)

    def test_call_observations_shape(self):
        result = self.obj(self.sampler)
        self.assertEqual(result.observations.shape[0], N_SAMPLES)

    def test_value_and_grad_output_type(self):
        out = self.obj.value_and_grad(self.sampler)
        self.assertIsInstance(out, ObjectiveFunctionOutput)

    def test_value_and_grad_o_loc_present(self):
        out = self.obj.value_and_grad(self.sampler)
        self.assertIsNotNone(out.o_loc)

    def test_value_and_grad_grad_present(self):
        out = self.obj.value_and_grad(self.sampler)
        self.assertIsNotNone(out.grad)

    def test_grad_log_psi_shape(self):
        out = self.obj.value_and_grad(self.sampler)
        self.assertEqual(out.grad_log_psi.observations.shape[0], N_SAMPLES)

    def test_value_and_grad_without_grad(self):
        out = self.obj.value_and_grad(self.sampler, compute_grad=False)
        self.assertIsNone(out.grad)
        self.assertIsNotNone(out.grad_log_psi)

    def test_grad_values_are_finite(self):
        out = self.obj.value_and_grad(self.sampler)
        self.assertTrue(np.all(np.isfinite(np.array(out.grad))))


class TestParametricObservable(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sampler, cls.psi = build_sampler()
        cls.sigma_z_0 = op.SigmaZ(0)
        cls.obj = ParametricObservable(cls.sigma_z_0, dummy_estimator_fn)

    def test_call_returns_sampled_obs(self):
        result = self.obj(self.sampler)
        self.assertIsInstance(result, SampledObs)

    def test_call_observations_shape(self):
        result = self.obj(self.sampler)
        self.assertEqual(result.observations.shape[0], N_SAMPLES)

    def test_value_and_grad_output_type(self):
        out = self.obj.value_and_grad(self.sampler)
        self.assertIsInstance(out, ObjectiveFunctionOutput)

    def test_grad_values_are_finite(self):
        out = self.obj.value_and_grad(self.sampler)
        self.assertTrue(np.all(np.isfinite(np.array(out.grad))))

    def test_grad_log_psi_shape(self):
        out = self.obj.value_and_grad(self.sampler)
        self.assertEqual(out.grad_log_psi.observations.shape[0], N_SAMPLES)

    def test_value_and_grad_without_grad(self):
        out = self.obj.value_and_grad(self.sampler, compute_grad=False)
        self.assertIsNone(out.grad)
        self.assertIsNotNone(out.grad_log_psi)

    def test_grad_differs_from_observable_only(self):
        # ParametricObservable grad = covar_term + explicit_grad / 2, so it must
        # differ from a plain Observable grad (the explicit contribution is non-zero)
        plain_out = Observable(self.sigma_z_0).value_and_grad(self.sampler)
        param_out = self.obj.value_and_grad(self.sampler)
        differs = not np.allclose(np.array(param_out.grad), np.array(plain_out.grad))

        self.assertTrue(
            differs, "ParametricObservable grad should include estimator contribution"
        )


class TestObservableBatchedJacobian(unittest.TestCase):
    """
    batched_jacobian=True routes grad_log_psi through LazySampledObs
    (batches of 3 samples per device over N_SAMPLES=16 -> uneven) instead of a
    materialized Jacobian. The physics must be identical either way.
    """

    @classmethod
    def setUpClass(cls):
        cls.sampler, cls.psi = build_sampler(batch_size=LAZY_BATCH_SIZE)
        cls.hamiltonian = build_hamiltonian()

    def test_grad_log_psi_is_lazy(self):
        out = Observable(self.hamiltonian, batched_jacobian=True).value_and_grad(
            self.sampler
        )
        self.assertIsInstance(out.grad_log_psi, LazySampledObs)

    def test_matches_dense_observable(self):
        dense_out = Observable(self.hamiltonian, batched_jacobian=False).value_and_grad(
            self.sampler
        )
        lazy_out = Observable(self.hamiltonian, batched_jacobian=True).value_and_grad(
            self.sampler
        )

        self.assertTrue(
            np.allclose(np.array(lazy_out.grad), np.array(dense_out.grad), atol=1e-10)
        )
        self.assertTrue(np.allclose(lazy_out.o_loc.mean, dense_out.o_loc.mean))


class TestEstimatorBatchedJacobian(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sampler, cls.psi = build_sampler(batch_size=LAZY_BATCH_SIZE)

    def test_matches_dense_estimator(self):
        dense_out = Estimator(
            dummy_estimator_fn, batched_jacobian=False
        ).value_and_grad(self.sampler)
        lazy_out = Estimator(dummy_estimator_fn, batched_jacobian=True).value_and_grad(
            self.sampler
        )

        self.assertTrue(
            np.allclose(np.array(lazy_out.grad), np.array(dense_out.grad), atol=1e-10)
        )


class TestParametricObservableBatchedJacobian(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sampler, cls.psi = build_sampler(batch_size=LAZY_BATCH_SIZE)
        cls.sigma_z_0 = op.SigmaZ(0)

    def test_grad_log_psi_is_lazy(self):
        obj = ParametricObservable(
            self.sigma_z_0, dummy_estimator_fn, batched_jacobian=True
        )
        out = obj.value_and_grad(self.sampler)
        self.assertIsInstance(out.grad_log_psi, LazySampledObs)

    def test_matches_dense_parametric_observable(self):
        dense_out = ParametricObservable(
            self.sigma_z_0, dummy_estimator_fn, batched_jacobian=False
        ).value_and_grad(self.sampler)
        lazy_out = ParametricObservable(
            self.sigma_z_0, dummy_estimator_fn, batched_jacobian=True
        ).value_and_grad(self.sampler)

        self.assertTrue(
            np.allclose(np.array(lazy_out.grad), np.array(dense_out.grad), atol=1e-10)
        )


class TestGradientFiniteDifferences(unittest.TestCase):
    """
    The gradient follows the convention dL/dtheta = 2 Re(grad), and is compared with central
    finite differences of the exact loss L = sum_s p(s) ell(s; theta). For ParametricObservable
    the operator O(theta) is built at the current parameters, where O_loc = ell.
    """

    def assert_finite_differences(self, build):
        """``build(psi)`` returns the objective function and its local loss ell(parameters, samples)"""
        for name, net in NETS.items():
            with self.subTest(net=name):
                psi = random_nqs(net(), seed=2)
                sampler = jVMC_exp.sampler.ExactSampler(psi, 2)
                sampler.sample()
                objective, ell = build(psi)
                out = objective.value_and_grad(sampler)

                self.assertAlmostEqual(
                    out.o_loc.mean.real.item(), exact_loss(psi, ell), places=10
                )
                grad = 2 * np.real(out.grad)
                self.assertTrue(
                    np.allclose(grad, finite_differences(psi, ell), atol=1e-7)
                )

    def test_estimator(self):
        self.assert_finite_differences(
            lambda psi: (Estimator(dummy_estimator_fn), dummy_estimator_fn)
        )

    def test_parametric_observable_diagonal(self):
        def build(psi):
            operator = float(param_norm(psi.parameters)) * op.SigmaZ(0)
            return ParametricObservable(operator, sigma_z_fn), sigma_z_fn

        self.assert_finite_differences(build)

    def test_parametric_observable_off_diagonal(self):
        def build(psi):
            ell = make_sigma_x_fn(psi)
            operator = float(param_norm(psi.parameters)) * op.SigmaX(0)
            return ParametricObservable(operator, ell), ell

        self.assert_finite_differences(build)


if __name__ == "__main__":
    unittest.main()
