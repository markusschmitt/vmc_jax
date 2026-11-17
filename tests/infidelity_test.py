import unittest
import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
from scipy.linalg import expm

import jVMC_exp
import jVMC_exp.nets as nets
import jVMC_exp.operator.discrete as op
from jVMC_exp import global_defs
from jVMC_exp.vqs import NQS
from jVMC_exp.sampler import ExactSampler, MCSampler
from jVMC_exp.objective_function import Infidelity
from jVMC_exp.objective_function.infidelity import MovingAverage
from jVMC_exp.util.util import matrix_to_jvmc_operator

L = 4
# 2 samples per device and batch: the connected configurations of a sample are evaluated in chunks
SMALL_BATCH_SIZE = 2 * jax.device_count()
# All configurations, with site 0 as the most significant bit, as in np.kron
BASIS = jnp.array((np.arange(2 ** L)[:, None] >> np.arange(L - 1, -1, -1)) & 1)
SX = np.array([[0, 1], [1, 0]])

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
            2, param_dtype=global_defs.DT_PARAMS_CPX, kernel_init=nets.initializers.cplx_init
        )(2 * s.ravel() - 1.0)

        return jnp.sum(jnp.log(jnp.cosh(theta))) + 0.1 * jnp.sum(jnp.abs(theta) ** 2)

def embed(matrix, site):
    """Dense matrix of ``matrix`` acting on the chain from ``site`` on"""
    n_sites = int(np.log2(matrix.shape[0]))

    return np.kron(np.kron(np.eye(2 ** site), matrix), np.eye(2 ** (L - site - n_sites)))

def random_unitary(seed):
    """Two-site unitary exp(-iH) with a random Hermitian H: its diagonal elements are large"""
    rng = np.random.default_rng(seed)
    h = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))

    return expm(-0.5j * (h + h.conj().T))

def random_nqs(net, seed, batch_size=2 ** L):
    """NQS with O(1) random parameters, so that different seeds give clearly different states"""
    psi = NQS(net, L, batchSize=batch_size, seed=seed)
    noise = jax.random.normal(jax.random.PRNGKey(seed), psi.parameters_flat.shape)
    psi.parameters = psi.parameters_flat + 0.5 * noise

    return psi

def state_vector(psi):
    return np.exp(np.asarray(psi(BASIS)).ravel())

def exact_infidelity(psi, chi, U):
    """1 - |<psi|U|chi>|^2 / (<psi|psi><chi|chi>) from the dense state vectors"""
    psi_v, chi_v = state_vector(psi), state_vector(chi)
    overlap = np.vdot(psi_v, U @ chi_v)

    return 1 - np.abs(overlap) ** 2 / (np.vdot(psi_v, psi_v).real * np.vdot(chi_v, chi_v).real)

def build_unitary():
    unitary = random_unitary(seed=3)
    U, U_dag = matrix_to_jvmc_operator(unitary, (1, 2), return_conjugate=True)

    return U, U_dag, embed(unitary, 1)

class TestInfidelityValue(unittest.TestCase):
    """
    With exact sampling the estimator must equal the exact infidelity. The operators have
    diagonal elements: their weight psi(s) / chi(s) is not 1, since psi and chi are different states.
    """
    @classmethod
    def setUpClass(cls):
        cls.chi = random_nqs(nets.CpxRBM(2, True), seed=1)
        cls.psi = random_nqs(nets.CpxRBM(2, True), seed=2)
        cls.U, cls.U_dag, cls.U_dense = build_unitary()

    def assert_exact(self, U_dense, chi=None, psi=None, **infidelity_kwargs):
        chi, psi = chi or self.chi, psi or self.psi
        sampler = ExactSampler(psi)
        value = Infidelity(ExactSampler(chi), **infidelity_kwargs)(sampler).mean.real.item()

        self.assertAlmostEqual(value, exact_infidelity(psi, chi, U_dense), places=10)

    def test_identity(self):
        self.assert_exact(np.eye(2 ** L))

    def test_unitary(self):
        self.assert_exact(self.U_dense, operator=self.U, conjugated_operator=self.U_dag)

    def test_off_diagonal_operator(self):
        flip = op.SigmaX(0) * op.SigmaX(1) * op.SigmaX(2) * op.SigmaX(3)
        flip_dense = embed(np.kron(np.kron(SX, SX), np.kron(SX, SX)), 0)
        self.assert_exact(flip_dense, operator=flip)

    def test_small_batch_size(self):
        chi = random_nqs(nets.CpxRBM(2, True), seed=1, batch_size=SMALL_BATCH_SIZE)
        psi = random_nqs(nets.CpxRBM(2, True), seed=2, batch_size=SMALL_BATCH_SIZE)
        self.assert_exact(
            self.U_dense, chi=chi, psi=psi, operator=self.U, conjugated_operator=self.U_dag
        )

    def test_control_variates(self):
        # For a unitary the control variate has zero mean, so the exact value is unchanged
        for control_variate in ("simple", "adaptive"):
            with self.subTest(control_variate=control_variate):
                self.assert_exact(
                    self.U_dense, operator=self.U, conjugated_operator=self.U_dag,
                    control_variate=control_variate
                )

    def test_adaptive_moving_average(self):
        # The coefficient changes from call to call while the moving average fills up
        sampler = ExactSampler(self.psi)
        loss = Infidelity(
            ExactSampler(self.chi), self.U, self.U_dag,
            control_variate="adaptive", moving_average_width=3
        )
        exact = exact_infidelity(self.psi, self.chi, self.U_dense)
        for _ in range(4):
            self.assertAlmostEqual(loss(sampler).mean.real.item(), exact, places=10)

    def test_time_dependent_operator(self):
        # The keyword arguments, e.g. the time t passed by the optimizers, reach the prefactors
        g = lambda **kwargs: jnp.cos(kwargs["t"])
        operator = op.IdentityOperator(2, 0) + g * op.SigmaX(0)
        loss = Infidelity(ExactSampler(self.chi), operator, operator)
        for t in (0.3, 1.1):
            exact = exact_infidelity(self.psi, self.chi, np.eye(2 ** L) + np.cos(t) * embed(SX, 0))
            with self.subTest(t=t):
                self.assertAlmostEqual(loss(ExactSampler(self.psi), t=t).mean.real.item(), exact, places=10)
                out = loss.value_and_grad(ExactSampler(self.psi), t=t)
                self.assertAlmostEqual(out.o_loc.mean.real.item(), exact, places=10)

    def test_conjugated_operator_without_operator(self):
        with self.assertRaises(ValueError):
            Infidelity(ExactSampler(self.chi), conjugated_operator=self.U_dag)

    def test_transformed_state(self):
        # A full configuration state holding U|chi> must have zero infidelity with U|chi>
        chi_v = state_vector(self.chi)
        chi, target = (NQS(nets.FullConfigurationState(L), L, batchSize=2 ** L) for _ in range(2))
        for state, v in ((chi, chi_v), (target, self.U_dense @ chi_v)):
            state.parameters = jnp.concatenate([v.real, v.imag])

        loss = Infidelity(ExactSampler(chi), self.U, self.U_dag)
        self.assertAlmostEqual(loss(ExactSampler(target)).mean.real.item(), 0, places=10)

    def test_monte_carlo(self):
        proposer = jVMC_exp.propose.SpinFlip()
        chi_sampler = MCSampler(self.chi, proposer, 123, 2 ** 5, 2 ** 13)
        psi_sampler = MCSampler(self.psi, proposer, 321, 2 ** 5, 2 ** 13)
        psi_sampler.sample()
        value = Infidelity(chi_sampler, self.U, self.U_dag)(psi_sampler).mean.real.item()

        self.assertAlmostEqual(value, exact_infidelity(self.psi, self.chi, self.U_dense), delta=0.02)

class TestMovingAverage(unittest.TestCase):
    def test_mean_of_last_values(self):
        average = MovingAverage(width=3)
        means = [float(average.update(x)) for x in range(1, 6)]

        # The window starts filled with zeros
        self.assertTrue(np.allclose(means, [1 / 3, 1, 2, 3, 4]))

class TestInfidelityGradient(unittest.TestCase):
    """
    The gradient follows the convention of Observable, dI/dtheta = 2 Re(grad), and is
    compared with central finite differences of the exact infidelity.
    """
    @classmethod
    def setUpClass(cls):
        cls.chi = random_nqs(nets.CpxRBM(2, True), seed=1)
        cls.U, cls.U_dag, cls.U_dense = build_unitary()

    def get_grad(self, psi, **infidelity_kwargs):
        sampler = ExactSampler(psi)
        sampler.sample()
        loss = Infidelity(ExactSampler(self.chi), self.U, self.U_dag, **infidelity_kwargs)

        return loss.value_and_grad(sampler).grad

    def assert_finite_differences(self, net):
        psi = random_nqs(net, seed=2)
        grad = 2 * np.real(self.get_grad(psi))

        theta, h = psi.parameters_flat, 1e-5
        finite_differences = []
        for k in range(len(theta)):
            shift = jnp.zeros_like(theta).at[k].set(h)
            infidelities = []
            for sign in (1, -1):
                psi.parameters = theta + sign * shift
                infidelities.append(exact_infidelity(psi, self.chi, self.U_dense))
            finite_differences.append((infidelities[0] - infidelities[1]) / (2 * h))
        psi.parameters = theta

        self.assertTrue(np.allclose(grad, finite_differences, atol=1e-7))

    def test_real_parameters(self):
        self.assert_finite_differences(ModulusPhaseNet())

    def test_holomorphic(self):
        self.assert_finite_differences(nets.CpxRBM(2, True))

    def test_non_holomorphic(self):
        self.assert_finite_differences(NonHolomorphicRBM())

    def test_batched_jacobian(self):
        psi = random_nqs(nets.CpxRBM(2, True), seed=2, batch_size=SMALL_BATCH_SIZE)
        dense = self.get_grad(psi, batched_jacobian=False)
        lazy = self.get_grad(psi, batched_jacobian=True)

        self.assertTrue(np.allclose(lazy, dense, atol=1e-10))

class TestInfidelityOptimization(unittest.TestCase):
    """
    Fit a trial state to U|chi> with exact sampling: the infidelity has to drop by orders of
    magnitude, and the estimate at the end has to equal the exact infidelity.
    """
    @classmethod
    def setUpClass(cls):
        cls.chi = random_nqs(nets.CpxRBM(2, True), seed=1)
        cls.U, cls.U_dag, cls.U_dense = build_unitary()

    def assert_converges(self, get_optimizer, steps, **search_kwargs):
        psi = NQS(nets.CpxRBM(6, True), L, batchSize=2 ** L, seed=4)
        sampler = ExactSampler(psi)
        loss = Infidelity(ExactSampler(self.chi), self.U, self.U_dag)
        initial = exact_infidelity(psi, self.chi, self.U_dense)

        get_optimizer(sampler, psi).ground_state_search(steps, loss, **search_kwargs)
        final = exact_infidelity(psi, self.chi, self.U_dense)
        sampler.sample()

        self.assertLess(final, 1e-2 * initial)
        self.assertAlmostEqual(loss(sampler).mean.real.item(), final, places=10)

    def test_sr(self):
        self.assert_converges(
            jVMC_exp.optimizer.SR, 200, stepper=jVMC_exp.stepper.Euler(1e-1)
        )

    def test_adam(self):
        self.assert_converges(
            lambda sampler, psi: jVMC_exp.optimizer.Adam(sampler, psi, learning_rate=1e-2), 500
        )

if __name__ == "__main__":
    unittest.main()
