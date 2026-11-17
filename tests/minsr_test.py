import unittest

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

import jVMC_exp
import jVMC_exp.operator.discrete as op
from jVMC_exp import nets, sampler
from jVMC_exp.objective_function.base import ObjectiveFunctionOutput
from jVMC_exp.stats import LazySampledObs, SampledObs
from jVMC_exp.vqs import NQS


class TestGsSearch(unittest.TestCase):
    def test_gs_search_cpx(self):
        L = 4
        J = -1.0
        hxs = [-1.3, -0.3]
        exact_energies = [-6.10160339, -4.09296160]

        batch_size = int(2**L)
        learning_rate = 1e-2
        num_steps = 300

        for hx, exact_energy in zip(hxs, exact_energies):
            # Set up variational wave function
            rbm = nets.CpxRBM(num_hidden=3, bias=False)
            psi = NQS(rbm, L, batch_size, seed=1234)

            # Set up hamiltonian for ground state search
            H = 0
            for l in range(L):
                H += J * op.SigmaZ(l) * op.SigmaZ((l + 1) % L) + hx * op.SigmaX(l)

            # Set up exact sampler
            exact_sampler = sampler.ExactSampler(psi)

            loss_function = jVMC_exp.objective_function.Observable(H)
            stepper = jVMC_exp.stepper.Euler(time_step=learning_rate)
            solver = jVMC_exp.solver.Pinv(pinv_cutoff=1e-6)
            opt = jVMC_exp.optimizer.MinSR(
                exact_sampler, psi, solver=solver, diagonal_shift=1e-3
            )

            opt.ground_state_search(num_steps, loss_function, stepper)

            E = exact_sampler(H)
            print(jnp.abs((E.mean.item() - exact_energy) / exact_energy))
            self.assertTrue(jnp.max(jnp.abs((E.mean.item() - exact_energy) / exact_energy)) < 1e-3)


class TestGsSearchBatchedJacobian(unittest.TestCase):
    def test_gs_search_batched_jacobian(self):
        """
        MinSR builds its (small, N_samples x N_samples) kernel matrix from
        the Jacobian via `_solve_lazy` when grad_log_psi is a LazySampledObs,
        so batched_jacobian=True must converge to the same ground state as
        the dense Jacobian path.
        """
        L = 4
        J = -1.0
        hx = -0.3
        exact_energy = -4.09296160

        batch_size = 3 * jax.device_count()
        learning_rate = 1e-2
        num_steps = 300

        rbm = nets.CpxRBM(num_hidden=3, bias=False)
        psi = NQS(rbm, L, batch_size, seed=1234)

        H = 0
        for l in range(L):
            H += J * op.SigmaZ(l) * op.SigmaZ((l + 1) % L) + hx * op.SigmaX(l)

        exact_sampler = sampler.ExactSampler(psi)

        loss_function = jVMC_exp.objective_function.Observable(H, batched_jacobian=True)
        stepper = jVMC_exp.stepper.Euler(time_step=learning_rate)
        solver = jVMC_exp.solver.Pinv(pinv_cutoff=1e-6)
        opt = jVMC_exp.optimizer.MinSR(
            exact_sampler, psi, solver=solver, diagonal_shift=1e-3
        )

        opt.ground_state_search(num_steps, loss_function, stepper)

        E = exact_sampler(H)
        eps_rel = jnp.abs((E.mean.item() - exact_energy) / exact_energy)
        self.assertTrue(eps_rel < 1e-3)


class _RealParamsComplexOut(nn.Module):
    """Real parameters and a complex output, i.e. a non-holomorphic network"""

    @nn.compact
    def __call__(self, s):
        x = nn.Dense(3)(2 * s.ravel() - 1)
        return jnp.sum(jnp.log(jnp.cosh(x))) + 1j * jnp.sum(nn.Dense(1)(x))


class TestUpdateBatchedJacobian(unittest.TestCase):
    """
    A single MinSR update from a batched (lazy) Jacobian must equal the update from the
    dense Jacobian. The batches are uneven and, on more than one device, do not consist of
    consecutive samples, so this checks that the tangent kernel, the local energies and the
    solution are lined up sample by sample, for real, holomorphic and non-holomorphic networks.
    """

    def test_update_matches_dense(self):
        L = 4
        H = 0
        for l in range(L):
            H += -1.0 * op.SigmaZ(l) * op.SigmaZ((l + 1) % L) - 0.7 * op.SigmaX(l)

        for name, net in (
            ("real", nets.RBM(num_hidden=3, bias=True)),
            ("holomorphic", nets.CpxRBM(num_hidden=3, bias=True)),
            ("non-holomorphic", _RealParamsComplexOut()),
        ):
            with self.subTest(net=name):
                psi = NQS(net, L, 3 * jax.device_count(), seed=1234)
                exact_sampler = sampler.ExactSampler(psi)
                opt = jVMC_exp.optimizer.MinSR(
                    exact_sampler,
                    psi,
                    solver=jVMC_exp.solver.Pinv(pinv_cutoff=1e-8),
                    diagonal_shift=1e-3,
                )

                updates = []
                for batched_jacobian in (False, True):
                    loss_function = jVMC_exp.objective_function.Observable(
                        H, batched_jacobian=batched_jacobian
                    )
                    updates.append(
                        opt.get_update(loss_function.value_and_grad(exact_sampler))
                    )

                self.assertTrue(np.allclose(updates[0], updates[1], atol=1e-10), name)

    @unittest.skipIf(
        jax.device_count() == 1, "Every number of samples is divisible by one device"
    )
    def test_update_matches_dense_indivisible_samples(self):
        """
        A number of samples that is not divisible by the number of devices is padded with
        zero-weight samples, both by SampledObs and by the batched Jacobian.
        """
        L = 4
        num_samples = 5 * jax.device_count() + 1
        k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(0), 4)
        s = jax.random.randint(k1, (num_samples, L), 0, 2).astype(
            jVMC_exp.global_defs.DT_SAMPLES
        )
        weights = jax.random.uniform(k2, (num_samples,))
        o_loc = SampledObs(
            jax.random.normal(k3, (num_samples,))
            + 1j * jax.random.normal(k4, (num_samples,)),
            weights,
        )

        for name, net in (
            ("holomorphic", nets.CpxRBM(num_hidden=3, bias=True)),
            ("non-holomorphic", _RealParamsComplexOut()),
        ):
            with self.subTest(net=name):
                psi = NQS(net, L, 2 * jax.device_count(), seed=1234)
                opt = jVMC_exp.optimizer.MinSR(
                    sampler.ExactSampler(psi),
                    psi,
                    solver=jVMC_exp.solver.Pinv(pinv_cutoff=1e-8),
                    diagonal_shift=1e-3,
                )
                dense = ObjectiveFunctionOutput(
                    o_loc=o_loc, grad_log_psi=SampledObs(psi.gradients(s), weights)
                )
                lazy = ObjectiveFunctionOutput(
                    o_loc=o_loc,
                    grad_log_psi=LazySampledObs(psi.lazy_gradients(s), weights),
                )

                self.assertTrue(
                    np.allclose(
                        opt.get_update(dense), opt.get_update(lazy), atol=1e-10
                    ),
                    name,
                )


class TestPinvSolve(unittest.TestCase):
    """
    MinSR used to pseudo-invert the tangent kernel and only then apply it to the local
    energies. ``Pinv`` instead solves in the eigenbasis, ``V (ev^-1 * (V^H b))``, which
    avoids reassembling ``A^+`` but must give the same answer.
    """

    def test_matches_explicit_pseudo_inverse(self):
        rng = np.random.default_rng(0)

        for N, complex_valued in ((64, True), (65, False), (128, True)):

            def draw(*shape):
                x = rng.standard_normal(shape)
                return x + 1j * rng.standard_normal(shape) if complex_valued else x

            # Hermitian positive semi-definite, like MinSR's tangent kernel
            X = draw(N, N)
            A = jnp.array(X @ X.conj().T) + 1e-3 * jnp.eye(N)
            b = jnp.array(draw(N))

            for cutoff in (1e-14, 1e-6, 1e-3):
                with self.subTest(N=N, complex_valued=complex_valued, cutoff=cutoff):
                    expected = jnp.linalg.pinv(A, rtol=cutoff, hermitian=True) @ b
                    x, info = jVMC_exp.solver.Pinv(pinv_cutoff=cutoff)(A, b)

                    self.assertTrue(jnp.allclose(x, expected, rtol=1e-10, atol=1e-10))
                    self.assertGreater(info["condition_number"], 1.0)

    def test_reports_discarded_residual(self):
        """
        ``residual`` is the fraction of ``b`` thrown away by the cutoff; cross validation
        normalizes against it, so it has to be present and to grow with the cutoff.
        """
        rng = np.random.default_rng(1)
        X = rng.standard_normal((64, 64))
        A = jnp.array(X @ X.T) + 1e-3 * jnp.eye(64)
        b = jnp.array(rng.standard_normal(64))

        loose = jVMC_exp.solver.Pinv(pinv_cutoff=1e-14)(A, b)[1]["residual"]
        tight = jVMC_exp.solver.Pinv(pinv_cutoff=1e-1)(A, b)[1]["residual"]

        self.assertLess(loose, 1e-12)
        self.assertGreater(tight, loose)


class TestSolverCompatibility(unittest.TestCase):
    def _psi_and_sampler(self):
        L = 4
        psi = NQS(nets.CpxRBM(num_hidden=3, bias=False), L, 2**L, seed=1234)
        return psi, sampler.ExactSampler(psi)

    def test_rejects_matrix_free_solver(self):
        psi, exact_sampler = self._psi_and_sampler()
        with self.assertRaises(ValueError):
            jVMC_exp.optimizer.MinSR(exact_sampler, psi, solver=jVMC_exp.solver.CG())

    def test_rejects_snr_regularization(self):
        psi, exact_sampler = self._psi_and_sampler()
        with self.assertRaises(ValueError):
            jVMC_exp.optimizer.MinSR(
                exact_sampler, psi, solver=jVMC_exp.solver.PinvSNR(snr_tol=1)
            )

    def test_accepts_pinv_snr_without_snr(self):
        """
        Only the SNR regularization is meaningless for MinSR; PinvSNR's adaptive
        eigenvalue cutoff is not, so snr_tol=0 has to go through and solve.
        """
        L = 4
        psi, exact_sampler = self._psi_and_sampler()

        H = 0
        for l in range(L):
            H += -1.0 * op.SigmaZ(l) * op.SigmaZ((l + 1) % L) + -0.3 * op.SigmaX(l)
        loss_function = jVMC_exp.objective_function.Observable(H)

        opt = jVMC_exp.optimizer.MinSR(
            exact_sampler, psi, solver=jVMC_exp.solver.PinvSNR(pinv_cutoff=1e-6)
        )
        new_parameters, _ = opt.step(
            0, jVMC_exp.stepper.Euler(time_step=1e-2), loss_function
        )

        self.assertEqual(new_parameters.shape, psi.parameters_flat.shape)
        self.assertTrue(bool(jnp.all(jnp.isfinite(jnp.abs(new_parameters)))))
        # info from the solve must reach the metadata MinSR reports
        self.assertIn("residual", opt.meta_data)


if __name__ == "__main__":
    unittest.main()
