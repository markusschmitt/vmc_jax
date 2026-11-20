import unittest

import jax.numpy as jnp

import jVMC
import jVMC.operator.discrete as op
from jVMC import nets, sampler
from jVMC.vqs import NQS


class TestGsSearch(unittest.TestCase):
    def test_gs_search_cpx(self):
        L = 4
        J = -1.0
        hxs = [-1.3, -0.3]
        exact_energies = [-6.10160339, -4.09296160]

        for hx, exact_energy in zip(hxs, exact_energies):
            # Set up variational wave function
            rbm = nets.CpxRBM(num_hidden=6, bias=False)
            psi = NQS(rbm, L, 2**4, seed=123)
            exact_sampler = sampler.ExactSampler(psi)

            # Set up hamiltonian for ground state search
            hamiltonian = 0
            for l in range(L):
                hamiltonian += J * (
                    op.SigmaZ(l) * op.SigmaZ((l + 1) % L)
                ) + hx * op.SigmaX(l)

            delta = 2
            loss_function = jVMC.objective_function.Observable(hamiltonian)
            solver = jVMC.solver.PinvSNR(snr_tol=1, pinv_tol=0.0, pinv_cutoff=1e-8)
            stepper = jVMC.stepper.Euler(5e-2)
            opt = jVMC.optimizer.SR(
                exact_sampler, psi, diagonal_shift=0, diagonal_scale=delta, solver=solver
            )

            opt.ground_state_search(500, loss_function, stepper)
            eps_rel = jnp.abs((exact_sampler(hamiltonian).mean.item() - exact_energy) / exact_energy)
            self.assertTrue(eps_rel < 1e-3)


class TestGsSearchBatchedJacobian(unittest.TestCase):
    def test_gs_search_batched_jacobian_dense_solver(self):
        L = 4
        J = -1.0
        hx = -0.3
        exact_energy = -4.09296160

        rbm = nets.CpxRBM(num_hidden=6, bias=False)
        psi = NQS(rbm, L, 2**4, seed=123)
        exact_sampler = sampler.ExactSampler(psi)

        hamiltonian = 0
        for l in range(L):
            hamiltonian += J * (op.SigmaZ(l) * op.SigmaZ((l + 1) % L)) + hx * op.SigmaX(
                l
            )

        delta = 2
        loss_function = jVMC.objective_function.Observable(
            hamiltonian, batched_jacobian=True
        )
        solver = jVMC.solver.PinvSNR(snr_tol=1, pinv_tol=0.0, pinv_cutoff=1e-8)
        stepper = jVMC.stepper.Euler(5e-2)
        opt = jVMC.optimizer.SR(
            exact_sampler, psi, diagonal_shift=0, diagonal_scale=delta, solver=solver
        )

        opt.ground_state_search(500, loss_function, stepper)
        eps_rel = jnp.abs((exact_sampler(hamiltonian).mean.item() - exact_energy) / exact_energy)
        self.assertTrue(eps_rel < 1e-3)

    def test_matrix_free_solver_rejects_batched_jacobian(self):
        """
        CG is matrix-free: it builds its matvec directly from the fully
        materialized Jacobian, so it cannot accept a batched (LazySampledObs)
        Jacobian -- this must raise loudly instead of silently misbehaving.
        """
        L = 4
        rbm = nets.CpxRBM(num_hidden=6, bias=False)
        psi = NQS(rbm, L, 2**4, seed=123)
        exact_sampler = sampler.ExactSampler(psi)

        hamiltonian = 0
        for l in range(L):
            hamiltonian += -1.0 * (
                op.SigmaZ(l) * op.SigmaZ((l + 1) % L)
            ) - 0.3 * op.SigmaX(l)

        loss_function = jVMC.objective_function.Observable(
            hamiltonian, batched_jacobian=True
        )
        solver = jVMC.solver.CG()
        stepper = jVMC.stepper.Euler(5e-2)
        opt = jVMC.optimizer.SR(exact_sampler, psi, diagonal_shift=0, solver=solver)

        with self.assertRaises(ValueError):
            opt.ground_state_search(1, loss_function, stepper)


if __name__ == "__main__":
    unittest.main()
