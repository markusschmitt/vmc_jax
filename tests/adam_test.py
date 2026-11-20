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

            loss_function = jVMC.objective_function.Observable(hamiltonian)
            opt = jVMC.optimizer.Adam(exact_sampler, psi, 1e-3)

            opt.ground_state_search(1000, loss_function)
            eps_rel = jnp.abs((exact_sampler(hamiltonian).mean.item() - exact_energy) / exact_energy)
            self.assertTrue(eps_rel < 1e-3)


if __name__ == "__main__":
    unittest.main()
