import unittest

import jax.numpy as jnp

import jVMC
import jVMC.nets as nets
from jVMC.vqs import NQS
import jVMC.operator as op
import jVMC.sampler as sampler

from jVMC.util import measure, ground_state_search


class TestGsSearch(unittest.TestCase):
    def test_gs_search_cpx(self):
        L = 4
        J = -1.0
        hxs = [-1.3, -0.3]

        exEs = [-6.10160339, -4.09296160]

        for hx, exE in zip(hxs, exEs):
            # Set up variational wave function
            rbm = nets.CpxRBM(numHidden=6, bias=False)
            psi = NQS(rbm)

            # Set up hamiltonian for ground state search
            hamiltonianGS = op.BranchFreeOperator()
            for l in range(L):
                hamiltonianGS.add(op.scal_opstr(J, (op.Sz(l), op.Sz((l + 1) % L))))
                hamiltonianGS.add(op.scal_opstr(hx, (op.Sx(l), )))

            # Set up exact sampler
            exactSampler = sampler.ExactSampler(psi, L)

            tdvpEquation = jVMC.util.MinSR(exactSampler, pinvTol=1e-8, makeReal='real')

            # Perform ground state search to get initial state
            ground_state_search(psi, hamiltonianGS, tdvpEquation, exactSampler, numSteps=100, stepSize=5e-2)

            obs = measure({"energy": hamiltonianGS}, psi, exactSampler)

            self.assertTrue(jnp.max(jnp.abs((obs['energy']['mean'] - exE) / exE)) < 1e-3)



if __name__ == "__main__":
    unittest.main()
