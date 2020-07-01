import sys
# Find jVMC package
sys.path.append(sys.path[0]+"/..")

import unittest

import jax
from jax.config import config
config.update("jax_enable_x64", True)

import jax.random as random
import flax.nn as nn
import jax.numpy as jnp

import numpy as np

import jVMC
import jVMC.stepper as jVMCstepper
import jVMC.nets as nets
from jVMC.vqs import NQS
import jVMC.operator as op
import jVMC.sampler as sampler
import jVMC.tdvp as tdvp

from jVMC.util import measure, ground_state_search

class TestGsSearch(unittest.TestCase):
    def test_gs_search(self):
        L=4
        J=-1.0
        hxs=[-1.3, -0.3]
        
        exEs=[-6.10160339, -4.09296160]

        for hx,exE in zip(hxs,exEs):
            # Set up variational wave function
            rbm = nets.CpxRBM.partial(L=L,numHidden=6,bias=False)
            _, params = rbm.init_by_shape(random.PRNGKey(0),[(1,L)])
            rbmModel = nn.Model(rbm,params)
            psi = NQS(rbmModel)

            # Set up hamiltonian for ground state search
            hamiltonianGS = op.Operator()
            for l in range(L):
                hamiltonianGS.add( op.scal_opstr( J, ( op.Sz(l), op.Sz((l+1)%L) ) ) )
                hamiltonianGS.add( op.scal_opstr( hx, ( op.Sx(l), ) ) )

            # Set up exact sampler
            exactSampler=sampler.ExactSampler(L)

            delta=2
            tdvpEquation = jVMC.tdvp.TDVP(exactSampler, snrTol=1, svdTol=1e-8, rhsPrefactor=1., diagonalShift=delta, makeReal='real')

            # Perform ground state search to get initial state
            ground_state_search(psi, hamiltonianGS, tdvpEquation, exactSampler, numSteps=150, stepSize=1e-2)
                
            obs = measure([hamiltonianGS], psi, exactSampler)

            self.assertTrue( jnp.max( jnp.abs( ( obs[0] - exE ) / exE) ) < 1e-3 )


if __name__ == "__main__":
    unittest.main()
