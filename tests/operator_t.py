import sys
# Find jVMC package
sys.path.append(sys.path[0]+"/..")

import unittest

import jax
from jax.config import config
config.update("jax_enable_x64", True)

import jax.random as random
import jax.numpy as jnp

import numpy as np

import jVMC
import jVMC.operator as op
import jVMC.sampler
import jVMC.nets as nets
from jVMC.vqs import NQS
import jVMC.global_defs as global_defs

def get_shape(shape):
    return (global_defs.device_count(),) + shape

class TestOperator(unittest.TestCase):

    def test_nonzeros(self):
        
        L=4
        lDim=2
        key = random.PRNGKey(3)
        s = random.randint(key, (24,L), 0, 2, dtype=np.int32).reshape(get_shape((-1, L)))

        h=op.BranchFreeOperator()

        h.add(op.scal_opstr(2., (op.Sp(0),)))
        h.add(op.scal_opstr(2., (op.Sp(1),)))
        h.add(op.scal_opstr(2., (op.Sp(2),)))

        sp,matEl=h.get_s_primes(s)

        logPsi=jnp.ones(s.shape[:-1])
        logPsiSP=jnp.ones(sp.shape[:-1])

        tmp = h.get_O_loc(logPsi,logPsiSP)

        self.assertTrue( jnp.sum(jnp.abs( tmp - 2. * jnp.sum(-(s[...,:3]-1), axis=-1) )) < 1e-7 )

    def test_batched_Oloc(self):
        
        L=4
        
        h=op.BranchFreeOperator()
        for i in range(L):
            h.add(op.scal_opstr(2., (op.Sx(i),)))
            h.add(op.scal_opstr(2., (op.Sy(i), op.Sz((i+1)%L))))
        
        rbm = nets.CpxRBM(numHidden=2,bias=False)
        psi = NQS(rbm)
        
        mcSampler=jVMC.sampler.MCSampler(psi, (L,), random.PRNGKey(0), updateProposer=jVMC.sampler.propose_spin_flip, numChains=1)

        numSamples = 100
        s, logPsi, _ = mcSampler.sample(numSamples=numSamples)

        sp, matEl = h.get_s_primes(s)
        logPsiSp = psi(sp)
        Oloc1 = h.get_O_loc(logPsi, logPsiSp)
        
        batchSize = 13
        Oloc2 = h.get_O_loc_batched(s, psi, logPsi, batchSize)

        self.assertTrue(jnp.abs(jnp.sum(Oloc1) - jnp.sum(Oloc2)) < 1e-5)

if __name__ == "__main__":
    unittest.main()
