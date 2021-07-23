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
import jVMC.operator as op
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


if __name__ == "__main__":
    unittest.main()
