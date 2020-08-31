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


class TestOperator(unittest.TestCase):

    def test_nonzeros(self):
        L=4
        lDim=2
        key = random.PRNGKey(3)
        s = random.randint(key, (24,L), 0, 2, dtype=np.int32).reshape((jax.local_device_count(), -1, L))

        h=op.Operator()

        h.add(op.scal_opstr(2., (op.Sp(0),)))
        h.add(op.scal_opstr(2., (op.Sp(1),)))
        h.add(op.scal_opstr(2., (op.Sp(2),)))

        sp,matEl=h.get_s_primes(s)

        logPsi=jax.pmap(lambda s: jnp.ones(s.shape[0]))(s)
        logPsiSP=jax.pmap(lambda sp: jnp.ones((sp.shape[0], sp.shape[1])))(sp) 

        tmp = h.get_O_loc(logPsi,logPsiSP)

        self.assertTrue( jnp.sum(jnp.abs( tmp - 2. * jnp.sum(-(s[:,:,:3]-1), axis=-1) )) < 1e-7 )


if __name__ == "__main__":
    unittest.main()
