import sys
# Find jVMC package
sys.path.append(sys.path[0]+"/..")

import unittest

import jVMC.nets as nets
from jVMC.vqs import NQS

import jax
import jax.random as random
import jax.numpy as jnp
import numpy as np
import flax.nn as nn

class TestGradients(unittest.TestCase):
    def test_gradients_cpx(self):
        rbm = nets.CpxRBM.partial(L=3,numHidden=2,bias=True)
        _,params = rbm.init_by_shape(random.PRNGKey(0),[(1,3)])
        rbmModel = nn.Model(rbm,params)
        s=2*jnp.zeros((2,3),dtype=np.int32)-1
        s=2*jnp.zeros((4,3),dtype=np.int32)-1
        s=jax.ops.index_update(s,jax.ops.index[0,1],1)
        s=jax.ops.index_update(s,jax.ops.index[2,2],1)
        
        psiC = NQS(rbmModel)
        psi0 = psiC(s)
        G = psiC.gradients(s)
        delta=1e-5
        for j in range(G.shape[1]):
            u = jax.ops.index_update(jnp.zeros(G.shape[1], dtype=np.float32), jax.ops.index[j], 1)
            psiC.update_parameters(delta * u)
            psi1 = psiC(s)
            psiC.set_parameters(params)

            # Finite difference gradients
            Gfd = (psi1-psi0) / delta

            with self.subTest(i=j):
                self.assertTrue( jnp.max( jnp.abs( Gfd - G[:,j] ) ) < 1e-2 )


if __name__ == "__main__":
    unittest.main()
