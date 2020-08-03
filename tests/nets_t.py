import sys
# Find jVMC package
sys.path.append(sys.path[0]+"/..")

import unittest

import jVMC
import jVMC.nets as nets

import jax
from jax.config import config
config.update("jax_enable_x64", True)
import jax.random as random
import jax.numpy as jnp
import numpy as np
import flax.nn as nn

class TestCNN(unittest.TestCase):
    
    def test_cnn_1d(self):
        cnn = nets.CNN.partial(F=(4,), channels=[3,2,5])
        _,params = cnn.init_by_shape(random.PRNGKey(0),[(5,)])
        cnnModel = nn.Model(cnn,params)
  
        S0=jnp.pad(jnp.array([1,0,1,1,0]),(0,4),'wrap')
        S=jnp.array(
                [S0[i:i+5]for i in range(5)]
            )
        psiS=jax.vmap(cnnModel)(S)
        psiS=psiS-psiS[0]

        self.assertTrue( jnp.max( jnp.abs( psiS ) ) < 1e-12 )
    

    def test_cnn_2d(self):
        cnn = nets.CNN.partial(F=(3,3), channels=[3,2,5], strides=[1,1])
        _,params = cnn.init_by_shape(random.PRNGKey(0),[(4,4)])
        cnnModel = nn.Model(cnn,params)
  
        S0=jnp.array(
                [[1,0,1,1],
                 [0,1,1,1],
                 [0,0,1,0],
                 [1,0,0,1]]
            )
        S0=jnp.pad(S0,[(0,3),(0,3)],'wrap')
        S=jnp.array(
                [S0[i:i+4, j:j+4] for i in range(4) for j in range(4)]
            )
        psiS=jax.vmap(cnnModel)(S)
        psiS=psiS-psiS[0]

        self.assertTrue( jnp.max( jnp.abs( psiS ) ) < 1e-12 )


if __name__ == "__main__":
    unittest.main()
