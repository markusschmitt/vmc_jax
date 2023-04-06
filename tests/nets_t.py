import unittest

import jVMC
import jVMC.nets as nets

import jax
from jax.config import config
config.update("jax_enable_x64", True)
import jax.random as random
import jax.numpy as jnp
import numpy as np

import jVMC.util.symmetries as symmetries


class TestCNN(unittest.TestCase):

    def test_cnn_1d(self):
        cnn = nets.CNN(F=(4,), channels=[3, 2, 5])
        params = cnn.init(random.PRNGKey(0), jnp.zeros((5,), dtype=np.int32))

        S0 = jnp.pad(jnp.array([1, 0, 1, 1, 0]), (0, 4), 'wrap')
        S = jnp.array(
            [S0[i:i + 5]for i in range(5)]
        )
        psiS = jax.vmap(lambda s: cnn.apply(params, s))(S)
        psiS = psiS - psiS[0]

        self.assertTrue(jnp.max(jnp.abs(psiS)) < 1e-12)

    def test_cnn_2d(self):
        cnn = nets.CNN(F=(3, 3), channels=[3, 2, 5], strides=[1, 1])
        params = cnn.init(random.PRNGKey(0), jnp.zeros((4, 4), dtype=np.int32))

        S0 = jnp.array(
            [[1, 0, 1, 1],
             [0, 1, 1, 1],
             [0, 0, 1, 0],
             [1, 0, 0, 1]]
        )
        S0 = jnp.pad(S0, [(0, 3), (0, 3)], 'wrap')
        S = jnp.array(
            [S0[i:i + 4, j:j + 4] for i in range(4) for j in range(4)]
        )
        psiS = jax.vmap(lambda s: cnn.apply(params, s))(S)
        psiS = psiS - psiS[0]

        self.assertTrue(jnp.max(jnp.abs(psiS)) < 1e-12)


class TestSymNet(unittest.TestCase):

    def test_sym_net(self):
        rbm = nets.RBM(numHidden=5)
        orbit = symmetries.get_orbit_1d(5, reflection=False, translation=True, z2sym=False)
        rbm_sym = nets.SymNet(net=rbm, orbit=orbit)
        params = rbm_sym.init(random.PRNGKey(0), jnp.zeros((5,), dtype=np.int32))

        S0 = jnp.pad(jnp.array([1, 0, 1, 1, 0]), (0, 4), 'wrap')
        S = jnp.array(
            [S0[i:i + 5]for i in range(5)]
        )
        psiS = jax.vmap(lambda s: rbm_sym.apply(params, s))(S)
        psiS = psiS - psiS[0]

        self.assertTrue(jnp.max(jnp.abs(psiS)) < 1e-12)


if __name__ == "__main__":
    unittest.main()
