import unittest
import jVMC_exp
import jVMC_exp.nets as nets
from jVMC_exp.symmetry import ProjectedOrbitNet
from jVMC_exp.symmetry.lattice_symetries import square_translation_symmetry

import jax
import jax.random as random
import jax.numpy as jnp
import numpy as np

class TestCNN(unittest.TestCase):

    def test_cnn_1d(self):
        cnn = nets.CNN(F=(4,), channels=[3, 2, 5])
        params = cnn.init(random.PRNGKey(0), jnp.zeros((5,), dtype=np.int32))

        S0 = jnp.pad(jnp.array([1, 0, 1, 1, 0]), (0, 4), 'wrap')
        S = jnp.array([S0[i:i + 5]for i in range(5)])
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
        S = jnp.array([S0[i:i + 4, j:j + 4] for i in range(4) for j in range(4)])
        psiS = jax.vmap(lambda s: cnn.apply(params, s))(S)
        psiS = psiS - psiS[0]

        self.assertTrue(jnp.max(jnp.abs(psiS)) < 1e-12)

class TestProjectedOrbitNet(unittest.TestCase):

    def test_projected_orbit_net(self):
        L = 5
        rbm = nets.RBM(numHidden=5)
        orbit = square_translation_symmetry(L, 1, "spin")
        rbm_sym = orbit * rbm
        params = rbm_sym.init(random.PRNGKey(0), jnp.zeros((L,), dtype=np.int32))

        S0 = jnp.pad(jnp.array([1, 0, 1, 1, 0]), (0, 4), 'wrap')
        S = jnp.array([S0[i:i + 5]for i in range(5)])
        psiS = jax.vmap(lambda s: rbm_sym.apply(params, s))(S)
        psiS = psiS - psiS[0]

        self.assertTrue(jnp.max(jnp.abs(psiS)) < 1e-12)

    def test_projected_orbit_net_generative(self):
        L=5
        rnn = nets.RNN1DGeneral(L=5)
        orbit = square_translation_symmetry(L, 1, "spin")
        rnn_sym = ProjectedOrbitNet(base_net=rnn, symmetry=orbit, symmetry_average="sep")
        params = rnn_sym.init(random.PRNGKey(0), jnp.zeros((5,), dtype=np.int32))

        S0 = jnp.pad(jnp.array([1, 0, 1, 1, 0]), (0, 4), 'wrap')
        S = jnp.array([S0[i:i + 5]for i in range(5)])
        psiS = jax.vmap(lambda s: rnn_sym.apply(params, s))(S)
        psiS = psiS - psiS[0]

        self.assertTrue(jnp.max(jnp.abs(psiS)) < 1e-12)


class TesDT_PARAMS_CPXNet(unittest.TestCase):

    def test_cpx_rnn_1d(self):
        rnn = nets.RNN1DGeneral(L=5, realValuedParams=False)
        params = rnn.init(random.PRNGKey(0), jnp.zeros((5,), dtype=np.int32))

        S0 = jnp.array([1, 0, 1, 1, 0])
        psiS0 = rnn.apply(params, S0)
        self.assertTrue(psiS0.dtype == np.complex128)

    # def test_cpx_rnn_2d(self):
    #     rnn = nets.RNN2DGeneral(L=4, realValuedParams=False)
    #     params = rnn.init(random.PRNGKey(0), jnp.zeros((4, 4), dtype=np.int32))

    #     S0 = jnp.array(
    #         [[1, 0, 1, 1],
    #          [0, 1, 1, 1],
    #          [0, 0, 1, 0],
    #          [1, 0, 0, 1]]
    #     )
    #     psiS0 = rnn.apply(params, S0)
    #     self.assertTrue(psiS0.dtype == np.complex128)

    def test_cpx_cnn_1d(self):
        cnn = nets.CpxCNN(F=(4,), channels=[3, 2, 5])
        params = cnn.init(random.PRNGKey(0), jnp.zeros((5,), dtype=np.int32))

        self.assertTrue(params['params']['Conv_1']['kernel'].dtype == jVMC_exp.global_defs.DT_PARAMS_CPX)
        self.assertTrue(params['params']['Conv_1']['bias'].dtype == jVMC_exp.global_defs.DT_PARAMS_CPX)

        S0 = jnp.pad(jnp.array([1, 0, 1, 1, 0]), (0, 4), 'wrap')
        S = jnp.array(
            [S0[i:i + 5]for i in range(5)]
        )
        psiS = jax.vmap(lambda s: cnn.apply(params, s))(S)
        psiS = psiS - psiS[0]

        self.assertTrue(jnp.max(jnp.abs(psiS)) < 1e-12)

if __name__ == "__main__":
    unittest.main()
