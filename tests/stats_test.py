import unittest

import jax
import jax.numpy as jnp

from jVMC.stats import SampledObs
import jVMC.mpi_wrapper as mpi


class TestStats(unittest.TestCase):

    def test_sampled_obs(self):
        
        Obs1Loc = jnp.array([[1,2,3]])
        Obs2Loc = jnp.array([[[1,4],[2,5],[3,7]]])
        p = (1./3) * jnp.ones(3)[None,...]

        obs1 = SampledObs(Obs1Loc, p)
        obs2 = SampledObs(Obs2Loc, p)

        self.assertAlmostEqual(obs1.mean()[0], 2.)
        self.assertAlmostEqual(obs1.var()[0], 2./3)

        self.assertTrue(jnp.allclose(obs2.covar(), jnp.array([[2./3, 1],[1.,14./9]])))

        self.assertTrue(jnp.allclose(obs2.mean(), jnp.array([2,16./3])))
        self.assertTrue(jnp.allclose(obs1.covar(obs2), jnp.array([2./3,1.])))

        self.assertTrue(jnp.allclose(obs1.covar(obs2), obs1.covar_data(obs2).mean()))

        self.assertTrue(jnp.allclose(obs1.covar_var(obs2), obs1.covar_data(obs2).var()))

        O = obs2._data.reshape((-1,2))
        self.assertTrue(jnp.allclose(obs2.tangent_kernel(), jnp.matmul(O, jnp.conj(jnp.transpose(O)))))

