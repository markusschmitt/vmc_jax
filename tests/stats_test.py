import unittest

import jax
import jax.numpy as jnp

from jVMC.stats import SampledObs
import jVMC.mpi_wrapper as mpi
from jVMC.global_defs import device_count


class TestStats(unittest.TestCase):

    def test_sampled_obs(self):
        
        Obs1Loc = jnp.array([[1, 2, 3]] * device_count())
        Obs2Loc = jnp.array([[[1, 4], [2, 5], [3, 7]]] * device_count())
        p = (1. / (3 * device_count())) * jnp.ones((device_count(), 3))

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

    def test_subset_function(self):

        N = 10
        Obs1 = jnp.reshape(jnp.arange(jax.device_count()*N), (jax.device_count(), N, 1))
        p = jax.random.uniform(jax.random.PRNGKey(123), (jax.device_count(),N))
        p = p / jnp.sum(p)

        obs1 = SampledObs(Obs1, p)
        obs2 = obs1.subset(0,N//2)

        self.assertTrue( jnp.allclose(obs1.mean(), jnp.sum(jnp.reshape(Obs1, (jax.device_count(), N)) * p)) )

        self.assertTrue(obs2.mean() - jnp.sum(jnp.reshape(Obs1, (jax.device_count(), N))[:,0:N//2] * p[:,0:N//2]) / jnp.sum(p[:,0:N//2]))


        obs3 = SampledObs(Obs1[:,0:N//2,:], p[:,0:N//2] / jnp.sum(p[:,0:N//2]))

        self.assertTrue( jnp.allclose(obs3.covar(), obs2.covar()))
