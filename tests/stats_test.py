import unittest

import jax
import jax.numpy as jnp

import jVMC
from jVMC.stats import SampledObs
import jVMC.operator as op
from jVMC.global_defs import device_count
import jVMC.mpi_wrapper as mpi


class TestStats(unittest.TestCase):

    def test_sampled_obs(self):
        
        Obs1Loc = jnp.array([[1, 2, 3]] * device_count())
        Obs2Loc = jnp.array([[[1, 4], [2, 5], [3, 7]]] * device_count())
        p = (1. / (3 * device_count() * mpi.commSize)) * jnp.ones((device_count(), 3))

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
        O = jnp.vstack([O,]*mpi.commSize)
        self.assertTrue(jnp.allclose(obs2.tangent_kernel(), jnp.matmul(O, jnp.conj(jnp.transpose(O)))))


    def test_estimator(self):

        L = 4
        
        for rbm in [jVMC.nets.CpxRBM(numHidden=1, bias=False), jVMC.nets.RBM(numHidden=1, bias=False)]:

            # Set up variational wave function
            orbit = jVMC.util.symmetries.get_orbit_1D(L, "translation", "reflection")
            net = jVMC.nets.sym_wrapper.SymNet(net=rbm, orbit=orbit)
            psi = jVMC.vqs.NQS(net)

            # Set up MCMC sampler
            # mcSampler = jVMC.sampler.MCSampler(psi, (L,), jax.random.PRNGKey(0), updateProposer=jVMC.sampler.propose_spin_flip, sweepSteps=L+1, numChains=777)

            exactSampler = jVMC.sampler.ExactSampler(psi, (L,))

            p0 = psi.parameters

            # configs, configsLogPsi, p = mcSampler.sample(numSamples=40000)

            configs, configsLogPsi, p = exactSampler.sample()

            h = op.BranchFreeOperator()
            for i in range(L):
                h.add(op.scal_opstr(2., (op.Sx(i),)))
                h.add(op.scal_opstr(2., (op.Sy(i), op.Sz((i + 1) % L))))

            op_estimator = h.get_estimator_function(psi)

            obs1 = SampledObs(configs, p, estimator=op_estimator)

            Oloc = h.get_O_loc(configs, psi)
            obs2 = SampledObs(Oloc, p)
            self.assertTrue( jnp.allclose( obs1.mean(p0), obs2.mean() ) )

            psiGrads = SampledObs(psi.gradients(configs), p)
            Eloc = h.get_O_loc(configs, psi, configsLogPsi)
            Eloc = SampledObs( Eloc, p )

            Egrad2 = 2*jnp.real( psiGrads.covar(Eloc) )

            self.assertTrue(jnp.allclose( jnp.real(obs1.mean_and_grad(psi, p0)[1]), Egrad2.ravel() ))
            
        
    def test_subset_function(self):

        N = 10
        Obs1 = jnp.reshape(jnp.arange(jax.device_count()*N), (jax.device_count(), N, 1))
        p = jax.random.uniform(jax.random.PRNGKey(123), (jax.device_count(),N))
        p = p / mpi.global_sum(p)

        obs1 = SampledObs(Obs1, p)
        obs2 = obs1.subset(0,N//2)

        self.assertTrue( jnp.allclose(obs1.mean()[0], mpi.global_sum(jnp.reshape(Obs1, (jax.device_count(), N)) * p)) )

        self.assertTrue( 
            jnp.allclose(
                obs2.mean(), 
                mpi.global_sum(jnp.reshape(Obs1, (jax.device_count(), N))[:,0:N//2] * p[:,0:N//2]) / jnp.sum(p[:,0:N//2])
                )
            )

        obs3 = SampledObs(Obs1[:,0:N//2,:], p[:,0:N//2] / mpi.global_sum(p[:,0:N//2]))

        self.assertTrue( jnp.allclose(obs3.covar(), obs2.covar()))
