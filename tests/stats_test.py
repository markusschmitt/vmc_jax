import unittest

import jax.numpy as jnp

import jVMC
from jVMC.stats import SampledObs
import jVMC.operator as op


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


        # obs1 = SampledObs(weights=pEx, configs=configsEx, estimator=op_estimator)
        # E0 = obs1.mean(psi.parameters)
        # p0 = psi.get_parameters()
        # dp = 1e-6
        # p0 = p0.at[0].add(dp)
        # psi.set_parameters(p0)
        # configsEx, configsLogPsiEx, pEx = exactSampler.sample(parameters=psi.params)
        # obs1 = SampledObs(weights=pEx, configs=configsEx, estimator=op_estimator)
        # E1 = obs1.mean(psi.parameters)
        # print((E1-E0)/dp)