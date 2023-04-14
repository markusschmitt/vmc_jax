import unittest

import jax
from jax.config import config
config.update("jax_enable_x64", True)
import jax.random as random
import jax.numpy as jnp

import numpy as np

import sys
sys.path.append(sys.path[0] + '/../')

import jVMC
import jVMC.nets as nets
from jVMC.vqs import NQS
import jVMC.sampler as sampler

import jVMC.global_defs as global_defs
from jVMC.util.symmetries import LatticeSymmetry

import time


def state_to_int(s):

    def for_fun(i, xs):
        return (xs[0] + xs[1][i] * (2**i), xs[1])

    x, _ = jax.lax.fori_loop(0, s.shape[-1], for_fun, (0, s))

    return x


class TestMC(unittest.TestCase):

    def test_MCMC_sampling(self):
        L = 4

        weights = jnp.array(
            [0.23898957, 0.12614753, 0.19479055, 0.17325271, 0.14619853, 0.21392751,
             0.19648707, 0.17103704, -0.15457255, 0.10954413, 0.13228065, -0.14935214,
             -0.09963073, 0.17610707, 0.13386381, -0.14836467]
        )

        # Set up variational wave function
        rbm = nets.CpxRBM(numHidden=2, bias=False)
        orbit = jVMC.util.symmetries.get_orbit_1d(L, translation=False, reflection=False, z2sym=False)
        net = nets.sym_wrapper.SymNet(net=rbm, orbit=orbit)
        psi = NQS(net)

        # Set up exact sampler
        exactSampler = sampler.ExactSampler(psi, L)

        # Set up MCMC sampler
        mcSampler = sampler.MCSampler(psi, (L,), random.PRNGKey(0), updateProposer=jVMC.sampler.propose_spin_flip, numChains=777)

        psi.set_parameters(weights)

        # Compute exact probabilities
        _, _, pex = exactSampler.sample()

        # Get samples from MCMC sampler
        numSamples = 500000
        smc, _, p = mcSampler.sample(numSamples=numSamples)

        smc = smc.reshape((smc.shape[0] * smc.shape[1], -1))

        self.assertTrue(smc.shape[0] >= numSamples)

        # Compute histogram of sampled configurations
        smcInt = jax.vmap(state_to_int)(smc)
        pmc, _ = np.histogram(smcInt, bins=np.arange(0, 17), weights=p[0])

        # Compare histogram to exact probabilities
        self.assertTrue(jnp.max(jnp.abs(pmc - pex.reshape((-1,))[:16])) < 2e-3)

    def test_MCMC_sampling_with_mu(self):
        mu = 1

        L = 4

        weights = jnp.array(
            [0.23898957, 0.12614753, 0.19479055, 0.17325271, 0.14619853, 0.21392751,
             0.19648707, 0.17103704, -0.15457255, 0.10954413, 0.13228065, -0.14935214,
             -0.09963073, 0.17610707, 0.13386381, -0.14836467]
        )

        # Set up variational wave function
        rbm = nets.CpxRBM(numHidden=2, bias=False)
        orbit = jVMC.util.symmetries.get_orbit_1d(L, translation=False, reflection=False, z2sym=False)
        net = nets.sym_wrapper.SymNet(net=rbm, orbit=orbit)
        psi = NQS(net)

        # Set up exact sampler
        exactSampler = sampler.ExactSampler(psi, L)

        # Set up MCMC sampler
        mcSampler = sampler.MCSampler(psi, (L,), random.PRNGKey(0), updateProposer=jVMC.sampler.propose_spin_flip, numChains=777, mu=mu)

        psi.set_parameters(weights)

        # Compute exact probabilities
        _, _, pex = exactSampler.sample()

        # Get samples from MCMC sampler
        numSamples = 500000
        smc, logPsi, weighting_probs = mcSampler.sample(numSamples=numSamples)

        smc = smc.reshape((smc.shape[0] * smc.shape[1], -1))

        self.assertTrue(smc.shape[0] >= numSamples)

        # Compute histogram of sampled configurations
        smcInt = jax.vmap(state_to_int)(smc)

        psi_log_basis = psi(exactSampler.basis)

        pmc, _ = np.histogram(smcInt, bins=np.arange(0, 17), weights=weighting_probs[0, :])

        # Compare histogram to exact probabilities
        self.assertTrue(jnp.max(jnp.abs(pmc - pex.reshape((-1,))[:16])) < 2e-3)

    def test_MCMC_sampling_with_two_nets(self):
        L = 4

        weights = jnp.array(
            [0.23898957, 0.12614753, 0.19479055, 0.17325271, 0.14619853, 0.21392751,
             0.19648707, 0.17103704, -0.15457255, 0.10954413, 0.13228065, -0.14935214,
             -0.09963073, 0.17610707, 0.13386381, -0.14836467]
        )

        # Set up variational wave function
        rbm1 = nets.RBM(numHidden=2, bias=False)
        rbm2 = nets.RBM(numHidden=2, bias=False)
        model = jVMC.nets.two_nets_wrapper.TwoNets(net1=rbm1, net2=rbm2)
        orbit = jVMC.util.symmetries.get_orbit_1d(L, translation=False, reflection=False, z2sym=False)
        net = nets.sym_wrapper.SymNet(net=model, orbit=orbit)
        psi = NQS(net)

        # Set up exact sampler
        exactSampler = sampler.ExactSampler(psi, L)

        # Set up MCMC sampler
        mcSampler = sampler.MCSampler(psi, (L,), random.PRNGKey(0), updateProposer=jVMC.sampler.propose_spin_flip, numChains=777)

        psi.set_parameters(jnp.concatenate([weights, weights]))

        # Compute exact probabilities
        _, _, pex = exactSampler.sample()

        # Get samples from MCMC sampler
        numSamples = 500000
        smc, _, p = mcSampler.sample(numSamples=numSamples)

        smc = smc.reshape((smc.shape[0] * smc.shape[1], -1))

        self.assertTrue(smc.shape[0] >= numSamples)

        # Compute histogram of sampled configurations
        smcInt = jax.vmap(state_to_int)(smc)
        pmc, _ = np.histogram(smcInt, bins=np.arange(0, 17), weights=p[0])

        # Compare histogram to exact probabilities
        self.assertTrue(jnp.max(jnp.abs(pmc - pex.reshape((-1,))[:16])) < 2e-3)

    def test_MCMC_sampling_with_integer_key(self):
        L = 4

        weights = jnp.array(
            [0.23898957, 0.12614753, 0.19479055, 0.17325271, 0.14619853, 0.21392751,
             0.19648707, 0.17103704, -0.15457255, 0.10954413, 0.13228065, -0.14935214,
             -0.09963073, 0.17610707, 0.13386381, -0.14836467]
        )

        # Set up variational wave function
        rbm = nets.CpxRBM(numHidden=2, bias=False)
        orbit = jVMC.util.symmetries.get_orbit_1d(L, translation=False, reflection=False, z2sym=False)
        net = nets.sym_wrapper.SymNet(net=rbm, orbit=orbit)
        psi = NQS(net)

        # Set up exact sampler
        exactSampler = sampler.ExactSampler(psi, L)

        # Set up MCMC sampler
        mcSampler = sampler.MCSampler(psi, (L,), 123, updateProposer=jVMC.sampler.propose_spin_flip, numChains=777)

        psi.set_parameters(weights)

        # Compute exact probabilities
        _, _, pex = exactSampler.sample()

        # Get samples from MCMC sampler
        numSamples = 500000
        smc, _, p = mcSampler.sample(numSamples=numSamples)

        smc = smc.reshape((smc.shape[0] * smc.shape[1], -1))

        self.assertTrue(smc.shape[0] >= numSamples)

        # Compute histogram of sampled configurations
        smcInt = jax.vmap(state_to_int)(smc)
        pmc, _ = np.histogram(smcInt, bins=np.arange(0, 17), weights=p[0])

        # Compare histogram to exact probabilities
        self.assertTrue(jnp.max(jnp.abs(pmc - pex.reshape((-1,))[:16])) < 2e-3)

    def test_autoregressive_sampling(self):

        L = 4

        # Set up variational wave function
        rnn = nets.RNN1DGeneral(L=4, hiddenSize=5, depth=2)
        rbm = nets.RBM(numHidden=2, bias=False)

        model = jVMC.nets.two_nets_wrapper.TwoNets(net1=rnn, net2=rbm)
        orbit = jVMC.util.symmetries.get_orbit_1d(L, translation=False, reflection=False, z2sym=False)
        net = nets.sym_wrapper.SymNet(net=model, orbit=orbit, avgFun=jVMC.nets.sym_wrapper.avgFun_Coefficients_Sep)
        psi = NQS(net)

        # Set up exact sampler
        exactSampler = sampler.ExactSampler(psi, L)

        # Set up MCMC sampler
        mcSampler = sampler.MCSampler(psi, (L,), random.PRNGKey(0), updateProposer=jVMC.sampler.propose_spin_flip, numChains=777)

        ps = psi.get_parameters()
        psi.update_parameters(ps)

        # Compute exact probabilities
        _, _, pex = exactSampler.sample()

        numSamples = 1000000
        smc, logPsi, p = mcSampler.sample(numSamples=numSamples)

        self.assertTrue(jnp.max(jnp.abs(jnp.real(psi(smc) - logPsi))) < 1e-12)

        smc = smc.reshape((smc.shape[0] * smc.shape[1], -1))

        self.assertTrue(smc.shape[0] >= numSamples)

        # Compute histogram of sampled configurations
        smcInt = jax.vmap(state_to_int)(smc)
        pmc, _ = np.histogram(smcInt, bins=np.arange(0, 17), weights=p[0])

        self.assertTrue(jnp.max(jnp.abs(pmc - pex.reshape((-1,))[:16])) < 1.1e-3)

    def test_autoregressive_sampling_with_symmetries(self):

        L = 4

        # Set up symmetry orbit
        orbit = LatticeSymmetry(jnp.array([jnp.roll(jnp.identity(L, dtype=np.int32), l, axis=1) for l in range(L)]))

        # Set up variational wave function
        rnn = nets.RNN1DGeneral(L=L, hiddenSize=5, realValuedOutput=True)
        rbm = nets.RBM(numHidden=2, bias=False)

        model = jVMC.nets.two_nets_wrapper.TwoNets(net1=rnn, net2=rbm)
        net = nets.sym_wrapper.SymNet(net=model, orbit=orbit, avgFun=jVMC.nets.sym_wrapper.avgFun_Coefficients_Sep)
        psi = NQS(net)

        # Set up exact sampler
        exactSampler = sampler.ExactSampler(psi, L)

        # Set up MCMC sampler
        mcSampler = sampler.MCSampler(psi, (L,), random.PRNGKey(0), numChains=777, updateProposer=jVMC.sampler.propose_spin_flip)

        # Compute exact probabilities
        _, logPsi, pex = exactSampler.sample()

        numSamples = 1000000
        smc, logPsi, p = mcSampler.sample(numSamples=numSamples)

        self.assertTrue(jnp.max(jnp.abs(jnp.real(psi(smc) - logPsi))) < 1e-12)

        smc = smc.reshape((smc.shape[0] * smc.shape[1], -1))

        self.assertTrue(smc.shape[0] >= numSamples)

        # Compute histogram of sampled configurations
        smcInt = jax.vmap(state_to_int)(smc)
        pmc, _ = np.histogram(smcInt, bins=np.arange(0, 17), weights=p[0])

        self.assertTrue(jnp.max(jnp.abs(pmc - pex.reshape((-1,))[:16])) < 2e-3)

    def test_autoregressive_sampling_with_lstm(self):

        L = 4

        # Set up symmetry orbit
        orbit = jnp.array([jnp.roll(jnp.identity(L, dtype=np.int32), l, axis=1) for l in range(L)])

        # Set up variational wave function
        rnn = nets.RNN1DGeneral(L=L, hiddenSize=5, cell="LSTM", realValuedParams=True, realValuedOutput=True, inputDim=2)
        rbm = nets.RBM(numHidden=2, bias=False)

        model = jVMC.nets.two_nets_wrapper.TwoNets(net1=rnn, net2=rbm)
        orbit = jVMC.util.symmetries.get_orbit_1d(L, translation=False, reflection=False, z2sym=False)
        net = nets.sym_wrapper.SymNet(net=model, orbit=orbit, avgFun=jVMC.nets.sym_wrapper.avgFun_Coefficients_Sep)
        psi = NQS(net)

        # Set up exact sampler
        exactSampler = sampler.ExactSampler(psi, L)

        # Set up MCMC sampler
        mcSampler = sampler.MCSampler(psi, (L,), random.PRNGKey(0), numChains=777)

        # Compute exact probabilities
        _, logPsi, pex = exactSampler.sample()

        numSamples = 1000000
        smc, logPsi, p = mcSampler.sample(numSamples=numSamples)

        self.assertTrue(jnp.max(jnp.abs(jnp.real(psi(smc) - logPsi))) < 1e-12)

        smc = smc.reshape((smc.shape[0] * smc.shape[1], -1))

        self.assertTrue(smc.shape[0] >= numSamples)

        # Compute histogram of sampled configurations
        smcInt = jax.vmap(state_to_int)(smc)
        pmc, _ = np.histogram(smcInt, bins=np.arange(0, 17), weights=p[0])

        self.assertTrue(jnp.max(jnp.abs(pmc - pex.reshape((-1,))[:16])) < 1e-3)

    def test_autoregressive_sampling_with_gru(self):

        L = 4

        # Set up symmetry orbit
        orbit = jnp.array([jnp.roll(jnp.identity(L, dtype=np.int32), l, axis=1) for l in range(L)])

        # Set up variational wave function
        rnn = nets.RNN1DGeneral(L=L, hiddenSize=5, cell="GRU", realValuedParams=True, realValuedOutput=True, inputDim=2)
        rbm = nets.RBM(numHidden=2, bias=False)

        model = jVMC.nets.two_nets_wrapper.TwoNets(net1=rnn, net2=rbm)
        orbit = jVMC.util.symmetries.get_orbit_1d(L, translation=False, reflection=False, z2sym=False)
        net = nets.sym_wrapper.SymNet(net=model, orbit=orbit, avgFun=jVMC.nets.sym_wrapper.avgFun_Coefficients_Sep)
        psi = NQS(net)

        # Set up exact sampler
        exactSampler = sampler.ExactSampler(psi, L)

        # Set up MCMC sampler
        mcSampler = sampler.MCSampler(psi, (L,), random.PRNGKey(0), numChains=777)

        # Compute exact probabilities
        _, logPsi, pex = exactSampler.sample()

        numSamples = 1000000
        smc, logPsi, p = mcSampler.sample(numSamples=numSamples)

        self.assertTrue(jnp.max(jnp.abs(jnp.real(psi(smc) - logPsi))) < 1e-12)

        smc = smc.reshape((smc.shape[0] * smc.shape[1], -1))

        self.assertTrue(smc.shape[0] >= numSamples)

        # Compute histogram of sampled configurations
        smcInt = jax.vmap(state_to_int)(smc)
        pmc, _ = np.histogram(smcInt, bins=np.arange(0, 17), weights=p[0])

        self.assertTrue(jnp.max(jnp.abs(pmc - pex.reshape((-1,))[:16])) < 1e-3)

    def test_autoregressive_sampling_with_rnn2d(self):

        L = 2

        # Set up variational wave function
        #rnn = nets.RNN2D( L=L, hiddenSize=5 )
        rnn = nets.RNN2DGeneral(L=L, hiddenSize=5, cell="RNN", realValuedParams=True, realValuedOutput=True)

        model = jVMC.nets.two_nets_wrapper.TwoNets(net1=rnn, net2=rnn)
        orbit = jVMC.util.symmetries.get_orbit_2d_square(L, translation=False, reflection=False, rotation=False, z2sym=False)
        net = nets.sym_wrapper.SymNet(net=model, orbit=orbit, avgFun=jVMC.nets.sym_wrapper.avgFun_Coefficients_Sep)
        psi = NQS(net)

        # Set up exact sampler
        exactSampler = sampler.ExactSampler(psi, (L, L))

        # Set up MCMC sampler
        mcSampler = sampler.MCSampler(psi, (L, L), random.PRNGKey(0), numChains=777)

        # Compute exact probabilities
        _, logPsi, pex = exactSampler.sample()

        self.assertTrue(jnp.abs(jnp.sum(pex) - 1.) < 1e-12)

        numSamples = 1000000
        smc, logPsi, p = mcSampler.sample(numSamples=numSamples)

        self.assertTrue(jnp.max(jnp.abs(jnp.real(psi(smc) - logPsi))) < 1e-12)

        smc = smc.reshape((smc.shape[0] * smc.shape[1], -1))

        self.assertTrue(smc.shape[0] >= numSamples)

        # Compute histogram of sampled configurations
        smcInt = jax.vmap(state_to_int)(smc)
        pmc, _ = np.histogram(smcInt, bins=np.arange(0, 17), weights=p[0])

        self.assertTrue(jnp.max(jnp.abs(pmc - pex.reshape((-1,))[:16])) < 1e-3)

    def test_autoregressive_sampling_with_lstm2d(self):

        L = 2

        # Set up variational wave function
        rnn = nets.RNN2DGeneral(L=L, hiddenSize=5, cell="LSTM", realValuedParams=True, realValuedOutput=True)

        model = jVMC.nets.two_nets_wrapper.TwoNets(net1=rnn, net2=rnn)
        orbit = jVMC.util.symmetries.get_orbit_2d_square(L, translation=False, reflection=False, rotation=False, z2sym=False)
        net = nets.sym_wrapper.SymNet(net=model, orbit=orbit, avgFun=jVMC.nets.sym_wrapper.avgFun_Coefficients_Sep)
        psi = NQS(net)

        # Set up exact sampler
        exactSampler = sampler.ExactSampler(psi, (L, L))

        # Set up MCMC sampler
        mcSampler = sampler.MCSampler(psi, (L, L), random.PRNGKey(0), numChains=777)

        # Compute exact probabilities
        _, logPsi, pex = exactSampler.sample()

        self.assertTrue(jnp.abs(jnp.sum(pex) - 1.) < 1e-12)

        numSamples = 1000000
        smc, logPsi, p = mcSampler.sample(numSamples=numSamples)

        self.assertTrue(jnp.max(jnp.abs(jnp.real(psi(smc) - logPsi))) < 1e-12)

        smc = smc.reshape((smc.shape[0] * smc.shape[1], -1))

        self.assertTrue(smc.shape[0] >= numSamples)

        # Compute histogram of sampled configurations
        smcInt = jax.vmap(state_to_int)(smc)
        pmc, _ = np.histogram(smcInt, bins=np.arange(0, 17), weights=p[0])

        self.assertTrue(jnp.max(jnp.abs(pmc - pex.reshape((-1,))[:16])) < 1e-3)


if __name__ == "__main__":
    unittest.main()
