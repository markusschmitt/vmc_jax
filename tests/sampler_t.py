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
import jVMC.nets as nets
from jVMC.vqs import NQS
import jVMC.sampler as sampler


def state_to_int(s):

    def for_fun(i,xs):
        return (xs[0] + xs[1][i]*(2**i), xs[1])

    x,_=jax.lax.fori_loop(0,s.shape[-1], for_fun, (0, s))

    return x


class TestMCMC(unittest.TestCase):

    def test_MCMC_sampling(self):
        L=4
        
        weights=jnp.array(
                [ 0.23898957,  0.12614753,  0.19479055,  0.17325271,  0.14619853,  0.21392751,
                  0.19648707,  0.17103704, -0.15457255,  0.10954413,  0.13228065, -0.14935214,
                 -0.09963073,  0.17610707,  0.13386381, -0.14836467]
                )

        # Set up variational wave function
        rbm = nets.CpxRBM.partial(numHidden=2,bias=False)
        _, params = rbm.init_by_shape(random.PRNGKey(0),[(L,)])
        rbmModel = nn.Model(rbm,params)
        psi = NQS(rbmModel)
        psi.set_parameters(weights)

        # Set up exact sampler
        exactSampler=sampler.ExactSampler(L)
        
        # Set up MCMC sampler
        mcSampler=sampler.MCMCSampler(random.PRNGKey(0),jVMC.sampler.propose_spin_flip, (L,), numChains=777)

        # Compute exact probabilities
        _, _, pex = exactSampler.sample(psi)

        # Get samples from MCMC sampler
        numSamples=500000
        smc, _, _ = mcSampler.sample(psi, numSamples=numSamples)

        # Compute histogram of sampled configurations
        smcInt = jax.vmap(state_to_int)(smc)
        pmc,_=np.histogram(smcInt, bins=np.arange(0,17))

        # Compare histogram to exact probabilities
        self.assertTrue( jnp.max( jnp.abs( pmc/numSamples - pex ) ) < 1e-3 )
    

    def test_autoregressive_sampling(self):

        L=4

        # Set up variational wave function
        rnn = nets.RNN.partial( L=4, units=[5] )
        _, params = rnn.init_by_shape( random.PRNGKey(0), [(L,)] )
        rnnModel = nn.Model(rnn,params)
        rbm = nets.RBM.partial(numHidden=2,bias=False)
        _, params = rbm.init_by_shape(random.PRNGKey(0),[(L,)])
        rbmModel = nn.Model(rbm,params)
        
        psi = NQS(rnnModel, rbmModel)
        
        # Set up exact sampler
        exactSampler=sampler.ExactSampler(L)
        
        # Set up MCMC sampler
        mcSampler=sampler.MCMCSampler(random.PRNGKey(0),jVMC.sampler.propose_spin_flip, (L,), numChains=777)
        
        # Compute exact probabilities
        _, _, pex = exactSampler.sample(psi)

        numSamples=500000
        smc,p,_=mcSampler.sample(psi, numSamples=numSamples)
        
        self.assertTrue( jnp.max( jnp.abs( jnp.real(psi(smc))-p) ) < 1e-12 )
        
        # Compute histogram of sampled configurations
        smcInt = jax.vmap(state_to_int)(smc)
        pmc,_=np.histogram(smcInt, bins=np.arange(0,17))

        self.assertTrue( jnp.max( jnp.abs( pmc/numSamples-pex ) ) < 1e-3 )

if __name__ == "__main__":
    unittest.main()
