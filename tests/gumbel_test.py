import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
## only one one gpu if more than one gpu is visible
##         if mpi.rank == 0:

import unittest

import jax
jax.config.update("jax_enable_x64", True)
import jax.random as random
import jax.numpy as jnp

import numpy as np

import sys
sys.path.append(sys.path[0] + '/../')

import jVMC
import jVMC.nets as nets
from jVMC.vqs import NQS
import jVMC.sampler as sampler
import jVMC.mpi_wrapper as mpi
import functools

def state_to_int(s,lDim=2):

    def for_fun(i, xs):
        return (xs[0] + xs[1][i] * (lDim**(xs[1].shape[0]-1 -i)), xs[1])
    def for_fun_2(i, xs):
        return (xs[0] + xs[1][i] * (lDim**(i)), xs[1])
    
    x, _ = jax.lax.fori_loop(0, s.shape[-1], for_fun if lDim > 2 else for_fun_2, (0, s))

    return x


class TestMC(unittest.TestCase):
    def test_gumbel_wrapper(self):
        L = 4
        lDim = 3
        # Set up variational wave function
        rwkv = nets.CpxRWKV(L=4, LocalHilDim=lDim, hidden_size=4,num_heads = 3, num_layers = 4,embedding_size = 4)
        net = jVMC.util.gumbel_wrapper(rwkv)
        psi = NQS(net)

        self.assertTrue(psi.is_gumbel)
        self.assertTrue(psi.net.is_gumbel)
    def test_gumbel_sampling(self):

        L = 4
        for lDim in [2,3,4]:
            print(f"Testing lDim: {lDim}")
            # Set up variational wave function
            rwkv = nets.CpxRWKV(L=4, LocalHilDim=lDim, hidden_size=4,num_heads = 3, num_layers = 4,embedding_size = 4)
            net = jVMC.util.gumbel_wrapper(rwkv)
            psi = NQS(net)

            # Set up exact sampler
            exactSampler = sampler.ExactSampler(psi, L,lDim=lDim)

            # Set up MCMC sampler
            mcSampler = sampler.MCSampler(psi, (L,), random.PRNGKey(0))

            ps = psi.get_parameters()
            psi.update_parameters(ps)

            # Compute exact probabilities
            _, _, pex = exactSampler.sample()
            pex = mpi.gather(pex)

            numSamples = min([(lDim**L)-1, 1e6]) ## more samples than the Hilbert space leads to a error
            
            smc, logPsi, p = mcSampler.sample(numSamples=numSamples)

            self.assertTrue(jnp.max(jnp.abs(jnp.real(psi(smc) - logPsi))) < 1e-12)

            smc = smc.reshape((smc.shape[0] * smc.shape[1], -1))
            # check sample size
            self.assertTrue(mpi.global_sum(jnp.array([smc.shape[0],])[None,None,...]) >= numSamples)

            # Compute histogram of sampled configurations
            
            smcInt = jax.vmap(functools.partial(state_to_int,lDim=lDim))(smc)
            ## check that the samples are unique: each index only appears once!
            self.assertTrue(mpi.global_sum(jnp.array([jnp.unique(smcInt).shape[0],])[None,None,...]) >= numSamples)

            pmc, _ = np.histogram(smcInt, bins=np.arange(0, lDim**L+1), weights=p[0])
            pmc = mpi.global_sum(jnp.array(pmc)[None,None,...])
            pmc = pmc / jnp.sum(pmc)
            # check relative sampling probabilities to agree with wave function probabilities
            self.assertTrue(jnp.max(jnp.abs(pmc - pex.reshape((-1,))[:lDim**L])) < 1.1e-3)
            
            
            psi1 = NQS(net, seed=98475)
            # Set up another MCMC sampler
            mcSampler = sampler.MCSampler(psi1, (L,), random.PRNGKey(0), updateProposer=jVMC.sampler.propose_spin_flip, numChains=777)
            s, psi_s, _ = mcSampler.sample(parameters=psi.get_parameters(), numSamples=numSamples)
            psi_s1 = psi(s)
            
            self.assertTrue(jnp.max(jnp.abs((psi_s - psi_s1) / psi_s)) < 1e-14)
        
        
if __name__ == "__main__":
    unittest.main()
