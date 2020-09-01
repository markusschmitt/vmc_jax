import sys
# Find jVMC package
sys.path.append(sys.path[0]+"/..")

import unittest

import jax
from jax.config import config
config.update("jax_enable_x64", True)

import jax.numpy as jnp

import numpy as np

import jVMC
import jVMC.mpi_wrapper as mpi


class TestMPI(unittest.TestCase):

    def test_mean(self):
        
        data=jnp.array(np.arange(720*4*jax.device_count()).reshape((jax.device_count()*720,4)))
        myNumSamples = mpi.distribute_sampling(jax.device_count()*720)

        myData=data[mpi.rank*myNumSamples:(mpi.rank+1)*myNumSamples].reshape((jax.device_count(),-1,4))

        self.assertTrue( jnp.sum(mpi.global_mean(myData)-jnp.mean(data,axis=0)) < 1e-10 )
    

    def test_var(self):
        
        data=jnp.array(np.arange(720*4*jax.device_count()).reshape((jax.device_count()*720,4)))
        myNumSamples = mpi.distribute_sampling(jax.device_count()*720)

        myData=data[mpi.rank*myNumSamples:(mpi.rank+1)*myNumSamples].reshape((jax.device_count(),-1,4))

        self.assertTrue( jnp.sum(mpi.global_variance(myData)-jnp.var(data,axis=0)) < 1e-10 )

if __name__ == "__main__":
    unittest.main()
