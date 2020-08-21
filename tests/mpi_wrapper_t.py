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
        
        data=jnp.array(np.arange(720*4).reshape((720,4)))
        myNumSamples = mpi.distribute_sampling(720)

        myData=data[mpi.rank*myNumSamples:(mpi.rank+1)*myNumSamples]

        self.assertTrue( jnp.sum(mpi.global_mean(myData)-jnp.mean(data,axis=0)) < 1e-10 )

if __name__ == "__main__":
    unittest.main()
