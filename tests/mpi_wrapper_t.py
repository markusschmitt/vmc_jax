import unittest

import jax
from jax.config import config
config.update("jax_enable_x64", True)

import jax.numpy as jnp

import numpy as np

import jVMC
import jVMC.mpi_wrapper as mpi

import jVMC.global_defs as global_defs

def get_shape(shape):
    return (global_defs.device_count(),) + shape

class TestMPI(unittest.TestCase):

    def test_mean(self):
        
        data=jnp.array(np.arange(720*4*global_defs.device_count()).reshape((global_defs.device_count()*720,4)))
        myNumSamples = mpi.distribute_sampling(global_defs.device_count()*720)

        myData=data[mpi.rank*myNumSamples:(mpi.rank+1)*myNumSamples].reshape(get_shape((-1,4)))

        self.assertTrue( jnp.sum(mpi.global_mean(myData)-jnp.mean(data,axis=0)) < 1e-10 )

    def test_var(self):
        
        data=jnp.array(np.arange(720*4*global_defs.device_count()).reshape((global_defs.device_count()*720,4)))
        myNumSamples = mpi.distribute_sampling(global_defs.device_count()*720)

        myData=data[mpi.rank*myNumSamples:(mpi.rank+1)*myNumSamples].reshape(get_shape((-1,4)))

        self.assertTrue( jnp.sum(mpi.global_variance(myData)-jnp.var(data,axis=0)) < 1e-10 )

    def test_bcast(self):
        data=np.zeros(10, dtype=np.int32)
        with self.assertRaises(TypeError) as context:
            mpi.bcast_unknown_size(data)

if __name__ == "__main__":
    unittest.main()
