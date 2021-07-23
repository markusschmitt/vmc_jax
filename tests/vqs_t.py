import sys
# Find jVMC package
sys.path.append(sys.path[0]+"/..")

import unittest

import jVMC
import jVMC.nets as nets
from jVMC.vqs import NQS

import jVMC.global_defs as global_defs

import jax
from jax.config import config
config.update("jax_enable_x64", True)
import jax.random as random
import jax.numpy as jnp
import numpy as np
import flax.nn as nn

def get_shape(shape):
    return (global_defs.myDeviceCount,) + shape

class TestGradients(unittest.TestCase):
    
    def test_gradients_cpx(self):

        dlist=[jax.devices()[0], jax.devices()]

        for ds in dlist:

            global_defs.set_pmap_devices(ds)

            rbmModel = nets.CpxRBM(numHidden=2,bias=True)
            s=jnp.zeros(get_shape((4,3)),dtype=np.int32)
            s=jax.ops.index_update(s,jax.ops.index[...,0,1],1)
            s=jax.ops.index_update(s,jax.ops.index[...,2,2],1)
            
            psiC = NQS(rbmModel)
            psi0 = psiC(s)
            G = psiC.gradients(s)
            delta=1e-5
            params = psiC.get_parameters()
            for j in range(G.shape[-1]):
                u = jax.ops.index_update(jnp.zeros(G.shape[-1], dtype=global_defs.tReal), jax.ops.index[j], 1)
                psiC.update_parameters(delta * u)
                psi1 = psiC(s)
                psiC.set_parameters(params)

                # Finite difference gradients
                Gfd = (psi1-psi0) / delta

                with self.subTest(i=j):
                    self.assertTrue( jnp.max( jnp.abs( Gfd - G[...,j] ) ) < 1e-2 )
    
    def test_gradients_real(self):
        
        dlist=[jax.devices()[0], jax.devices()]

        for ds in dlist:

            global_defs.set_pmap_devices(ds)

            rbmModel1 = nets.RBM(numHidden=2,bias=True)
            rbmModel2 = nets.RBM(numHidden=3,bias=True)

            s=jnp.zeros(get_shape((4,3)),dtype=np.int32)
            s=jax.ops.index_update(s,jax.ops.index[...,0,1],1)
            s=jax.ops.index_update(s,jax.ops.index[...,2,2],1)
            
            psi = NQS((rbmModel1,rbmModel2))
            psi0 = psi(s)
            G = psi.gradients(s)
            delta=1e-5
            params = psi.get_parameters()
            for j in range(G.shape[-1]):
                u = jax.ops.index_update(jnp.zeros(G.shape[-1], dtype=jVMC.global_defs.tReal), jax.ops.index[j], 1)
                psi.update_parameters(delta * u)
                psi1 = psi(s)
                psi.set_parameters(params)

                # Finite difference gradients
                Gfd = (psi1-psi0) / delta

                with self.subTest(i=j):
                    self.assertTrue( jnp.max( jnp.abs( Gfd - G[...,j] ) ) < 1e-2 )
    
    def test_gradients_nonhermitian(self):
        
        dlist=[jax.devices()[0], jax.devices()]

        for ds in dlist:

            global_defs.set_pmap_devices(ds)

            model = nets.CpxRNN(L=3)

            s=jnp.zeros(get_shape((4,3)),dtype=np.int32)
            s=jax.ops.index_update(s,jax.ops.index[...,0,1],1)
            s=jax.ops.index_update(s,jax.ops.index[...,2,2],1)
            
            psi = NQS(model)
            psi0 = psi(s)
            G = psi.gradients(s)
            delta=1e-5
            params = psi.get_parameters()
            for j in range(G.shape[-1]):
                u = jax.ops.index_update(jnp.zeros(G.shape[-1], dtype=jVMC.global_defs.tReal), jax.ops.index[j], 1)
                psi.update_parameters(delta * u)
                psi1 = psi(s)
                psi.set_parameters(params)

                # Finite difference gradients
                Gfd = (psi1-psi0) / delta

                with self.subTest(i=j):
                    self.assertTrue( jnp.max( jnp.abs( Gfd - G[...,j] ) ) < 1e-2 )

class TestEvaluation(unittest.TestCase):

    def test_evaluation_cpx(self):

        dlist=[jax.devices()[0], jax.devices()]

        for ds in dlist:

            global_defs.set_pmap_devices(ds)

            rbmModel = nets.CpxRBM(numHidden=2,bias=True)
            s=jnp.zeros(get_shape((4,3)),dtype=np.int32)
            s=jax.ops.index_update(s,jax.ops.index[...,0,1],1)
            s=jax.ops.index_update(s,jax.ops.index[...,2,2],1)
            
            psiC = NQS(rbmModel)
            cpxCoeffs = psiC(s)
            realCoeffs = psiC.real_coefficients(s)
            
            self.assertTrue( jnp.linalg.norm(jnp.real(cpxCoeffs) - realCoeffs) < 1e-6 )

if __name__ == "__main__":
    unittest.main()
