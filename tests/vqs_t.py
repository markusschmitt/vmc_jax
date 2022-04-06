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
from math import isclose

def get_shape(shape):
    return (global_defs.myDeviceCount,) + shape

class TestGradients(unittest.TestCase):
    
    def test_gradients_cpx(self):

        dlist=[jax.devices()[0], jax.devices()]

        for ds in dlist:

            global_defs.set_pmap_devices(ds)

            rbmModel = nets.CpxRBM(numHidden=2,bias=True)
            s=jnp.zeros(get_shape((4,3)),dtype=np.int32)
            s=s.at[...,0,1].set(1)
            s=s.at[...,2,2].set(1)

            psiC = NQS(rbmModel)
            psi0 = psiC(s)
            G = psiC.gradients(s)
            delta=1e-6
            params = psiC.get_parameters()
            for j in range(G.shape[-1]):
                u = jnp.zeros(G.shape[-1], dtype=global_defs.tReal).at[j].set(1)
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
            s=s.at[...,0,1].set(1)
            s=s.at[...,2,2].set(1)
            
            psi = NQS((rbmModel1,rbmModel2))
            psi0 = psi(s)
            G = psi.gradients(s)
            delta=1e-5
            params = psi.get_parameters()
            for j in range(G.shape[-1]):
                u = jnp.zeros(G.shape[-1], dtype=jVMC.global_defs.tReal).at[j].set(1)
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

            model = nets.RNN1DGeneral(L=3)

            s=jnp.zeros(get_shape((4,3)),dtype=np.int32)
            s=s.at[...,0,1].set(1)
            s=s.at[...,2,2].set(1)
            
            psi = NQS(model)
            psi0 = psi(s)
            G = psi.gradients(s)
            delta=1e-5
            params = psi.get_parameters()
            for j in range(G.shape[-1]):
                u = jnp.zeros(G.shape[-1], dtype=jVMC.global_defs.tReal).at[j].set(1)
                psi.update_parameters(delta * u)
                psi1 = psi(s)
                psi.set_parameters(params)

                # Finite difference gradients
                Gfd = (psi1-psi0) / delta

                with self.subTest(i=j):
                    self.assertTrue( jnp.max( jnp.abs( Gfd - G[...,j] ) ) < 1e-2 )
    
    
    def test_gradient_dict(self):

        net = jVMC.nets.CpxRBM(numHidden=8, bias=False)
        psi = jVMC.vqs.NQS(net, seed=1234)  # Variational wave function

        s = jnp.zeros((1,3,4))
        psi(s)

        g1 = psi.gradients(s)
        g2 = psi.gradients_dict(s)["Dense_0"]["kernel"]

        self.assertTrue(isclose(jnp.linalg.norm(g1-g2),0.0))

class TestEvaluation(unittest.TestCase):

    def test_evaluation_cpx(self):

        dlist=[jax.devices()[0], jax.devices()]

        for ds in dlist:

            global_defs.set_pmap_devices(ds)

            rbmModel = nets.CpxRBM(numHidden=2,bias=True)
            s=jnp.zeros(get_shape((4,3)),dtype=np.int32)
            s=s.at[...,0,1].set(1)
            s=s.at[...,2,2].set(1)
            
            psiC = NQS(rbmModel)
            cpxCoeffs = psiC(s)
            f,p = psiC.get_sampler_net()
            realCoeffs = global_defs.pmap_for_my_devices(lambda x: jax.vmap(lambda y: f(p,y))(x))(s)
            
            self.assertTrue( jnp.linalg.norm(jnp.real(cpxCoeffs) - realCoeffs) < 1e-6 )

if __name__ == "__main__":
    unittest.main()
