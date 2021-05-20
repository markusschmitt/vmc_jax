import sys
# Find jVMC package
sys.path.append(sys.path[0]+"/..")

import unittest

import jVMC.util.stepper as st

import jax
import jax.random as random
import jax.numpy as jnp
import numpy as np


class TestIntegrationHeun(unittest.TestCase):

    # Test adaptive integrator with multi-dimensional linear ODE
    def test_integration(self):

        def f(y,t,**args):
            return args['mat'].dot(y)

        def norm(y):
            return jnp.real(jnp.conjugate(y).dot(y))

        N = 4

        np.random.seed(123)
        stepper = st.AdaptiveHeun()
        mat = jnp.array(np.random.rand(N,N) + 1.j * np.random.rand(N,N))

        y0 = jnp.array(np.random.rand(N))
        y = y0.copy()
        t=0
        diffs=[]
        for k in range(100):
            y, dt = stepper.step(t,f,y,normFunction=norm,mat=mat)
            t+=dt
            yExact = jax.scipy.linalg.expm(t * mat).dot(y0)
            diff = y - yExact
            diffs.append(norm(diff)/N)

        self.assertTrue( jnp.max(jnp.array(diffs)) < 1e-5 )


if __name__ == "__main__":
    unittest.main()
