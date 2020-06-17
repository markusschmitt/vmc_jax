import jax
import jax.numpy as jnp

class EigenSolver:

    def __init__(self, snrTol=2):
        self.snrTol=snrTol

    def __call__(self, A, b):

        ev, V = jnp.linalg.eigh(A)

        self.Vtb = jnp.dot(jnp.transpose(V),b)
        self.invEv = 1. / ev

        return V.dot(self.invEv * self.Vtb)
