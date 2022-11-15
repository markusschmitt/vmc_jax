import jax
import jax.numpy as jnp
import flax.linen as nn
from jVMC.util.symmetries import LatticeSymmetry

class SymNet(nn.Module):
    """
    Wrapper module for symmetrization.
    This is a wrapper module for the incorporation of lattice symmetries. 
    The given plain ansatz :math:`\\psi_\\theta` is symmetrized as

        :math:`\\Psi_\\theta(s)=\\frac{1}{|\\mathcal S|}\\sum_{\\tau\\in\\mathcal S}\\psi_\\theta(\\tau(s))`

    where :math:`\\mathcal S` denotes the set of symmetry operations (``orbit`` in our nomenclature).

    Initialization arguments:
        * ``orbit``: orbits which define the symmetry operations (instance of ``util.symmetries.LatticeSymmetry``)
        * ``net``: Flax module defining the plain ansatz.

    """
    orbit: LatticeSymmetry
    net: callable

    @nn.compact
    def __call__(self, x):

        inShape = x.shape
        x = jax.vmap(lambda o, s: jnp.dot(o, s.ravel()).reshape(inShape), in_axes=(0, None))(self.orbit.orbit, x)

        def evaluate(x):
            return self.net(x)

        res = jnp.mean(jax.vmap(evaluate)(x), axis=0)

        return res

# ** end class SymNet