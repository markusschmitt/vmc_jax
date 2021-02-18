import jax
from jax import jit, vmap, grad, partial
import jax.numpy as jnp
import numpy as np

import sys
# Find jVMC package
sys.path.append(sys.path[0]+"/../..")

import jVMC.global_defs as global_defs
from . import Operator

import functools

opDtype = global_defs.tCpx

class POVMOperator(Operator):
    """This class provides functionality to compute operator matrix elements

    Initializer arguments:
        
        * ``lDim``: Dimension of local Hilbert space.
    """

    def __init__(self,):
        """Initialize ``Operator``.
        """

        super().__init__()


    def compile(self):
        """Compiles a operator mapping function
        """
        return None

