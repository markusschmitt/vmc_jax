import jax
import jax.numpy as jnp
import numpy as np

import sys
# Find jVMC package
sys.path.append(sys.path[0]+"/..")
import jVMC.global_defs as global_defs

import jVMC.operator

import functools

class SxOperator(jVMC.operator.Operator):
    """Define a `\hat\sigma_l^x` operator."""

    def __init__(self, siteIdx):

        self.siteIdx = siteIdx

        super().__init__() # Constructor of base class Operator has to be called!


    def compile(self):

        def get_s_primes(s, idx):

            sp = s.copy()

            configShape = sp.shape
            sp = sp.ravel()

            # Define matrix element
            matEl = jnp.array([1.,], dtype=global_defs.tCpx)
            # Define mapping of Sx: 0->1, 1->0 
            sMap = jnp.array([1,0])
            # Perform mapping
            sp = jax.ops.index_update(sp, jax.ops.index[idx], sMap[s[idx]])

            return sp.reshape(configShape), matEl

        # Create a pure function that takes only a basis configuration as argument
        map_function = functools.partial(get_s_primes, idx=self.siteIdx)

        return map_function


mySx = SxOperator(siteIdx=1)


# Get operator mapping function
testFunction = mySx.compile()

# Simple test configuration s=|0000>
testConfig=jnp.array([0,0,0,0], dtype=np.int32)

# Get connected s' and matrix elements <s'|Sx|s>
sp, matEl = testFunction(testConfig)
