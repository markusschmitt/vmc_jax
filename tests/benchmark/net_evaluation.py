import sys
# Find jVMC package
sys.path.append(sys.path[0]+"/../..")

import jVMC
import jVMC.nets as nets
import numpy as np
import jax
import jax.numpy as jnp
import flax
import flax.nn as nn

import time

numStates=1000

timingReps = 10


L=128
states = jnp.array(jax.random.bernoulli(jax.random.PRNGKey(0), shape=(numStates,L)), dtype=np.int32)

print("* CNN 1D")
print("  - evaluation")

cnn1D = nets.CNN.partial(F=(8,), channels=[6,5,4])
_,params = cnn1D.init_by_shape(jax.random.PRNGKey(0),[(1,L)])
cnn1DModel = nn.Model(cnn1D,params)

t0 = time.perf_counter()
cnn1DModel(states)
t1 = time.perf_counter()
print("  Time elapsed (incl. jit): %f seconds" % (t1-t0))

t=0
for i in range(timingReps):
    t0 = time.perf_counter()
    cnn1DModel(states).block_until_ready()
    t1 = time.perf_counter()
    t += t1-t0
print("  Avg. time elapsed (jit'd, %d repetitions): %f seconds" % (timingReps, t/timingReps))


L=10
states = jnp.array(jax.random.bernoulli(jax.random.PRNGKey(0), shape=(numStates,L,L)), dtype=np.int32)

print("* CNN 2D")
print("  - evaluation")

cnn2D = nets.CNN.partial(F=(6,6), channels=[6,5,4], strides=[1,1])
_,params = cnn2D.init_by_shape(jax.random.PRNGKey(0),[(1,L,L)])
cnn2DModel = nn.Model(cnn2D,params)

t0 = time.perf_counter()
cnn2DModel(states)
t1 = time.perf_counter()
print("  Time elapsed (incl. jit): %f seconds" % (t1-t0))

t=0
for i in range(timingReps):
    t0 = time.perf_counter()
    cnn2DModel(states).block_until_ready()
    t1 = time.perf_counter()
    t += t1-t0
print("  Avg. time elapsed (jit'd, %d repetitions): %f seconds" % (timingReps, t/timingReps))

