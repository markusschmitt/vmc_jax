import sys
# Find jVMC package
sys.path.append(sys.path[0]+"/../..")

import jVMC
import jVMC.operator as op
import numpy as np
import jax
import jax.numpy as jnp
import time
        

L=128
J=-1.0
hx=-0.3

numStates=1000

timingReps = 10

hamiltonian = op.Operator()
for l in range(L):
    hamiltonian.add( op.scal_opstr( J, ( op.Sz(l), op.Sz((l+1)%L) ) ) )
    hamiltonian.add( op.scal_opstr( hx, ( op.Sx(l), ) ) )


states = jnp.array(jax.random.bernoulli(jax.random.PRNGKey(0), shape=(numStates,L)), dtype=np.int32)

print("* Compute off-diagonal configurations")

t0 = time.perf_counter()
sp, me = hamiltonian.get_s_primes(states)
sp.block_until_ready()
t1 = time.perf_counter()
print("  Time elapsed (incl. jit): %f seconds" % (t1-t0))

t=0
for i in range(timingReps):
    t0 = time.perf_counter()
    sp, me = hamiltonian.get_s_primes(states)
    sp.block_until_ready()
    t1 = time.perf_counter()
    t += t1-t0
print("  Avg. time elapsed (jit'd, %d repetitions): %f seconds" % (timingReps, t/timingReps))

logPsi = jax.random.normal(jax.random.PRNGKey(0), shape=(len(states),)) + 1.j * jax.random.normal(jax.random.PRNGKey(1), shape=(len(states),))
logPsip = jax.random.normal(jax.random.PRNGKey(2), shape=(len(sp),)) + 1.j * jax.random.normal(jax.random.PRNGKey(3), shape=(len(sp),))


print("* Compute local energy")

t0 = time.perf_counter()
Eloc = hamiltonian.get_O_loc(logPsi, logPsip).block_until_ready()
t1 = time.perf_counter()
print("  Time elapsed (incl. jit): %f seconds" % (t1-t0))

t=0
for i in range(timingReps):
    t0 = time.perf_counter()
    Eloc = hamiltonian.get_O_loc(logPsi, logPsip).block_until_ready()
    t1 = time.perf_counter()
    t += t1-t0
print("  Avg. time elapsed (jit'd, %d repetitions): %f seconds" % (timingReps, t/timingReps))
