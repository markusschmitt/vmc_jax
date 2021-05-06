import os
import sys
# Find jVMC package
sys.path.append(sys.path[0]+"/..")

import jax
from jax.config import config
config.update("jax_enable_x64", True)

import jax.random as random
import jax.numpy as jnp
import numpy as np

import jVMC

L = 50
g = -0.7

# Initialize net
# net = jVMC.nets.CpxCNN(F=[15,], channels=[100], bias=False)
orbit = jnp.array([jnp.roll(jnp.identity(L,dtype=np.int32), l, axis=1) for l in range(L)])
net = jVMC.nets.RNNsym(hiddenSize=15, L=L, depth=5, orbit=orbit)
params = net.init(jax.random.PRNGKey(1234),jnp.zeros((L,), dtype=np.int32))

psi = jVMC.vqs.NQS(net,params, batchSize=500) # Variational wave function
print(f"The variational ansatz has {psi.numParameters} parameters.")

# Set up hamiltonian
hamiltonian = jVMC.operator.BranchFreeOperator()
for l in range(L):
    hamiltonian.add( jVMC.operator.scal_opstr( -1., ( jVMC.operator.Sz(l), jVMC.operator.Sz((l+1)%L) ) ) )
    hamiltonian.add( jVMC.operator.scal_opstr( g, ( jVMC.operator.Sx(l), ) ) )

# Set up sampler
sampler = jVMC.sampler.MCMCSampler( random.PRNGKey(4321), psi, jVMC.sampler.propose_spin_flip_Z2, (L,),
                                    numChains=50, sweepSteps=L,
                                    numSamples=300000, thermalizationSweeps=0)

# Set up TDVP
tdvpEquation = jVMC.tdvp.TDVP(sampler, rhsPrefactor=1.,
                              svdTol=1e-8, diagonalShift=10, makeReal='real')

stepper = jVMC.stepper.Euler(timeStep=1e-2) # ODE integrator

# Set up OutputManager
wdir = "./benchmarks/"
if jVMC.mpi_wrapper.rank == 0:
    try:
        os.makedirs(wdir)
    except OSError:
        print("Creation of the directory %s failed" % wdir)
    else:
        print("Successfully created the directory %s " % wdir)
outp = jVMC.util.OutputManager("./benchmarks/data.hdf5", append=False)

res = []
for n in range(3):

    dp, _ = stepper.step(0, tdvpEquation, psi.get_parameters(), hamiltonian=hamiltonian, psi=psi, numSamples=None, outp=outp)
    psi.set_parameters(dp)

    print("Benchmarking data")
    total = 0
    for key, value in outp.timings.items():
        print("\taverage and latest timings of ", key)
        print("\t", value["total"] / value["count"])
        print("\t", value["newest"])
        total += value["newest"]
    print("\t=== Total: ", total)
