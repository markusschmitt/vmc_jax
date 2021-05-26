import sys
# Find jVMC package
sys.path.append(sys.path[0] + "/..")

import jax
from jax.config import config
config.update("jax_enable_x64", True)

import jax.random as random
import jax.numpy as jnp
import numpy as np

import jVMC

L = 10
g = -0.7

# Initialize net
net = jVMC.nets.CpxCNN(F=[10, ], channels=[6], bias=False)
params = net.init(jax.random.PRNGKey(1234), jnp.zeros((L,), dtype=np.int32))

psi = jVMC.vqs.NQS(net, params)  # Variational wave function

# Set up hamiltonian
hamiltonian = jVMC.operator.BranchFreeOperator()
for l in range(L):
    hamiltonian.add(jVMC.operator.scal_opstr(-1., (jVMC.operator.Sz(l), jVMC.operator.Sz((l + 1) % L))))
    hamiltonian.add(jVMC.operator.scal_opstr(g, (jVMC.operator.Sx(l), )))

# Set up sampler
sampler = jVMC.sampler.MCSampler(psi, (L,), random.PRNGKey(4321), updateProposer=jVMC.sampler.propose_spin_flip_Z2,
                                   numChains=100, sweepSteps=L,
                                   numSamples=1000, thermalizationSweeps=25)

# Set up TDVP
tdvpEquation = jVMC.tdvp.TDVP(sampler, rhsPrefactor=1.,
                              svdTol=1e-8, diagonalShift=10, makeReal='real')

stepper = jVMC.stepper.Euler(timeStep=1e-2)  # ODE integrator

res = []
for n in range(200):

    dp, _ = stepper.step(0, tdvpEquation, psi.get_parameters(), hamiltonian=hamiltonian, psi=psi, numSamples=None)
    psi.set_parameters(dp)

    print(n, jax.numpy.real(tdvpEquation.ElocMean0) / L, tdvpEquation.ElocVar0 / L**2)

    res.append([n, jax.numpy.real(tdvpEquation.ElocMean0) / L, tdvpEquation.ElocVar0 / L**2])

import numpy as np
res = np.array(res)
import matplotlib.pyplot as plt
plt.plot(res[:, 0], res[:, 1] + 1.1272225, '-')
plt.legend()
plt.xlabel('iteration')
plt.ylabel(r'$(E-E_0)/L$')
plt.tight_layout()
plt.savefig('gs_search.pdf')
