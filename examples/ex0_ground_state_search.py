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

DMRG_energies = {"10": -1.0465761512947138, "20":-1.0851894140492975, "100":-1.1160796689826018}

L = 100
g = -0.7

# Initialize net
net = jVMC.nets.CpxRBM(numHidden=8, bias=False)
params = net.init(jax.random.PRNGKey(1234), jnp.zeros((L,), dtype=np.int32))

psi = jVMC.vqs.NQS(net, params)  # Variational wave function

# Set up hamiltonian
hamiltonian = jVMC.operator.BranchFreeOperator()
for l in range(L-1):
    hamiltonian.add(jVMC.operator.scal_opstr(-1., (jVMC.operator.Sz(l), jVMC.operator.Sz((l + 1) % L))))
    hamiltonian.add(jVMC.operator.scal_opstr(g, (jVMC.operator.Sx(l), )))
hamiltonian.add(jVMC.operator.scal_opstr(g, (jVMC.operator.Sx(L-1), )))

# Set up sampler
sampler = jVMC.sampler.MCSampler(psi, (L,), random.PRNGKey(4321), updateProposer=jVMC.sampler.propose_spin_flip_Z2,
                                 numChains=100, sweepSteps=L,
                                 numSamples=5000, thermalizationSweeps=25)

# Set up TDVP
tdvpEquation = jVMC.util.tdvp.TDVP(sampler, rhsPrefactor=1.,
                                   svdTol=1e-8, diagonalShift=10, makeReal='real')

stepper = jVMC.util.stepper.Euler(timeStep=1e-2)  # ODE integrator

res = []
for n in range(300):

    dp, _ = stepper.step(0, tdvpEquation, psi.get_parameters(), hamiltonian=hamiltonian, psi=psi, numSamples=None)
    psi.set_parameters(dp)

    print(n, jax.numpy.real(tdvpEquation.ElocMean0) / L, tdvpEquation.ElocVar0 / L)

    res.append([n, jax.numpy.real(tdvpEquation.ElocMean0) / L, tdvpEquation.ElocVar0 / L])

import numpy as np
res = np.array(res)
import matplotlib.pyplot as plt

fig, ax = plt.subplots(2,1, sharex=True, figsize=[4.8,4.8])
if str(L) in DMRG_energies:
    ax[0].semilogy(res[:, 0], res[:, 1] - DMRG_energies[str(L)], '-', label=r"$L="+str(L)+"$")
    ax[0].set_ylabel(r'$(E-E_0)/L$')
else:
    ax[0].plot(res[:, 0], res[:, 1], '-')
    ax[0].set_ylabel(r'$E/L$')

ax[1].semilogy(res[:, 0], res[:, 2], '-')
ax[1].set_ylabel(r'Var$(E)/L$')
ax[0].legend()
plt.xlabel('iteration')
plt.tight_layout()
plt.savefig('gs_search.pdf')
