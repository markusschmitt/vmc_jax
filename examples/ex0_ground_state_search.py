#!/usr/bin/env python
# coding: utf-8


import sys
# Find jVMC package
sys.path.append(sys.path[0] + "/..")

import jax
from jax.config import config
config.update("jax_enable_x64", True)

import jax.random as random
import jax.numpy as jnp
import flax.linen as nn

import numpy as np
import matplotlib.pyplot as plt

import jVMC

L = 10
g = -0.7

GPU_avail = True
# Initialize net
if GPU_avail:
    # reproduces results in Fig. 3 of the paper
    # estimated run_time in colab (GPU enabled): ~26 minutes
    def myActFun(x):
        return 1 + nn.elu(x)
    net = jVMC.nets.CNN(F=(L,), channels=(16,), strides=(1,), periodicBoundary=True, actFun=(myActFun,))
    n_steps = 1000
    n_Samples = 40000
else:
    # may be used to obtain results on Laptop CPUs
    # estimated run_time: ~100 seconds
    net = jVMC.nets.CpxRBM(numHidden=8, bias=False)
    n_steps = 300
    n_Samples = 5000


psi = jVMC.vqs.NQS(net, seed=1234)  # Variational wave function


def energy_single_p_mode(h_t, P):
    return np.sqrt(1 + h_t**2 - 2 * h_t * np.cos(P))


def ground_state_energy_per_site(h_t, N):
    Ps = 0.5 * np.arange(- (N - 1), N - 1 + 2, 2)
    Ps = Ps * 2 * np.pi / N
    energies_p_modes = np.array([energy_single_p_mode(h_t, P) for P in Ps])
    return - 1 / N * np.sum(energies_p_modes)


exact_energy = ground_state_energy_per_site(g, L)
print(exact_energy)

# Set up hamiltonian
hamiltonian = jVMC.operator.BranchFreeOperator()
for l in range(L):
    hamiltonian.add(jVMC.operator.scal_opstr(-1., (jVMC.operator.Sz(l), jVMC.operator.Sz((l + 1) % L))))
    hamiltonian.add(jVMC.operator.scal_opstr(g, (jVMC.operator.Sx(l), )))

# Set up sampler
sampler = jVMC.sampler.MCSampler(psi, (L,), random.PRNGKey(4321), updateProposer=jVMC.sampler.propose_spin_flip_Z2,
                                 numChains=100, sweepSteps=L,
                                 numSamples=n_Samples, thermalizationSweeps=25)

# Set up TDVP
tdvpEquation = jVMC.util.tdvp.TDVP(sampler, rhsPrefactor=1.,
                                   svdTol=1e-8, diagonalShift=10, makeReal='real')

stepper = jVMC.util.stepper.Euler(timeStep=1e-2)  # ODE integrator

res = []
for n in range(n_steps):

    dp, _ = stepper.step(0, tdvpEquation, psi.get_parameters(), hamiltonian=hamiltonian, psi=psi, numSamples=None)
    psi.set_parameters(dp)

    print(n, jax.numpy.real(tdvpEquation.ElocMean0) / L, tdvpEquation.ElocVar0 / L)

    res.append([n, jax.numpy.real(tdvpEquation.ElocMean0) / L, tdvpEquation.ElocVar0 / L])

res = np.array(res)

fig, ax = plt.subplots(2, 1, sharex=True, figsize=[4.8, 4.8])
ax[0].semilogy(res[:, 0], res[:, 1] - exact_energy, '-', label=r"$L=" + str(L) + "$")
ax[0].set_ylabel(r'$(E-E_0)/L$')

ax[1].semilogy(res[:, 0], res[:, 2], '-')
ax[1].set_ylabel(r'Var$(E)/L$')
ax[0].legend()
plt.xlabel('iteration')
plt.tight_layout()
plt.savefig('gs_search.pdf')
