import sys
# Find jVMC package
sys.path.append(sys.path[0] + "/..")

import os

import jax
from jax.config import config
config.update("jax_enable_x64", True)

import jax.random as random
import flax
import jax.numpy as jnp

import numpy as np

import time

import jVMC
from jVMC.util import measure
import jVMC.operator as op

import matplotlib.pyplot as plt


L = 6
g = -0.7
h = 0.1

dt = 1e-3  # Initial time step
integratorTol = 1e-4  # Adaptive integrator tolerance
tmax = 2  # Final time

# Set up variational wave function
net = jVMC.nets.CpxRBM(numHidden=10, bias=True)

psi = jVMC.vqs.NQS(net, seed=1234)  # Variational wave function

# Set up hamiltonian
hamiltonian = jVMC.operator.BranchFreeOperator()
for l in range(L):
    hamiltonian.add(op.scal_opstr(-1., (op.Sz(l), op.Sz((l + 1) % L))))
    hamiltonian.add(op.scal_opstr(g, (op.Sx(l), )))
    hamiltonian.add(op.scal_opstr(h, (op.Sz(l),)))

# Set up observables
observables = {
    "energy": hamiltonian,
    "X": jVMC.operator.BranchFreeOperator(),
}
for l in range(L):
    observables["X"].add(op.scal_opstr(1. / L, (op.Sx(l), )))

sampler = None
# Set up exact sampler
sampler = jVMC.sampler.ExactSampler(psi, L)

# Set up TDVP
tdvpEquation = jVMC.util.tdvp.TDVP(sampler, svdTol=1e-8,
                                   rhsPrefactor=1.j,
                                   makeReal='imag')

t = 0.0  # Initial time

# Set up stepper
stepper = jVMC.util.stepper.AdaptiveHeun(timeStep=dt, tol=integratorTol)

# Measure initial observables
obs = measure(observables, psi, sampler)
data = []
data.append([t, obs["energy"]["mean"][0], obs["X"]["mean"][0]])

plt.ion()
plt.xlim(0, tmax)
plt.ylim(0, 1)
plt.legend()
plt.ylabel(r"Transverse magnetization $\langle X\rangle$")
plt.xlabel(r"Time $\langle Jt\rangle$")

while t < tmax:
    tic = time.perf_counter()
    print(">  t = %f\n" % (t))

    # TDVP step
    dp, dt = stepper.step(0, tdvpEquation, psi.get_parameters(), hamiltonian=hamiltonian, psi=psi)
    psi.set_parameters(dp)
    t += dt

    # Measure observables
    obs = measure(observables, psi, sampler)
    data.append([t, obs["energy"]["mean"][0], obs["X"]["mean"][0]])

    # Write some meta info to screen
    print("   Time step size: dt = %f" % (dt))
    tdvpErr, tdvpRes = tdvpEquation.get_residuals()
    print("   Residuals: tdvp_err = %.2e, solver_res = %.2e" % (tdvpErr, tdvpRes))
    print("    Energy = %f +/- %f" % (obs["energy"]["mean"], obs["energy"]["MC_error"]))
    toc = time.perf_counter()
    print("   == Total time for this step: %fs\n" % (toc - tic))

    # Plot data
    npdata = np.array(data)
    plt.plot(npdata[:, 0], npdata[:, 2], c="red")
    plt.pause(0.05)
