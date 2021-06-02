import sys
# Find jVMC package
sys.path.append(sys.path[0] + "/..")

import os

import jax
from jax.config import config
config.update("jax_enable_x64", True)

#print(" rank %d"%(rank), jax.devices())

import jax.random as random
import flax
import flax.nn as nn
import jax.numpy as jnp

import numpy as np

import time
import json

import jVMC
import jVMC.operator as op
from jVMC.util import measure, ground_state_search, OutputManager, init_net
import jVMC.mpi_wrapper as mpi
import jVMC.activation_functions as act_funs
import jVMC.global_defs as global_defs

from functools import partial


class ZPolShortTime(nn.Module):

    def apply(self, s, J, hx, hz, DeltaT):

        N = s.shape[0]
        nUp = jnp.sum(s)
        nDown = N - nUp

        ZZ = jnp.sum((2 * s - 1) * (2 * jnp.roll(s, 1) - 1))

        dummyParam = self.param("dummy", (1,), jax.random.normal)

        return 1.j * 0.5 * DeltaT * (J * (N - ZZ) + hz * (N - (nUp - nDown))) + nUp * jnp.log(jnp.cos(hx * DeltaT)) + nDown * jnp.log(-1.j * jnp.sin(hx * DeltaT))

# ** end class ZPolShortTime


def first_step(params, psi, hx, hz, dt, sampler, L):

    net = ZPolShortTime.partial(J=-1.0, hx=hx, hz=hz, DeltaT=dt)
    _, params = net.init_by_shape(random.PRNGKey(123), [(L,)])
    psiTarget = jVMC.vqs.NQS(nn.Model(net, params), batchSize=1000)

    # Get sample
    sampleConfigs, sampleLogPsi, p = sampler.sample(psiTarget)

    for k in range(1500):
        obs1 = jax.pmap(lambda x, y: jnp.exp(x - y))(psi(sampleConfigs), sampleLogPsi)
        gradDenom = mpi.global_mean(obs1, p)
        gradNum = mpi.global_mean(jax.pmap(lambda a, b: a * b[:, None])(psi.gradients(sampleConfigs), obs1), p)
        grad = -2. * jnp.real(gradNum / gradDenom)

        psi.update_parameters(-0.05 * grad)

        if k % 10 == 0:
            if mpi.rank == 0:
                print(gradDenom)
            if gradDenom > 0.96:
                break


inp = None
if len(sys.argv) > 1:
    # if an input file is given
    with open(sys.argv[1], 'r') as f:
        inp = json.load(f)
else:

    if mpi.rank == 0:
        print("Error: No input file given.")
        exit()

wdir = inp["general"]["working_directory"]
if mpi.rank == 0:
    try:
        os.makedirs(wdir)
    except OSError:
        print("Creation of the directory %s failed" % wdir)
    else:
        print("Successfully created the directory %s " % wdir)

global_defs.set_pmap_devices(jax.devices()[mpi.rank % jax.device_count()])
print(" -> Rank %d working with device %s" % (mpi.rank, global_defs.devices()), flush=True)

L = inp["system"]["L"]

# Initialize output manager
outp = OutputManager(wdir + inp["general"]["data_output"], append=inp["general"]["append_data"])

# Set up variational wave function
psi = init_net(inp["network"], [(L,)])

outp.print("** Network properties")
outp.print("    Number of parameters: %d" % (len(psi.get_parameters())))

# Set up hamiltonian for ground state search
hamiltonianGS = op.Operator()
hz0 = 0.0
if "hz0" in inp["system"].keys():
    hz0 = inp["system"]["hz0"]
for l in range(L):
    hamiltonianGS.add(op.scal_opstr(inp["system"]["J0"], (op.Sz(l), op.Sz((l + 1) % L))))
    hamiltonianGS.add(op.scal_opstr(inp["system"]["hx0"], (op.Sx(l), )))
    if np.abs(hz0) > 1e-10:
        hamiltonianGS.add(op.scal_opstr(hz0, (op.Sz(l), )))

# Set up hamiltonian
hamiltonian = op.Operator()
lbda = 0.0
if "lambda" in inp["system"].keys():
    lbda = inp["system"]["lambda"]
hz = 0.0
if "hz" in inp["system"].keys():
    hz = inp["system"]["hz"]
for l in range(L):
    hamiltonian.add(op.scal_opstr(inp["system"]["J"], (op.Sz(l), op.Sz((l + 1) % L))))
    hamiltonian.add(op.scal_opstr(inp["system"]["hx"], (op.Sx(l), )))
    if np.abs(lbda) > 1e-10:
        hamiltonian.add(op.scal_opstr(lbda * inp["system"]["J"], (op.Sz(l), op.Sz((l + 2) % L))))
    if np.abs(hz) > 1e-10:
        hamiltonian.add(op.scal_opstr(inp["system"]["hz"], (op.Sz(l),)))

# Set up observables
observables = {
    "energy": hamiltonianGS,
    "X": op.Operator(),
    "Z": op.Operator(),
    "ZZ": [op.Operator() for d in range((L + 1) // 2)]
}
for l in range(L):
    observables["X"].add(op.scal_opstr(1. / L, (op.Sx(l), )))
    observables["Z"].add(op.scal_opstr(1. / L, (op.Sz(l), )))
    for d in range((L + 1) // 2):
        observables["ZZ"][d].add(op.scal_opstr(1. / L, (op.Sz(l), op.Sz((l + d + 1) % L))))

sampler = None
if inp["sampler"]["type"] == "MC":
    # Set up MCMC sampler
    sampler = jVMC.sampler.MCSampler(random.PRNGKey(inp["sampler"]["seed"]), jVMC.sampler.propose_spin_flip_Z2, (L,),
                                     numChains=inp["sampler"]["numChains"],
                                     numSamples=inp["sampler"]["numSamples"],
                                     thermalizationSweeps=inp["sampler"]["num_thermalization_sweeps"],
                                     sweepSteps=4 * L)
else:
    # Set up exact sampler
    sampler = jVMC.sampler.ExactSampler(L)

tdvpEquation = jVMC.util.tdvp.TDVP(sampler, snrTol=inp["time_evol"]["snr_tolerance"],
                                   svdTol=inp["time_evol"]["svd_tolerance"],
                                   rhsPrefactor=1.,
                                   diagonalShift=inp["gs_search"]["init_regularizer"], makeReal='real')

t = inp["time_evol"]["t_init"]

fromCheckpoint = False
if t < 0:
    outp.set_group("time_evolution")

    t, weights = outp.get_network_checkpoint(t)

    psi.set_parameters(weights)

    fromCheckpoint = True

else:
    # Perform ground state search to get initial state
    outp.print("** Ground state search")
    outp.set_group("ground_state_search")

    if "numSamplesGS" in inp["sampler"]:
        sampler.set_number_of_samples(inp["sampler"]["numSamplesGS"])
    ground_state_search(psi, hamiltonianGS, tdvpEquation, sampler,
                        numSteps=inp["gs_search"]["num_steps"], varianceTol=inp["gs_search"]["convergence_variance"] * L**2,
                        stepSize=1e-2, observables=observables, outp=outp)

    sampler.set_number_of_samples(inp["sampler"]["numSamples"])

# Time evolution
outp.print("** Time evolution")
outp.set_group("time_evolution")

reim = 'imag'
if "tdvp_make_real" in inp["time_evol"]:
    reim = inp["time_evol"]["tdvp_make_real"]

observables["energy"] = hamiltonian
tdvpEquation = jVMC.tdvp.TDVP(sampler, snrTol=inp["time_evol"]["snr_tolerance"],
                              svdTol=inp["time_evol"]["svd_tolerance"],
                              rhsPrefactor=1.j, diagonalShift=0., makeReal=reim)


def norm_fun(v, df=lambda x: x):
    return jnp.real(jnp.conj(jnp.transpose(v)).dot(df(v)))


stepper = jVMC.util.stepper.AdaptiveHeun(timeStep=inp["time_evol"]["time_step"], tol=inp["time_evol"]["stepper_tolerance"])

tmax = inp["time_evol"]["t_final"]

if not fromCheckpoint:
    first_step(psi.get_parameters(), psi, inp["system"]["hx"], hz, 0.1, sampler, L)
    t += 0.1

if not fromCheckpoint:
    outp.start_timing("measure observables")
    obs = measure(observables, psi, sampler)
    outp.stop_timing("measure observables")

    outp.write_observables(t, **obs)

while t < tmax:
    tic = time.perf_counter()
    outp.print(">  t = %f\n" % (t))

    # TDVP step
    dp, dt = stepper.step(0, tdvpEquation, psi.get_parameters(), hamiltonian=hamiltonian, psi=psi, numSamples=inp["sampler"]["numSamples"], outp=outp, normFunction=partial(norm_fun, df=tdvpEquation.S_dot))
    psi.set_parameters(dp)
    t += dt
    outp.print("   Time step size: dt = %f" % (dt))
    tdvpErr, tdvpRes = tdvpEquation.get_residuals()
    outp.print("   Residuals: tdvp_err = %.2e, solver_res = %.2e" % (tdvpErr, tdvpRes))

    # Measure observables
    outp.start_timing("measure observables")
    obs = measure(observables, psi, sampler)
    outp.stop_timing("measure observables")

    # Write observables
    outp.write_observables(t, **obs)
    # Write metadata
    outp.write_metadata(t, tdvp_error=tdvpErr,
                        tdvp_residual=tdvpRes,
                        SNR=tdvpEquation.get_snr(),
                        spectrum=tdvpEquation.get_spectrum())
    # Write network parameters
    outp.write_network_checkpoint(t, psi.get_parameters())

    outp.print("    Energy = %f +/- %f" % (obs["energy"]["mean"], obs["energy"]["MC_error"]))

    outp.print_timings(indent="   ")

    toc = time.perf_counter()
    outp.print("   == Total time for this step: %fs\n" % (toc - tic))
