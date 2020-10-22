import sys
# Find jVMC package
sys.path.append(sys.path[0]+"/..")

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
from jVMC.util import measure, ground_state_search, OutputManager
import jVMC.mpi_wrapper as mpi
import jVMC.activation_functions as act_funs
import jVMC.global_defs as global_defs

from functools import partial

def init_net(descr, dims, seed=0):

    def get_activation_functions(actFuns):

        if type(actFuns) is list:
            return [act_funs.activationFunctions[fn] for fn in actFuns] 
        
        return act_funs.activationFunctions[actFuns]
   
 
    netTypesReal = {
        "RBM" : jVMC.nets.RBM,
        "FFN" : jVMC.nets.FFN,
        "CNN" : jVMC.nets.CNN,
        "LSTM" : jVMC.nets.LSTM,
        "LSTMsym" : jVMC.nets.LSTMsym,
        "PhaseRNN" : jVMC.nets.RNN,
        "PhaseRNNsym" : jVMC.nets.RNNsym,
        "RNN" : jVMC.nets.RNN,
        "RNNsym" : jVMC.nets.RNNsym
    }
    netTypesCpx = {
        "CpxRBM" : jVMC.nets.CpxRBM,
        "CpxCNN" : jVMC.nets.CpxCNN
    }

    def get_net(descr, dims, seed, netTypes=None):

        net = netTypes[descr["type"]].partial(**descr["parameters"])
        _, params = net.init_by_shape(random.PRNGKey(seed),dims)
        return nn.Model(net,params)

    get_real_net=partial(get_net, netTypes=netTypesReal)
    get_cpx_net=partial(get_net, netTypes=netTypesCpx)


    if "actFun" in descr["net1"]["parameters"]:

        descr["net1"]["parameters"]["actFun"] = get_activation_functions(descr["net1"]["parameters"]["actFun"])

    if descr["net1"]["type"][-3:] == "sym":
        # Generate orbit of 1D translations for RNNsym net
        L = descr["net1"]["parameters"]["L"]
        descr["net1"]["parameters"]["orbit"] = jnp.array([jnp.roll(jnp.identity(L,dtype=np.int32), l, axis=1) for l in range(L)])
    if descr["net2"]["type"][-3:] == "sym":
        # Generate orbit of 1D translations for RNNsym net
        L = descr["net2"]["parameters"]["L"]
        descr["net2"]["parameters"]["orbit"] = jnp.array([jnp.roll(jnp.identity(L,dtype=np.int32), l, axis=1) for l in range(L)])

    if not "net2" in descr:

        model = get_cpx_net(descr["net1"], dims, seed)
    
        return jVMC.vqs.NQS(model, batchSize=descr["gradient_batch_size"])

    else:

        if "actFun" in descr["net2"]["parameters"]:

            descr["net2"]["parameters"]["actFun"] = get_activation_functions(descr["net2"]["parameters"]["actFun"])
        
        model1 = get_real_net(descr["net1"], dims, seed)
        model2 = get_real_net(descr["net2"], dims, seed)

        return jVMC.vqs.NQS((model1, model2), batchSize=descr["gradient_batch_size"])


inp = None
if len(sys.argv) > 1:
    # if an input file is given
    with open(sys.argv[1],'r') as f:
        inp = json.load(f)
else:
    # otherwise, set up default input dict
    inp = {}
    inp["general"] = {
        "working_directory": "./data/devel/",
        "data_output" : "data.hdf5",
        "append_data" : False
    }
    inp["system"] = {
        "L" : 4,
        "J0" : -1.0,
        "hx0" : -2.5,
        "J" : -1.0,
        "hx" : -0.3
    }

    inp["sampler"] = {
        "type" : "exact",
        "numSamples" : 50000,
        "numChains" : 500,
        "num_thermalization_sweeps": 20,
        "seed": 1234
    }
    
    inp["gs_search"] = {
        "num_steps" : 10,
        "init_regularizer" : 5
    }
    
    inp["time_evol"] = {
        "t_init": 0.,
        "t_final": 1.,
        "time_step": 1e-3,
        "snr_tolerance": 5,
        "svd_tolerance": 1e-6,
        "stepper_tolerance": 1e-4
    }


wdir=inp["general"]["working_directory"]
if mpi.rank == 0:
    try:
        os.makedirs(wdir)
    except OSError:
        print ("Creation of the directory %s failed" % wdir)
    else:
        print ("Successfully created the directory %s " % wdir)

global_defs.set_pmap_devices(jax.devices()[mpi.rank % jax.device_count()])
print(" -> Rank %d working with device %s" % (mpi.rank, global_defs.devices()), flush=True)

L = inp["system"]["L"]

# Initialize output manager
outp = OutputManager(wdir+inp["general"]["data_output"], append=inp["general"]["append_data"])

# Set up variational wave function
psi = init_net(inp["network"], [(L,)])

outp.print("** Network properties")
outp.print("    Number of parameters: %d" % (len(psi.get_parameters())))

# Set up hamiltonian for ground state search
hamiltonianGS = op.Operator()
for l in range(L):
    hamiltonianGS.add( op.scal_opstr( inp["system"]["J0"], ( op.Sz(l), op.Sz((l+1)%L) ) ) )
    hamiltonianGS.add( op.scal_opstr( inp["system"]["hx0"], ( op.Sx(l), ) ) )

# Set up hamiltonian
hamiltonian = op.Operator()
lbda = 0.0
if "lambda" in inp["system"].keys():
    lbda = inp["system"]["lambda"]
for l in range(L):
    hamiltonian.add( op.scal_opstr( inp["system"]["J"], ( op.Sz(l), op.Sz((l+1)%L) ) ) )
    hamiltonian.add( op.scal_opstr( inp["system"]["hx"], ( op.Sx(l), ) ) )
    if np.abs(lbda) > 1e-10:
        hamiltonian.add( op.scal_opstr( lbda * inp["system"]["J"], ( op.Sz(l), op.Sz((l+2)%L) ) ) )

# Set up observables
observables = {
    "energy" : hamiltonianGS,
    "X" : op.Operator(),
    "Z" : op.Operator(),
    "ZZ" : [op.Operator() for d in range((L+1)//2)]
}
for l in range(L):
    observables["X"].add( op.scal_opstr( 1./L, ( op.Sx(l), ) ) )
    observables["Z"].add( op.scal_opstr( 1./L, ( op.Sz(l), ) ) )
    for d in range((L+1)//2):
        observables["ZZ"][d].add( op.scal_opstr( 1./L, ( op.Sz(l), op.Sz((l+d+1)%L ) ) ) )

sampler = None
if inp["sampler"]["type"] == "MC":
    # Set up MCMC sampler
    sampler = jVMC.sampler.MCMCSampler( random.PRNGKey(inp["sampler"]["seed"]), jVMC.sampler.propose_spin_flip_Z2, (L,),
                                        numChains=inp["sampler"]["numChains"],
                                        numSamples=inp["sampler"]["numSamples"],
                                        thermalizationSweeps=inp["sampler"]["num_thermalization_sweeps"],
                                        sweepSteps=4*L )
else:
    # Set up exact sampler
    sampler=jVMC.sampler.ExactSampler(L)

tdvpEquation = jVMC.tdvp.TDVP(sampler, snrTol=inp["time_evol"]["snr_tolerance"], 
                                       svdTol=inp["time_evol"]["svd_tolerance"],
                                       rhsPrefactor=1., 
                                       diagonalShift=inp["gs_search"]["init_regularizer"], makeReal='real')

t=inp["time_evol"]["t_init"]

fromCheckpoint = False
if t<0:
    outp.set_group("time_evolution")

    t, weights = outp.get_network_checkpoint(t)

    psi.set_parameters(weights)

    fromCheckpoint = True

else:
    # Perform ground state search to get initial state
    outp.print("** Ground state search")
    outp.set_group("ground_state_search")

    ground_state_search(psi, hamiltonianGS, tdvpEquation, sampler,
                        numSteps=inp["gs_search"]["num_steps"], varianceTol=inp["gs_search"]["convergence_variance"]*L**2,
                        stepSize=1e-2, observables=observables, outp=outp)

# Time evolution
outp.print("** Time evolution")
outp.set_group("time_evolution")

observables["energy"] = hamiltonian
tdvpEquation = jVMC.tdvp.TDVP(sampler, snrTol=inp["time_evol"]["snr_tolerance"], 
                                       svdTol=inp["time_evol"]["svd_tolerance"],
                                       rhsPrefactor=1.j, diagonalShift=0., makeReal='imag')

def norm_fun(v, df=lambda x:x):
    return jnp.real(jnp.conj(jnp.transpose(v)).dot(df(v)))

stepper = jVMC.stepper.AdaptiveHeun(timeStep=inp["time_evol"]["time_step"], tol=inp["time_evol"]["stepper_tolerance"])

tmax=inp["time_evol"]["t_final"]

if not fromCheckpoint:
    outp.start_timing("measure observables")
    obs = measure(observables, psi, sampler)
    outp.stop_timing("measure observables")

    outp.write_observables(t, **obs)

while t<tmax:
    tic = time.perf_counter()
    outp.print( ">  t = %f\n" % (t) )

    # TDVP step
    dp, dt = stepper.step(0, tdvpEquation, psi.get_parameters(), hamiltonian=hamiltonian, psi=psi, numSamples=inp["sampler"]["numSamples"], outp=outp, normFunction=partial(norm_fun, df=tdvpEquation.S_dot))
    psi.set_parameters(dp)
    t += dt
    outp.print( "   Time step size: dt = %f" % (dt) )
    tdvpErr, tdvpRes = tdvpEquation.get_residuals()
    outp.print( "   Residuals: tdvp_err = %.2e, solver_res = %.2e" % (tdvpErr, tdvpRes) )

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
    outp.print( "   == Total time for this step: %fs\n" % (toc-tic) )
