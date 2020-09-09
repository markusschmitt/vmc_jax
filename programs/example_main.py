import sys
# Find jVMC package
sys.path.append(sys.path[0]+"/..")

import jax
from jax.config import config
config.update("jax_enable_x64", True)

import jax.random as random
import flax
import flax.nn as nn
import jax.numpy as jnp

import numpy as np

import time
import os
import json

import jVMC
import jVMC.operator as op
from jVMC.util import measure, ground_state_search, OutputManager
import jVMC.mpi_wrapper as mpi
import jVMC.activation_functions as act_funs

from functools import partial


def init_net(descr, dims, seed=0):

    def get_activation_functions(actFuns):

        return [act_funs.activationFunctions[fn] for fn in actFuns] 
   
 
    netTypesReal = {
        "RBM" : (jVMC.nets.RBM, (1,)+dims),
        "FFN" : (jVMC.nets.FFN, (1,)+dims),
        "CNN" : (jVMC.nets.CNN, dims),
        "RNN" : (jVMC.nets.FFN, (1,)+dims),
    }
    netTypesCpx = {
        "CpxRBM" : (jVMC.nets.CpxRBM, (1,)+dims),
        "CpxCNN" : (jVMC.nets.CpxCNN, dims),
    }


    def get_net(descr, dims, seed, netTypes=None):

        net = netTypes[descr["type"]][0].partial(**descr["parameters"])
        _, params = net.init_by_shape(random.PRNGKey(seed),[netTypes[descr["type"]][1]])
        return nn.Model(net,params)

    get_real_net=partial(get_net, netTypes=netTypesReal)
    get_cpx_net=partial(get_net, netTypes=netTypesCpx)


    if "actFun" in descr["net1"]["parameters"]:

        descr["net1"]["parameters"]["actFun"] = get_activation_functions(descr["net1"]["parameters"]["actFun"])

    if not "net2" in descr:

        model = get_cpx_net(descr["net1"], dims, seed)
    
        return jVMC.vqs.NQS(model)

    else:

        if "actFun" in descr["net2"]["parameters"]:

            descr["net2"]["parameters"]["actFun"] = get_activation_functions(descr["net2"]["parameters"]["actFun"])
        
        model1 = get_real_net(descr["net1"], dims, seed)
        model2 = get_real_net(descr["net2"], dims, seed)
    
        return jVMC.vqs.NQS(model1, model2)


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
try:
    os.makedirs(wdir)
except OSError:
    print ("Creation of the directory %s failed" % wdir)
else:
    print ("Successfully created the directory %s " % wdir)


L = inp["system"]["L"]

# Initialize output manager
outp = OutputManager(wdir+inp["general"]["data_output"], append=inp["general"]["append_data"])

# Set up variational wave function
##rbm = jVMC.nets.CpxRBM.partial(numHidden=10,bias=False)
#rbm = jVMC.nets.CpxRBM.partial(**inp["network"]["parameters"])
#_, params = rbm.init_by_shape(random.PRNGKey(0),[(1,inp["system"]["L"])])
#rbmModel = nn.Model(rbm,params)
#
#rbm1 = jVMC.nets.RBM.partial(numHidden=6,bias=False)
#_, params1 = rbm1.init_by_shape(random.PRNGKey(123),[(1,inp["system"]["L"])])
#rbmModel1 = nn.Model(rbm1,params1)
##rbm2 = jVMC.nets.FFN.partial(layers=[5,5],bias=False)
#rbm2 = jVMC.nets.RBM.partial(numHidden=6,bias=False)
#_, params2 = rbm2.init_by_shape(random.PRNGKey(321),[(1,inp["system"]["L"])])
#rbmModel2 = nn.Model(rbm2,params2)
#
#psi = jVMC.vqs.NQS(rbmModel)
##psi = jVMC.vqs.NQS(rbmModel1, rbmModel2)

psi = init_net(inp["network"], (L,))

# Set up hamiltonian for ground state search
hamiltonianGS = op.Operator()
for l in range(L):
    hamiltonianGS.add( op.scal_opstr( inp["system"]["J0"], ( op.Sz(l), op.Sz((l+1)%L) ) ) )
    hamiltonianGS.add( op.scal_opstr( inp["system"]["hx0"], ( op.Sx(l), ) ) )

# Set up hamiltonian
hamiltonian = op.Operator()
for l in range(L):
    hamiltonian.add( op.scal_opstr( inp["system"]["J"], ( op.Sz(l), op.Sz((l+1)%L) ) ) )
    hamiltonian.add( op.scal_opstr( inp["system"]["hx"], ( op.Sx(l), ) ) )

# Set up observables
observables = [hamiltonianGS, op.Operator(), op.Operator(), op.Operator()]
for l in range(L):
    observables[1].add( ( op.Sx(l), ) )
    observables[2].add( ( op.Sz(l), op.Sz((l+1)%L) ) )
    observables[3].add( ( op.Sz(l), op.Sz((l+2)%L) ) )

sampler = None
if inp["sampler"]["type"] == "MC":
    # Set up MCMC sampler
    sampler = jVMC.sampler.MCMCSampler( random.PRNGKey(inp["sampler"]["seed"]), jVMC.sampler.propose_spin_flip, [L],
                                        numChains=inp["sampler"]["numChains"],
                                        numSamples=inp["sampler"]["numSamples"],
                                        thermalizationSweeps=inp["sampler"]["num_thermalization_sweeps"],
                                        sweepSteps=L )
else:
    # Set up exact sampler
    sampler=jVMC.sampler.ExactSampler(L)

tdvpEquation = jVMC.tdvp.TDVP(sampler, snrTol=inp["time_evol"]["snr_tolerance"], 
                                       svdTol=inp["time_evol"]["svd_tolerance"],
                                       rhsPrefactor=1., 
                                       diagonalShift=inp["gs_search"]["init_regularizer"], makeReal='real')

# Perform ground state search to get initial state
outp.print("** Ground state search")
outp.set_group("ground_state_search")

ground_state_search(psi, hamiltonianGS, tdvpEquation, sampler, numSteps=inp["gs_search"]["num_steps"], stepSize=1e-2, observables=observables, outp=outp)

# Time evolution
outp.print("** Time evolution")
outp.set_group("time_evolution")

observables[0] = hamiltonian
tdvpEquation = jVMC.tdvp.TDVP(sampler, snrTol=inp["time_evol"]["snr_tolerance"], 
                                       svdTol=inp["time_evol"]["svd_tolerance"],
                                       rhsPrefactor=1.j, diagonalShift=0., makeReal='imag')

stepper = jVMC.stepper.AdaptiveHeun(timeStep=inp["time_evol"]["time_step"], tol=inp["time_evol"]["stepper_tolerance"])

t=inp["time_evol"]["t_init"]
tmax=inp["time_evol"]["t_final"]

outp.start_timing("measure observables")
obs, err = measure(observables, psi, sampler)
outp.stop_timing("measure observables")

outp.write_observables(t, energy=obs[0], X=obs[1]/L, ZZ=obs[2:]/L)

while t<tmax:
    tic = time.perf_counter()
    outp.print( ">  t = %f\n" % (t) )

    # TDVP step
    dp, dt = stepper.step(0, tdvpEquation, psi.get_parameters(), hamiltonian=hamiltonian, psi=psi, numSamples=inp["sampler"]["numSamples"], outp=outp)
    psi.set_parameters(dp)
    t += dt
    outp.print( "   Time step size: dt = %f" % (dt) )
    tdvpErr, tdvpRes = tdvpEquation.get_residuals()
    outp.print( "   Residuals: tdvp_err = %.2e, solver_res = %.2e" % (tdvpErr, tdvpRes) )

    # Measure observables
    outp.start_timing("measure observables")
    obs, err = measure(observables, psi, sampler)
    outp.stop_timing("measure observables")

    # Write observables
    outp.write_observables(t, energy=obs[0], X=obs[1]/L, ZZ=obs[2:]/L)
    # Write metadata
    outp.write_metadata(t, tdvp_error=tdvpErr, tdvp_residual=tdvpRes, SNR=tdvpEquation.get_snr(), spectrum=tdvpEquation.get_spectrum())
    # Write network parameters
    outp.write_network_checkpoint(t, psi.get_parameters())

    outp.print("    Energy = %f +/- %f" % (obs[0], err[0]))

    outp.print_timings(indent="   ")

    toc = time.perf_counter()
    outp.print( "   == Total time for this step: %fs\n" % (toc-tic) )
