import sys
# Find jVMC package
sys.path.append(sys.path[0]+"/..")

import jax
from jax.config import config
config.update("jax_enable_x64", True)

import jax.numpy as jnp

import time

import jVMC.stepper as jVMCstepper
import jVMC.mpi_wrapper as mpi

import collections

def get_iterable(x):
    if isinstance(x, collections.abc.Iterable):
        return x
    else:
        return (x,)

def measure(observables, psi, sampler, numSamples=None):
    
    # Get sample
    sampleConfigs, sampleLogPsi, p =  sampler.sample( psi, numSamples )
    #sampleConfigs = jax.pmap(lambda x: jnp.concatenate([x, 1-x], axis=0))(sampleConfigs)
    #sampleLogPsi = jax.pmap(lambda x: jnp.concatenate([x, x], axis=0))(sampleLogPsi)
    #mpi.globNumSamples *= 2

    result = {}
    
    for name, ops in observables.items():

        tmpMeans = []
        tmpVariances = []
        tmpErrors = []

        for op in get_iterable(ops):
            sampleOffdConfigs, matEls = op.get_s_primes(sampleConfigs)
            sampleLogPsiOffd = psi(sampleOffdConfigs)
            Oloc = op.get_O_loc(sampleLogPsi,sampleLogPsiOffd)

            if p is not None:
                tmpMeans.append( mpi.global_mean(Oloc, p) )
                tmpVariances.append( mpi.global_variance(Oloc, p) )
                tmpErrors.append( 0. )
            else:
                tmpMeans.append( mpi.global_mean(Oloc) )
                tmpVariances.append( mpi.global_variance(Oloc) )
                tmpErrors.append( tmpVariances[-1] / jnp.sqrt(sampler.get_last_number_of_samples()) )

        result[name] = {}
        result[name]["mean"] = jnp.real(jnp.array(tmpMeans))
        result[name]["variance"] = jnp.real(jnp.array(tmpVariances))
        result[name]["MC_error"] = jnp.real(jnp.array(tmpErrors))

    return result 


def ground_state_search(psi, ham, tdvpEquation, sampler, numSteps=200, varianceTol=1e-10, stepSize=1e-2, observables=None, outp=None):

    delta = tdvpEquation.diagonalShift

    stepper = jVMCstepper.Euler(timeStep=stepSize)

    n=0
    if outp is not None:
        if observables is not None:
            obs = measure(observables, psi, sampler)
            outp.write_observables(n, **obs)

    varE = 1.0

    while n<numSteps and varE>varianceTol:

        tic = time.perf_counter()
    
        dp, _ = stepper.step(0, tdvpEquation, psi.get_parameters(), hamiltonian=ham, psi=psi, numSamples=None, outp=outp)
        psi.set_parameters(dp)
        n += 1

        varE = tdvpEquation.get_energy_variance()

        if outp is not None:
            if observables is not None:
                obs = measure(observables, psi, sampler)
                outp.write_observables(n, **obs)

        delta=0.95*delta
        tdvpEquation.set_diagonal_shift(delta)

        if outp is not None:
            outp.print(" STEP %d" % (n) )
            outp.print("   Energy mean: %f" % (tdvpEquation.get_energy_mean()) )
            outp.print("   Energy variance: %f" % (varE) )
            outp.print_timings(indent="   ")
            outp.print("   == Time for step: %fs" % (time.perf_counter() - tic) )

import h5py

class OutputManager:

    def __init__(self, dataFileName, group="/", append=False):

        self.fn = dataFileName
        
        self.currentGroup = "/"

        self.append = 'w'
        if append:
            self.append = 'a'

        self.set_group(group)

        self.timings = {}


    def set_group(self, group):
        
        if group != "/":
            self.currentGroup = "/" + group

        if mpi.rank == 0:                
            with h5py.File(self.fn, self.append) as f:
                if not self.currentGroup in f:
                    f.create_group(self.currentGroup)

        self.append = 'a'
        

    def write_observables(self, time, **kwargs):

        if mpi.rank == 0:                
            
            with h5py.File(self.fn, "a") as f:
                
                if not "observables" in f[self.currentGroup]:
                    f.create_group(self.currentGroup+"/observables")

                if not "times" in f[self.currentGroup+"/observables"]:
                    f.create_dataset(self.currentGroup+"/observables/times", (0,), maxshape=(None,), dtype='f8', chunks=True)
                
                newLen = len(f[self.currentGroup+"/observables/times"]) + 1
                f[self.currentGroup+"/observables/times"].resize((newLen,))
                f[self.currentGroup+"/observables/times"][-1] = time

                for key, obsDict in kwargs.items():

                    for name, value in obsDict.items():

                        datasetName = self.currentGroup+"/observables/"+key+"/"+name

                        if not key in f[self.currentGroup+"/observables"]:

                            f.create_group(self.currentGroup+"/observables/"+key)

                        if not name in f[self.currentGroup+"/observables/"+key]:

                            f.create_dataset(datasetName, (0,)+value.shape, maxshape=(None,)+value.shape, dtype='f8', chunks=True)

                        newLen = len(f[datasetName]) + 1
                        f[datasetName].resize((newLen,)+value.shape)
                        f[datasetName][-1] = value


    def write_metadata(self, time, **kwargs):

        if mpi.rank == 0:                
            
            groupname="metadata"

            with h5py.File(self.fn, "a") as f:
                
                if not groupname in f[self.currentGroup]:
                    f.create_group(self.currentGroup+"/"+groupname)

                if not "times" in f[self.currentGroup+"/"+groupname]:
                    f.create_dataset(self.currentGroup+"/"+groupname+"/times", (0,), maxshape=(None,), dtype='f8', chunks=True)
                
                newLen = len(f[self.currentGroup+"/"+groupname+"/times"]) + 1
                f[self.currentGroup+"/"+groupname+"/times"].resize((newLen,))
                f[self.currentGroup+"/"+groupname+"/times"][-1] = time

                for key, value in kwargs.items():

                    if not key in f[self.currentGroup+"/"+groupname]:

                        f.create_dataset(self.currentGroup+"/"+groupname+"/"+key, (0,)+value.shape, maxshape=(None,)+value.shape, dtype='f8', chunks=True)

                    newLen = len(f[self.currentGroup+"/"+groupname+"/"+key]) + 1
                    f[self.currentGroup+"/"+groupname+"/"+key].resize((newLen,)+value.shape)
                    f[self.currentGroup+"/"+groupname+"/"+key][-1] = value
 

    def write_network_checkpoint(self, time, weights):

        if mpi.rank == 0:                
            
            groupname="network_checkpoints"

            with h5py.File(self.fn, "a") as f:
                
                if not groupname in f[self.currentGroup]:
                    f.create_group(self.currentGroup+"/"+groupname)

                if not "times" in f[self.currentGroup+"/"+groupname]:
                    f.create_dataset(self.currentGroup+"/"+groupname+"/times", (0,), maxshape=(None,), dtype='f8', chunks=True)
                
                newLen = len(f[self.currentGroup+"/"+groupname+"/times"]) + 1
                f[self.currentGroup+"/"+groupname+"/times"].resize((newLen,))
                f[self.currentGroup+"/"+groupname+"/times"][-1] = time

                key = "checkpoints"
                value = weights

                if not key in f[self.currentGroup+"/"+groupname]:

                    f.create_dataset(self.currentGroup+"/"+groupname+"/"+key, (0,)+value.shape, maxshape=(None,)+value.shape, dtype='f8', chunks=True)

                newLen = len(f[self.currentGroup+"/"+groupname+"/"+key]) + 1
                f[self.currentGroup+"/"+groupname+"/"+key].resize((newLen,)+value.shape)
                f[self.currentGroup+"/"+groupname+"/"+key][-1] = value


    def get_network_checkpoint(self, time=-1, idx=-1):
        
        groupname="network_checkpoints"
        key = "checkpoints"

        weights = None

        if mpi.rank == 0:
            
            if time < 0:

                with h5py.File(self.fn, "r") as f:
                    weights = f[self.currentGroup+"/"+groupname+"/"+key][idx]
                    time = f[self.currentGroup+"/"+groupname+"/times"][idx]
        
            mpi.comm.bcast(time)

        else:

            time = mpi.comm.bcast(None)

        weights = mpi.bcast_unknown_size(weights)

        return time, weights


    def start_timing(self, name):
        
        if not name in self.timings:
            self.timings[name] = {"total": 0.0, "last_total": 0.0, "newest": 0.0, "count": 0, "init": 0.0}

        self.timings[name]["init"] = time.perf_counter()


    def stop_timing(self, name):

        toc = time.perf_counter()

        if not name in self.timings:
            self.timings[name] = {"total": 0.0, "last_total": 0.0, "newest": 0.0, "count": 0, "init": 0.0}

        elapsed = toc - self.timings[name]["init"]

        self.timings[name]["total"] += elapsed
        self.timings[name]["newest"] = elapsed
        self.timings[name]["count"] += 1
    
    
    def add_timing(self, name, elapsed):
        
        if not name in self.timings:
            self.timings[name] = {"total": 0.0, "last_total": 0.0, "newest": 0.0, "count": 0, "init": 0.0}

        self.timings[name]["total"] += elapsed
        self.timings[name]["newest"] = elapsed
        self.timings[name]["count"] += 1


    def print_timings(self, indent=""):

        self.print("%sRecorded timings:" % indent)
        for key, item in self.timings.items():
            self.print( "%s  * %s: %fs" % ( indent, key, item["total"]-item["last_total"] ) )
            
        for key in self.timings:
            self.timings[key]["last_total"] = self.timings[key]["total"]


    def print(self, text):
        if mpi.rank == 0:
            print(text, flush=True)

import numpy as np

if __name__ == "__main__":

    outp = OutputManager("tmp.hdf5", group="gs_search")
    
    outp.start_timing("bla")

    outp.write_observables(1.2, X=np.array([1.234]), ZZ=np.arange(1,5,.3))

    outp.stop_timing("bla")
    outp.print_timings()
    outp.start_timing("bla")

    outp.write_observables(1.2, X=np.array([1.234]), ZZ=np.arange(1,5,.3))

    outp.stop_timing("bla")

    outp.print_timings()
