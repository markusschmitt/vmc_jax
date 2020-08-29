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

def measure(ops, psi, sampler, numSamples=None):
    
    # Get sample
    sampleConfigs, sampleLogPsi, p =  sampler.sample( psi, numSamples )

    means = []
    errors = []
    
    for op in ops:

        sampleOffdConfigs, matEls = op.get_s_primes(sampleConfigs)
        sampleLogPsiOffd = psi(sampleOffdConfigs)
        Oloc = op.get_O_loc(sampleLogPsi,sampleLogPsiOffd)

        if p is not None:
            means.append( mpi.global_sum( jnp.array([jnp.dot(p, Oloc)]) ) )
            errors.append(0.0)
        else:
            means.append( mpi.global_mean(Oloc) )
            errors.append( mpi.global_variance(Oloc) / jnp.sqrt(sampler.get_last_number_of_samples()) )

    return jnp.real(jnp.array(means)), jnp.real(jnp.array(errors))


def ground_state_search(psi, ham, tdvpEquation, sampler, numSteps=200, stepSize=1e-2, observables=None, outp=None):

    delta = tdvpEquation.diagonalShift

    stepper = jVMCstepper.Euler(timeStep=stepSize)

    n=0
    if outp is not None:
        if observables is not None:
            obs, err = measure(observables, psi, sampler)
            outp.print("{0:d} {1:.6f} {2:.6f} {3:.6f}".format(n, obs[0], obs[1], obs[2]))
    while n<numSteps:

        tic = time.perf_counter()
    
        dp, _ = stepper.step(0, tdvpEquation, psi.get_parameters(), hamiltonian=ham, psi=psi, numSamples=None)
        psi.set_parameters(dp)
        n += 1

        if outp is not None:
            if observables is not None:
                obs, err = measure(observables, psi, sampler)
                outp.print("{0:d} {1:.6f} {2:.6f} {3:.6f}".format(n, obs[0], obs[1], obs[2]))

        delta=0.95*delta
        tdvpEquation.set_diagonal_shift(delta)

        if outp is not None:
            outp.print("   == Time for step: %fs" % (time.perf_counter() - tic) )

import h5py

class OutputManager:

    def __init__(self, dataFileName, group="/"):

        self.fn = dataFileName
        
        self.currentGroup = "/"

        self.set_group(group)

        self.timings = {}


    def set_group(self, group):
        
        if group != "/":
            self.currentGroup = "/" + group

        if mpi.rank == 0:                
            with h5py.File(self.fn, 'a') as f:
                if not self.currentGroup in f:
                    f.create_group(self.currentGroup)
        

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

                for key, obs in kwargs.items():

                    if not key in f[self.currentGroup+"/observables"]:

                        f.create_dataset(self.currentGroup+"/observables/"+key+"/mean", (0,)+obs.shape, maxshape=(None,)+obs.shape, dtype='f8', chunks=True)
                        f.create_dataset(self.currentGroup+"/observables/"+key+"/variance", (0,)+obs.shape, maxshape=(None,)+obs.shape, dtype='f8', chunks=True)

                    newLen = len(f[self.currentGroup+"/observables/"+key+"/mean"]) + 1
                    f[self.currentGroup+"/observables/"+key+"/mean"].resize((newLen,)+obs.shape)
                    f[self.currentGroup+"/observables/"+key+"/mean"][-1] = obs


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
