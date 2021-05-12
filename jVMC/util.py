import sys
# Find jVMC package
sys.path.append(sys.path[0] + "/..")

import jax
from jax.config import config
config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as random
import flax
import flax.nn as nn

import time

import jVMC
import jVMC.stepper as jVMCstepper
import jVMC.mpi_wrapper as mpi
import jVMC.activation_functions as act_funs

import collections


def get_iterable(x):
    if isinstance(x, collections.abc.Iterable):
        return x
    else:
        return (x,)


def get_point_orbit(L):

    trafos = []

    idx = np.arange(L*L).reshape((L,L))

    for _ in range(2):
        for _ in range(4):
            idx = np.array(list(zip(*idx[::-1]))) # rotation
            trafos.append(idx)
        idx = np.transpose(idx) # reflection

    orbit = []

    idx = np.arange(L*L)

    for t in trafos:

        o = np.zeros((L*L,L*L), dtype=np.int32)

        o[idx,t.ravel()] = 1

        orbit.append(o)

    orbit = jnp.array(orbit)

    return orbit

def get_translation_orbit(L):

    idx = np.arange(L**2, dtype=np.int32).reshape((L,L))

    trafos = []

    for lx in range(L):
        for ly in range(L):

            trafos.append(idx)

            idx = np.roll(idx, 1, axis=1)

        idx = np.roll(idx, 1, axis=0)

    orbit = []

    idx = np.arange(L*L)

    for t in trafos:

        o = np.zeros((L*L,L*L), dtype=np.int32)

        o[idx,t.ravel()] = 1

        orbit.append(o)

    orbit = jnp.array(orbit)

    return orbit

def get_2d_orbit(L):

    po = get_point_orbit(L)

    to = get_translation_orbit(L)

    orbit = jax.vmap(lambda x,y: jax.vmap(lambda a,b: jnp.dot(b,a), in_axes=(None,0))(x,y), in_axes=(0,None))(to,po)

    orbit = orbit.reshape((-1,L**2,L**2))

    newOrbit = [tuple(x.ravel()) for x in orbit]

    uniqueOrbit = np.unique(newOrbit,axis=0).reshape(-1,L**2,L**2)

    return jnp.array(uniqueOrbit)


def init_net(descr, dims, seed=0):

    def get_activation_functions(actFuns):

        if type(actFuns) is list:
            return [act_funs.activationFunctions[fn] for fn in actFuns]

        return act_funs.activationFunctions[actFuns]

    netTypes = {
        "RBM": jVMC.nets.RBM,
        "FFN": jVMC.nets.FFN,
        "CNN": jVMC.nets.CNN,
        "LSTM": jVMC.nets.LSTM,
        "LSTMsym": jVMC.nets.LSTMsym,
        "PhaseRNN": jVMC.nets.RNN,
        "PhaseRNNsym": jVMC.nets.RNNsym,
        "CpxRNN": jVMC.nets.CpxRNN,
        "RNN": jVMC.nets.RNN,
        "RNN2D": jVMC.nets.RNN2D,
        "RNNsym": jVMC.nets.RNNsym,
        "RNN2Dsym": jVMC.nets.RNN2Dsym,
        "CpxRBM": jVMC.nets.CpxRBM,
        "CpxCNN": jVMC.nets.CpxCNN
    }

    def get_net(descr, dims, seed):

        net = netTypes[descr["type"]](**descr["parameters"])
        params = net.init(random.PRNGKey(seed), jnp.zeros(dims, dtype=np.int32))
        return net, params

    if "actFun" in descr["net1"]["parameters"]:
        descr["net1"]["parameters"]["actFun"] = get_activation_functions(descr["net1"]["parameters"]["actFun"])

    if descr["net1"]["type"][-3:] == "sym":
        L = descr["net1"]["parameters"]["L"]
        if descr["net1"]["type"][-5:-3] == "2D":
            descr["net1"]["parameters"]["orbit"] = get_2d_orbit(L)
        else:
            # Generate orbit of 1D translations for RNNsym net
            descr["net1"]["parameters"]["orbit"] = jnp.array([jnp.roll(jnp.identity(L, dtype=np.int32), l, axis=1) for l in range(L)])

    if "net2" in descr:
        if descr["net2"]["type"][-3:] == "sym":
            L = descr["net2"]["parameters"]["L"]
            if descr["net2"]["type"][-5:-3] == "2D":
                # Generate orbit of 2D translations for RNNsym net
                descr["net2"]["parameters"]["orbit"] = get_2d_orbit(L)
            else:
                # Generate orbit of 1D translations for RNNsym net
                descr["net2"]["parameters"]["orbit"] = jnp.array([jnp.roll(jnp.identity(L, dtype=np.int32), l, axis=1) for l in range(L)])

    if not "net2" in descr:

        model, params = get_net(descr["net1"], dims, seed)

        return jVMC.vqs.NQS(model, params, batchSize=descr["gradient_batch_size"])

    else:

        if "actFun" in descr["net2"]["parameters"]:

            descr["net2"]["parameters"]["actFun"] = get_activation_functions(descr["net2"]["parameters"]["actFun"])

        model1, params1 = get_net(descr["net1"], dims, seed)
        model2, params2 = get_net(descr["net2"], dims, seed)

        return jVMC.vqs.NQS((model1, model2), (params1, params2), batchSize=descr["gradient_batch_size"])


def measure(observables, psi, sampler, numSamples=None):

    # Get sample
    sampleConfigs, sampleLogPsi, p = sampler.sample(numSamples)

    result = {}

    for name, ops in observables.items():

        tmpMeans = []
        tmpVariances = []
        tmpErrors = []

        for op in get_iterable(ops):
            sampleOffdConfigs, matEls = op.get_s_primes(sampleConfigs)
            sampleLogPsiOffd = psi(sampleOffdConfigs)
            Oloc = op.get_O_loc(sampleLogPsi, sampleLogPsiOffd)

            if p is not None:
                tmpMeans.append(mpi.global_mean(Oloc, p))
                tmpVariances.append(mpi.global_variance(Oloc, p))
                tmpErrors.append(0.)
            else:
                tmpMeans.append(mpi.global_mean(Oloc))
                tmpVariances.append(mpi.global_variance(Oloc))
                tmpErrors.append(jnp.sqrt(tmpVariances[-1]) / jnp.sqrt(sampler.get_last_number_of_samples()))

        result[name] = {}
        result[name]["mean"] = jnp.real(jnp.array(tmpMeans))
        result[name]["variance"] = jnp.real(jnp.array(tmpVariances))
        result[name]["MC_error"] = jnp.real(jnp.array(tmpErrors))

    return result


def ground_state_search(psi, ham, tdvpEquation, sampler, numSteps=200, varianceTol=1e-10, stepSize=1e-2, observables=None, outp=None):

    delta = tdvpEquation.diagonalShift

    stepper = jVMCstepper.Euler(timeStep=stepSize)

    n = 0
    if outp is not None:
        if observables is not None:
            obs = measure(observables, psi, sampler)
            outp.write_observables(n, **obs)

    varE = 1.0

    while n < numSteps and varE > varianceTol:

        tic = time.perf_counter()

        dp, _ = stepper.step(0, tdvpEquation, psi.get_parameters(), hamiltonian=ham, psi=psi, numSamples=None, outp=outp)
        psi.set_parameters(dp)
        n += 1

        varE = tdvpEquation.get_energy_variance()

        if outp is not None:
            if observables is not None:
                obs = measure(observables, psi, sampler)
                outp.write_observables(n, **obs)

        delta = 0.95 * delta
        tdvpEquation.set_diagonal_shift(delta)

        if outp is not None:
            outp.print(" STEP %d" % (n))
            outp.print("   Energy mean: %f" % (tdvpEquation.get_energy_mean()))
            outp.print("   Energy variance: %f" % (varE))
            outp.print_timings(indent="   ")
            outp.print("   == Time for step: %fs" % (time.perf_counter() - tic))


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
                    f.create_group(self.currentGroup + "/observables")

                if not "times" in f[self.currentGroup + "/observables"]:
                    f.create_dataset(self.currentGroup + "/observables/times", (0,), maxshape=(None,), dtype='f8', chunks=True)

                newLen = len(f[self.currentGroup + "/observables/times"]) + 1
                f[self.currentGroup + "/observables/times"].resize((newLen,))
                f[self.currentGroup + "/observables/times"][-1] = time

                for key, obsDict in kwargs.items():

                    for name, value in obsDict.items():

                        datasetName = self.currentGroup + "/observables/" + key + "/" + name

                        if not key in f[self.currentGroup + "/observables"]:

                            f.create_group(self.currentGroup + "/observables/" + key)

                        if not name in f[self.currentGroup + "/observables/" + key]:

                            f.create_dataset(datasetName, (0,) + value.shape, maxshape=(None,) + value.shape, dtype='f8', chunks=True)

                        newLen = len(f[datasetName]) + 1
                        f[datasetName].resize((newLen,) + value.shape)
                        f[datasetName][-1] = value

    def write_metadata(self, time, **kwargs):

        if mpi.rank == 0:

            groupname = "metadata"

            with h5py.File(self.fn, "a") as f:

                if not groupname in f[self.currentGroup]:
                    f.create_group(self.currentGroup + "/" + groupname)

                if not "times" in f[self.currentGroup + "/" + groupname]:
                    f.create_dataset(self.currentGroup + "/" + groupname + "/times", (0,), maxshape=(None,), dtype='f8', chunks=True)

                newLen = len(f[self.currentGroup + "/" + groupname + "/times"]) + 1
                f[self.currentGroup + "/" + groupname + "/times"].resize((newLen,))
                f[self.currentGroup + "/" + groupname + "/times"][-1] = time

                for key, value in kwargs.items():

                    if not key in f[self.currentGroup + "/" + groupname]:

                        f.create_dataset(self.currentGroup + "/" + groupname + "/" + key, (0,) + value.shape, maxshape=(None,) + value.shape, dtype='f8', chunks=True)

                    newLen = len(f[self.currentGroup + "/" + groupname + "/" + key]) + 1
                    f[self.currentGroup + "/" + groupname + "/" + key].resize((newLen,) + value.shape)
                    f[self.currentGroup + "/" + groupname + "/" + key][-1] = value

    def write_network_checkpoint(self, time, weights):

        if mpi.rank == 0:

            groupname = "network_checkpoints"

            with h5py.File(self.fn, "a") as f:

                if not groupname in f[self.currentGroup]:
                    f.create_group(self.currentGroup + "/" + groupname)

                if not "times" in f[self.currentGroup + "/" + groupname]:
                    f.create_dataset(self.currentGroup + "/" + groupname + "/times", (0,), maxshape=(None,), dtype='f8', chunks=True)

                newLen = len(f[self.currentGroup + "/" + groupname + "/times"]) + 1
                f[self.currentGroup + "/" + groupname + "/times"].resize((newLen,))
                f[self.currentGroup + "/" + groupname + "/times"][-1] = time

                key = "checkpoints"
                value = weights

                if not key in f[self.currentGroup + "/" + groupname]:

                    f.create_dataset(self.currentGroup + "/" + groupname + "/" + key, (0,) + value.shape, maxshape=(None,) + value.shape, dtype='f8', chunks=True)

                newLen = len(f[self.currentGroup + "/" + groupname + "/" + key]) + 1
                f[self.currentGroup + "/" + groupname + "/" + key].resize((newLen,) + value.shape)
                f[self.currentGroup + "/" + groupname + "/" + key][-1] = value

    def get_network_checkpoint(self, time=-1, idx=-1):

        groupname = "network_checkpoints"
        key = "checkpoints"

        weights = None

        if mpi.rank == 0:

            if time < 0:

                with h5py.File(self.fn, "r") as f:
                    weights = f[self.currentGroup + "/" + groupname + "/" + key][idx]
                    time = f[self.currentGroup + "/" + groupname + "/times"][idx]

            mpi.comm.bcast(time)

        else:

            time = mpi.comm.bcast(None)

        weights = mpi.bcast_unknown_size(weights)

        return time, weights

    def write_error_data(self, name, data, mpiRank=0):

        if mpi.rank == mpiRank:

            groupname = "error_data"

            with h5py.File(self.fn, "a") as f:

                if not groupname in f["/"]:
                    f.create_group("/" + groupname)

                f.create_dataset("/" + groupname + "/" + name, data=np.array(data))

    def write_dataset(self, name, data, groupname="/", mpiRank=0):

        if mpi.rank == mpiRank:

            with h5py.File(self.fn, "a") as f:

                if not groupname == "/":
                    if not groupname in f["/"]:
                        f.create_group("/" + groupname)

                print(data.shape)
                f.create_dataset("/" + groupname + "/" + name, data=np.array(data))

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
            self.print("%s  * %s: %fs" % (indent, key, item["total"] - item["last_total"]))

        for key in self.timings:
            self.timings[key]["last_total"] = self.timings[key]["total"]

    def print(self, text):
        if mpi.rank == 0:
            print(text, flush=True)


import numpy as np

if __name__ == "__main__":

    outp = OutputManager("tmp.hdf5", group="gs_search")

    outp.start_timing("bla")

    outp.write_observables(1.2, X=np.array([1.234]), ZZ=np.arange(1, 5, .3))

    outp.stop_timing("bla")
    outp.print_timings()
    outp.start_timing("bla")

    outp.write_observables(1.2, X=np.array([1.234]), ZZ=np.arange(1, 5, .3))

    outp.stop_timing("bla")

    outp.print_timings()
