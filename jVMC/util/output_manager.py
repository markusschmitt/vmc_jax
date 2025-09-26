import h5py
import numpy as np
import jax.numpy as jnp
import jVMC.mpi_wrapper as mpi
import time

class OutputManager:
    '''This class provides functionality for I/O and timing.
    '''

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

                        value = self.to_array(value)

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

                    value = self.to_array(value)

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

            with h5py.File(self.fn, "r") as f:
                times = np.array(f[self.currentGroup + "/" + groupname + "/times"])

            if time >= 0:
                idx = np.argmin(np.abs(times-time))

            with h5py.File(self.fn, "r") as f:
                weights = f[self.currentGroup + "/" + groupname + "/" + key][idx]
            time = times[idx]

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


    def to_array(self, x):

        if not isinstance(x, (np.ndarray, jnp.ndarray)):
            x = np.array([x])

        return x
