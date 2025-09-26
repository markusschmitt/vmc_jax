import unittest

import jax
import jax.numpy as jnp
import numpy as np

import jVMC
from jVMC.util.output_manager import OutputManager
import jVMC.mpi_wrapper as mpi
import h5py
import os


class TestOutputManger(unittest.TestCase):

    def test_output(self):

        if mpi.rank == 0:
            outp = OutputManager("test.h5", append=False)

            outp.set_group("test")

            with h5py.File("test.h5") as f:
                self.assertTrue(len(f.keys()) == 1)
                self.assertTrue("test" in f.keys())

            x = np.array([13.2])
            outp.write_metadata(0.3, bla=x)
            with h5py.File("test.h5") as f:
                self.assertTrue("metadata" in f["test"].keys())
                self.assertTrue("bla" in f["test"]["metadata"].keys())
                self.assertTrue(np.allclose(f["test"]["metadata"]["bla"][0], x))

            y = 99.1
            outp.write_metadata(0.5, bla=y)
            with h5py.File("test.h5") as f:
                print(np.array(f["test"]["metadata"]["bla"]))
                self.assertTrue(np.allclose(f["test"]["metadata"]["bla"], np.array([x, [y]])))


            x = np.random.uniform(1,2,size=(13,))
            y = np.random.uniform(-1,1,size=(3,))
            outp.write_observables(0.1, obs1={"mean":x}, obs2={"mean":y})
            with h5py.File("test.h5") as f:
                self.assertTrue("observables" in f["test"].keys())
                self.assertTrue("obs1" in f["test"]["observables"].keys())
                self.assertTrue(np.allclose(f["test"]["observables"]["obs1"]["mean"][0], x))
                self.assertTrue(np.allclose(f["test"]["observables"]["obs2"]["mean"][0], y))

            y = 99.1
            outp.write_observables(0.5, bla={"mean":y})
            with h5py.File("test.h5") as f:
                self.assertTrue(np.allclose(f["test"]["observables"]["bla"]["mean"][0], np.array([y])))

            os.remove("test.h5")