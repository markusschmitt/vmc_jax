import os
import time
import unittest

import h5py
import numpy as np

from jVMC_exp.util.input_manager import InputManager
from jVMC_exp.util.output_manager import OutputManager


class TestOutputManager(unittest.TestCase):
    def setUp(self):
        self.h5 = "test_output.h5"
        self.outp = OutputManager()

    def tearDown(self):
        if os.path.exists(self.h5):
            os.remove(self.h5)

    def test_metadata_times_recorded(self):
        self.outp.write_metadata(0.3, bla=1.0)
        self.outp.write_metadata(0.5, bla=2.0)
        self.outp.save_to_h5(self.h5)

        with h5py.File(self.h5) as f:
            times = f["metadata/times"][()]
            self.assertTrue(np.allclose(times, [0.3, 0.5]))

    def test_observables_nested_dict(self):
        """Nested dict kwargs must create sub-groups in the HDF5 file."""
        x = np.random.uniform(1, 2, size=(13,))
        y = np.random.uniform(-1, 1, size=(3,))
        self.outp.write_observables(0.1, obs1={"mean": x}, obs2={"mean": y})
        self.outp.save_to_h5(self.h5)

        with h5py.File(self.h5) as f:
            self.assertIn("observables", f)
            self.assertIn("obs1", f["observables"])
            self.assertIn("obs2", f["observables"])
            self.assertTrue(np.allclose(f["observables/obs1/mean"][0], x))
            self.assertTrue(np.allclose(f["observables/obs2/mean"][0], y))

    def test_observables_times_and_values(self):
        """Times and values accumulate across multiple write_observables calls."""
        self.outp.write_observables(0.1, energy=1.0)
        self.outp.write_observables(0.5, energy=2.0)
        self.outp.save_to_h5(self.h5)

        with h5py.File(self.h5) as f:
            self.assertTrue(np.allclose(f["observables/times"][()], [0.1, 0.5]))
            self.assertTrue(np.allclose(f["observables/energy"][()], [1.0, 2.0]))

    def test_observables_scalar_nested(self):
        """Scalar inside a nested dict is stored as a single-element row."""
        self.outp.write_observables(0.5, bla={"mean": 99.1})
        self.outp.save_to_h5(self.h5)

        with h5py.File(self.h5) as f:
            self.assertTrue(np.allclose(f["observables/bla/mean"][0], [99.1]))

    def test_network_checkpoint_roundtrip(self):
        """Checkpoint stored in memory and retrieved by index and time."""
        weights = np.random.uniform(size=(5,))
        self.outp.write_network_checkpoint(1.0, weights)

        t, w = self.outp.get_network_checkpoint()
        self.assertAlmostEqual(t, 1.0)
        self.assertTrue(np.allclose(w, weights))

        t, w = self.outp.get_network_checkpoint(time=1.0)
        self.assertAlmostEqual(t, 1.0)
        self.assertTrue(np.allclose(w, weights))

    def test_network_checkpoint_saved_to_h5(self):
        weights = np.random.uniform(size=(4,))
        self.outp.write_network_checkpoint(2.0, weights)
        self.outp.save_to_h5(self.h5)

        with h5py.File(self.h5) as f:
            self.assertIn("network_checkpoints", f)
            self.assertTrue(
                np.allclose(f["network_checkpoints/checkpoints"][0], weights)
            )

    def test_save_append_overwrites_datasets(self):
        """Calling save_to_h5 twice with append=True should replace datasets."""
        self.outp.write_observables(0.1, energy=1.0)
        self.outp.save_to_h5(self.h5)
        self.outp.write_observables(0.5, energy=2.0)
        self.outp.save_to_h5(self.h5, append=True)

        with h5py.File(self.h5) as f:
            self.assertEqual(len(f["observables/times"][()]), 2)

    def test_no_path_buffers_without_creating_file(self):
        self.outp.write_observables(0.1, energy=1.0)
        self.outp.write_metadata(model="RBM")
        self.outp.write_dataset("samples", np.arange(4), group="diagnostics")

        self.assertFalse(os.path.exists(self.h5))
        self.assertIsNone(self.outp.path)
        self.assertIn("observables", self.outp.data)
        self.assertIn("diagnostics", self.outp.data)

    def test_save_path_persists_for_later_saves(self):
        self.outp.write_observables(0.1, energy=1.0)
        self.outp.save_to_h5(self.h5)
        self.assertEqual(str(self.outp.path), self.h5)

        self.outp.write_observables(0.2, energy=2.0)
        self.outp.save_to_h5(append=True)

        with h5py.File(self.h5) as f:
            self.assertTrue(np.allclose(f["observables/times"][()], [0.1, 0.2]))
            self.assertTrue(np.allclose(f["observables/energy"][()], [1.0, 2.0]))

    def test_load_from_h5_roundtrip(self):
        """load_from_h5 should reconstruct the same arrays."""
        self.outp.write_observables(0.1, energy=1.0)
        self.outp.write_observables(0.5, energy=2.0)
        self.outp.save_to_h5(self.h5)

        loaded = InputManager.load_from_h5(self.h5)
        self.assertTrue(np.allclose(loaded["observables"]["times"], [0.1, 0.5]))
        self.assertTrue(np.allclose(loaded["observables"]["energy"], [1.0, 2.0]))

    def test_parameter_roundtrip_uses_input_manager(self):
        params = {"dense": {"kernel": np.arange(4).reshape(2, 2), "bias": np.ones(2)}}
        template = {"dense": {"kernel": np.zeros((2, 2)), "bias": np.zeros(2)}}

        self.outp.write_parameters(0, params, attrs={"optimizer": "SR"})
        self.assertFalse(os.path.exists(self.h5))
        self.outp.save_to_h5(self.h5)

        inp = InputManager(self.h5)
        loaded = inp.load_parameters(template, step="latest")
        self.assertTrue(
            np.allclose(loaded["dense"]["kernel"], params["dense"]["kernel"])
        )
        self.assertTrue(np.allclose(loaded["dense"]["bias"], params["dense"]["bias"]))

    def test_timings_accumulate(self):
        self.outp.start_timing("step")
        time.sleep(0.02)
        elapsed = self.outp.stop_timing("step")
        self.assertGreater(elapsed, 0.01)
        self.assertGreater(self.outp._timings["step"]["total"], 0.0)
        self.assertEqual(self.outp._timings["step"]["count"], 1)

    def test_add_timing(self):
        self.outp.add_timing("manual", 0.5)
        self.outp.add_timing("manual", 0.3)
        self.assertAlmostEqual(self.outp._timings["manual"]["total"], 0.8, places=9)
        self.assertEqual(self.outp._timings["manual"]["count"], 2)


if __name__ == "__main__":
    unittest.main()
