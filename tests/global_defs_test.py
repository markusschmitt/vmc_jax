import unittest
import jax.numpy as jnp
import numpy as np

from jVMC_exp import global_defs

_SETTINGS = (
    "DT_PARAMS_REAL", "DT_PARAMS_CPX", "DT_OUT_REAL", "DT_OUT_CPX",
    "DT_OPERATORS_REAL", "DT_OPERATORS_CPX", "BATCH_LOOP", "MATMUL_CHUNK_BYTES",
)

class TestUpdate(unittest.TestCase):
    def setUp(self):
        saved = {name: getattr(global_defs, name) for name in _SETTINGS}
        self.addCleanup(lambda: [setattr(global_defs, k, v) for k, v in saved.items()])

    def test_accepts_all_allowed_dtypes(self):
        for dtype in global_defs.allowed_dtypes_real.values():
            global_defs.update(dt_params_real=dtype, dt_out_real=dtype)
            self.assertEqual(global_defs.DT_PARAMS_REAL, jnp.dtype(dtype))
            self.assertEqual(global_defs.DT_OUT_REAL, jnp.dtype(dtype))
            self.assertEqual(global_defs.DT_OPERATORS_REAL, jnp.dtype(dtype))

        for dtype in global_defs.allowed_dtypes_cpx.values():
            global_defs.update(dt_params_cpx=dtype, dt_out_cpx=dtype)
            self.assertEqual(global_defs.DT_PARAMS_CPX, jnp.dtype(dtype))
            self.assertEqual(global_defs.DT_OUT_CPX, jnp.dtype(dtype))
            self.assertEqual(global_defs.DT_OPERATORS_CPX, jnp.dtype(dtype))

    def test_rejects_other_dtypes_and_keeps_the_setting(self):
        before = global_defs.DT_PARAMS_REAL

        for dtype in (jnp.int32, jnp.bool_, jnp.complex64, "f64", "f16"):
            with self.assertRaises(TypeError):
                global_defs.update(dt_params_real=dtype)
            self.assertEqual(global_defs.DT_PARAMS_REAL, before)

        for dtype in (jnp.int32, jnp.float32, "c64"):
            with self.assertRaises(TypeError):
                global_defs.update(dt_params_cpx=dtype)

    def test_use_distributed_cannot_be_updated(self):
        with self.assertRaises(TypeError):
            global_defs.update(use_distributed=True)

class TestReadDtype(unittest.TestCase):
    def test_names_count_bits(self):
        self.assertEqual(global_defs._read_dtype("f16", True), jnp.dtype(jnp.float16))
        self.assertEqual(global_defs._read_dtype("F32", True), jnp.dtype(jnp.float32))
        self.assertEqual(global_defs._read_dtype("c64", False), jnp.dtype(jnp.complex64))

    def test_rejects_wrong_kind_and_unknown_names(self):
        for name, real in (("c64", True), ("f32", False), ("f8", True), ("", False)):
            with self.assertRaises(ValueError):
                global_defs._read_dtype(name, real)

if __name__ == "__main__":
    unittest.main()
