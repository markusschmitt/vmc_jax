import unittest
import jax
import jax.numpy as jnp
import numpy as np

from jax.sharding import PartitionSpec

from jVMC_exp import sharding_config
from jVMC_exp.sharding_config import (
    BatchLayout, DEVICE_SHARDING, DEVICE_SPEC, REPLICATED_SHARDING, sharded, pad_to_devices,
    _take_batch, _trim_batch, _put_batch,
)

D = jax.device_count()
COLLECTIVES = ("all-to-all", "all-gather", "collective-permute", "all-reduce", "reduce-scatter")

# (samples per device, samples per device in a batch): several full batches, a partial
# last batch, a single full batch and fewer samples than one batch
LOCAL_SIZES = ((12, 4), (10, 4), (4, 4), (3, 8))

def _data(num_samples, seed=0):
    x = jax.random.normal(jax.random.PRNGKey(seed), (num_samples, 3))
    return jax.device_put(x, DEVICE_SHARDING)

def _interleave(x):
    return jnp.stack([x, -x], axis=1).reshape((2 * x.shape[0],) + x.shape[1:])

class TestBatchLayout(unittest.TestCase):

    def test_put_inverts_split(self):
        for n, b in LOCAL_SIZES:
            with self.subTest(n=n, b=b):
                layout = BatchLayout(n * D, b * D)
                x = _data(n * D)
                out = None
                for i, piece in enumerate(layout.split(x)):
                    out = layout.alloc(piece) if out is None else out
                    out = layout.put(out, piece, i)

                self.assertTrue(jnp.array_equal(out, x))
                self.assertTrue(out.sharding.is_equivalent_to(DEVICE_SHARDING, out.ndim))

    def test_put_accepts_padded_last_batch(self):
        layout = BatchLayout(10 * D, 4 * D)
        x = _data(10 * D)
        out = layout.alloc(x)
        for i in range(layout.n_batches):
            out = layout.put(out, layout.take(x, i), i)

        self.assertTrue(jnp.array_equal(out, x))

    def test_batch_sizes_match_consecutive_batches(self):
        for n, b in LOCAL_SIZES:
            with self.subTest(n=n, b=b):
                layout = BatchLayout(n * D, b * D)
                sizes = [piece.shape[0] for piece in layout.split(_data(n * D))]
                expected = [min(b * D, n * D - i * b * D) for i in range(-(-n // b))]

                self.assertEqual(sizes, expected)
                self.assertEqual(layout.n_batches, len(expected))

    def test_batch_positions(self):
        for n, b in LOCAL_SIZES:
            with self.subTest(n=n, b=b):
                layout = BatchLayout(n * D, b * D)
                x = _data(n * D)
                in_batch_order = jnp.concatenate(layout.split(x))

                self.assertTrue(jnp.array_equal(in_batch_order[layout.batch_positions()], x))

    def test_scaled_layout_splits_interleaved_rows(self):
        layout = BatchLayout(10 * D, 4 * D)
        x = _data(10 * D)
        for piece, interleaved_piece in zip(layout.split(x), layout.scaled(2).split(_interleave(x))):
            self.assertTrue(jnp.array_equal(_interleave(piece), interleaved_piece))

    def test_helpers_do_not_communicate(self):
        layout = BatchLayout(10 * D, 4 * D)
        x = _data(10 * D)
        piece = layout.take(x, 0)
        compiled = {
            "take": _take_batch.lower(x, 0, b=4, n_batches=layout.n_batches),
            "trim": _trim_batch.lower(piece, rows=2),
            "put": _put_batch.lower(x, piece, 0, b=4, rows=4),
            "put last": _put_batch.lower(x, piece, 2, b=4, rows=2),
        }
        for name, lowered in compiled.items():
            hlo = lowered.compile().as_text()
            with self.subTest(helper=name):
                self.assertFalse([c for c in COLLECTIVES if c in hlo])

    def test_replicated_results_stay_replicated(self):
        layout = BatchLayout(10 * D, 4 * D)
        x = _data(10 * D)
        out = None
        for i in range(layout.n_batches):
            piece = jax.device_put(layout.take(x, i), REPLICATED_SHARDING)
            out = layout.alloc(piece) if out is None else out
            out = layout.put(out, piece, i)

        self.assertTrue(jnp.array_equal(out, x))
        self.assertTrue(out.sharding.is_equivalent_to(REPLICATED_SHARDING, out.ndim))

    def test_invalid_sizes_raise(self):
        with self.assertRaises(ValueError):
            BatchLayout(4 * D, 0)
        if D > 1:
            with self.assertRaises(ValueError):
                BatchLayout(4 * D + 1, 4 * D)
            with self.assertRaises(ValueError):
                BatchLayout(4 * D, 4 * D + 1)

class _Model:
    @sharded()
    def double(self, x, *, batch_size):
        return 2 * x

    @sharded()
    def combine(self, x, y, *, batch_size):
        return {"sum": x + y, "prod": x * y}

    @sharded(yield_iter=True)
    def lazy_double(self, x, *, batch_size):
        return 2 * x

    @sharded(in_specs=(DEVICE_SPEC,))
    def double_custom_specs(self, x, *, batch_size):
        return 2 * x

class TestShardedDecorator(unittest.TestCase):
    """
    The decorator evaluates local batches. Non-lazy results must keep the order of the
    input samples, and lazy results must line up with the iterable's layout.
    """
    def test_results_keep_sample_order(self):
        model = _Model()
        for n, b in LOCAL_SIZES:
            with self.subTest(n=n, b=b):
                x = _data(n * D)
                out = model.double(x, batch_size=b * D)

                self.assertTrue(jnp.array_equal(out, 2 * x))
                self.assertTrue(out.sharding.is_equivalent_to(DEVICE_SHARDING, out.ndim))

    def test_pytree_output_and_several_arguments(self):
        model = _Model()
        x, y = _data(10 * D, seed=0), _data(10 * D, seed=1)
        out = model.combine(x, y, batch_size=4 * D)

        self.assertTrue(jnp.array_equal(out["sum"], x + y))
        self.assertTrue(jnp.array_equal(out["prod"], x * y))

    def test_arguments_of_different_lengths_raise(self):
        model = _Model()
        x, y = _data(10 * D, seed=0), _data(6 * D, seed=1)
        for a, b in ((x, y), (y, x)):
            with self.subTest(lengths=(a.shape[0], b.shape[0])):
                with self.assertRaises(ValueError):
                    model.combine(a, b, batch_size=4 * D)

    def test_host_input(self):
        model = _Model()
        x = np.arange(10 * D * 3, dtype=np.float64).reshape(10 * D, 3)

        self.assertTrue(np.array_equal(model.double(x, batch_size=4 * D), 2 * x))

    @unittest.skipIf(D == 1, "Every number of samples is divisible by one device")
    def test_indivisible_number_of_samples_is_padded(self):
        model = _Model()
        x = jax.random.normal(jax.random.PRNGKey(0), (10 * D + 1, 3))
        out = model.double(x, batch_size=4 * D)

        self.assertEqual(out.shape, x.shape)
        self.assertTrue(jnp.array_equal(out, 2 * x))

    def test_lazy_batches_line_up_with_layout(self):
        model = _Model()
        for n, b in LOCAL_SIZES:
            with self.subTest(n=n, b=b):
                x = _data(n * D)
                iterable = model.lazy_double(x, batch_size=b * D)

                self.assertEqual(iterable.layout, BatchLayout(n * D, b * D))
                self.assertEqual(len(iterable), iterable.layout.n_batches)
                for piece, expected in zip(iterable, iterable.layout.split(2 * x)):
                    self.assertTrue(jnp.array_equal(piece, expected))

    def test_lazy_without_batch_size(self):
        model = _Model()
        x = _data(10 * D)
        pieces = list(model.lazy_double(x, batch_size=None))

        self.assertEqual(len(pieces), 1)
        self.assertTrue(jnp.array_equal(pieces[0], 2 * x))

    def test_customised_specs_only_without_batch_size(self):
        model = _Model()
        x = _data(10 * D)

        self.assertTrue(jnp.array_equal(model.double_custom_specs(x, batch_size=None), 2 * x))
        with self.assertRaises(ValueError):
            model.double_custom_specs(x, batch_size=4 * D)

    @unittest.skipIf(D == 1, "Every number of samples is divisible by one device")
    def test_lazy_indivisible_number_of_samples_is_padded(self):
        model = _Model()
        x = jax.random.normal(jax.random.PRNGKey(0), (10 * D + 1, 3))
        iterable = model.lazy_double(x, batch_size=4 * D)

        self.assertEqual(iterable.layout, BatchLayout(len(pad_to_devices(x)), 4 * D))
        for piece, expected in zip(iterable, iterable.layout.split(pad_to_devices(2 * x))):
            self.assertTrue(jnp.array_equal(piece, expected))

class TestModule(unittest.TestCase):
    def test_P_is_partition_spec(self):
        self.assertIs(sharding_config.P, PartitionSpec)

if __name__ == "__main__":
    unittest.main()
