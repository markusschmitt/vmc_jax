import unittest
import jax
import jax.numpy as jnp
import numpy as np

from jVMC_exp.geometry import HyperRectangle

class TestUniformPopulate(unittest.TestCase):
    def test_positions_are_uniform_over_the_domain_with_different_extents(self):
        for PBC in (False, True):
            geometry = HyperRectangle(3, 2, PBC=PBC, extent=(6., 4.))
            num_samples = 20000
            x = geometry.uniform_populate(jax.random.PRNGKey(0), (num_samples, 6), jnp.float64)
            x = np.asarray(x).reshape(num_samples, 3, 2)

            for d, L in enumerate(geometry.extent):
                for p in range(geometry.n_particles):
                    counts, _ = np.histogram(x[:, p, d], bins=4, range=(-L / 2, L / 2))
                    # All positions are inside the domain and spread evenly over it
                    self.assertEqual(counts.sum(), num_samples)
                    np.testing.assert_allclose(counts / num_samples, 0.25, atol=0.02)

class TestPBC(unittest.TestCase):
    def test_accepted_forms_are_stored_as_a_tuple_of_bools(self):
        for PBC in [(True, False), [True, False], np.array([True, False]), jnp.array([True, False])]:
            geometry = HyperRectangle(2, 2, PBC=PBC)
            self.assertEqual(geometry.PBC, (True, False))
            self.assertTrue(all(type(p) is bool for p in geometry.PBC))
            self.assertEqual(geometry, HyperRectangle(2, 2, PBC=(True, False)))
            self.assertEqual(hash(geometry), hash(HyperRectangle(2, 2, PBC=(True, False))))

            x = jnp.array([0.7, 0.7, 0.2, 0.2])
            np.testing.assert_allclose(geometry.apply_PBC(x), [-0.3, 0.7, 0.2, 0.2])

        for PBC in [True, np.bool_(True), np.array(True), jnp.array(True)]:
            geometry = HyperRectangle(2, 2, PBC=PBC)
            self.assertEqual(geometry.PBC, (True, True))
            self.assertTrue(all(type(p) is bool for p in geometry.PBC))

class TestEquality(unittest.TestCase):
    def test_equal_geometries_have_equal_hashes(self):
        a = HyperRectangle(2, 2)
        b = HyperRectangle(2, 2, PBC=(True, True), extent=(1., 1.))

        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))
        self.assertEqual(len({a, b}), 1)

    def test_different_geometries_are_not_equal(self):
        class Box(HyperRectangle):
            pass

        a = HyperRectangle(2, 2)
        self.assertNotEqual(a, Box(2, 2))
        self.assertNotEqual(a, HyperRectangle(3, 2))
        self.assertNotEqual(a, HyperRectangle(2, 2, PBC=(True, False)))
        self.assertNotEqual(a, HyperRectangle(2, 2, extent=(1., 2.)))

if __name__ == '__main__':
    unittest.main()
