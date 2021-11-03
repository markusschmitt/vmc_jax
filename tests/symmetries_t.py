import sys
# Find jVMC package
sys.path.append(sys.path[0] + "/..")

import unittest

import jax
from jax.config import config
config.update("jax_enable_x64", True)
import jax.random as random
import jax.numpy as jnp

import numpy as np

import jVMC
import jVMC.util.symmetries as symmetries

import jVMC.global_defs as global_defs

import time


class TestSymmetries(unittest.TestCase):

    def test_symmetries2D(self):
        L = 3
        rotation_f = 4
        reflection_f = 2
        translation_f = L**2
        for rotation in [True, False]:
            for reflection in [True, False]:
                for translation in [True, False]:
                    orbit = symmetries.get_orbit_2d_square(L, rotation=rotation, reflection=reflection, translation=translation)
                    self.assertTrue(orbit.shape[0] == (rotation_f if rotation else 1) * (reflection_f if reflection else 1) * (translation_f if translation else 1))
                    self.assertTrue(np.issubdtype(orbit.dtype, np.integer))

    def test_symmetries1D(self):
        L = 3
        reflection_f = 2
        translation_f = L
        for translation in [True, False]:
            for reflection in [True, False]:
                orbit = symmetries.get_orbit_1d(L, reflection=reflection, translation=translation)
                self.assertTrue(orbit.shape[0] == (reflection_f if reflection else 1) * (translation_f if translation else 1))
                self.assertTrue(np.issubdtype(orbit.dtype, np.integer))


if __name__ == "__main__":
    unittest.main()
