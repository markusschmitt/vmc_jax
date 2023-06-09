import unittest

import jax
from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp

import numpy as np

import jVMC
import jVMC.util.symmetries as symmetries
import jVMC.nets as nets
from jVMC.vqs import NQS


class TestSymmetries(unittest.TestCase):

    def test_symmetries2D(self):
        L = 3
        for rotation in [True, False]:
            for reflection in [True, False]:
                for translation in [True, False]:
                    for spinflip in [True, False]:
                        symms = ("rotation" if rotation else None, 
                                 "reflection" if reflection else None,
                                 "translation" if translation else None,
                                 "spinflip" if spinflip else None)
                        
                        orbit = symmetries.get_orbit_2D_square(L, *symms)
                        self.assertTrue(np.issubdtype(orbit.dtype, np.integer))

    def test_symmetries1D(self):
        L = 3
        reflection_f = 2
        translation_f = L
        spinflip_f = 2
        for translation in [True, False]:
            for reflection in [True, False]:
                for spinflip in [True, False]:
                    symms = ("reflection" if reflection else None,
                             "translation" if translation else None,
                             "spinflip" if spinflip else None)
                    orbit = symmetries.get_orbit_1D(L, *symms)
                    self.assertTrue(orbit.shape[0] == (reflection_f if reflection else 1) * (translation_f if translation else 1) * (spinflip_f if spinflip else 1))
                    self.assertTrue(np.issubdtype(orbit.dtype, np.integer))


class TestSymmetrySector(unittest.TestCase):

    def test_symmetry_sector_1D(self):

        for reflection, translation, spinflip in [[True, False, False], [False, True, False], [False, False, True]]:
            L = 10
            symms = {"reflection": {"use": reflection, "factor": -1},
                     "translation": {"use": translation, "factor": jnp.exp(1j * 2 * jnp.pi * L / 2 / L)},
                     "spinflip": {"use": spinflip, "factor": -1},
                     }
            symms = ("reflection" if reflection else None,
                     "translation" if translation else None,
                     "spinflip" if spinflip else None)
            orbit = symmetries.get_orbit_1D(L, *symms,
                                              reflection_factor=-1.,
                                              translation_factor=jnp.exp(1j * 2 * jnp.pi * L / 2 / L),
                                              spinflip_factor=-1.)

            rbm = nets.CpxRBM_Nospinflip(numHidden=2, bias=False)
            net = nets.sym_wrapper.SymNet(net=rbm, orbit=orbit)

            psi = NQS(net)

            config_1 = jax.random.choice(jax.random.PRNGKey(0), jnp.array([0, 1] * (L // 2)), shape=(L,), replace=False)

            if reflection:
                config_2 = config_1[::-1]

            if translation:
                config_2 = jnp.roll(config_1, 1, axis=0)

            if spinflip:
                config_2 = 1 - config_1

            coeff_1 = psi(config_1[None, None, ...])
            coeff_2 = psi(config_2[None, None, ...])

            self.assertTrue(jnp.abs(1 + jnp.exp(coeff_1 - coeff_2)) < 1e-10)

    def test_symmetry_sector_2D(self):

        for rotation, reflection, translation, spinflip in [[True, False, False, False], [False, True, False, False], [False, False, True, False], [False, False, False, True]]:

            L = 4
            symms = ("rotation" if rotation else None, 
                     "reflection" if reflection else None,
                     "translation" if translation else None,
                     "spinflip" if spinflip else None)
            orbit = symmetries.get_orbit_2D_square(L, *symms,
                                                      rotation_factor=-1.,
                                                      reflection_factor=-1.,
                                                      translation_factor=jnp.exp(1j * 2 * jnp.pi * L / 2 / L),
                                                      spinflip_factor=-1.)

            rbm = nets.CpxRBM_Nospinflip(numHidden=2, bias=False)
            net = nets.sym_wrapper.SymNet(net=rbm, orbit=orbit)

            psi = NQS(net)

            config_1 = jax.random.choice(jax.random.PRNGKey(0), jnp.array([0, 1] * (L**2 // 2)), shape=(L, L), replace=False)

            if rotation:
                config_2 = config_1[:, ::-1].T

            if reflection:
                config_2 = config_1[::-1, :]

            if translation:
                config_2 = jnp.roll(config_1, 1, axis=0)

            if spinflip:
                config_2 = 1 - config_1

            coeff_1 = psi(config_1[None, None, ...])
            coeff_2 = psi(config_2[None, None, ...])

            self.assertTrue(jnp.abs(1 + jnp.exp(coeff_1 - coeff_2)) < 1e-10)


if __name__ == "__main__":
    unittest.main()
