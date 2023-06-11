import numpy as np
import jax.numpy as jnp
import jax
from dataclasses import dataclass

import warnings


class LatticeSymmetry:
    '''A wrapper class for lattice symmetries.

    Wrapping the ``orbit`` in this class is a workaround to make the ``jax.numpy.array``
    hashable so that they can be passed as arguments to Flax modules.

    Initializer arguments:
        * ``orbit``: A ``jax.numpy.array`` of rank three, which contains the permutation matrix \
                     of lattice indices for each symmetry operation.
        * ``factor``: A ``jax.numpy.array`` of rank one, which contains the phase factor associated with \
                    each element in the ``orbit`` due to the chosen quantum number.
    '''

    def __init__(self, orbit, factor):
        self._orbit = orbit
        self._factor = factor

    @property
    def factor(self):
        return self._factor

    @property
    def orbit(self):
        return self._orbit

    @property
    def shape(self):
        return self._orbit.shape

    @property
    def dtype(self):
        return self._orbit.dtype


def get_2D_index(L):
    return np.arange(L**2).reshape((L, L))


def get_1D_index(L):
    return np.arange(L)


def get_orbits_from_maps_2D(L, translation, rotation, reflection):
    """
    Given the site ids that are displaced by the desired symmetry operation,
    obtain the matrices that perform the desired symmetry operation.
    """
    idx = get_2D_index(L)

    orbits, factors = [], []
    for map_t, fac_t in zip(translation["maps"], translation["factor"]):
        for map_ref, fac_ref in zip(reflection["maps"], reflection["factor"]):
            for map_rot, fac_rot in zip(rotation["maps"], rotation["factor"]):
                orbit_t = np.zeros((L**2, L**2), dtype=np.int32)
                orbit_t[idx.ravel(), map_t.ravel()] = 1

                orbit_ref = np.zeros((L**2, L**2), dtype=np.int32)
                orbit_ref[idx.ravel(), map_ref.ravel()] = 1

                orbit_rot = np.zeros((L**2, L**2), dtype=np.int32)
                orbit_rot[idx.ravel(), map_rot.ravel()] = 1

                orbits.append(orbit_t @ orbit_ref @ orbit_rot)
                factors.append(fac_t * fac_ref * fac_rot)

    return jnp.array(orbits), jnp.array(factors)


def get_maps_2D(L, **kwargs):
    """
    Compute the effect of symmetry operations on a 2D square lattice.
    """
    idx = get_2D_index(L)

    rotation = {"maps": [idx], "factor": [1.]}
    reflection = {"maps": [idx], "factor": [1.]}
    translation = {"maps": [idx], "factor": [1.]}

    if kwargs["rotation"]["use"]:
        rotation["maps"].append(idx[::-1, :].T)
        rotation["factor"].append(kwargs["rotation"]["factor"])
        rotation["maps"].append(idx[::-1, ::-1])
        rotation["factor"].append(kwargs["rotation"]["factor"]**2)
        rotation["maps"].append(idx[:, ::-1].T)
        rotation["factor"].append(kwargs["rotation"]["factor"]**3)

    if kwargs["reflection"]["use"]:
        reflection["maps"].append(idx[::-1, :])
        reflection["factor"].append(kwargs["reflection"]["factor"])
        reflection["maps"].append(idx[:, ::-1])
        reflection["factor"].append(kwargs["reflection"]["factor"])
        reflection["maps"].append(idx[::-1, ::-1])
        reflection["factor"].append(kwargs["reflection"]["factor"]**2)

    if kwargs["translation"]["use"]:
        for x in range(L):
            for y in range(L):
                if x == 0 and y == 0:
                    continue
                translation["maps"].append(np.roll(idx, (y, x), axis=(0, 1)))
                translation["factor"].append(kwargs["translation"]["factor"]**(x + y))

    return translation, reflection, rotation


def get_orbit_2D_square(L, *args, translation_factor=1., reflection_factor=1., rotation_factor=1., spinflip_factor=1.):
    ''' This function generates the group of lattice symmetries in a two-dimensional square lattice.

    Arguments:
        * ``L``: Linear dimension of the lattice.
        * ``*args``: Any choice of "reflection", "rotation", "translation", or "spinflip".
        * ``*_factor``: Prefactor for the symmetrization corresponding to the quantum number. \
                        Replace ``*`` by the name of the symmetry ("reflection", "rotation", "translation", or "spinflip").

    Returns:
        An object of type ``LatticeSymmetry``.
    '''
    
    kwargs = {"reflection": {"use": ("reflection" in args), "factor": reflection_factor},
              "translation": {"use": ("translation" in args), "factor": translation_factor},
              "rotation": {"use": ("rotation" in args), "factor": rotation_factor},
              "spinflip": {"use": ("spinflip" in args), "factor": spinflip_factor},
             }

    translation, reflection, rotation = get_maps_2D(L, **kwargs)
    orbits, factors = get_orbits_from_maps_2D(L, translation, reflection, rotation)

    if kwargs["spinflip"]["use"]:
        warnings.warn("Symmetry 'spinflip' assumes that spin up/down is encoded with numerical values +1/-1.")
        orbits = jnp.concatenate([orbits, - orbits], axis=0)
        factors = jnp.concatenate([factors, kwargs["spinflip"]["factor"] * factors])

    uniqueOrbit, indices = np.unique(orbits.reshape((-1, L**4)), return_index=True, axis=0)
    factors = factors[indices]
    return LatticeSymmetry(jnp.array(uniqueOrbit).reshape(-1, L**2, L**2), factors)


def get_orbits_from_maps_1D(L, translation, reflection):
    """
    Given the site ids that are displaced by the desired symmetry operation,
    obtain the matrices that perform the desired symmetry operation.
    """
    idx = get_1D_index(L)

    orbits, factors = [], []
    for map_t, fac_t in zip(translation["maps"], translation["factor"]):
        for map_ref, fac_ref in zip(reflection["maps"], reflection["factor"]):
            orbit_t = np.zeros((L, L), dtype=np.int32)
            orbit_t[idx.ravel(), map_t.ravel()] = 1

            orbit_ref = np.zeros((L, L), dtype=np.int32)
            orbit_ref[idx.ravel(), map_ref.ravel()] = 1

            orbits.append(orbit_t @ orbit_ref)
            factors.append(fac_t * fac_ref)

    return jnp.array(orbits), jnp.array(factors)


def get_maps_1D(L, **kwargs):
    """
    Compute the effect of symmetry operations on a 2D square lattice.
    """
    idx = get_1D_index(L)

    reflection = {"maps": [idx], "factor": [1.]}
    translation = {"maps": [idx], "factor": [1.]}

    if kwargs["reflection"]["use"]:
        reflection["maps"].append(idx[::-1])
        reflection["factor"].append(kwargs["reflection"]["factor"])

    if kwargs["translation"]["use"]:
        for x in range(L):
            if x == 0:
                continue
            translation["maps"].append(np.roll(idx, x, axis=(0,)))
            translation["factor"].append(kwargs["translation"]["factor"]**x)

    return translation, reflection


def get_orbit_1D(L, *args, translation_factor=1., reflection_factor=1., spinflip_factor=1.):
    ''' This function generates the group of lattice symmetries in a one-dimensional lattice.

    Arguments:
        * ``L``: Linear dimension of the lattice.
        * ``*args``: Any choice of "reflection", "rotation", "translation", or "spinflip".
        * ``*_factor``: Prefactor for the symmetrization corresponding to the quantum number. \
                        Replace ``*`` by the name of the symmetry ("reflection", "rotation", "translation", or "spinflip").

    Returns:
        An object of type ``LatticeSymmetry``.
    '''

    kwargs = {"reflection": {"use": ("reflection" in args), "factor": reflection_factor},
              "translation": {"use": ("translation" in args), "factor": translation_factor},
              "spinflip": {"use": ("spinflip" in args), "factor": spinflip_factor},
             }

    translation, reflection = get_maps_1D(L, **kwargs)
    orbits, factors = get_orbits_from_maps_1D(L, translation, reflection)

    if kwargs["spinflip"]["use"]:
        warnings.warn("Symmetry 'spinflip' assumes that spin up/down is encoded with numerical values +1/-1.")
        orbits = jnp.concatenate([orbits, - orbits], axis=0)
        factors = jnp.concatenate([factors, kwargs["spinflip"]["factor"] * factors])

    uniqueOrbit, indices = np.unique(orbits.reshape((-1, L**2)), return_index=True, axis=0)
    factors = factors[indices]
    return LatticeSymmetry(jnp.array(uniqueOrbit).reshape(-1, L, L), factors)


if __name__ == "__main__":
    L = 3
    syms = {"rotation": {"use": True, "factor": 1},
            "reflection": {"use": False, "factor": 1},
            "translation": {"use": False, "factor": 1},
            "spinflip": {"use": False, "factor": 1}}

    latsym = get_orbit_2D_square(L, **syms)
    print(latsym.orbit.shape)

    idx = np.arange(L**2).reshape((L, L))
    for o in latsym.orbit:
        print((o @ idx.ravel()).reshape((L, L)))
