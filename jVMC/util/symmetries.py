import numpy as np
import jax.numpy as jnp
import jax


def get_point_orbit_2d_square(L, rotation, reflection):
    ''' This function generates the group of point symmetries in a two-dimensional square lattice.

    Arguments:
        * ``L``: Linear dimension of the lattice.

    Returns:
        A three-dimensional ``jax.numpy.array``, where the first dimension corresponds to the different
        symmetry operations and the following two dimensions correspond to the corresponding permuation matrix.
    '''

    trafos = []

    idx = np.arange(L * L).reshape((L, L))

    for _ in range(2 if reflection else 1):
        for _ in range(4 if rotation else 1):
            trafos.append(idx)
            idx = np.array(list(zip(*idx[::-1])))  # rotation
        idx = np.transpose(idx)  # reflection

    orbit = []

    idx = np.arange(L * L)

    for t in trafos:

        o = np.zeros((L * L, L * L), dtype=np.int32)

        o[idx, t.ravel()] = 1

        orbit.append(o)

    orbit = jnp.array(orbit)

    return orbit


def get_translation_orbit_2d_square(L, translation):
    ''' This function generates the group of translations in a two-dimensional square lattice.

    Arguments:
        * ``L``: Linear dimension of the lattice.

    Returns:
        A three-dimensional ``jax.numpy.array``, where the first dimension corresponds to the different
        translations and the following two dimensions correspond to the corresponding permuation matrix.
    '''

    idx = np.arange(L**2, dtype=np.int32).reshape((L, L))

    trafos = []

    for lx in range(L if translation else 1):
        for ly in range(L if translation else 1):

            trafos.append(idx)

            idx = np.roll(idx, 1, axis=1)

        idx = np.roll(idx, 1, axis=0)

    orbit = []

    idx = np.arange(L * L)

    for t in trafos:

        o = np.zeros((L * L, L * L), dtype=np.int32)

        o[idx, t.ravel()] = 1

        orbit.append(o)

    orbit = jnp.array(orbit)

    return orbit


def get_orbit_2d_square(L, rotation=True, reflection=True, translation=True):
    ''' This function generates the group of lattice symmetries in a two-dimensional square lattice.

    Arguments:
        * ``L``: Linear dimension of the lattice.

    Returns:
        A three-dimensional ``jax.numpy.array``, where the first dimension corresponds to the different
        symmetry operations and the following two dimensions correspond to the corresponding permuation matrix.
    '''

    po = get_point_orbit_2d_square(L, rotation, reflection)

    to = get_translation_orbit_2d_square(L, translation)

    orbit = jax.vmap(lambda x, y: jax.vmap(lambda a, b: jnp.dot(b, a), in_axes=(None, 0))(x, y), in_axes=(0, None))(to, po)

    orbit = orbit.reshape((-1, L**2, L**2))

    newOrbit = [tuple(x.ravel()) for x in orbit]

    uniqueOrbit = np.unique(newOrbit, axis=0).reshape(-1, L**2, L**2)

    return jnp.array(uniqueOrbit)


def get_orbit_1d(L, translation=True, reflection=True, **kwargs):
    ''' This function generates the group of lattice symmetries in a one-dimensional lattice.

    Arguments:
        * ``L``: Linear dimension of the lattice.

    Returns:
        A three-dimensional ``jax.numpy.array``, where the first dimension corresponds to the different
        symmetry operations and the following two dimensions correspond to the corresponding permuation matrix.
    '''

    def get_point_orbit_1D(L, reflection):
        return jnp.array([jnp.eye(L), jnp.fliplr(jnp.eye(L))]) if reflection else jnp.array([jnp.eye(L)])

    def get_translation_orbit_1D(L, translation):
        to = np.array([np.eye(L)] * L)
        for idx, t in enumerate(to):
            to[idx] = np.roll(t, idx, axis=1)
        return jnp.array(to) if translation else jnp.array([jnp.eye(L)])

    po = get_point_orbit_1D(L, reflection)
    to = get_translation_orbit_1D(L, translation)
    orbit = jax.vmap(lambda x, y: jax.vmap(lambda a, b: jnp.dot(a, b), in_axes=(None, 0))(x, y), in_axes=(0, None))(to, po)

    orbit = orbit.reshape((-1, L, L))
    return orbit
