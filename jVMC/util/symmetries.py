import numpy as np
import jax.numpy as jnp

def get_point_orbit_2d_square(L):
    ''' This function generates the group of point symmetries in a two-dimensional square lattice.

    Arguments:
        * ``L``: Linear dimension of the lattice.

    Returns:
        A three-dimensional ``jax.numpy.array``, where the first dimension corresponds to the different
        symmetry operations and the following two dimensions correspond to the corresponding permuation matrix.
    '''

    trafos = []

    idx = np.arange(L*L).reshape((L,L))

    for _ in range(2):
        for _ in range(4):
            idx = np.array(list(zip(*idx[::-1]))) # rotation
            trafos.append(idx)
        idx = np.transpose(idx) # reflection

    orbit = []

    idx = np.arange(L*L)

    for t in trafos:

        o = np.zeros((L*L,L*L), dtype=np.int32)

        o[idx,t.ravel()] = 1

        orbit.append(o)

    orbit = jnp.array(orbit)

    return orbit


def get_translation_orbit_2d_square(L):
    ''' This function generates the group of translations in a two-dimensional square lattice.

    Arguments:
        * ``L``: Linear dimension of the lattice.

    Returns:
        A three-dimensional ``jax.numpy.array``, where the first dimension corresponds to the different
        translations and the following two dimensions correspond to the corresponding permuation matrix.
    '''

    idx = np.arange(L**2, dtype=np.int32).reshape((L,L))

    trafos = []

    for lx in range(L):
        for ly in range(L):

            trafos.append(idx)

            idx = np.roll(idx, 1, axis=1)

        idx = np.roll(idx, 1, axis=0)

    orbit = []

    idx = np.arange(L*L)

    for t in trafos:

        o = np.zeros((L*L,L*L), dtype=np.int32)

        o[idx,t.ravel()] = 1

        orbit.append(o)

    orbit = jnp.array(orbit)

    return orbit


def get_orbit_2d_square(L):
    ''' This function generates the group of lattice symmetries in a two-dimensional square lattice.

    Arguments:
        * ``L``: Linear dimension of the lattice.

    Returns:
        A three-dimensional ``jax.numpy.array``, where the first dimension corresponds to the different
        symmetry operations and the following two dimensions correspond to the corresponding permuation matrix.
    '''

    po = get_point_orbit_2d_square(L)

    to = get_translation_orbit_2d_square(L)

    orbit = jax.vmap(lambda x,y: jax.vmap(lambda a,b: jnp.dot(b,a), in_axes=(None,0))(x,y), in_axes=(0,None))(to,po)

    orbit = orbit.reshape((-1,L**2,L**2))

    newOrbit = [tuple(x.ravel()) for x in orbit]

    uniqueOrbit = np.unique(newOrbit,axis=0).reshape(-1,L**2,L**2)

    return jnp.array(uniqueOrbit)
