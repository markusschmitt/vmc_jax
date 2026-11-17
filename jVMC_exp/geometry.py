import jax.numpy as jnp
import numpy as np
import jax
from abc import ABC, abstractmethod

class AbstractGeometry(ABC):
    """
    Interface of the geometries of systems of particles in continuous space.

    A configuration holds the positions of all the particles in a flat array of shape \
    ``(n_particles * n_dim,)``, particle after particle: the coordinates of particle ``i`` are at the \
    indices ``i * n_dim`` to ``(i + 1) * n_dim - 1``, so ``x.reshape(n_particles, n_dim)`` has one \
    row per particle.

    Geometries compare equal when they are of the same class and have the same number of particles, \
    number of dimensions, boundary conditions and extent. Continuous operators can only be combined \
    if their geometries are equal.

    Args:
        * ``n_particles``: Number of particles.
        * ``n_dim``: Number of spatial dimensions.
        * ``PBC``: Whether the boundary conditions are periodic, either a ``bool`` for all the \
          directions or a sequence (tuple, list or array) with one ``bool`` per direction.
        * ``extent``: Tuple with the length of the system along each direction. If ``None``, the \
          length is 1 along every direction.
    """
    def __init__(self, n_particles, n_dim, PBC=True, extent=None):
        self._n_particles = n_particles
        self._n_dim = n_dim 

        if np.ndim(PBC) == 0:
            PBC = (PBC, ) * self.n_dim
        self._PBC = tuple(bool(p) for p in PBC)

        if extent is None:
            extent = (1,) * self.n_dim
        elif not isinstance(extent, tuple):
            raise ValueError(f'Extent has to be a tuple, got {extent}.')
        self._extent = extent

    def __eq__(self, other):
        if not isinstance(other, AbstractGeometry):
            return NotImplemented
        return (
            type(self) is type(other) and
            self.n_particles == other.n_particles and
            self.n_dim == other.n_dim and
            self.PBC == other.PBC and
            self.extent == other.extent
        )

    def __hash__(self):
        return hash((type(self), self.n_particles, self.n_dim, self.PBC, self.extent))

    @property
    def n_particles(self):
        """Number of particles."""
        return self._n_particles
    
    @property
    def n_dim(self):
        """Number of spatial dimensions."""
        return self._n_dim
    
    @property
    def PBC(self):
        """Tuple with one ``bool`` per direction, ``True`` where the boundary conditions are periodic."""
        return self._PBC
    
    @property
    def extent(self):
        """Tuple with the length of the system along each direction."""
        return self._extent
    
    def get_absolute_distance(self, x):
        """
        Distances between all pairs of particles, with the minimum image convention along the \
        periodic directions.

        Args:
            * ``x``: Configuration, of shape ``(n_particles * n_dim,)``.

        Returns:
            Array of shape ``(n_particles * (n_particles - 1) / 2,)`` with the distances \
            :math:`|r_i - r_j|` for :math:`i < j`, in the order of \
            ``jnp.triu_indices(n_particles, 1)``.
        """
        diff = self.get_distance(x)
        sqrd_dist = jnp.sum(diff**2, axis=-1)

        return jnp.sqrt(sqrd_dist[jnp.triu_indices(self.n_particles, 1)])
    
    @property
    @abstractmethod
    def domain(self):
        """
        Region of space of the system, as an array of shape ``(n_dim, 2)`` with the lower and the \
        upper bound along each direction.
        """
        pass
    
    @abstractmethod
    def apply_PBC(self, x):
        """
        Map the coordinates along the periodic directions back into the domain.

        Args:
            * ``x``: Configuration, or array of configurations, of shape \
              ``(..., n_particles * n_dim)``.

        Returns:
            Array of the same shape as ``x``, unchanged along the directions without periodic \
            boundary conditions.
        """
        pass

    @abstractmethod
    def uniform_populate(self, key, shape, dtype):
        """
        Draw configurations with the positions of the particles distributed uniformly over the \
        domain.

        ``MCSamplerCont`` calls it to start its chains.

        Args:
            * ``key``: An instance of ``jax.random.PRNGKey``.
            * ``shape``: Shape of the output, ``(..., n_particles * n_dim)``.
            * ``dtype``: Data type of the output.

        Returns:
            Array of configurations of shape ``shape``.
        """
        pass

    @abstractmethod
    def get_distance(self, x):
        """
        Displacement vectors between all pairs of particles, with the minimum image convention \
        along the periodic directions.

        Args:
            * ``x``: Configuration, of shape ``(n_particles * n_dim,)``.

        Returns:
            Array of shape ``(n_particles, n_particles, n_dim)`` whose entry ``[i, j]`` is \
            :math:`r_i - r_j`.
        """

class HyperRectangle(AbstractGeometry):
    """
    Hyperrectangle centered at the origin, with side lengths ``extent``.

    The domain is :math:`[-L_d/2, L_d/2]` along each direction :math:`d`, with :math:`L_d` the \
    entry ``d`` of ``extent``. Along the periodic directions, distances follow the minimum image \
    convention and ``apply_PBC()`` maps the coordinates into :math:`[-L_d/2, L_d/2)`. Along the \
    other directions, the particles can leave the domain, and the extent only sets the region where \
    the chains of ``MCSamplerCont`` start, the default step sizes of the proposers and the grid of \
    ``count_particles_in_cell()``.

    See :class:`~jVMC_exp.geometry.AbstractGeometry` for the arguments.

    Example:
        Three particles in a two-dimensional box, periodic along the first direction and open \
        along the second::

            geometry = jVMC_exp.geometry.HyperRectangle(3, 2, PBC=(True, False), extent=(4., 2.))
    """
    @property
    def domain(self):
        """
        Array of shape ``(n_dim, 2)`` with the bounds ``-extent / 2`` and ``extent / 2`` along each \
        direction.
        """
        ext = jnp.array(self.extent)

        return jnp.stack([-ext / 2, ext / 2], axis=1)
    
    def apply_PBC(self, x):
        """
        Map the coordinates along the periodic directions into :math:`[-L_d/2, L_d/2)`.

        Args:
            * ``x``: Configuration, or array of configurations, of shape \
              ``(..., n_particles * n_dim)``.

        Returns:
            Array of the same shape as ``x``, unchanged along the directions without periodic \
            boundary conditions.
        """
        extent = jnp.tile(jnp.array(self.extent), self.n_particles)
        PBC = jnp.tile(jnp.array(self.PBC, dtype=jnp.bool_), self.n_particles)

        return jnp.where(PBC, ((x + extent/2) % extent) - extent/2, x)
    
    def uniform_populate(self, key, shape, dtype=jnp.float64):
        """
        Draw configurations with each coordinate distributed uniformly between the bounds of the \
        domain along its direction.

        Args:
            * ``key``: An instance of ``jax.random.PRNGKey``.
            * ``shape``: Shape of the output, ``(..., n_particles * n_dim)``.
            * ``dtype``: Data type of the output.

        Returns:
            Array of configurations of shape ``shape``.
        """
        low = self.domain[:, 0]
        samples = jax.random.uniform(key, shape, dtype=dtype)

        low_flat = jnp.tile(low, self.n_particles)
        extent_flat = jnp.tile(jnp.array(self.extent), self.n_particles)

        return self.apply_PBC(low_flat + samples * extent_flat)
    
    def get_distance(self, x):
        x = x.reshape((self.n_particles, self.n_dim))
        diff = x[:, None, :] - x[None, :, :]
        extent = jnp.array(self.extent)
        PBC = jnp.array(self.PBC, dtype=bool)

        return diff - PBC * extent * jnp.round(diff / extent)
    
    def count_particles_in_cell(self, sample, linear_partition):
        """
        Count the particles in each cell of a regular grid over the domain.

        Along each direction ``d``, the domain is divided into ``linear_partition[d]`` cells of \
        equal length. A particle on the upper bound, or outside the domain, is counted in the \
        closest cell along each direction.

        Args:
            * ``sample``: Configuration, of shape ``(n_particles * n_dim,)``.
            * ``linear_partition``: Tuple with the number of cells along each direction.

        Returns:
            Integer array of shape ``linear_partition`` with the number of particles in each cell, \
            indexed by the cell indices along each direction.
        """
        sample = sample.reshape((self.n_particles, self.n_dim))
        delta = jnp.array(self.extent) / jnp.array(linear_partition)
        shifted = sample - self.domain[:, 0]              
        cell_indices = jnp.floor(shifted / delta).astype(jnp.int32)      

        # clip to handle particles on the upper boundary
        cell_indices = jnp.clip(
            cell_indices,
            0,
            jnp.array(linear_partition) - 1
        )

        flat_indices = jnp.ravel_multi_index(
            cell_indices.T,
            dims=linear_partition,
            mode='clip'
        )

        return jnp.bincount(flat_indices, length=np.prod(linear_partition)).reshape(linear_partition)