from __future__ import annotations
import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
from dataclasses import dataclass
import warnings

from .util import deduplicate_symmetry_arrays, array_tuple, vector_tuple, ParticleType, normalize_particle_type
from .average_fun import average_coefficients, SymmetryAverage
from jVMC_exp.util.util import has_callable_attr

def _has_autoregressive_sampler(module) -> bool:
    if bool(getattr(module, "sampler", False)):
        return True
    if callable(getattr(module, "sample", None)):
        return True
    wrapped = getattr(module, "net", None)
    return wrapped is not None and _has_autoregressive_sampler(wrapped)

def _local_dimension(symm: SymmetryProjector) -> int | None:
    if symm.local_map is not None:
        return int(symm.local_map_array.shape[1])
    if symm.local_phase is not None:
        return int(symm.local_phase_array.shape[1])
    return None

def _compose_local_maps(left: SymmetryProjector, right: SymmetryProjector):
    left_map = left.local_map_array
    right_map = right.local_map_array
    if left_map is None and right_map is None:
        return None
    if left_map is None:
        return jnp.repeat(right_map[None, :, :], left.nsymm, axis=0).reshape(left.nsymm * right.nsymm, -1)
    if right_map is None:
        return jnp.repeat(left_map[:, None, :], right.nsymm, axis=1).reshape(left.nsymm * right.nsymm, -1)
    if left_map.shape[1] != right_map.shape[1]:
        raise ValueError("Cannot compose local maps with different local dimensions.")
    rows = []
    for left_idx in range(left.nsymm):
        for right_idx in range(right.nsymm):
            rows.append(right_map[right_idx][left_map[left_idx]])
    
    return jnp.stack(rows, axis=0)

def _compose_local_phases(left: SymmetryProjector, right: SymmetryProjector):
    if left.local_phase is None and right.local_phase is None:
        return None
    left_dim = _local_dimension(left)
    right_dim = _local_dimension(right)
    local_dim = left_dim if left_dim is not None else right_dim
    if right_dim is not None and local_dim != right_dim:
        raise ValueError("Cannot compose local phases with different local dimensions.")

    identity = jnp.arange(local_dim, dtype=jnp.int32)
    left_map = left.local_map_array
    if left_map is None:
        left_map = jnp.repeat(identity[None, :], left.nsymm, axis=0)
    left_phase = left.local_phase_array
    if left_phase is None:
        left_phase = jnp.ones((left.nsymm, local_dim), dtype=jnp.complex128)
    right_phase = right.local_phase_array
    if right_phase is None:
        right_phase = jnp.ones((right.nsymm, local_dim), dtype=jnp.complex128)

    rows = []
    for left_idx in range(left.nsymm):
        for right_idx in range(right.nsymm):
            rows.append(left_phase[left_idx] * right_phase[right_idx][left_map[left_idx]])
    
    return jnp.stack(rows, axis=0)

def _compose_local_signs(left: SymmetryProjector, right: SymmetryProjector):
    return jnp.bitwise_xor(left.local_sign_array[:, None], right.local_sign_array[None, :]).reshape(-1)

def compose_projectors(
        left: SymmetryProjector, right: SymmetryProjector,
        *, name: str = "composed"
    ) -> SymmetryProjector:
    """
    Product of two symmetry projectors.

    The elements of the result are the pairs ``(l, r)`` of the elements of ``left`` and \
    ``right``, with ``r`` running fastest: a configuration is transformed by the element ``l`` \
    and then by the element ``r``, with the character :math:`\\chi_l\\chi_r`. The boundary signs \
    of the element ``(l, r)`` on the site ``i`` are \
    ``left.perm_sign[l, right.perm[r, i]] * right.perm_sign[r, i]``. Elements that coincide are \
    merged, as in ``deduplicate_symmetry_arrays()``. Without boundary signs, the elements are \
    the products :math:`A_lA_r` of the elements of ``left`` and ``right`` (see \
    ``SymmetryProjector``), and the projection is the product :math:`P_lP_r` of the two \
    projections, which is again a projection if they commute. ``left * right`` is a shorthand, \
    with the name ``"<left>_x_<right>"``.

    Args:
        * ``left``: ``SymmetryProjector``.
        * ``right``: ``SymmetryProjector`` on the same lattice, with the same particle type.
        * ``name``: Name of the result.

    Returns:
        The composed ``SymmetryProjector``. Raises ``ValueError`` if the lattices or particle \
        types differ.
    """
    if (left.Lx, left.Ly, left.particle_type) != (right.Lx, right.Ly, right.particle_type):
        raise ValueError("Only compatible symmetries can be composed.")
    left_perm = left.perm_array
    right_perm = right.perm_array
    perm = left_perm[:, right_perm].reshape(left.nsymm * right.nsymm, left.nmodes)
    character = (left.character_array[:, None] * right.character_array[None, :]).reshape(-1)
    perm_sign = left.perm_sign_array[:, right_perm] * right.perm_sign_array[None, :, :]
    perm_sign = perm_sign.reshape(left.nsymm * right.nsymm, left.nmodes)
    local_map = _compose_local_maps(left, right)
    local_phase = _compose_local_phases(left, right)
    local_sign = _compose_local_signs(left, right)

    keep, character_unique = deduplicate_symmetry_arrays(
        perm,
        character,
        local_map=local_map,
        local_phase=local_phase,
        local_sign=local_sign,
        perm_sign=perm_sign,
    )

    return SymmetryProjector(
        Lx=left.Lx,
        Ly=left.Ly,
        particle_type=left.particle_type,
        perm=array_tuple(perm[keep], jnp.int32),
        character=vector_tuple(character_unique, jnp.complex128),
        perm_sign=array_tuple(perm_sign[keep], jnp.int8),
        local_map=None if local_map is None else array_tuple(local_map[keep], jnp.int32),
        local_phase=None if local_phase is None else array_tuple(local_phase[keep], jnp.complex128),
        local_sign=tuple(jnp.asarray(local_sign[keep], dtype=jnp.int8).tolist()),
        name=name,
    )

def split_spinful_site_state(s, n_sites: int):
    """
    Split a spinful fermion configuration into its up and down occupations.

    A spinful configuration has one local state per site: 0 for an empty site, 1 for spin up, \
    2 for spin down and 3 for a doubly occupied site. The fermionic signs use this \
    representation expanded into the modes ``(N-1, down), (N-1, up), ..., (0, down), (0, up)``.

    Args:
        * ``s``: Configuration with ``n_sites`` local states.
        * ``n_sites``: Number of sites.

    Returns:
        ``up, down``: Occupations (0 or 1) of the two spins on each site, of shape ``(n_sites,)``.
    """
    site_state = jnp.asarray(s, dtype=jnp.int32).reshape(int(n_sites)) # flatten and adjust type
    up = jnp.bitwise_and(site_state, 1)
    down = jnp.bitwise_and(jnp.right_shift(site_state, 1), 1)

    return up, down

def _spinful_local_occupancy(s, n_sites: int):
    up, down = split_spinful_site_state(s, n_sites)
    site_order = jnp.arange(int(n_sites) - 1, -1, -1)

    return jnp.stack([down[site_order], up[site_order]], axis=-1).reshape(2 * int(n_sites)) > 0

def _spinless_local_occupancy(s, n_sites: int):
    site_order = jnp.arange(int(n_sites) - 1, -1, -1)

    return jnp.asarray(s, dtype=jnp.int32).reshape(int(n_sites))[site_order] > 0

def _site_perm_to_spinless_orbital_perm(site_perm, n_sites: int):
    site_perm = jnp.asarray(site_perm, dtype=jnp.int32)
    out_sites = jnp.arange(int(n_sites) - 1, -1, -1)

    return int(n_sites) - 1 - site_perm[..., out_sites]

def _site_perm_to_spinful_orbital_perm(site_perm, n_sites: int):
    site_perm = jnp.asarray(site_perm, dtype=jnp.int32)
    out_sites = jnp.arange(int(n_sites) - 1, -1, -1)
    input_sites = site_perm[..., out_sites]
    down_pos = 2 * (int(n_sites) - 1 - input_sites)
    up_pos = down_pos + 1

    return jnp.stack([down_pos, up_pos], axis=-1).reshape(*site_perm.shape[:-1], 2 * int(n_sites))

def _spinful_site_sign_to_orbital_sign(site_sign, n_sites: int):
    site_sign = jnp.asarray(site_sign, dtype=jnp.int32)
    site_order = jnp.arange(int(n_sites) - 1, -1, -1)

    return jnp.repeat(site_sign[..., site_order], 2, axis=-1)

def _fermion_particle_hole_sign_from_occ(occ):
    r"""
    Fock basis sign for the fermionic particle-hole transform

    P_{ph} c_i P_{ph}^{-1} = c_i^\dagger
    P_{ph} c_i^\dagger P_{ph}^{-1} = c_i
    where P_{ph} acts on the state as 
    P_{ph} |0> = |full>
    and the full reference state
    |full> = c_0^\dagger c_1^\dagger ... c_{M-1}^\dagger |0>
    """
    occ = jnp.asarray(occ, dtype=jnp.int32)
    mode_idx = jnp.arange(occ.shape[-1], dtype=jnp.int32)
    exponent = jnp.sum(occ * mode_idx, axis=-1)

    return (1 - 2 * jnp.bitwise_and(exponent, 1)).astype(jnp.complex128)

@dataclass(frozen=True)
class SymmetryProjector:
    """
    Projector on a symmetry sector, acting on configurations, with fermionic signs.

    The projector is defined by the ``nsymm`` elements :math:`g` of a symmetry group and their \
    characters :math:`\\chi_g`. The element :math:`g` maps a configuration :math:`s` to \
    :math:`g\\cdot s`: the sites are permuted, ``s[perm[g]]``, and then the local states are \
    mapped by ``local_map[g]``. ``ProjectedOrbitNet`` projects a wave function as

    .. math::

        \\psi_P(s)=\\frac{1}{n_\\text{symm}}\\sum_g w_g(s)\\,\\psi(g\\cdot s),\\qquad
        w_g(s)=\\chi_g\\,\\sigma_g(s).

    For spins and bosons the sign :math:`\\sigma_g(s)` is one. For fermions it is the sign of \
    the permutation of the occupied modes, in the mode order of the fermionic operators of \
    ``jVMC_exp.operator``, ``(N-1, down), (N-1, up), ..., (0, down), (0, up)`` for ``N`` sites, \
    times the boundary signs ``perm_sign``, the local phases ``local_phase`` and, for the \
    elements with ``local_sign``, the particle-hole sign. Without local maps, the matrix \
    :math:`A_g` with :math:`(A_g\\psi)(s)=\\sigma_g(s)\\psi(g\\cdot s)` then maps the fermion \
    operators as :math:`A_g\\hat c_iA_g^\\dagger=\\hat c_{\\mathrm{perm}[g,i]}`, for each spin.

    The functions of ``jVMC_exp.symmetry.lattice_symetries`` build the projectors of the usual \
    lattice symmetries. ``projector_1 * projector_2`` composes two projectors (see \
    ``compose_projectors()``), and ``projector * net`` returns the ``ProjectedOrbitNet`` of a \
    network.

    A configuration has one local state per site, with the sites of a two-dimensional lattice \
    in the order of its site enumeration (see ``jVMC_exp.symmetry.lattice_symetries``). For \
    spinful fermions the local states 0, 1, 2 and 3 stand for an empty site, a spin up, a spin \
    down and a doubly occupied site.

    The fields are stored as tuples, so that a projector is hashable and can be a field of a \
    Flax module. The ``*_array`` properties return them as arrays.

    Args:
        * ``Lx``, ``Ly``: Size of the lattice.
        * ``particle_type``: ``"spin"``, ``"boson"``, ``"spinless_fermion"`` or \
          ``"spinful_fermion"``. The plural forms are accepted.
        * ``perm``: Site permutations, of shape ``(nsymm, n_sites)``: the element ``g`` maps \
          ``s`` to ``s[perm[g]]``.
        * ``character``: Characters :math:`\\chi_g`, of shape ``(nsymm,)``.
        * ``perm_sign`` [optional]: Boundary signs, +1 or -1, of shape ``(nsymm, n_sites)``: \
          under the element ``g``, a fermion on the site ``i`` of ``s`` gets the sign \
          ``perm_sign[g, i]``, e.g. -1 for the fermions that cross an antiperiodic boundary. \
          All +1 by default.
        * ``local_map`` [optional]: Maps of the local states, of shape ``(nsymm, local_dim)``, \
          applied after the permutation: the local state ``v`` becomes ``local_map[g, v]``.
        * ``local_phase`` [optional]: Phases of the local states, of shape \
          ``(nsymm, local_dim)``: for fermions, the element ``g`` multiplies \
          :math:`\\sigma_g(s)` by the product of ``local_phase[g, v]`` over the local states \
          ``v`` of ``s[perm[g]]``.
        * ``local_sign`` [optional]: 1 for the elements that include the fermionic \
          particle-hole transformation, 0 for the others, of shape ``(nsymm,)``.
        * ``name``: Name of the projector.

    Raises ``ValueError`` if the shapes do not match, or if ``local_sign`` is not 0 or 1.
    """
    Lx: int
    Ly: int
    particle_type: ParticleType
    perm: tuple[tuple[int, ...], ...]
    character: tuple[complex, ...]
    perm_sign: tuple[tuple[int, ...], ...] | None = None
    local_map: tuple[tuple[int, ...], ...] | None = None
    local_phase: tuple[tuple[complex, ...], ...] | None = None
    local_sign: tuple[int, ...] | None = None
    name: str = "custom"

    def __post_init__(self):
        particle_type = normalize_particle_type(self.particle_type)
        perm_arr = jnp.asarray(self.perm, dtype=jnp.int32)
        character_arr = jnp.asarray(self.character, dtype=jnp.complex128)
        if perm_arr.ndim != 2:
            raise ValueError("perm must have shape (nsymm, nmodes).")
        if character_arr.shape != (perm_arr.shape[0],):
            raise ValueError("character must have shape (nsymm,).")
        if self.perm_sign is None:
            perm_sign_arr = jnp.ones_like(perm_arr, dtype=jnp.int8)
        else:
            perm_sign_arr = jnp.asarray(self.perm_sign, dtype=jnp.int8)
            if perm_sign_arr.shape != perm_arr.shape:
                raise ValueError("perm_sign must have the same shape as perm.")
        if self.local_map is None:
            local_map_arr = None
        else:
            local_map_arr = jnp.asarray(self.local_map, dtype=jnp.int32)
            if local_map_arr.ndim != 2 or local_map_arr.shape[0] != perm_arr.shape[0]:
                raise ValueError("local_map must have shape (nsymm, local_dim).")
        if self.local_phase is None:
            local_phase_arr = None
        else:
            local_phase_arr = jnp.asarray(self.local_phase, dtype=jnp.complex128)
            if local_phase_arr.ndim != 2 or local_phase_arr.shape[0] != perm_arr.shape[0]:
                raise ValueError("local_phase must have shape (nsymm, local_dim).")
            if local_map_arr is not None and local_phase_arr.shape[1] != local_map_arr.shape[1]:
                raise ValueError("local_phase and local_map must have the same local dimension.")
        if self.local_sign is None:
            local_sign_arr = jnp.zeros((perm_arr.shape[0],), dtype=jnp.int8)
        else:
            local_sign_arr = jnp.asarray(self.local_sign, dtype=jnp.int8)
            if local_sign_arr.shape != (perm_arr.shape[0],):
                raise ValueError("local_sign must have shape (nsymm,).")
            if bool(jnp.any((local_sign_arr != 0) & (local_sign_arr != 1))):
                raise ValueError("local_sign entries must be 0 or 1.")
        object.__setattr__(self, "perm", array_tuple(perm_arr, jnp.int32))
        object.__setattr__(self, "character", vector_tuple(character_arr, jnp.complex128))
        object.__setattr__(self, "perm_sign", array_tuple(perm_sign_arr, jnp.int8))
        if local_map_arr is not None:
            object.__setattr__(self, "local_map", array_tuple(local_map_arr, jnp.int32))
        if local_phase_arr is not None:
            object.__setattr__(self, "local_phase", array_tuple(local_phase_arr, jnp.complex128))
        object.__setattr__(self, "local_sign", tuple(jnp.asarray(local_sign_arr, dtype=jnp.int8).tolist()))
        object.__setattr__(self, "particle_type", particle_type)

    @property
    def perm_array(self):
        """``perm`` as an array of shape ``(nsymm, n_sites)``."""
        return jnp.asarray(self.perm, dtype=jnp.int32)

    @property
    def character_array(self):
        """``character`` as an array of shape ``(nsymm,)``."""
        return jnp.asarray(self.character, dtype=jnp.complex128)

    @property
    def perm_sign_array(self):
        """``perm_sign`` as an array of shape ``(nsymm, n_sites)``."""
        return jnp.asarray(self.perm_sign, dtype=jnp.int8)

    @property
    def local_map_array(self):
        """``local_map`` as an array of shape ``(nsymm, local_dim)``, or ``None``."""
        if self.local_map is None:
            return None
        return jnp.asarray(self.local_map, dtype=jnp.int32)

    @property
    def local_phase_array(self):
        """``local_phase`` as an array of shape ``(nsymm, local_dim)``, or ``None``."""
        if self.local_phase is None:
            return None
        return jnp.asarray(self.local_phase, dtype=jnp.complex128)

    @property
    def local_sign_array(self):
        """``local_sign`` as an array of shape ``(nsymm,)``."""
        return jnp.asarray(self.local_sign, dtype=jnp.int8)

    @property
    def nsymm(self) -> int:
        """Number of elements."""
        return int(len(self.perm))

    @property
    def nmodes(self) -> int:
        """Number of sites ``n_sites``, i.e. of local states in a configuration."""
        return int(len(self.perm[0]))

    @property
    def is_fermion(self) -> bool:
        """Whether the particles are fermions, spinless or spinful."""
        return self.particle_type in {"spinless_fermion", "spinful_fermion"}

    # Useful to determine if  autoregressive networks can be used
    @property
    def has_nontrivial_projector_weights(self) -> bool:
        """
        Whether a character, boundary sign or local phase differs from one, an element includes \
        the particle-hole transformation, or the sites of fermions are permuted.

        Autoregressive networks can then not be projected (see ``ProjectedOrbitNet``).
        """
        character = np.asarray(self.character, dtype=np.complex128)
        perm = np.asarray(self.perm, dtype=np.int32)
        perm_sign = np.asarray(self.perm_sign, dtype=np.int8)
        local_sign = np.asarray(self.local_sign, dtype=np.int8)
        local_phase = None if self.local_phase is None else np.asarray(self.local_phase, dtype=np.complex128)
        identity = np.arange(self.nmodes, dtype=np.int32)
        return bool(
            (not np.allclose(character, 1.0 + 0.0j))
            or np.any(perm_sign != 1)
            or np.any(local_sign != 0)
            or (local_phase is not None and not np.allclose(local_phase, 1.0 + 0.0j))
            or (self.is_fermion and np.any(perm != identity[None, :]))
        )

    # Used to determine if autoregressive networks are egligible for symmetrization
    @property
    def has_nontrivial_factors(self) -> bool:
        """Same as ``has_nontrivial_projector_weights``."""
        return self.has_nontrivial_projector_weights

    def _apply_local_map(self, states, idx=None):
        if self.local_map is None:
            return states
        states = jnp.asarray(states, dtype=jnp.int32)
        if idx is None:
            local_map = self.local_map_array
            return jnp.take_along_axis(local_map, states, axis=1)
        return self.local_map_array[idx][states]

    def transformed_states(self, s):
        """
        All the transformed configurations :math:`g\\cdot s`.

        Args:
            * ``s``: Configuration with ``n_sites`` local states, of any shape.

        Returns:
            Array of shape ``(nsymm, n_sites)``.
        """
        s = jnp.asarray(s).reshape(self.nmodes)
        return self._apply_local_map(s[self.perm_array])

    def transformed_state_at(self, s, idx):
        """
        Transformed configuration :math:`g\\cdot s` for the element ``idx``.

        Args:
            * ``s``: Configuration with ``n_sites`` local states, of any shape.
            * ``idx``: Index of the element.

        Returns:
            Array of shape ``(n_sites,)``.
        """
        s = jnp.asarray(s).reshape(self.nmodes)
        return self._apply_local_map(s[self.perm_array[idx]], idx)

    def random_transformed_state(self, s, key):
        """
        Transformed configuration :math:`g\\cdot s` for an element drawn uniformly at random.

        Args:
            * ``s``: Configuration with ``n_sites`` local states, of any shape.
            * ``key``: ``jax.random`` key.

        Returns:
            Array of shape ``(n_sites,)``.
        """
        idx = jax.random.randint(key, (), 0, self.nsymm)
        return self.transformed_state_at(s, idx)

    def random_transformed_states(self, samples, key):
        """
        ``random_transformed_state()`` for each configuration of a batch, with independent elements.

        Args:
            * ``samples``: Configurations, of shape ``(N, ...)``.
            * ``key``: ``jax.random`` key.

        Returns:
            Array of shape ``(N, n_sites)``.
        """
        keys = jax.random.split(key, samples.shape[0])
        return jax.vmap(self.random_transformed_state)(samples, keys)

    def _fermion_occupancy_perm_sign(self, s, idx=None):
        if self.particle_type == "spinful_fermion":
            occ = _spinful_local_occupancy(s, self.nmodes)
            site_perm = self.perm_array if idx is None else self.perm_array[idx]
            perm = _site_perm_to_spinful_orbital_perm(site_perm, self.nmodes)
            sign = self.perm_sign_array if idx is None else self.perm_sign_array[idx]
            perm_sign = _spinful_site_sign_to_orbital_sign(sign, self.nmodes)
        else:
            occ = _spinless_local_occupancy(s, self.nmodes)
            perm = self.perm_array if idx is None else self.perm_array[idx]
            perm = _site_perm_to_spinless_orbital_perm(perm, self.nmodes)
            sign = self.perm_sign_array if idx is None else self.perm_sign_array[idx]
            site_order = jnp.arange(self.nmodes - 1, -1, -1)
            perm_sign = sign.astype(jnp.int32)[..., site_order]

        return occ, perm, perm_sign

    def _fermion_local_sign_at(self, s, idx):
        if not self.is_fermion:
            return jnp.asarray(1.0 + 0.0j, dtype=jnp.complex128)
        local_sign = self.local_sign_array[idx]
        pre_local_state = jnp.asarray(s).reshape(self.nmodes)[self.perm_array[idx]]
        if self.particle_type == "spinful_fermion":
            occ = _spinful_local_occupancy(pre_local_state, self.nmodes)
        else:
            occ = _spinless_local_occupancy(pre_local_state, self.nmodes)
        ph_sign = _fermion_particle_hole_sign_from_occ(occ)
        phase = jnp.asarray(1.0 + 0.0j, dtype=jnp.complex128)
        if self.local_phase is not None:
            phase = jnp.prod(self.local_phase_array[idx][pre_local_state])

        return (phase * jnp.where(local_sign == 1, ph_sign, 1.0 + 0.0j)).astype(jnp.complex128)

    def _fermion_local_signs(self, s):
        if not self.is_fermion:
            return jnp.ones((self.nsymm,), dtype=jnp.complex128)
        states = jnp.asarray(s).reshape(self.nmodes)[self.perm_array]
        if self.particle_type == "spinful_fermion":
            occ = jax.vmap(lambda x: _spinful_local_occupancy(x, self.nmodes))(states)
        else:
            occ = jax.vmap(lambda x: _spinless_local_occupancy(x, self.nmodes))(states)
        ph_sign = _fermion_particle_hole_sign_from_occ(occ)
        phase = jnp.ones((self.nsymm,), dtype=jnp.complex128)
        if self.local_phase is not None:
            phase = jnp.prod(jnp.take_along_axis(self.local_phase_array, states, axis=1), axis=1)

        return (phase * jnp.where(self.local_sign_array == 1, ph_sign, 1.0 + 0.0j)).astype(jnp.complex128)

    def fermion_sign_at(self, s, idx):
        """
        Sign :math:`\\sigma_g(s)` of the element ``idx``, one for spins and bosons.

        Args:
            * ``s``: Configuration with ``n_sites`` local states, of any shape.
            * ``idx``: Index of the element.

        Returns:
            Complex scalar.
        """
        if not self.is_fermion:
            return jnp.asarray(1.0 + 0.0j, dtype=jnp.complex128)
        occ, perm, perm_sign = self._fermion_occupancy_perm_sign(s, idx)
        inv_perm = jnp.argsort(perm)
        occ_pair = occ[:, None] & occ[None, :]
        inversions = (inv_perm[:, None] > inv_perm[None, :]) & occ_pair
        inversions = jnp.sum(jnp.triu(inversions.astype(jnp.int32), k=1))
        parity = 1 - 2 * jnp.bitwise_and(inversions, 1)
        boundary = jnp.prod(jnp.where(occ, perm_sign, 1))

        return (parity * boundary).astype(jnp.complex128) * self._fermion_local_sign_at(s, idx)

    def weight_at(self, s, idx):
        """
        Weight :math:`w_g(s)=\\chi_g\\sigma_g(s)` of the element ``idx``.

        Args:
            * ``s``: Configuration with ``n_sites`` local states, of any shape.
            * ``idx``: Index of the element.

        Returns:
            Complex scalar.
        """
        return self.character_array[idx] * self.fermion_sign_at(s, idx)

    def fermion_signs(self, s):
        """
        Signs :math:`\\sigma_g(s)` of all the elements, one for spins and bosons.

        Args:
            * ``s``: Configuration with ``n_sites`` local states, of any shape.

        Returns:
            Complex array of shape ``(nsymm,)``.
        """
        if not self.is_fermion:
            return jnp.ones((self.nsymm,), dtype=jnp.complex128)
        occ, perm, perm_sign = self._fermion_occupancy_perm_sign(s, None)
        inv_perm = jnp.argsort(perm, axis=1)
        occ_pair = occ[None, :, None] & occ[None, None, :]
        inversions = (inv_perm[:, :, None] > inv_perm[:, None, :]) & occ_pair
        inversions = jnp.sum(jnp.triu(inversions.astype(jnp.int32), k=1), axis=(1, 2))
        parity = 1 - 2 * jnp.bitwise_and(inversions, 1)
        boundary = jnp.prod(jnp.where(occ[None, :], perm_sign, 1), axis=1)

        return (parity * boundary).astype(jnp.complex128) * self._fermion_local_signs(s)

    def weights(self, s):
        """
        Weights :math:`w_g(s)=\\chi_g\\sigma_g(s)` of all the elements.

        Args:
            * ``s``: Configuration with ``n_sites`` local states, of any shape.

        Returns:
            Complex array of shape ``(nsymm,)``.
        """
        return self.character_array * self.fermion_signs(s)

    def __mul__(self, other):
        """
        ``projector * other``: ``compose_projectors(projector, other)`` for another \
        ``SymmetryProjector``, and ``ProjectedOrbitNet(base_net=other, symmetry=projector)`` \
        for a Flax module.
        """
        if isinstance(other, SymmetryProjector):
            return compose_projectors(self, other, name=f"{self.name}_x_{other.name}")
        if isinstance(other, nn.Module):
            return ProjectedOrbitNet(base_net=other, symmetry=self)
        return NotImplemented

class ProjectedOrbitNet(nn.Module):
    """
    Network projected on a symmetry sector.

    Evaluates ``base_net`` on all the transformed configurations :math:`g\\cdot s` of a \
    ``SymmetryProjector`` and combines the logarithmic coefficients :math:`\\ln\\psi(g\\cdot s)` \
    into :math:`\\ln\\psi_P(s)`, as set by ``symmetry_average``:

    * ``"exp"``: :math:`\\ln\\big(\\frac{1}{n_\\text{symm}}\\sum_g w_g(s)\\psi(g\\cdot s)\\big)`, \
      the projection with the weights :math:`w_g(s)` of the projector \
      (``SymmetryProjector.weights()``). If the terms cancel, the result is :math:`-\\infty`.
    * ``"log"``: :math:`\\frac{1}{n_\\text{symm}}\\sum_g\\ln\\psi(g\\cdot s)`.
    * ``"sep"``: \
      :math:`\\frac12\\ln\\big(\\frac{1}{n_\\text{symm}}\\sum_g|\\psi(g\\cdot s)|^2\\big)+i\\arg\\sum_g e^{i\\,\\mathrm{Im}\\ln\\psi(g\\cdot s)}`, \
      which averages the probabilities and the phases separately. For a network normalized \
      with ``logProbFactor=0.5``, :math:`\\psi_P` is normalized too, and ``sample()`` draws \
      configurations from :math:`|\\psi_P|^2`.
    * A callable ``f(log_amplitudes, weights)``, which gets the arrays of the ``nsymm`` \
      logarithmic coefficients and weights and returns :math:`\\ln\\psi_P(s)`.

    ``"log"`` and ``"sep"`` do not use the weights, so for a projector whose weights are not all \
    one (``SymmetryProjector.has_nontrivial_projector_weights``) they are not a linear \
    projection, and a ``UserWarning`` is issued. A network with a ``sample()`` method (e.g. an \
    autoregressive network) requires ``"sep"`` or a callable, and a projector whose weights are \
    all one: otherwise a ``ValueError`` is raised when the module is set up.

    The optional coefficient networks add a logarithmic coefficient :math:`c_g(s)` to each \
    term, :math:`\\ln\\psi(g\\cdot s)\\to\\ln\\psi(g\\cdot s)+c_g(s)`: ``coefficient_net(s)`` \
    returns the ``nsymm`` coefficients for the configuration :math:`s`, and \
    ``coefficient_state_net(g·s)`` one coefficient for each transformed configuration. Only one \
    of the two can be given.

    The transformed configurations are passed to the networks with the shape of :math:`s`. The \
    output is ``complex128``. ``NQS(net, ..., orbit=projector)`` wraps ``net`` into a \
    ``ProjectedOrbitNet``, and so does ``projector * net``.

    Attributes:
        * ``base_net``: Network of :math:`\\ln\\psi`, for a single configuration.
        * ``symmetry``: ``SymmetryProjector``.
        * ``coefficient_net`` [optional]: Network of the ``nsymm`` coefficients :math:`c_g(s)`.
        * ``coefficient_state_net`` [optional]: Network of the coefficient of one transformed \
          configuration.
        * ``symmetry_average``: ``"exp"``, ``"log"``, ``"sep"`` or a callable.
    """
    base_net: nn.Module
    symmetry: SymmetryProjector
    coefficient_net: nn.Module | None = None
    coefficient_state_net: nn.Module | None = None
    symmetry_average: SymmetryAverage = "exp"

    def setup(self):
        if self.symmetry_average not in ("exp", "log", "sep") and not callable(self.symmetry_average):
            raise ValueError("symmetry_average must be 'exp', 'log', 'sep', or a callable.")
        if self.coefficient_net is not None and self.coefficient_state_net is not None:
            raise ValueError("Pass only one of coefficient_net or coefficient_state_net.")
        if self.symmetry_average in ("log", "sep") and self.symmetry.has_nontrivial_projector_weights:
            warnings.warn(
                f"symmetry_average='{self.symmetry_average}' does not apply nontrivial projector weights "
                "as a linear wavefunction projector. Use symmetry_average='exp' for character/sign-weighted "
                "projection, or proceed only if this non-linear orbit average is intentional.",
                UserWarning,
                stacklevel=2,
            )
        if _has_autoregressive_sampler(self.base_net):
            if self.symmetry_average != "sep" and not callable(self.symmetry_average):
                raise ValueError("Autoregressive/sampling networks require symmetry_average='sep' or a custom callable.")
            if self.symmetry.has_nontrivial_projector_weights:
                raise ValueError(
                    "Autoregressive/sampling networks are only compatible with symmetry projectors "
                    "whose bare configuration action has no nontrivial character, boundary sign, "
                    "fermion permutation sign, or particle-hole sign."
                )

    @nn.compact
    def __call__(self, s):
        """
        Logarithmic coefficient :math:`\\ln\\psi_P(s)` of a single configuration.

        :meta public:
        """
        s = jnp.asarray(s)
        states = self.symmetry.transformed_states(s).reshape((self.symmetry.nsymm,) + s.shape)

        # TODO: This can create OOM issues since it is batched externally
        base_logs = jax.vmap(self.base_net)(states)
        # TODO: why complex128 always? 
        base_logs = jnp.asarray(base_logs, dtype=jnp.complex128)

        if self.coefficient_net is not None:
            coeff_logs = jnp.asarray(self.coefficient_net(s), dtype=jnp.complex128)
        elif self.coefficient_state_net is not None:
            coeff_logs = jnp.asarray(jax.vmap(self.coefficient_state_net)(states), dtype=jnp.complex128)
        else:
            coeff_logs = jnp.zeros((self.symmetry.nsymm,), dtype=jnp.complex128)
        if coeff_logs.shape != (self.symmetry.nsymm,):
            raise ValueError(
                "coefficient_net and coefficient_state_net must "
                "return one log coefficient per symmetry element."
            )
        
        return average_coefficients(
            base_logs + coeff_logs,
            self.symmetry.weights(s),
            self.symmetry_average,
        )

    def sample(self, key):
        """
        Draw a configuration: ``base_net.sample()`` draws :math:`s'`, and an element :math:`g` \
        drawn uniformly at random maps it to :math:`g\\cdot s'`.

        With ``symmetry_average="sep"`` and without coefficient networks, the configurations \
        are distributed as :math:`|\\psi_P|^2`.

        Args:
            * ``key``: ``jax.random`` key.

        Returns:
            Configuration of the shape returned by ``base_net.sample()``. Raises \
            ``AttributeError`` if ``base_net`` has no ``sample()`` method.
        """
        if not has_callable_attr(self.base_net, "sample"):
            raise AttributeError(
                f"{type(self.base_net).__name__} does not have a 'sample' method. "
                "ProjectedOrbitNet.sample requires base_net to implement sampling."
            )
        
        sample_key, orbit_key = jax.random.split(key)
        sample = self.base_net.sample(sample_key)

        return self.symmetry.random_transformed_state(sample, orbit_key).reshape(jnp.shape(sample))