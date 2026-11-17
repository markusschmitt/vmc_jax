"""
Projectors of the usual lattice symmetries.

Each function returns a ``SymmetryProjector`` for a lattice of ``Lx`` x ``Ly`` sites with periodic
boundary conditions, ``Ly=1`` for a chain. Projectors are composed with ``*``, e.g.
``square_translation_symmetry(L, L, "spin") * square_d4_symmetry(L, L, "spin")``.

A configuration holds the local states of the sites in a flat array. The site enumeration gives
the index in this array of the site at the coordinates ``(x, y)``:

* ``site_order="direct"``: row-major order, the index ``x + Lx * y``.
* ``site_order="snake"``: column-snake order, ``np.arange(Lx * Ly).reshape(Lx, Ly)`` with the even
  columns reversed, i.e. the index ``x * Ly + y`` for odd ``x`` and ``x * Ly + Ly - 1 - y`` for
  even ``x``.
* ``site_order="custom"``: the array ``site_enum`` of the indices, of shape ``(Ly, Lx)`` and indexed
  as ``site_enum[y, x]`` with ``site_enum_axes="yx"``, or of shape ``(Lx, Ly)`` and indexed as
  ``site_enum[x, y]`` with ``site_enum_axes="xy"``. It must contain each index from 0 to
  ``Lx * Ly - 1`` once.

For spinful fermions the local states 0, 1, 2 and 3 stand for an empty site, a spin up, a spin
down and a doubly occupied site, and the fermionic signs use the mode order
``(N-1, down), (N-1, up), ..., (0, down), (0, up)`` of ``jVMC_exp.operator``. The fermionic
particle-hole transformation :math:`C` satisfies :math:`C\\hat c_iC^{-1}=\\hat c_i^\\dagger` for
every mode, with the sign given in ``particle_hole_symmetry()``.
"""

from typing import Literal, Sequence
import jax.numpy as jnp
import cmath
import math

from .util import ParticleType, normalize_particle_type, array_tuple, vector_tuple, deduplicate_symmetry_arrays
from .projector import SymmetryProjector

#: Custom site enumeration: an array of the site indices, used with ``site_order="custom"``.
SiteEnumeration = Sequence[Sequence[int]] | jnp.ndarray | None
#: Site enumeration: ``"direct"``, ``"snake"`` or ``"custom"``.
SiteOrder = Literal["direct", "snake", "custom"]
#: Axes of a custom site enumeration: ``"yx"`` for ``site_enum[y, x]``, ``"xy"`` for ``site_enum[x, y]``.
SiteEnumerationAxes = Literal["yx", "xy"]

def _validate_lattice_shape(Lx: int, Ly: int) -> tuple[int, int]:
    Lx, Ly = int(Lx), int(Ly)
    if Lx <= 0 or Ly <= 0:
        raise ValueError("Lx and Ly must both be positive. Use Ly=1 for one-dimensional chains.")
    
    return Lx, Ly

def _site_enum_array(
    Lx: int,
    Ly: int,
    site_order: SiteOrder = "direct",
    site_enum: SiteEnumeration = None,
    site_enum_axes: SiteEnumerationAxes = "yx",
):
    if site_order not in ("direct", "snake", "custom"):
        raise ValueError("site_order must be 'direct', 'snake', or 'custom'.")
    if site_enum_axes not in ("yx", "xy"):
        raise ValueError("site_enum_axes must be 'yx' or 'xy'.")
    if site_order != "custom" and site_enum is not None:
        raise ValueError("site_enum is only accepted with site_order='custom'.")
    
    if site_order == "direct":
        enum = jnp.arange(Lx * Ly, dtype=jnp.int32).reshape(Ly, Lx)
    elif site_order == "snake":
        enum_xy = jnp.arange(Lx * Ly, dtype=jnp.int32).reshape(Lx, Ly)
        enum_xy = enum_xy.at[::2, :].set(enum_xy[::2, ::-1])
        enum = enum_xy.T
    else:
        if site_enum is None:
            raise ValueError("site_order='custom' requires site_enum.")
        enum = jnp.asarray(site_enum, dtype=jnp.int32)
        if site_enum_axes == "xy":
            if enum.shape != (Lx, Ly):
                raise ValueError("site_enum with site_enum_axes='xy' must have shape (Lx, Ly).")
            enum = enum.T
        else:
            if enum.shape != (Ly, Lx):
                raise ValueError("site_enum with site_enum_axes='yx' must have shape (Ly, Lx).")
            
    flat = jnp.ravel(enum)
    expected = jnp.arange(Lx * Ly, dtype=jnp.int32)
    if bool(jnp.any(jnp.sort(flat) != expected)):
        raise ValueError("site enumeration must contain each site index from 0 to Lx*Ly-1 exactly once.")
    
    return enum

def _site_enum_lookup(
    Lx: int,
    Ly: int,
    site_order: SiteOrder = "direct",
    site_enum: SiteEnumeration = None,
    site_enum_axes: SiteEnumerationAxes = "yx",
):
    enum = _site_enum_array(Lx, Ly, site_order, site_enum, site_enum_axes)
    coords = jnp.zeros((Lx * Ly, 2), dtype=jnp.int32)
    for y in range(Ly):
        for x in range(Lx):
            coords = coords.at[int(enum[y, x])].set(jnp.asarray([x, y], dtype=jnp.int32))

    return enum, coords

def square_translation_symmetry(
    Lx: int,
    Ly: int,
    particle_type: ParticleType,
    q: tuple[int, int] = (0, 0),
    site_order: SiteOrder = "direct",
    site_enum: SiteEnumeration = None,
    site_enum_axes: SiteEnumerationAxes = "yx",
) -> SymmetryProjector:
    """
    Translations of the lattice, projected on the momentum ``q``.

    The element ``(sx, sy)``, with the index ``sx + Lx * sy``, shifts the configuration by \
    ``(sx, sy)``: the local state of the site at ``(x, y)`` moves to ``(x + sx, y + sy)``, \
    modulo the size of the lattice. Its character is \
    :math:`e^{-2\\pi i(q_xs_x/L_x+q_ys_y/L_y)}`, so that the projected state picks up the phase \
    :math:`e^{2\\pi i(q_xa_x/L_x+q_ya_y/L_y)}` under the element ``a``: for spins and bosons, \
    :math:`\\psi_P` of the configuration shifted by ``a`` is this phase times :math:`\\psi_P(s)`.

    Args:
        * ``Lx``, ``Ly``: Size of the lattice, ``Ly=1`` for a chain.
        * ``particle_type``: Particle type, see ``SymmetryProjector``.
        * ``q``: Momentum as two integers ``(qx, qy)``, for the wave vector \
          :math:`(2\\pi q_x/L_x, 2\\pi q_y/L_y)`. They are taken modulo ``Lx`` and ``Ly``.
        * ``site_order``, ``site_enum``, ``site_enum_axes``: Site enumeration, see the module.

    Returns:
        ``SymmetryProjector`` named ``"translation"``, with ``Lx * Ly`` elements.
    """
    Lx, Ly = _validate_lattice_shape(Lx, Ly)
    particle_type = normalize_particle_type(particle_type)
    enum, coords = _site_enum_lookup(Lx, Ly, site_order, site_enum, site_enum_axes)
    qx, qy = int(q[0]) % Lx, int(q[1]) % Ly
    site_perms = []
    characters = []
    for sy in range(Ly):
        for sx in range(Lx):
            site_perm = []
            for sample_idx in range(Lx * Ly):
                x, y = map(int, coords[sample_idx].tolist())
                site_perm.append(int(enum[(y - sy) % Ly, (x - sx) % Lx]))
            site_perms.append(site_perm)
            phase = -2j * math.pi * (qx * sx / Lx + qy * sy / Ly)
            characters.append(complex(cmath.exp(phase)))

    perm = jnp.asarray(site_perms, dtype=jnp.int32)
    return SymmetryProjector(
        Lx=Lx,
        Ly=Ly,
        particle_type=particle_type,
        perm=array_tuple(perm, jnp.int32),
        character=tuple(characters),
        name="translation",
    )

def square_d4_symmetry(
    Lx: int,
    Ly: int,
    particle_type: ParticleType = "spin",
    site_order: SiteOrder = "direct",
    site_enum: SiteEnumeration = None,
    site_enum_axes: SiteEnumerationAxes = "yx",
) -> SymmetryProjector:
    """
    Point group :math:`D_4` of a square lattice.

    The eight elements are the identity, the rotations by 90, 180 and 270 degrees, and the \
    reflections about the two axes and the two diagonals, through the center of the lattice. \
    All the characters are one.

    Args:
        * ``Lx``, ``Ly``: Size of the lattice, which must be square.
        * ``particle_type``: Particle type, see ``SymmetryProjector``.
        * ``site_order``, ``site_enum``, ``site_enum_axes``: Site enumeration, see the module.

    Returns:
        ``SymmetryProjector`` named ``"d4"``. Raises ``ValueError`` if ``Lx != Ly``.
    """
    Lx, Ly = _validate_lattice_shape(Lx, Ly)
    if Lx != Ly:
        raise ValueError("D4 symmetry requires a square lattice.")
    particle_type = normalize_particle_type(particle_type)
    enum, coords = _site_enum_lookup(Lx, Ly, site_order, site_enum, site_enum_axes)
    transforms = (
        lambda x, y: (x, y),
        lambda x, y: (Lx - 1 - y, x),
        lambda x, y: (Lx - 1 - x, Ly - 1 - y),
        lambda x, y: (y, Ly - 1 - x),
        lambda x, y: (x, Ly - 1 - y),
        lambda x, y: (Lx - 1 - y, Ly - 1 - x),
        lambda x, y: (Lx - 1 - x, y),
        lambda x, y: (y, x),
    )
    site_perms = []
    for transform in transforms:
        site_perm = []
        for sample_idx in range(Lx * Ly):
            x, y = map(int, coords[sample_idx].tolist())
            xp, yp = transform(x, y)
            site_perm.append(int(enum[yp % Ly, xp % Lx]))
        site_perms.append(site_perm)
    site_perm = jnp.asarray(site_perms, dtype=jnp.int32)
    perm = site_perm

    return SymmetryProjector(
        Lx=Lx,
        Ly=Ly,
        particle_type=particle_type,
        perm=array_tuple(perm, jnp.int32),
        character=tuple([1.0 + 0.0j] * int(perm.shape[0])),
        name="d4",
    )

def _unique_symmetry_projector(
    Lx: int,
    Ly: int,
    particle_type: ParticleType,
    perm,
    character,
    *,
    name: str,
    local_map=None,
    local_phase=None,
    local_sign=None,
) -> SymmetryProjector:
    particle_type = normalize_particle_type(particle_type)
    perm = jnp.asarray(perm, dtype=jnp.int32)
    character = jnp.asarray(character, dtype=jnp.complex128)
    local_map_arr = None if local_map is None else jnp.asarray(local_map, dtype=jnp.int32)
    local_phase_arr = None if local_phase is None else jnp.asarray(local_phase, dtype=jnp.complex128)
    local_sign_arr = jnp.zeros((perm.shape[0],), dtype=jnp.int8) if local_sign is None else jnp.asarray(local_sign, dtype=jnp.int8)
    keep, character_unique = deduplicate_symmetry_arrays(
        perm,
        character,
        local_map=local_map_arr,
        local_phase=local_phase_arr,
        local_sign=local_sign_arr,
    )

    return SymmetryProjector(
        Lx=Lx,
        Ly=Ly,
        particle_type=particle_type,
        perm=array_tuple(perm[keep], jnp.int32),
        character=vector_tuple(character_unique, jnp.complex128),
        local_map=None if local_map_arr is None else array_tuple(local_map_arr[keep], jnp.int32),
        local_phase=None if local_phase_arr is None else array_tuple(local_phase_arr[keep], jnp.complex128),
        local_sign=tuple(jnp.asarray(local_sign_arr[keep], dtype=jnp.int8).tolist()),
        name=name,
    )

def rectangle_d2_symmetry(
    Lx: int,
    Ly: int,
    particle_type: ParticleType = "spin",
    site_order: SiteOrder = "direct",
    site_enum: SiteEnumeration = None,
    site_enum_axes: SiteEnumerationAxes = "yx",
) -> SymmetryProjector:
    """
    Point group :math:`D_2` of a rectangular lattice.

    The elements are the identity, the rotation by 180 degrees and the reflections \
    :math:`x\\to L_x-1-x` and :math:`y\\to L_y-1-y`, all with character one. Elements that \
    coincide are merged, so that a chain, ``Ly=1``, has the two elements of the reflection.

    Args:
        * ``Lx``, ``Ly``: Size of the lattice.
        * ``particle_type``: Particle type, see ``SymmetryProjector``.
        * ``site_order``, ``site_enum``, ``site_enum_axes``: Site enumeration, see the module.

    Returns:
        ``SymmetryProjector`` named ``"d2"``.
    """
    Lx, Ly = _validate_lattice_shape(Lx, Ly)
    particle_type = normalize_particle_type(particle_type)
    enum, coords = _site_enum_lookup(Lx, Ly, site_order, site_enum, site_enum_axes)
    transforms = (
        lambda x, y: (x, y),
        lambda x, y: (Lx - 1 - x, Ly - 1 - y),
        lambda x, y: (Lx - 1 - x, y),
        lambda x, y: (x, Ly - 1 - y),
    )
    site_perms = []
    for transform in transforms:
        site_perm = []
        for sample_idx in range(Lx * Ly):
            x, y = map(int, coords[sample_idx].tolist())
            xp, yp = transform(x, y)
            site_perm.append(int(enum[yp % Ly, xp % Lx]))
        site_perms.append(site_perm)

    return _unique_symmetry_projector(
        Lx,
        Ly,
        particle_type,
        site_perms,
        [1.0 + 0.0j] * len(site_perms),
        name="d2",
    )

def chain_reflection_symmetry(
    L: int,
    particle_type: ParticleType = "spin",
    site_order: SiteOrder = "direct",
    site_enum: SiteEnumeration = None,
    site_enum_axes: SiteEnumerationAxes = "yx",
) -> SymmetryProjector:
    """
    Reflection :math:`x\\to L-1-x` of a chain, ``rectangle_d2_symmetry(L, 1, ...)``.

    Args:
        * ``L``: Number of sites.
        * ``particle_type``: Particle type, see ``SymmetryProjector``.
        * ``site_order``, ``site_enum``, ``site_enum_axes``: Site enumeration, see the module.

    Returns:
        ``SymmetryProjector`` named ``"d2"``, with two elements.
    """
    return rectangle_d2_symmetry(
        int(L),
        1,
        particle_type,
        site_order=site_order,
        site_enum=site_enum,
        site_enum_axes=site_enum_axes,
    )

def point_group_symmetry(
    Lx: int,
    Ly: int,
    particle_type: ParticleType = "spin",
    site_order: SiteOrder = "direct",
    site_enum: SiteEnumeration = None,
    site_enum_axes: SiteEnumerationAxes = "yx",
) -> SymmetryProjector:
    """
    Point group of the lattice: ``square_d4_symmetry()`` for a square lattice with more than \
    one site, ``rectangle_d2_symmetry()`` otherwise.

    Args:
        * ``Lx``, ``Ly``: Size of the lattice.
        * ``particle_type``: Particle type, see ``SymmetryProjector``.
        * ``site_order``, ``site_enum``, ``site_enum_axes``: Site enumeration, see the module.

    Returns:
        ``SymmetryProjector`` named ``"d4"`` or ``"d2"``.
    """
    Lx, Ly = _validate_lattice_shape(Lx, Ly)
    if Lx == Ly and Ly > 1:
        return square_d4_symmetry(
            Lx,
            Ly,
            particle_type,
            site_order=site_order,
            site_enum=site_enum,
            site_enum_axes=site_enum_axes,
        )
    return rectangle_d2_symmetry(
        Lx,
        Ly,
        particle_type,
        site_order=site_order,
        site_enum=site_enum,
        site_enum_axes=site_enum_axes,
    )

def particle_hole_symmetry(
    Lx: int,
    Ly: int,
    particle_type: ParticleType,
    local_dim: int | None = None,
) -> SymmetryProjector:
    """
    Particle-hole transformation of all the sites.

    The two elements are the identity and the transformation :math:`C`, both with character \
    one. For bosons the occupation ``n`` becomes ``local_dim - 1 - n``. For fermions every mode \
    is emptied if occupied and occupied if empty: the local states ``0`` and ``1`` are \
    exchanged for spinless fermions; an empty and a doubly occupied site, and a spin up and a \
    spin down, for spinful fermions. A configuration with the occupations :math:`n_m` of the \
    modes :math:`m=0,\\dots,M-1`, numbered in the mode order of the module, gets the sign \
    :math:`(-1)^{\\sum_m m\\,n_m}`, so that :math:`C\\hat c_mC^{-1}=\\hat c_m^\\dagger` for every \
    mode.

    Args:
        * ``Lx``, ``Ly``: Size of the lattice.
        * ``particle_type``: ``"boson"``, ``"spinless_fermion"`` or ``"spinful_fermion"``. For \
          spins, see ``spin_flip_symmetry()``.
        * ``local_dim``: Local dimension, required for bosons.

    Returns:
        ``SymmetryProjector`` named ``"particle_hole"``. Raises ``ValueError`` for spins, and for \
        bosons without a positive ``local_dim``.
    """
    Lx, Ly = _validate_lattice_shape(Lx, Ly)
    particle_type = normalize_particle_type(particle_type)
    n_sites = Lx * Ly
    perm = [list(range(n_sites)), list(range(n_sites))]
    character = [1.0 + 0.0j, 1.0 + 0.0j]
    if particle_type == "boson":
        if local_dim is None:
            raise ValueError("particle_hole_symmetry for bosons requires local_dim.")
        local_dim = int(local_dim)
        if local_dim <= 0:
            raise ValueError("local_dim must be positive.")
        ph_map = list(range(local_dim - 1, -1, -1))
    elif particle_type == "spinless_fermion":
        ph_map = [1, 0]
    elif particle_type == "spinful_fermion":
        ph_map = [3, 2, 1, 0]
    else:
        raise ValueError("particle_hole_symmetry is defined for bosons and fermions, not spins.")
    local_sign = [0, 1] if particle_type in {"spinless_fermion", "spinful_fermion"} else [0, 0]

    return SymmetryProjector(
        Lx=Lx,
        Ly=Ly,
        particle_type=particle_type,
        perm=array_tuple(perm, jnp.int32),
        character=tuple(character),
        local_map=array_tuple([list(range(len(ph_map))), ph_map], jnp.int32),
        local_sign=tuple(local_sign),
        name="particle_hole",
    )

def spin_flip_symmetry(Lx: int, Ly: int, particle_type: ParticleType = "spin") -> SymmetryProjector:
    """
    Spin flip of all the sites.

    The two elements are the identity and the spin flip :math:`F`, with character one. For \
    spins, the local states ``0`` and ``1`` are exchanged. For spinful fermions, the spins up \
    and down are exchanged on every site, with a sign -1 for each doubly occupied site, so that \
    :math:`F\\hat c_{i\\uparrow}F^{-1}=\\hat c_{i\\downarrow}` and \
    :math:`F\\hat c_{i\\downarrow}F^{-1}=\\hat c_{i\\uparrow}`.

    Args:
        * ``Lx``, ``Ly``: Size of the lattice.
        * ``particle_type``: ``"spin"`` or ``"spinful_fermion"``.

    Returns:
        ``SymmetryProjector`` named ``"spin_flip"``. Raises ``ValueError`` for other particle \
        types.
    """
    Lx, Ly = _validate_lattice_shape(Lx, Ly)
    particle_type = normalize_particle_type(particle_type)
    if particle_type == "spin":
        local_map = [[0, 1], [1, 0]]
        local_phase = None
    elif particle_type == "spinful_fermion":
        local_map = [[0, 1, 2, 3], [0, 2, 1, 3]]
        local_phase = [[1.0 + 0.0j] * 4, [1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j, -1.0 + 0.0j]]
    else:
        raise ValueError("spin_flip_symmetry is defined for spins and spinful fermions.")
    n_sites = Lx * Ly
    perm = [list(range(n_sites)), list(range(n_sites))]

    return SymmetryProjector(
        Lx=Lx,
        Ly=Ly,
        particle_type=particle_type,
        perm=array_tuple(perm, jnp.int32),
        character=(1.0 + 0.0j, 1.0 + 0.0j),
        local_map=array_tuple(local_map, jnp.int32),
        local_phase=None if local_phase is None else array_tuple(local_phase, jnp.complex128),
        name="spin_flip",
    )