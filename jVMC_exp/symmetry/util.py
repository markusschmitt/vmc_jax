import jax.numpy as jnp
from typing import Literal

#: Particle types accepted by the symmetry projectors (the plural forms are accepted too).
ParticleType = Literal["spin", "boson", "bosons", "spinless_fermion", "spinful_fermion"]
_PARTICLE_TYPES = {"spin", "boson", "spinless_fermion", "spinful_fermion"}

def deduplicate_symmetry_arrays(perm, character, local_map=None, local_phase=None, local_sign=None, perm_sign=None):
    """
    Merge the elements of a projector that act identically.

    Two elements are the same if their ``perm``, ``perm_sign``, ``local_map``, ``local_phase`` \
    and ``local_sign`` agree. The characters of merged elements are added, and all the \
    characters are rescaled by the number of distinct elements over the number of elements, so \
    that the projection, an average over the elements, does not change.

    Args:
        * ``perm``: Site permutations, of shape ``(n, n_sites)``.
        * ``character``: Characters, of shape ``(n,)``.
        * ``local_map``, ``local_phase``, ``local_sign``, ``perm_sign`` [optional]: The other \
          arrays of the elements, as in ``SymmetryProjector``.

    Returns:
        ``keep, character``: Indices of the first occurrence of each distinct element, in their \
        original order, and the characters of the distinct elements.
    """
    n_original = int(perm.shape[0])
    local_sign = jnp.zeros((n_original,), dtype=jnp.int8) if local_sign is None else jnp.asarray(local_sign, dtype=jnp.int8)
    buckets = {}
    order = []
    for idx, row in enumerate(perm.tolist()):
        key = (
            tuple(row),
            None if perm_sign is None else tuple(perm_sign[idx].tolist()),
            None if local_map is None else tuple(local_map[idx].tolist()),
            None if local_phase is None else tuple(local_phase[idx].tolist()),
            int(local_sign[idx]),
        )
        if key not in buckets:
            buckets[key] = {"idx": idx, "character": 0.0 + 0.0j}
            order.append(key)
        buckets[key]["character"] += complex(character[idx])

    keep = jnp.asarray([buckets[key]["idx"] for key in order], dtype=jnp.int32)
    scale = len(order) / n_original
    character_unique = jnp.asarray([scale * buckets[key]["character"] for key in order], dtype=jnp.complex128)

    return keep, character_unique

def array_tuple(x, dtype):
    """Two-dimensional array as a tuple of tuples of Python scalars, with the type ``dtype``."""
    return tuple(map(tuple, jnp.asarray(x, dtype=dtype).tolist()))

def vector_tuple(x, dtype):
    """One-dimensional array as a tuple of Python scalars, with the type ``dtype``."""
    return tuple(jnp.asarray(x, dtype=dtype).tolist())

def normalize_particle_type(particle_type: ParticleType) -> str:
    """
    Singular form of a particle type: ``"spin"``, ``"boson"``, ``"spinless_fermion"`` or \
    ``"spinful_fermion"``. Raises ``ValueError`` for any other value.
    """
    particle_type = str(particle_type)
    if particle_type == "bosons":
        particle_type = "boson"
    if particle_type == "spins":
        particle_type = "spin"
    if particle_type == "spinless_fermions":
        particle_type = "spinless_fermion"
    if particle_type == "spinful_fermions":
        particle_type = "spinful_fermion"
    if particle_type not in _PARTICLE_TYPES:
        raise ValueError("particle_type must be 'spin', 'boson', 'spinless_fermion', or 'spinful_fermion'.")
    return particle_type