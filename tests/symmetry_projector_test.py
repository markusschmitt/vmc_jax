from __future__ import annotations
import unittest
import warnings
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from jVMC_exp.symmetry import ProjectedOrbitNet, SymmetryProjector, compose_projectors
from jVMC_exp.symmetry.lattice_symetries import (
    chain_reflection_symmetry,
    particle_hole_symmetry,
    point_group_symmetry,
    rectangle_d2_symmetry,
    spin_flip_symmetry,
    square_d4_symmetry,
    square_translation_symmetry,
    _unique_symmetry_projector
)

class _DummyNet(nn.Module):
    @nn.compact
    def __call__(self, s):
        del s
        return jnp.asarray(0.0 + 0.0j, dtype=jnp.complex128)


class _FirstSiteNet(nn.Module):
    @nn.compact
    def __call__(self, s):
        s = jnp.asarray(s, dtype=jnp.float64).reshape(-1)
        return 0.3 * s[0] + 0.7j * s[-1]


class _StateCoefficientNet(nn.Module):
    @nn.compact
    def __call__(self, s):
        s = jnp.asarray(s, dtype=jnp.float64).reshape(-1)
        return 0.11 * jnp.sum(s) - 0.23j * s[0]


class _OrbitCoefficientNet(nn.Module):
    nsymm: int

    @nn.compact
    def __call__(self, s):
        offset = jnp.asarray(0.01 * jnp.sum(s), dtype=jnp.complex128)
        return offset + 0.05 * jnp.arange(self.nsymm, dtype=jnp.float64)


class _AutoregressiveLikeNet(nn.Module):
    @nn.compact
    def __call__(self, s):
        return jnp.asarray(jnp.sum(s), dtype=jnp.complex128)

    def sample(self, numSamples: int, key):
        del key
        return jnp.zeros((int(numSamples), 4), dtype=jnp.int32)


class _TwoDimensionalNet(nn.Module):
    """Accepts only configurations of shape (Ly, Lx) = (2, 3), like a 2D convolutional network."""
    @nn.compact
    def __call__(self, s):
        if jnp.shape(s) != (2, 3):
            raise ValueError(f"expected a configuration of shape (2, 3), got {jnp.shape(s)}")
        s = jnp.asarray(s, dtype=jnp.float64)
        return 0.3 * jnp.sum(s[0]) + 0.2j * jnp.sum(s[:, 0]) + 0.1 * s[1, 2]


class _TwoDimensionalOrbitCoefficientNet(nn.Module):
    nsymm: int

    @nn.compact
    def __call__(self, s):
        if jnp.shape(s) != (2, 3):
            raise ValueError(f"expected a configuration of shape (2, 3), got {jnp.shape(s)}")
        return 0.01 * jnp.sum(s[0]) + 0.05j * jnp.arange(self.nsymm, dtype=jnp.float64)


class _TwoDimensionalGenerator(nn.Module):
    @nn.compact
    def __call__(self, s):
        return jnp.asarray(0.0 + 0.0j, dtype=jnp.complex128)

    def sample(self, key):
        return jax.random.bernoulli(key, 0.5, (2, 3)).astype(jnp.int32)


def _weighted_sum_average(log_amplitudes, weights):
    return jnp.log(jnp.sum(weights * jnp.exp(log_amplitudes)))


def _site_idx(x: int, y: int, Lx: int, Ly: int) -> int:
    return (x % Lx) + Lx * (y % Ly)


def _translation_perm(Lx: int, Ly: int, sx: int, sy: int) -> np.ndarray:
    return np.asarray([_site_idx(x - sx, y - sy, Lx, Ly) for y in range(Ly) for x in range(Lx)], dtype=np.int32)


def _coords_from_enum(enum_yx: np.ndarray) -> np.ndarray:
    Ly, Lx = enum_yx.shape
    coords = np.zeros((Lx * Ly, 2), dtype=np.int32)
    for y in range(Ly):
        for x in range(Lx):
            coords[enum_yx[y, x]] = (x, y)
    return coords


def _translation_perm_from_enum(enum_yx: np.ndarray, sx: int, sy: int) -> np.ndarray:
    Ly, Lx = enum_yx.shape
    coords = _coords_from_enum(enum_yx)
    row = []
    for sample_idx in range(Lx * Ly):
        x, y = coords[sample_idx]
        row.append(enum_yx[(y - sy) % Ly, (x - sx) % Lx])
    return np.asarray(row, dtype=np.int32)


def _d4_perms(L: int) -> np.ndarray:
    base = np.arange(L * L, dtype=np.int32).reshape(L, L)
    flip = np.flip(base, axis=0)
    transforms = (
        base,
        np.rot90(base, 1),
        np.rot90(base, 2),
        np.rot90(base, 3),
        flip,
        np.rot90(flip, 1),
        np.rot90(flip, 2),
        np.rot90(flip, 3),
    )
    return np.stack([t.reshape(-1) for t in transforms], axis=0)


def _rectangle_d2_perms(Lx: int, Ly: int) -> np.ndarray:
    transforms = (
        lambda x, y: (x, y),
        lambda x, y: (Lx - 1 - x, Ly - 1 - y),
        lambda x, y: (Lx - 1 - x, y),
        lambda x, y: (x, Ly - 1 - y),
    )
    rows = []
    for transform in transforms:
        row = []
        for y in range(Ly):
            for x in range(Lx):
                xp, yp = transform(x, y)
                row.append(_site_idx(xp, yp, Lx, Ly))
        if row not in rows:
            rows.append(row)
    return np.asarray(rows, dtype=np.int32)


def _spinless_occ_jvmc(state: np.ndarray) -> np.ndarray:
    return np.asarray(state, dtype=np.int32)[::-1] > 0


def _spinful_occ_jvmc(state: np.ndarray) -> np.ndarray:
    state = np.asarray(state, dtype=np.int32).reshape(-1)
    up = state & 1
    down = (state >> 1) & 1
    order = np.arange(state.size - 1, -1, -1)
    return np.stack([down[order], up[order]], axis=-1).reshape(2 * state.size) > 0


def _spinless_orbital_perm_jvmc(site_perm: np.ndarray) -> np.ndarray:
    site_perm = np.asarray(site_perm, dtype=np.int32)
    n_sites = site_perm.size
    out_sites = np.arange(n_sites - 1, -1, -1)
    return n_sites - 1 - site_perm[out_sites]


def _spinful_orbital_perm_jvmc(site_perm: np.ndarray) -> np.ndarray:
    site_perm = np.asarray(site_perm, dtype=np.int32)
    n_sites = site_perm.size
    out_sites = np.arange(n_sites - 1, -1, -1)
    input_sites = site_perm[out_sites]
    down_pos = 2 * (n_sites - 1 - input_sites)
    up_pos = down_pos + 1
    return np.stack([down_pos, up_pos], axis=-1).reshape(2 * n_sites)


def _permutation_sign_from_occ(occ: np.ndarray, orbital_perm: np.ndarray) -> complex:
    inv_perm = np.argsort(orbital_perm)
    occupied = np.flatnonzero(occ)
    restricted = inv_perm[occupied]
    inversions = sum(int(restricted[i] > restricted[j]) for i in range(len(restricted)) for j in range(i + 1, len(restricted)))
    return complex(1 - 2 * (inversions & 1))


def _particle_hole_sign_from_operator_identity(occ: np.ndarray) -> complex:
    exponent = sum(int(mode) for mode in np.flatnonzero(occ))
    return complex(1 - 2 * (exponent & 1))


def _bits_from_state_index(state_idx: int, n_modes: int) -> np.ndarray:
    return np.asarray([(state_idx >> mode) & 1 for mode in range(n_modes)], dtype=np.int32)


def _state_index_from_bits(bits: np.ndarray) -> int:
    return sum(int(bit) << mode for mode, bit in enumerate(np.asarray(bits, dtype=np.int32)))


def _bits_from_spinful_state_jvmc(state: np.ndarray) -> np.ndarray:
    return _spinful_occ_jvmc(np.asarray(state, dtype=np.int32)).astype(np.int32)


def _spinful_state_from_bits_jvmc(bits: np.ndarray) -> np.ndarray:
    bits = np.asarray(bits, dtype=np.int32)
    n_sites = bits.size // 2
    state = np.zeros(n_sites, dtype=np.int32)
    for site in range(n_sites):
        down = bits[2 * (n_sites - 1 - site)]
        up = bits[2 * (n_sites - 1 - site) + 1]
        state[site] = up + 2 * down
    return state


def _spinful_state_index_jvmc(state: np.ndarray) -> int:
    return _state_index_from_bits(_bits_from_spinful_state_jvmc(state))


def _spinful_mode_index(site: int, spin: str, n_sites: int) -> int:
    down = 2 * (n_sites - 1 - int(site))
    if spin == "down":
        return down
    if spin == "up":
        return down + 1
    raise ValueError(spin)


def _particle_hole_matrix(n_modes: int) -> np.ndarray:
    dim = 1 << n_modes
    matrix = np.zeros((dim, dim), dtype=np.complex128)
    for state_idx in range(dim):
        occ = _bits_from_state_index(state_idx, n_modes)
        dst = _state_index_from_bits(1 - occ)
        matrix[dst, state_idx] = _particle_hole_sign_from_operator_identity(occ)
    return matrix


def _spinful_spin_flip_matrix(n_sites: int) -> np.ndarray:
    dim = 1 << (2 * n_sites)
    matrix = np.zeros((dim, dim), dtype=np.complex128)
    for state_idx in range(dim):
        state = _spinful_state_from_bits_jvmc(_bits_from_state_index(state_idx, 2 * n_sites))
        up = state & 1
        down = (state >> 1) & 1
        dst = down + 2 * up
        sign = 1 - 2 * (int(np.sum(state == 3)) & 1)
        matrix[_spinful_state_index_jvmc(dst), state_idx] = sign
    return matrix


def _annihilation_matrix(n_modes: int, mode: int) -> np.ndarray:
    dim = 1 << n_modes
    matrix = np.zeros((dim, dim), dtype=np.complex128)
    for state_idx in range(dim):
        occ = _bits_from_state_index(state_idx, n_modes)
        if occ[mode] == 0:
            continue
        sign = 1 - 2 * (int(occ[:mode].sum()) & 1)
        dst_occ = occ.copy()
        dst_occ[mode] = 0
        matrix[_state_index_from_bits(dst_occ), state_idx] = sign
    return matrix


def _assert_allclose(got, expected):
    np.testing.assert_allclose(np.asarray(got), np.asarray(expected), atol=1e-12, rtol=1e-12)


class Test(unittest.TestCase):
    def test_translation_permutations_characters_and_fermion_signs(self):
        Lx, Ly = 3, 2
        q = (1, 1)
        state_spinless = np.array([1, 0, 1, 1, 0, 0], dtype=np.int32)
        state_spinful = np.array([3, 1, 2, 0, 1, 2], dtype=np.int32)

        symm = square_translation_symmetry(Lx, Ly, "spinless_fermion", q=q)
        expected_perm = np.stack([_translation_perm(Lx, Ly, sx, sy) for sy in range(Ly) for sx in range(Lx)], axis=0)
        expected_char = np.asarray([np.exp(-2j * np.pi * (q[0] * sx / Lx + q[1] * sy / Ly)) for sy in range(Ly) for sx in range(Lx)])

        _assert_allclose(symm.perm_array, expected_perm)
        _assert_allclose(symm.character_array, expected_char)
        _assert_allclose(symm.transformed_states(jnp.asarray(state_spinless)), state_spinless[expected_perm])

        signs = np.asarray(symm.fermion_signs(jnp.asarray(state_spinless)))
        occ = _spinless_occ_jvmc(state_spinless)
        expected_signs = [_permutation_sign_from_occ(occ, _spinless_orbital_perm_jvmc(row)) for row in expected_perm]
        _assert_allclose(signs, expected_signs)

        spinful_symm = square_translation_symmetry(Lx, Ly, "spinful_fermion", q=q)
        signs = np.asarray(spinful_symm.fermion_signs(jnp.asarray(state_spinful)))
        occ = _spinful_occ_jvmc(state_spinful)
        expected_signs = [_permutation_sign_from_occ(occ, _spinful_orbital_perm_jvmc(row)) for row in expected_perm]
        _assert_allclose(signs, expected_signs)


    def test_snake_and_custom_site_enumeration_for_translation_and_validation(self):
        Lx, Ly = 4, 3
        enum_xy = np.arange(Lx * Ly, dtype=np.int32).reshape(Lx, Ly)
        enum_xy[::2, :] = enum_xy[::2, ::-1]
        enum_yx = enum_xy.T
        state = np.arange(Lx * Ly, dtype=np.int32)

        symm = square_translation_symmetry(Lx, Ly, "spin", q=(1, 2), site_order="snake")
        expected_perm = np.stack([_translation_perm_from_enum(enum_yx, sx, sy) for sy in range(Ly) for sx in range(Lx)], axis=0)
        expected_char = np.asarray([np.exp(-2j * np.pi * (sx / Lx + 2 * sy / Ly)) for sy in range(Ly) for sx in range(Lx)])
        _assert_allclose(symm.perm_array, expected_perm)
        _assert_allclose(symm.character_array, expected_char)
        _assert_allclose(symm.transformed_states(jnp.asarray(state)), state[expected_perm])

        custom = square_translation_symmetry(
            Lx,
            Ly,
            "spin",
            q=(1, 2),
            site_order="custom",
            site_enum=enum_xy,
            site_enum_axes="xy",
        )
        _assert_allclose(custom.perm_array, expected_perm)

        bad_enum = enum_xy.copy()
        bad_enum[0, 0] = bad_enum[0, 1]
        try:
            square_translation_symmetry(Lx, Ly, "spin", site_order="custom", site_enum=bad_enum, site_enum_axes="xy")
        except ValueError as exc:
            assert "exactly once" in str(exc)
        else:
            raise AssertionError("duplicate custom site enumeration should fail")

        try:
            square_translation_symmetry(Lx, Ly, "spin", site_enum=enum_xy, site_enum_axes="xy")
        except ValueError as exc:
            assert "site_order='custom'" in str(exc)
        else:
            raise AssertionError("site_enum without site_order='custom' should fail")


    def test_d4_permutations_and_point_group_fallbacks(self):
        state = np.arange(9, dtype=np.int32)
        d4 = square_d4_symmetry(3, 3, "spin")
        expected = _d4_perms(3)
        _assert_allclose(d4.perm_array, expected)
        _assert_allclose(d4.transformed_states(jnp.asarray(state)), state[expected])
        _assert_allclose(d4.character_array, np.ones(8, dtype=np.complex128))

        assert point_group_symmetry(3, 3, "spin").name == "d4"
        assert point_group_symmetry(4, 2, "spin").name == "d2"


    def test_rectangle_d2_and_chain_reflection(self):
        state = np.arange(8, dtype=np.int32)
        d2 = rectangle_d2_symmetry(4, 2, "spin")
        expected = _rectangle_d2_perms(4, 2)
        _assert_allclose(d2.perm_array, expected)
        _assert_allclose(d2.transformed_states(jnp.asarray(state)), state[expected])
        _assert_allclose(d2.character_array, np.ones(4, dtype=np.complex128))

        chain = chain_reflection_symmetry(5, "spin")
        expected_chain = _rectangle_d2_perms(5, 1)
        _assert_allclose(chain.perm_array, expected_chain)
        assert chain.nsymm == 2


    def test_unique_symmetry_projector_accumulates_duplicate_characters(self):
        symm = _unique_symmetry_projector(
            2,
            1,
            "spin",
            perm=[[0, 1], [0, 1], [1, 0]],
            character=[1.0 + 0.0j, -1.0 + 0.0j, 1.0 + 0.0j],
            name="duplicate_character_test",
        )

        assert symm.nsymm == 2
        _assert_allclose(symm.perm_array, np.asarray([[0, 1], [1, 0]], dtype=np.int32))
        _assert_allclose(symm.character_array, np.asarray([0.0 + 0.0j, 2.0 / 3.0 + 0.0j], dtype=np.complex128))


    def test_spin_flip_and_boson_particle_hole_local_maps(self):
        spin_state = np.array([0, 1, 1, 0], dtype=np.int32)
        spin_flip = spin_flip_symmetry(4, 1)
        _assert_allclose(spin_flip.transformed_states(jnp.asarray(spin_state)), np.asarray([spin_state, 1 - spin_state]))
        _assert_allclose(spin_flip.weights(jnp.asarray(spin_state)), np.ones(2, dtype=np.complex128))

        boson_state = np.array([0, 1, 3, 2], dtype=np.int32)
        boson_ph = particle_hole_symmetry(4, 1, "boson", local_dim=4)
        _assert_allclose(boson_ph.transformed_states(jnp.asarray(boson_state)), np.asarray([boson_state, 3 - boson_state]))
        _assert_allclose(boson_ph.weights(jnp.asarray(boson_state)), np.ones(2, dtype=np.complex128))


    def test_spinful_fermion_spin_flip_local_map_and_doublon_sign(self):
        state = np.array([0, 1, 2, 3], dtype=np.int32)
        spin_flip = spin_flip_symmetry(4, 1, "spinful_fermion")

        expected_states = np.asarray([state, np.array([0, 2, 1, 3], dtype=np.int32)])
        _assert_allclose(spin_flip.transformed_states(jnp.asarray(state)), expected_states)
        _assert_allclose(spin_flip.fermion_signs(jnp.asarray(state)), np.asarray([1.0 + 0.0j, -1.0 + 0.0j]))
        _assert_allclose(spin_flip.weights(jnp.asarray(state)), np.asarray([1.0 + 0.0j, -1.0 + 0.0j]))

        even_doublon_state = np.array([3, 1, 2, 3], dtype=np.int32)
        _assert_allclose(spin_flip.fermion_signs(jnp.asarray(even_doublon_state)), np.asarray([1.0 + 0.0j, 1.0 + 0.0j]))


    def test_spinless_particle_hole_uses_canonical_fock_sign(self):
        state = np.array([1, 0, 1, 0], dtype=np.int32)
        symm = particle_hole_symmetry(4, 1, "spinless_fermion")

        expected_states = np.asarray([state, 1 - state])
        _assert_allclose(symm.transformed_states(jnp.asarray(state)), expected_states)

        expected_sign = _particle_hole_sign_from_operator_identity(_spinless_occ_jvmc(state))
        _assert_allclose(symm.fermion_signs(jnp.asarray(state)), np.asarray([1.0 + 0.0j, expected_sign]))


    def test_plural_spinful_particle_type_and_projector_weight_alias(self):
        symm = particle_hole_symmetry(2, 1, "spinful_fermions")
        assert symm.particle_type == "spinful_fermion"
        assert symm.has_nontrivial_projector_weights
        assert symm.has_nontrivial_factors == symm.has_nontrivial_projector_weights


    def test_spinful_particle_hole_uses_canonical_fock_sign(self):
        state = np.array([3, 1, 2, 0], dtype=np.int32)
        symm = particle_hole_symmetry(2, 2, "spinful_fermion")

        expected_states = np.asarray([state, 3 - state])
        _assert_allclose(symm.transformed_states(jnp.asarray(state)), expected_states)

        expected_sign = _particle_hole_sign_from_operator_identity(_spinful_occ_jvmc(state))
        _assert_allclose(symm.fermion_signs(jnp.asarray(state)), np.asarray([1.0 + 0.0j, expected_sign]))


    def test_spinful_particle_hole_matches_dense_fock_operator(self):
        n_sites = 2
        symm = particle_hole_symmetry(n_sites, 1, "spinful_fermion")
        C = _particle_hole_matrix(2 * n_sites)

        for state_idx in range(1 << (2 * n_sites)):
            state = _spinful_state_from_bits_jvmc(_bits_from_state_index(state_idx, 2 * n_sites))
            states = np.asarray(symm.transformed_states(jnp.asarray(state)))
            signs = np.asarray(symm.fermion_signs(jnp.asarray(state)))

            expected_idx = int(np.flatnonzero(np.abs(C[:, state_idx]) > 0)[0])
            expected_state = _spinful_state_from_bits_jvmc(_bits_from_state_index(expected_idx, 2 * n_sites))
            expected_sign = C[expected_idx, state_idx]

            _assert_allclose(states[0], state)
            _assert_allclose(signs[0], 1.0 + 0.0j)
            _assert_allclose(states[1], expected_state)
            _assert_allclose(signs[1], expected_sign)


    def test_particle_hole_matrix_interchanges_creation_and_annihilation(self):
        n_modes = 4
        C = _particle_hole_matrix(n_modes)
        C_inv = C.conj().T
        for mode in range(n_modes):
            c = _annihilation_matrix(n_modes, mode)
            _assert_allclose(C @ c @ C_inv, c.conj().T)
            _assert_allclose(C @ c.conj().T @ C_inv, c)


    def test_spinful_spin_flip_matches_dense_fock_operator(self):
        n_sites = 2
        symm = spin_flip_symmetry(n_sites, 1, "spinful_fermion")
        F = _spinful_spin_flip_matrix(n_sites)
        F_inv = F.conj().T

        for site in range(n_sites):
            c_up = _annihilation_matrix(2 * n_sites, _spinful_mode_index(site, "up", n_sites))
            c_down = _annihilation_matrix(2 * n_sites, _spinful_mode_index(site, "down", n_sites))
            _assert_allclose(F @ c_up @ F_inv, c_down)
            _assert_allclose(F @ c_down @ F_inv, c_up)

        for state_idx in range(1 << (2 * n_sites)):
            state = _spinful_state_from_bits_jvmc(_bits_from_state_index(state_idx, 2 * n_sites))
            states = np.asarray(symm.transformed_states(jnp.asarray(state)))
            signs = np.asarray(symm.fermion_signs(jnp.asarray(state)))

            expected_idx = int(np.flatnonzero(np.abs(F[:, state_idx]) > 0)[0])
            expected_state = _spinful_state_from_bits_jvmc(_bits_from_state_index(expected_idx, 2 * n_sites))
            expected_sign = F[expected_idx, state_idx]

            _assert_allclose(states[0], state)
            _assert_allclose(signs[0], 1.0 + 0.0j)
            _assert_allclose(states[1], expected_state)
            _assert_allclose(signs[1], expected_sign)


    def test_composed_translation_particle_hole_states_and_signs(self):
        Lx, Ly = 2, 2
        state = np.array([3, 1, 2, 0], dtype=np.int32)
        translation = square_translation_symmetry(Lx, Ly, "spinful_fermion")
        ph = particle_hole_symmetry(Lx, Ly, "spinful_fermion")
        composed = compose_projectors(translation, ph, name="translation_x_particle_hole")

        states = np.asarray(composed.transformed_states(jnp.asarray(state)))
        signs = np.asarray(composed.fermion_signs(jnp.asarray(state)))
        for row, got_state, got_sign, local_sign in zip(np.asarray(composed.perm_array), states, signs, np.asarray(composed.local_sign_array)):
            pre_local = state[row]
            expected_state = 3 - pre_local if local_sign else pre_local
            _assert_allclose(got_state, expected_state)

            perm_sign = _permutation_sign_from_occ(_spinful_occ_jvmc(state), _spinful_orbital_perm_jvmc(row))
            ph_sign = _particle_hole_sign_from_operator_identity(_spinful_occ_jvmc(pre_local)) if local_sign else 1.0 + 0.0j
            _assert_allclose(got_sign, perm_sign * ph_sign)


    def test_projector_multiplication_ux(self):
        translation = square_translation_symmetry(2, 2, "spinful_fermion")
        ph = particle_hole_symmetry(2, 2, "spinful_fermion")

        composed = translation * ph
        explicit = compose_projectors(translation, ph, name="translation_x_particle_hole")
        _assert_allclose(composed.perm_array, explicit.perm_array)
        _assert_allclose(composed.local_sign_array, explicit.local_sign_array)

        raw_net = _DummyNet()
        projected = composed * raw_net
        assert projected.base_net is raw_net
        assert projected.symmetry is composed

        try:
            _ = raw_net * composed
        except TypeError:
            pass
        else:
            raise AssertionError("right-acting raw_net * symmetry should remain unsupported")


    def test_projected_orbit_net_average_modes(self):
        state = jnp.asarray([0, 1, 1, 0], dtype=jnp.int32)
        symm = spin_flip_symmetry(4, 1)
        base_net = _FirstSiteNet()
        states = np.asarray(symm.transformed_states(state))
        coeffs = np.asarray([base_net.apply({}, jnp.asarray(row)) for row in states], dtype=np.complex128)

        exp_expected = np.log(np.mean(np.exp(coeffs)))
        log_expected = np.mean(coeffs)
        sep_expected = 0.5 * np.log(np.mean(np.exp(2 * np.real(coeffs)))) + 1j * np.angle(np.mean(np.exp(1j * np.imag(coeffs))))

        for average, expected in (("exp", exp_expected), ("log", log_expected), ("sep", sep_expected)):
            projected = ProjectedOrbitNet(base_net=base_net, symmetry=symm, symmetry_average=average)
            _assert_allclose(projected.apply({}, state), expected)


    def test_non_projector_averages_warn_for_nontrivial_weights(self):
        state = jnp.asarray([0, 1, 1, 0], dtype=jnp.int32)
        symm = square_translation_symmetry(4, 1, "spin", q=(1, 0))
        base_net = _FirstSiteNet()

        for average in ("log", "sep"):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                ProjectedOrbitNet(base_net=base_net, symmetry=symm, symmetry_average=average).init({}, state)
            assert any("does not apply nontrivial projector weights" in str(warning.message) for warning in caught)


    def test_exp_average_includes_nontrivial_momentum_weights(self):
        state = jnp.asarray([0, 1, 1, 0], dtype=jnp.int32)
        symm = square_translation_symmetry(4, 1, "spin", q=(1, 0))
        base_net = _FirstSiteNet()

        states = np.asarray(symm.transformed_states(state))
        weights = np.asarray(symm.weights(state))
        coeffs = np.asarray([base_net.apply({}, jnp.asarray(row)) for row in states], dtype=np.complex128)
        expected = np.log(np.mean(weights * np.exp(coeffs)))

        projected = ProjectedOrbitNet(base_net=base_net, symmetry=symm, symmetry_average="exp")
        _assert_allclose(projected.apply({}, state), expected)


    def test_projected_orbit_net_accepts_custom_average_function(self):
        state = jnp.asarray([0, 1, 1, 0], dtype=jnp.int32)
        symm = square_translation_symmetry(4, 1, "spin", q=(1, 0))
        base_net = _FirstSiteNet()

        states = np.asarray(symm.transformed_states(state))
        weights = np.asarray(symm.weights(state))
        coeffs = np.asarray([base_net.apply({}, jnp.asarray(row)) for row in states], dtype=np.complex128)
        expected = np.log(np.sum(weights * np.exp(coeffs)))

        projected = ProjectedOrbitNet(base_net=base_net, symmetry=symm, symmetry_average=_weighted_sum_average)
        _assert_allclose(projected.apply({}, state), expected)


    def test_coefficient_state_net_is_vectorized_over_transformed_states(self):
        state = jnp.asarray([0, 1, 1, 0], dtype=jnp.int32)
        symm = square_translation_symmetry(4, 1, "spin")
        base_net = _FirstSiteNet()
        coefficient_state_net = _StateCoefficientNet()

        states = np.asarray(symm.transformed_states(state))
        base_logs = np.asarray([base_net.apply({}, jnp.asarray(row)) for row in states], dtype=np.complex128)
        coeff_logs = np.asarray([coefficient_state_net.apply({}, jnp.asarray(row)) for row in states], dtype=np.complex128)
        expected = np.log(np.mean(np.exp(base_logs + coeff_logs)))

        projected = ProjectedOrbitNet(
            base_net=base_net,
            symmetry=symm,
            coefficient_state_net=coefficient_state_net,
            symmetry_average="exp",
        )
        _assert_allclose(projected.apply({}, state), expected)


    def test_projected_orbit_net_rejects_both_coefficient_interfaces(self):
        state = jnp.asarray([0, 1, 1, 0], dtype=jnp.int32)
        symm = square_translation_symmetry(4, 1, "spin")
        projected = ProjectedOrbitNet(
            base_net=_FirstSiteNet(),
            symmetry=symm,
            coefficient_net=_OrbitCoefficientNet(symm.nsymm),
            coefficient_state_net=_StateCoefficientNet(),
        )

        try:
            projected.init({}, state)
        except ValueError as exc:
            assert "only one" in str(exc)
        else:
            raise AssertionError("using both coefficient interfaces should fail")


    def test_exp_average_preserves_exact_projector_cancellation(self):
        state = jnp.asarray([0, 0], dtype=jnp.int32)
        symm = SymmetryProjector(
            Lx=2,
            Ly=1,
            particle_type="spin",
            perm=((0, 1), (1, 0)),
            character=(1.0 + 0.0j, -1.0 + 0.0j),
            name="exact_cancellation",
        )

        projected = ProjectedOrbitNet(base_net=_DummyNet(), symmetry=symm, symmetry_average="exp")
        got = np.asarray(projected.apply({}, state))
        assert np.isneginf(np.real(got))


    def test_autoregressive_sep_average_uses_fixed_half_log_prob_factor(self):
        state = jnp.asarray([0, 1, 1, 0], dtype=jnp.int32)
        symm = spin_flip_symmetry(4, 1)
        base_net = _AutoregressiveLikeNet()
        states = np.asarray(symm.transformed_states(state))
        coeffs = np.asarray([base_net.apply({}, jnp.asarray(row)) for row in states], dtype=np.complex128)
        expected = 0.5 * np.log(np.mean(np.exp(2 * np.real(coeffs)))) + 1j * np.angle(np.mean(np.exp(1j * np.imag(coeffs))))

        projected = ProjectedOrbitNet(base_net=base_net, symmetry=symm, symmetry_average="sep")
        _assert_allclose(projected.apply({}, state), expected)


    def test_autoregressive_projector_rejects_nontrivial_factors(self):
        state = jnp.asarray([0, 1, 1, 0], dtype=jnp.int32)
        ar_net = _AutoregressiveLikeNet()

        trivial = square_translation_symmetry(4, 1, "spin", q=(0, 0))
        try:
            ProjectedOrbitNet(base_net=ar_net, symmetry=trivial).init({}, state)
        except ValueError as exc:
            assert "symmetry_average='sep'" in str(exc)
        else:
            raise AssertionError("autoregressive projection with non-sep averaging should fail")

        ProjectedOrbitNet(base_net=ar_net, symmetry=trivial, symmetry_average="sep").init({}, state)

        nontrivial_character = square_translation_symmetry(4, 1, "spin", q=(1, 0))
        try:
            ProjectedOrbitNet(base_net=ar_net, symmetry=nontrivial_character, symmetry_average="sep").init({}, state)
        except ValueError as exc:
            assert "Autoregressive" in str(exc)
        else:
            raise AssertionError("autoregressive projection with nontrivial character should fail")

        fermion_ph = particle_hole_symmetry(4, 1, "spinless_fermion")
        try:
            ProjectedOrbitNet(base_net=ar_net, symmetry=fermion_ph, symmetry_average="sep").init({}, state)
        except ValueError as exc:
            assert "Autoregressive" in str(exc)
        else:
            raise AssertionError("autoregressive projection with fermionic particle-hole sign should fail")

    def test_projected_orbit_net_keeps_the_shape_of_the_configurations(self):
        """
        A network for two-dimensional configurations, e.g. a CNN with sampleShape (Ly, Lx),
        gets the transformed configurations with the shape of s, and so do the coefficient
        networks.
        """
        Lx, Ly = 3, 2
        state = jnp.asarray([[1, 0, 1], [0, 0, 1]], dtype=jnp.int32)
        symm = square_translation_symmetry(Lx, Ly, "spinless_fermion", q=(1, 0))
        base_net = _TwoDimensionalNet()

        # With the direct site order, the flat index is x + Lx * y: a row reshaped to (Ly, Lx)
        states = np.asarray(symm.transformed_states(state)).reshape(symm.nsymm, Ly, Lx)
        weights = np.asarray(symm.weights(state))
        base_logs = np.asarray([base_net.apply({}, jnp.asarray(row)) for row in states], dtype=np.complex128)
        expected = np.log(np.mean(weights * np.exp(base_logs)))
        projected = ProjectedOrbitNet(base_net=base_net, symmetry=symm)
        _assert_allclose(projected.apply({}, state), expected)

        coeff_logs = np.asarray([base_net.apply({}, jnp.asarray(row)) for row in states], dtype=np.complex128)
        expected = np.log(np.mean(weights * np.exp(base_logs + coeff_logs)))
        projected = ProjectedOrbitNet(base_net=base_net, symmetry=symm, coefficient_state_net=_TwoDimensionalNet())
        _assert_allclose(projected.apply({}, state), expected)

        coefficient_net = _TwoDimensionalOrbitCoefficientNet(symm.nsymm)
        coeff_logs = np.asarray(coefficient_net.apply({}, state), dtype=np.complex128)
        expected = np.log(np.mean(weights * np.exp(base_logs + coeff_logs)))
        projected = ProjectedOrbitNet(base_net=base_net, symmetry=symm, coefficient_net=coefficient_net)
        _assert_allclose(projected.apply({}, state), expected)

    def test_projected_orbit_net_sample_keeps_the_shape_of_the_configurations(self):
        symm = square_translation_symmetry(3, 2, "spin")
        generator = _TwoDimensionalGenerator()
        projected = ProjectedOrbitNet(base_net=generator, symmetry=symm, symmetry_average="sep")

        for seed in range(4):
            key = jax.random.PRNGKey(seed)
            sample = np.asarray(projected.apply({}, key, method=ProjectedOrbitNet.sample))
            base_sample = generator.apply({}, jax.random.split(key)[0], method=_TwoDimensionalGenerator.sample)
            orbit = {tuple(row) for row in np.asarray(symm.transformed_states(base_sample)).tolist()}

            self.assertEqual(sample.shape, (2, 3))
            self.assertIn(tuple(sample.reshape(-1).tolist()), orbit)

if __name__ == "__main__":
    unittest.main()