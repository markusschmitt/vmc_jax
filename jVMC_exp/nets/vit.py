from __future__ import annotations

import math
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp

from jVMC_exp import global_defs

PATCH_MIXING_ATTENTION = "patch-mixing-attention"
VITERITTI_DISTANCE_ATTENTION = "viteritti-distance-attention"
_VIT_ATTENTION_ALIASES = {
    "patch-mixing-attention": PATCH_MIXING_ATTENTION,
    "patch_mixing_attention": PATCH_MIXING_ATTENTION,
    "patch-mixing": PATCH_MIXING_ATTENTION,
    "patch_mixing": PATCH_MIXING_ATTENTION,
    "viteritti-distance-attention": VITERITTI_DISTANCE_ATTENTION,
    "viteritti_distance_attention": VITERITTI_DISTANCE_ATTENTION,
    "viteritt-distance-attention": VITERITTI_DISTANCE_ATTENTION,
    "viteritt_distance_attention": VITERITTI_DISTANCE_ATTENTION,
    "distance-attention": VITERITTI_DISTANCE_ATTENTION,
    "distance_attention": VITERITTI_DISTANCE_ATTENTION,
}


def _param_dtype(param_dtype):
    return jnp.dtype(global_defs.DT_PARAMS_REAL if param_dtype is None else param_dtype)


def _compute_dtype(compute_dtype, param_dtype):
    return jnp.dtype(param_dtype if compute_dtype is None else compute_dtype)


def _out_real_dtype():
    return jnp.dtype(global_defs.DT_OUT_REAL)


def _out_cpx_dtype():
    return jnp.dtype(global_defs.DT_OUT_CPX)


def _complex_logmeanexp(z):
    z = jnp.asarray(z, dtype=_out_cpx_dtype())
    shift = jnp.max(jnp.real(z))
    return shift + jnp.log(jnp.mean(jnp.exp(z - shift)))


def _d4_square_transforms(s01, Lx: int, Ly: int):
    if Lx != Ly:
        raise ValueError("D4 symmetry averaging requires a square lattice.")
    a = jnp.asarray(s01).reshape(Ly, Lx)
    flip = jnp.flip(a, axis=0)
    return (
        a,
        jnp.rot90(a, 1),
        jnp.rot90(a, 2),
        jnp.rot90(a, 3),
        flip,
        jnp.rot90(flip, 1),
        jnp.rot90(flip, 2),
        jnp.rot90(flip, 3),
    )


def _translation_project_log_square(
    eval_log_one,
    s01,
    Lx: int,
    Ly: int,
    q: tuple[int, int],
    representative_shape: tuple[int, int] | None = None,
):
    qx, qy = int(q[0]) % Lx, int(q[1]) % Ly
    if representative_shape is None:
        rep_x, rep_y = Lx, Ly
    else:
        rep_x, rep_y = int(representative_shape[0]), int(representative_shape[1])
        if rep_x <= 0 or rep_y <= 0 or Lx % rep_x != 0 or Ly % rep_y != 0:
            raise ValueError(
                "translation representative shape must divide the lattice."
            )
        if (qx, qy) != (0, 0) and (rep_x, rep_y) != (Lx, Ly):
            raise ValueError(
                "Reduced translation representatives are only valid in the q=(0,0) sector."
            )
    a = jnp.asarray(s01).reshape(Ly, Lx)
    vals = []
    for sy in range(rep_y):
        for sx in range(rep_x):
            phase_arg = (qx * sx / Lx) + (qy * sy / Ly)
            phase = -2j * jnp.pi * phase_arg
            shifted = jnp.roll(a, shift=(sy, sx), axis=(0, 1)).reshape(Lx * Ly)
            vals.append(
                jnp.asarray(phase, dtype=_out_cpx_dtype()) + eval_log_one(shifted)
            )
    return _complex_logmeanexp(jnp.asarray(vals, dtype=_out_cpx_dtype()))


def _stable_log_cosh_real(x, compute_dtype):
    x = jnp.asarray(x, dtype=compute_dtype)
    ax = jnp.abs(x)
    return ax + jax.nn.softplus(-2.0 * ax) - jnp.log(jnp.asarray(2.0, dtype=x.dtype))


def _stable_log_cosh_complex(z):
    z = jnp.asarray(z, dtype=_out_cpx_dtype())
    log2 = jnp.log(jnp.asarray(2.0, dtype=z.real.dtype))
    return jnp.where(
        jnp.real(z) >= 0.0,
        z + jnp.log1p(jnp.exp(-2.0 * z)) - log2,
        -z + jnp.log1p(jnp.exp(2.0 * z)) - log2,
    )


def _translation_invariant_attention_alpha(alpha, n_patches: int):
    side = math.isqrt(int(n_patches))
    if side * side != n_patches:
        raise ValueError(
            "Translation-invariant ViT attention requires a square patch grid."
        )
    alpha_map = alpha.reshape(alpha.shape[0], side, side)
    rows = []
    for iy in range(side):
        for ix in range(side):
            rows.append(
                jnp.roll(jnp.roll(alpha_map, iy, axis=-2), ix, axis=-1).reshape(
                    alpha.shape[0], n_patches
                )
            )
    return jnp.stack(rows, axis=1)


def canonical_vit_attention_kind(attention_kind: str) -> str:
    key = str(attention_kind).strip().lower()
    try:
        return _VIT_ATTENTION_ALIASES[key]
    except KeyError as exc:
        allowed = ", ".join(
            sorted({PATCH_MIXING_ATTENTION, VITERITTI_DISTANCE_ATTENTION})
        )
        raise ValueError(f"attention_kind must be one of: {allowed}.") from exc


def _inverse_softplus(x: float):
    x = float(x)
    if x <= 0.0:
        raise ValueError("spatial attention gamma init must be positive.")
    xj = jnp.asarray(x, dtype=jnp.float32)
    return xj + jnp.log(-jnp.expm1(-xj))


def _periodic_patch_distance_matrix(n_patches: int):
    side = math.isqrt(int(n_patches))
    if side * side != n_patches:
        raise ValueError("Viteritti distance attention requires a square patch grid.")
    coords = jnp.asarray(
        [(iy, ix) for iy in range(side) for ix in range(side)], dtype=jnp.float32
    )
    dy = jnp.abs(coords[:, None, 0] - coords[None, :, 0])
    dx = jnp.abs(coords[:, None, 1] - coords[None, :, 1])
    dy = jnp.minimum(dy, side - dy)
    dx = jnp.minimum(dx, side - dx)
    return jnp.sqrt(dx * dx + dy * dy).astype(jnp.float32)


def _constant_init(value):
    def init(key, shape, dtype=jnp.float32):
        del key, shape
        return jnp.asarray(value, dtype=dtype)

    return init


class ViterittiFactoredAttention(nn.Module):
    embed_dim: int
    heads: int
    n_patches: int
    attention_kind: str = PATCH_MIXING_ATTENTION
    distance_gamma_init: float = 3.0
    param_dtype: Any = None
    compute_dtype: Any = None

    @nn.compact
    def __call__(self, x):
        param_dtype = _param_dtype(self.param_dtype)
        compute_dtype = _compute_dtype(self.compute_dtype, param_dtype)
        if self.embed_dim % self.heads != 0:
            raise ValueError(
                "ViterittiFactoredAttention requires embed_dim divisible by heads."
            )
        head_dim = self.embed_dim // self.heads
        x = jnp.asarray(x, dtype=compute_dtype)
        attention_kind = canonical_vit_attention_kind(self.attention_kind)
        if attention_kind == PATCH_MIXING_ATTENTION:
            alpha = self.param(
                "alpha",
                nn.initializers.xavier_uniform(),
                (self.heads, self.n_patches),
                param_dtype,
            )
            weights = _translation_invariant_attention_alpha(
                jnp.asarray(alpha, dtype=compute_dtype), self.n_patches
            )
        elif attention_kind == VITERITTI_DISTANCE_ATTENTION:
            alpha = self.param(
                "alpha",
                nn.initializers.ones,
                (self.heads, self.n_patches),
                param_dtype,
            )
            alpha_weights = _translation_invariant_attention_alpha(
                jnp.asarray(alpha, dtype=compute_dtype), self.n_patches
            )
            raw_gamma = self.param(
                "raw_gamma",
                _constant_init(
                    jnp.full((self.heads,), _inverse_softplus(self.distance_gamma_init))
                ),
                (self.heads,),
                param_dtype,
            )
            gamma = jax.nn.softplus(jnp.asarray(raw_gamma, dtype=compute_dtype))
            distances = jnp.asarray(
                _periodic_patch_distance_matrix(self.n_patches), dtype=compute_dtype
            )
            distance_kernel = jax.nn.softmax(
                -gamma[:, None, None] * distances[None, :, :], axis=-1
            )
            weights = distance_kernel * alpha_weights
        else:
            raise ValueError(f"Unsupported attention_kind: {self.attention_kind}")
        weights = jnp.asarray(weights, dtype=compute_dtype)

        values = nn.Dense(
            self.embed_dim,
            param_dtype=param_dtype,
            kernel_init=nn.initializers.xavier_uniform(),
            name="value",
        )(x)
        values = values.reshape(self.n_patches, self.heads, head_dim)
        values = jnp.transpose(values, (1, 0, 2))
        attended = jnp.einsum("hij,hjr->hir", weights, values)
        attended = jnp.transpose(attended, (1, 0, 2)).reshape(
            self.n_patches, self.embed_dim
        )
        return nn.Dense(
            self.embed_dim,
            param_dtype=param_dtype,
            kernel_init=nn.initializers.xavier_uniform(),
            name="out",
        )(attended)


class ViterittiEncoderBlock(nn.Module):
    embed_dim: int
    heads: int
    n_patches: int
    attention_kind: str = PATCH_MIXING_ATTENTION
    distance_gamma_init: float = 3.0
    param_dtype: Any = None
    compute_dtype: Any = None

    @nn.compact
    def __call__(self, x):
        param_dtype = _param_dtype(self.param_dtype)
        compute_dtype = _compute_dtype(self.compute_dtype, param_dtype)
        y = nn.LayerNorm(param_dtype=param_dtype, name="ln_attn")(
            jnp.asarray(x, dtype=compute_dtype)
        )
        y = ViterittiFactoredAttention(
            embed_dim=self.embed_dim,
            heads=self.heads,
            n_patches=self.n_patches,
            attention_kind=self.attention_kind,
            distance_gamma_init=self.distance_gamma_init,
            param_dtype=param_dtype,
            compute_dtype=compute_dtype,
            name="attention",
        )(y)
        x = x + y
        y = nn.LayerNorm(param_dtype=param_dtype, name="ln_ff")(x)
        y = nn.Dense(
            4 * self.embed_dim,
            param_dtype=param_dtype,
            kernel_init=nn.initializers.xavier_uniform(),
            name="ff_up",
        )(y)
        y = jax.nn.gelu(y)
        y = nn.Dense(
            self.embed_dim,
            param_dtype=param_dtype,
            kernel_init=nn.initializers.xavier_uniform(),
            name="ff_down",
        )(y)
        return x + y


class ViterittiOutputHead(nn.Module):
    embed_dim: int
    theta0: float = 0.0
    phase_scale: float = 0.0
    param_dtype: Any = None
    compute_dtype: Any = None

    @nn.compact
    def __call__(self, x):
        param_dtype = _param_dtype(self.param_dtype)
        compute_dtype = _compute_dtype(self.compute_dtype, param_dtype)
        z = jnp.sum(jnp.asarray(x, dtype=compute_dtype), axis=0)
        z = nn.LayerNorm(param_dtype=param_dtype, name="out_ln")(z)
        out_real = nn.Dense(
            self.embed_dim,
            param_dtype=param_dtype,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=jax.nn.initializers.zeros,
            name="out_real_dense",
        )(z)
        out_real = nn.LayerNorm(param_dtype=param_dtype, name="out_real_ln")(out_real)
        if abs(float(self.phase_scale)) <= 0.0:
            return jnp.sum(_stable_log_cosh_real(out_real, compute_dtype)).astype(
                _out_real_dtype()
            ) + jnp.asarray(self.theta0, dtype=_out_real_dtype())
        out_imag = nn.Dense(
            self.embed_dim,
            param_dtype=param_dtype,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=jax.nn.initializers.zeros,
            name="out_imag_dense",
        )(z)
        out_imag = nn.LayerNorm(param_dtype=param_dtype, name="out_imag_ln")(out_imag)
        out = out_real.astype(_out_cpx_dtype()) + 1j * jnp.asarray(
            self.phase_scale, dtype=_out_real_dtype()
        ) * out_imag.astype(_out_cpx_dtype())
        return jnp.sum(_stable_log_cosh_complex(out)) + jnp.asarray(
            self.theta0, dtype=_out_cpx_dtype()
        )


class CPMPhaseLayer(nn.Module):
    Lx: int
    Ly: int
    n_sites: int
    embed_dim: int
    pair_init_scale: float = 1e-2
    context_init_scale: float = 1e-2
    param_dtype: Any = None
    compute_dtype: Any = None

    @nn.compact
    def __call__(self, x, s01):
        param_dtype = _param_dtype(self.param_dtype)
        compute_dtype = _compute_dtype(self.compute_dtype, param_dtype)
        z = jnp.sum(jnp.asarray(x, dtype=compute_dtype), axis=0)
        z = nn.LayerNorm(param_dtype=param_dtype, name="cpm_ln")(z)
        local_bias = self.param("local_bias", nn.initializers.zeros, (2,), param_dtype)
        context_kernel = self.param(
            "context_kernel",
            nn.initializers.normal(stddev=self.context_init_scale),
            (2, self.embed_dim),
            param_dtype,
        )
        pair_table = self.param(
            "pair_table",
            nn.initializers.normal(stddev=self.pair_init_scale),
            (self.Ly, self.Lx, 2, 2),
            param_dtype,
        )

        s = jnp.asarray(s01, dtype=jnp.int32).reshape(self.Ly, self.Lx)
        local_terms = jnp.asarray(local_bias, dtype=compute_dtype)[s]
        context_terms = jnp.einsum(
            "...d,d->...", jnp.asarray(context_kernel, dtype=compute_dtype)[s], z
        )

        pair = jnp.asarray(0.0, dtype=compute_dtype)
        tables = jnp.asarray(pair_table, dtype=compute_dtype)
        for dy in range(self.Ly):
            for dx in range(self.Lx):
                if dx == 0 and dy == 0:
                    continue
                shifted = jnp.roll(s, shift=(-dy, -dx), axis=(0, 1))
                pair = pair + jnp.sum(tables[dy, dx][s, shifted])
        pair = pair / jnp.asarray(2 * self.n_sites, dtype=compute_dtype)

        phase = jnp.sum(local_terms + context_terms) / jnp.sqrt(
            jnp.asarray(self.n_sites, dtype=compute_dtype)
        )
        phase = phase + pair
        return phase.astype(_out_real_dtype())


class ViterittiSpatialViTReal(nn.Module):
    Lx: int
    Ly: int
    patch_size: int
    layers: int
    embed_dim: int
    heads: int
    theta0: float = 0.0
    phase_scale: float = 0.0
    cpm_phase_layer: bool = False
    attention_kind: str = PATCH_MIXING_ATTENTION
    distance_gamma_init: float = 3.0
    param_dtype: Any = None
    compute_dtype: Any = None

    @nn.compact
    def __call__(self, s_spin, s01=None):
        param_dtype = _param_dtype(self.param_dtype)
        compute_dtype = _compute_dtype(self.compute_dtype, param_dtype)
        if self.patch_size <= 0:
            raise ValueError("vit_patch_size must be positive.")
        if self.Lx % self.patch_size != 0 or self.Ly % self.patch_size != 0:
            raise ValueError("vit_patch_size must divide both Lx and Ly.")
        if self.layers < 1:
            raise ValueError("ViterittiSpatialViTReal requires at least one layer.")
        if self.embed_dim % self.heads != 0:
            raise ValueError("vit_embed_dim/channels must be divisible by vit_heads.")
        nx = self.Lx // self.patch_size
        ny = self.Ly // self.patch_size
        n_patches = nx * ny
        patch_dim = self.patch_size * self.patch_size

        x = jnp.asarray(s_spin, dtype=compute_dtype).reshape(self.Ly, self.Lx)
        x = x.reshape(ny, self.patch_size, nx, self.patch_size)
        x = jnp.transpose(x, (0, 2, 1, 3)).reshape(n_patches, patch_dim)
        x = nn.Dense(
            self.embed_dim,
            param_dtype=param_dtype,
            kernel_init=nn.initializers.xavier_uniform(),
            name="patch_embedding",
        )(x)
        for i in range(self.layers):
            x = ViterittiEncoderBlock(
                embed_dim=self.embed_dim,
                heads=self.heads,
                n_patches=n_patches,
                attention_kind=self.attention_kind,
                distance_gamma_init=self.distance_gamma_init,
                param_dtype=param_dtype,
                compute_dtype=compute_dtype,
                name=f"encoder_{i}",
            )(x)
        log_value = ViterittiOutputHead(
            embed_dim=self.embed_dim,
            theta0=self.theta0,
            phase_scale=self.phase_scale,
            param_dtype=param_dtype,
            compute_dtype=compute_dtype,
            name="output_head",
        )(x)
        if not self.cpm_phase_layer:
            return log_value
        if s01 is None:
            raise ValueError(
                "CPM phase layer requires the original 0/1 spin configuration."
            )
        phase = CPMPhaseLayer(
            Lx=self.Lx,
            Ly=self.Ly,
            n_sites=self.Lx * self.Ly,
            embed_dim=self.embed_dim,
            param_dtype=param_dtype,
            compute_dtype=compute_dtype,
            name="cpm_phase",
        )(x, s01)
        return log_value.astype(_out_cpx_dtype()) + 1j * phase.astype(_out_cpx_dtype())


class LogViterittiSpatialViTJVMC(nn.Module):
    Lx: int
    Ly: int
    patch_size: int
    layers: int
    embed_dim: int
    heads: int
    sign_rule: str = "none"
    symmetry_average: str = "none"
    translation_q: tuple[int, int] = (0, 0)
    translation_project: bool = False
    theta0: float = 0.0
    phase_scale: float = 0.0
    cpm_phase_layer: bool = False
    attention_kind: str = PATCH_MIXING_ATTENTION
    distance_gamma_init: float = 3.0
    param_dtype: Any = None
    compute_dtype: Any = None

    @nn.compact
    def __call__(self, s):
        param_dtype = _param_dtype(self.param_dtype)
        compute_dtype = _compute_dtype(self.compute_dtype, param_dtype)
        s01 = jnp.asarray(s, dtype=compute_dtype)
        parity = (jnp.indices((self.Ly, self.Lx)).sum(axis=0) % 2).reshape(
            self.Lx * self.Ly
        )
        log_amplitude_net = ViterittiSpatialViTReal(
            Lx=self.Lx,
            Ly=self.Ly,
            patch_size=self.patch_size,
            layers=self.layers,
            embed_dim=self.embed_dim,
            heads=self.heads,
            theta0=self.theta0,
            phase_scale=self.phase_scale,
            cpm_phase_layer=self.cpm_phase_layer,
            attention_kind=self.attention_kind,
            distance_gamma_init=self.distance_gamma_init,
            param_dtype=param_dtype,
            compute_dtype=compute_dtype,
            name="log_amplitude",
        )

        def eval_log_one(s01_one):
            sign_phase = jnp.asarray(0.0, dtype=_out_real_dtype())
            if self.sign_rule == "marshall":
                exponent = jnp.sum(
                    jnp.asarray(s01_one.reshape(self.Lx * self.Ly), dtype=jnp.int32)
                    * parity
                )
                sign_phase = jnp.pi * jnp.asarray(exponent % 2, dtype=_out_real_dtype())
            elif self.sign_rule != "none":
                raise ValueError("sign_rule must be 'none' or 'marshall'.")
            s_spin = (2.0 * s01_one - 1.0).reshape(self.Lx * self.Ly)
            log_value = log_amplitude_net(s_spin, s01_one)
            return log_value.astype(_out_cpx_dtype()) + 1j * sign_phase.astype(
                _out_cpx_dtype()
            )

        def translation_project(s01_one, q):
            representative_shape = None
            if tuple(q) == (0, 0):
                representative_shape = (self.patch_size, self.patch_size)
            return _translation_project_log_square(
                eval_log_one,
                s01_one,
                self.Lx,
                self.Ly,
                q,
                representative_shape,
            )

        flat = s01.reshape(self.Lx * self.Ly)
        if self.symmetry_average == "none":
            if not self.translation_project:
                return eval_log_one(flat)
            return translation_project(flat, self.translation_q)
        if self.symmetry_average == "d4":
            if tuple(self.translation_q) != (0, 0):
                raise ValueError(
                    "D4 averaging and nonzero momentum projection cannot be combined in one sector."
                )
            vals = [
                translation_project(t.reshape(self.Lx * self.Ly), (0, 0))
                if self.translation_project
                else eval_log_one(t.reshape(self.Lx * self.Ly))
                for t in _d4_square_transforms(flat, self.Lx, self.Ly)
            ]
            return _complex_logmeanexp(jnp.asarray(vals, dtype=_out_cpx_dtype()))
        raise ValueError("symmetry_average must be 'none' or 'd4'.")


ViterittiViT = LogViterittiSpatialViTJVMC
