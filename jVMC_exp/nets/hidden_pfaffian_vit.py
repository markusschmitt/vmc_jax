from __future__ import annotations

from typing import Any

import flax.linen as nn
import jax.numpy as jnp

from jVMC_exp import global_defs
from jVMC_exp.nets.vit import ViterittiEncoderBlock, ViterittiOutputHead
from jVMC_exp.symmetry import ProjectedOrbitNet, SymmetryProjector
from jVMC_exp.symmetry.projector import split_spinful_site_state


def _param_dtype(param_dtype):
    return jnp.dtype(global_defs.DT_PARAMS_REAL if param_dtype is None else param_dtype)


def _compute_dtype(compute_dtype, param_dtype):
    return jnp.dtype(param_dtype if compute_dtype is None else compute_dtype)


def _out_cpx_dtype():
    return jnp.dtype(global_defs.DT_OUT_CPX)


class RawSpinfulSiteViTLogNet(nn.Module):
    Lx: int
    Ly: int
    patch_size: int
    layers: int
    embed_dim: int
    heads: int
    theta0: float = 0.0
    attention_kind: str = "patch-mixing-attention"
    distance_gamma_init: float = 3.0
    param_dtype: Any = None
    compute_dtype: Any = None

    @nn.compact
    def __call__(self, s):
        param_dtype = _param_dtype(self.param_dtype)
        compute_dtype = _compute_dtype(self.compute_dtype, param_dtype)
        n_sites = self.Lx * self.Ly
        up_flat, dn_flat = split_spinful_site_state(s, n_sites)
        up = up_flat.reshape(self.Ly, self.Lx)
        dn = dn_flat.reshape(self.Ly, self.Lx)
        features = jnp.stack([up, dn], axis=-1).astype(compute_dtype)
        features = 2.0 * features - 1.0
        if self.Lx % self.patch_size != 0 or self.Ly % self.patch_size != 0:
            raise ValueError("patch_size must divide both Lx and Ly.")
        nx = self.Lx // self.patch_size
        ny = self.Ly // self.patch_size
        n_patches = nx * ny
        patch_dim = self.patch_size * self.patch_size * 2
        x = features.reshape(ny, self.patch_size, nx, self.patch_size, 2)
        x = jnp.transpose(x, (0, 2, 1, 3, 4)).reshape(n_patches, patch_dim)
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
        return ViterittiOutputHead(
            embed_dim=self.embed_dim,
            theta0=self.theta0,
            phase_scale=0.0,
            param_dtype=param_dtype,
            compute_dtype=compute_dtype,
            name="output_head",
        )(x).astype(_out_cpx_dtype())


class SingletPfaffianSpinful(nn.Module):
    Lx: int
    Ly: int
    particles_per_spin: int | None = None
    param_dtype: Any = None

    @nn.compact
    def __call__(self, s):
        param_dtype = _param_dtype(self.param_dtype)
        out_dtype = _out_cpx_dtype()
        n_sites = self.Lx * self.Ly
        n = (
            self.particles_per_spin
            if self.particles_per_spin is not None
            else n_sites // 2
        )
        up_raw, dn_raw = split_spinful_site_state(s, n_sites)
        up = up_raw > 0
        dn = dn_raw > 0
        idx_up = jnp.nonzero(up, size=n, fill_value=0)[0]
        idx_dn = jnp.nonzero(dn, size=n, fill_value=0)[0]
        valid = (jnp.sum(up) == n) & (jnp.sum(dn) == n)
        f_re = self.param(
            "F_up_down_re",
            nn.initializers.normal(stddev=0.05),
            (n_sites, n_sites),
            param_dtype,
        )
        f_im = self.param(
            "F_up_down_im",
            nn.initializers.normal(stddev=0.05),
            (n_sites, n_sites),
            param_dtype,
        )
        F = f_re.astype(out_dtype) + 1j * f_im.astype(out_dtype)
        F_occ = F[idx_up[:, None], idx_dn[None, :]]
        sign, logabs = jnp.linalg.slogdet(F_occ)
        singlet_sign = (-1) ** (n * (n - 1) // 2)
        log_value = jnp.log(sign.astype(out_dtype) * singlet_sign) + logabs.astype(
            out_dtype
        )
        return jnp.where(
            valid, log_value, jnp.asarray(-jnp.inf + 0.0j, dtype=out_dtype)
        )


class HiddenPfaffianViT(nn.Module):
    Lx: int
    Ly: int
    symmetry: SymmetryProjector
    patch_size: int = 1
    layers: int = 2
    embed_dim: int = 16
    heads: int = 2
    particles_per_spin: int | None = None
    attention_kind: str = "patch-mixing-attention"
    distance_gamma_init: float = 3.0
    symmetry_average: str = "exp"
    param_dtype: Any = None
    compute_dtype: Any = None

    @nn.compact
    def __call__(self, s):
        param_dtype = _param_dtype(self.param_dtype)
        compute_dtype = _compute_dtype(self.compute_dtype, param_dtype)
        coeff_base = RawSpinfulSiteViTLogNet(
            Lx=self.Lx,
            Ly=self.Ly,
            patch_size=self.patch_size,
            layers=self.layers,
            embed_dim=self.embed_dim,
            heads=self.heads,
            attention_kind=self.attention_kind,
            distance_gamma_init=self.distance_gamma_init,
            param_dtype=param_dtype,
            compute_dtype=compute_dtype,
            name="coefficient_vit_base",
        )
        base_net = SingletPfaffianSpinful(
            Lx=self.Lx,
            Ly=self.Ly,
            particles_per_spin=self.particles_per_spin,
            param_dtype=param_dtype,
            name="singlet_pfaffian",
        )
        return ProjectedOrbitNet(
            base_net=base_net,
            symmetry=self.symmetry,
            coefficient_state_net=coeff_base,
            symmetry_average=self.symmetry_average,
            name="hidden_pfaffian_projector",
        )(s)
