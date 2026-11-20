from __future__ import annotations

from itertools import product
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import random

from jVMC import global_defs

RWKV_PHASE_MODES = {
    "legacy_transformer",
    "shared_neck",
    "cpm",
}


def _param_dtype(param_dtype):
    return jnp.dtype(global_defs.DT_PARAMS_REAL if param_dtype is None else param_dtype)


def _compute_dtype(compute_dtype, param_dtype):
    return jnp.dtype(param_dtype if compute_dtype is None else compute_dtype)


def _out_real_dtype():
    return jnp.dtype(global_defs.DT_OUT_REAL)


def _out_cpx_dtype():
    return jnp.dtype(global_defs.DT_OUT_CPX)


class RWKV2DCausalBlock(nn.Module):
    layer_num: int = 0
    num_layers: int = 1
    embedding_size: int = 32
    hidden_size: int = 64
    init_variance: float = 0.1
    param_dtype: Any = None
    compute_dtype: Any = None

    @nn.compact
    def __call__(self, x, left_h, up_h):
        param_dtype = _param_dtype(self.param_dtype)
        compute_dtype = _compute_dtype(self.compute_dtype, param_dtype)
        kernel_init = nn.initializers.variance_scaling(
            self.init_variance,
            mode="fan_in",
            distribution="truncated_normal",
        )
        x = jnp.asarray(x, dtype=compute_dtype)
        left_h = jnp.asarray(left_h, dtype=compute_dtype)
        up_h = jnp.asarray(up_h, dtype=compute_dtype)
        ctx = jnp.concatenate((x, left_h, up_h, left_h * up_h), axis=-1)
        ctx = nn.LayerNorm(param_dtype=param_dtype, name="context_ln")(ctx)
        gates = nn.Dense(
            3 * int(self.embedding_size),
            param_dtype=param_dtype,
            kernel_init=kernel_init,
            name="gates",
        )(ctx)
        gate_x, gate_left, gate_up = jnp.split(
            nn.sigmoid(jnp.asarray(gates, dtype=compute_dtype)), 3, axis=-1
        )
        candidate = nn.Dense(
            int(self.hidden_size),
            param_dtype=param_dtype,
            kernel_init=kernel_init,
            name="candidate_in",
        )(ctx)
        candidate = nn.gelu(jnp.asarray(candidate, dtype=compute_dtype))
        candidate = nn.Dense(
            int(self.embedding_size),
            param_dtype=param_dtype,
            kernel_init=kernel_init,
            name="candidate_out",
        )(candidate)
        mixed = (
            gate_x * x
            + gate_left * left_h
            + gate_up * up_h
            + jnp.asarray(candidate, dtype=compute_dtype)
        )
        mixed = nn.LayerNorm(param_dtype=param_dtype, name="mixed_ln")(mixed)
        update = nn.Dense(
            int(self.hidden_size),
            param_dtype=param_dtype,
            kernel_init=kernel_init,
            name="ffn_in",
        )(mixed)
        update = nn.gelu(jnp.asarray(update, dtype=compute_dtype))
        update = nn.Dense(
            int(self.embedding_size),
            param_dtype=param_dtype,
            kernel_init=kernel_init,
            name="ffn_out",
        )(update)
        return jnp.asarray(
            nn.LayerNorm(param_dtype=param_dtype, name="out_ln")(mixed + update),
            dtype=compute_dtype,
        )


class RWKV2DGraphCausalBlock(nn.Module):
    layer_num: int = 0
    num_layers: int = 1
    embedding_size: int = 32
    hidden_size: int = 64
    init_variance: float = 0.1
    param_dtype: Any = None
    compute_dtype: Any = None

    @nn.compact
    def __call__(self, x, left_h, up_h, graph_by_type):
        param_dtype = _param_dtype(self.param_dtype)
        compute_dtype = _compute_dtype(self.compute_dtype, param_dtype)
        kernel_init = nn.initializers.variance_scaling(
            self.init_variance,
            mode="fan_in",
            distribution="truncated_normal",
        )
        x = jnp.asarray(x, dtype=compute_dtype)
        left_h = jnp.asarray(left_h, dtype=compute_dtype)
        up_h = jnp.asarray(up_h, dtype=compute_dtype)
        graph_by_type = jnp.asarray(graph_by_type, dtype=compute_dtype).reshape(
            3 * int(self.embedding_size)
        )
        graph_h = nn.Dense(
            int(self.embedding_size),
            param_dtype=param_dtype,
            kernel_init=kernel_init,
            name="graph_context",
        )(graph_by_type)
        graph_h = nn.LayerNorm(param_dtype=param_dtype, name="graph_ln")(graph_h)
        graph_h = jnp.asarray(graph_h, dtype=compute_dtype)
        ctx = jnp.concatenate((x, left_h, up_h, graph_h, left_h * up_h), axis=-1)
        ctx = nn.LayerNorm(param_dtype=param_dtype, name="context_ln")(ctx)
        gates = nn.Dense(
            4 * int(self.embedding_size),
            param_dtype=param_dtype,
            kernel_init=kernel_init,
            name="gates",
        )(ctx)
        gate_x, gate_left, gate_up, gate_graph = jnp.split(
            nn.sigmoid(jnp.asarray(gates, dtype=compute_dtype)), 4, axis=-1
        )
        candidate = nn.Dense(
            int(self.hidden_size),
            param_dtype=param_dtype,
            kernel_init=kernel_init,
            name="candidate_in",
        )(ctx)
        candidate = nn.gelu(jnp.asarray(candidate, dtype=compute_dtype))
        candidate = nn.Dense(
            int(self.embedding_size),
            param_dtype=param_dtype,
            kernel_init=kernel_init,
            name="candidate_out",
        )(candidate)
        mixed = (
            gate_x * x
            + gate_left * left_h
            + gate_up * up_h
            + gate_graph * graph_h
            + jnp.asarray(candidate, dtype=compute_dtype)
        )
        mixed = nn.LayerNorm(param_dtype=param_dtype, name="mixed_ln")(mixed)
        update = nn.Dense(
            int(self.hidden_size),
            param_dtype=param_dtype,
            kernel_init=kernel_init,
            name="ffn_in",
        )(mixed)
        update = nn.gelu(jnp.asarray(update, dtype=compute_dtype))
        update = nn.Dense(
            int(self.embedding_size),
            param_dtype=param_dtype,
            kernel_init=kernel_init,
            name="ffn_out",
        )(update)
        return jnp.asarray(
            nn.LayerNorm(param_dtype=param_dtype, name="out_ln")(mixed + update),
            dtype=compute_dtype,
        )


class RWKVPhaseTransformerBlock(nn.Module):
    embedding_size: int
    heads: int
    init_variance: float = 0.1
    param_dtype: Any = None
    compute_dtype: Any = None

    @nn.compact
    def __call__(self, x):
        param_dtype = _param_dtype(self.param_dtype)
        compute_dtype = _compute_dtype(self.compute_dtype, param_dtype)
        kernel_init = nn.initializers.variance_scaling(
            self.init_variance,
            mode="fan_in",
            distribution="truncated_normal",
        )
        x = jnp.asarray(x, dtype=compute_dtype)
        y = nn.MultiHeadDotProductAttention(
            int(self.heads),
            param_dtype=param_dtype,
            kernel_init=kernel_init,
        )(x, x)
        x = x + jnp.asarray(y, dtype=compute_dtype)
        y = nn.Dense(
            4 * int(self.embedding_size),
            param_dtype=param_dtype,
            kernel_init=kernel_init,
            name="dense_in",
        )(x)
        y = nn.gelu(jnp.asarray(y, dtype=compute_dtype))
        y = nn.Dense(
            int(self.embedding_size),
            param_dtype=param_dtype,
            kernel_init=kernel_init,
            name="dense_out",
        )(y)
        return x + jnp.asarray(y, dtype=compute_dtype)


class LogRWKV2DAutoregressiveJVMC(nn.Module):
    """Two-dimensional causal autoregressive RWKV log-amplitude network."""

    L: int
    l_hil_dim: int = 2
    patch_size: int = 1
    grid_Lx: int = 1
    grid_Ly: int = 1
    hidden_size: int = 64
    num_heads: int = 1
    num_layers: int = 1
    embedding_size: int = 32
    log_prob_factor: float = 0.5
    one_hot: bool = False
    bias: bool = False
    linear_output: bool = False
    position_embedding: bool = False
    conditional_logits: bool = False
    conditional_phase: bool = False
    translation_shared: bool = False
    temperature: float = 1.0
    logit_clip: float = 0.0
    init_variance: float = 0.1
    flag_phase: bool = False
    phase_scale: float = 1.0
    num_layers_phase: int = 1
    phase_mode: str = "cpm"
    cpm_bias_init_std: float = 3.141592653589793
    cpm_weight_init_std: float = 0.1
    sign_rule: str = "none"
    graph_context: bool = False
    graph_edge_src: tuple[int, ...] = ()
    graph_edge_dst: tuple[int, ...] = ()
    graph_edge_type: tuple[int, ...] = ()
    site_permutation: tuple[int, ...] = ()
    fixed_n_up: int = -1
    param_dtype: Any = None
    compute_dtype: Any = None

    def setup(self):
        param_dtype = _param_dtype(self.param_dtype)
        if int(self.L) <= 0:
            raise ValueError("2D RWKV system size must be positive.")
        if int(self.l_hil_dim) <= 1:
            raise ValueError("2D RWKV local Hilbert dimension must be at least 2.")
        if int(self.patch_size) <= 0 or int(self.L) % int(self.patch_size) != 0:
            raise ValueError("2D RWKV patch_size must be positive and divide L.")
        self.n_patches = int(self.L) // int(self.patch_size)
        self.PL = self.n_patches
        if int(self.grid_Lx) <= 0 or int(self.grid_Ly) <= 0:
            raise ValueError("2D RWKV grid_Lx and grid_Ly must be positive.")
        if int(self.grid_Lx) * int(self.grid_Ly) != self.n_patches:
            raise ValueError(
                "2D RWKV grid shape must match the number of autoregressive tokens: "
                f"grid_Lx*grid_Ly={int(self.grid_Lx) * int(self.grid_Ly)}, tokens={self.n_patches}."
            )
        if int(self.num_layers) < 1:
            raise ValueError("2D RWKV requires at least one layer.")
        if int(self.embedding_size) <= 0 or int(self.hidden_size) <= 0:
            raise ValueError("2D RWKV embedding_size and hidden_size must be positive.")
        if int(self.num_heads) <= 0:
            raise ValueError("2D RWKV num_heads must be positive.")
        if float(self.temperature) <= 0.0:
            raise ValueError("2D RWKV sampling temperature must be positive.")
        if float(self.logit_clip) < 0.0:
            raise ValueError("2D RWKV logit_clip must be nonnegative.")
        if int(self.fixed_n_up) >= 0:
            if int(self.l_hil_dim) != 2:
                raise ValueError(
                    "2D RWKV fixed_n_up currently requires binary local Hilbert space."
                )
            if int(self.fixed_n_up) > int(self.L):
                raise ValueError("2D RWKV fixed_n_up cannot exceed L.")
            if bool(self.linear_output):
                raise ValueError(
                    "2D RWKV fixed_n_up requires linear_output=False for exact masking."
                )
        phase_mode = str(self.phase_mode).lower().replace("-", "_")
        if phase_mode not in RWKV_PHASE_MODES:
            allowed = "', '".join(sorted(RWKV_PHASE_MODES))
            raise ValueError(f"phase_mode must be one of: '{allowed}'.")
        if self.sign_rule not in {"none", "honeycomb_ab_z_phase"}:
            raise ValueError(
                "2D RWKV sign_rule must be 'none' or 'honeycomb_ab_z_phase'."
            )
        if self.sign_rule == "honeycomb_ab_z_phase" and int(self.patch_size) != 2:
            raise ValueError(
                "honeycomb_ab_z_phase for 2D RWKV requires one A/B supersite per token."
            )
        if bool(self.graph_context):
            if not (
                len(self.graph_edge_src)
                == len(self.graph_edge_dst)
                == len(self.graph_edge_type)
            ):
                raise ValueError(
                    "2D graph RWKV edge source, destination, and type arrays must have equal length."
                )
            if len(self.graph_edge_src) == 0:
                raise ValueError(
                    "2D graph RWKV requires at least one causal graph edge."
                )
        if len(self.site_permutation) == 0:
            site_permutation = tuple(range(int(self.L)))
        else:
            site_permutation = tuple(int(site) for site in self.site_permutation)
            if len(site_permutation) != int(self.L):
                raise ValueError(
                    "2D RWKV site_permutation must contain one entry per physical site."
                )
            if sorted(site_permutation) != list(range(int(self.L))):
                raise ValueError(
                    "2D RWKV site_permutation must be a permutation of physical site indices."
                )
        self.site_permutation_array = jnp.asarray(site_permutation, dtype=jnp.int32)

        patch_states = list(
            product(range(int(self.l_hil_dim)), repeat=int(self.patch_size))
        )
        self.patch_states = jnp.asarray(patch_states, dtype=jnp.int32)
        self.patch_bit_counts = jnp.sum(self.patch_states, axis=-1).astype(jnp.int32)
        self.local_hilbert_dim = int(self.l_hil_dim) ** int(self.patch_size)
        self.local_hil_dim = self.local_hilbert_dim
        self.null_token = self.local_hilbert_dim
        self.index_weights = jnp.asarray(
            int(self.l_hil_dim) ** jnp.arange(int(self.patch_size) - 1, -1, -1),
            dtype=jnp.int32,
        )
        if bool(self.position_embedding):
            if bool(self.translation_shared):
                raise ValueError(
                    "2D RWKV translation_shared is incompatible with position_embedding."
                )
            self.token_position_embedding = self.param(
                "token_position_embedding",
                nn.initializers.normal(stddev=1e-2),
                (self.n_patches, int(self.embedding_size)),
                param_dtype,
            )
        if bool(self.conditional_logits):
            zero_init = nn.initializers.zeros
            self.position_logit_bias = self.param(
                "position_logit_bias",
                zero_init,
                (self.local_hilbert_dim,)
                if bool(self.translation_shared)
                else (self.n_patches, self.local_hilbert_dim),
                param_dtype,
            )
            self.left_conditional_logits = self.param(
                "left_conditional_logits",
                zero_init,
                (self.local_hilbert_dim, self.local_hilbert_dim),
                param_dtype,
            )
            self.up_conditional_logits = self.param(
                "up_conditional_logits",
                zero_init,
                (self.local_hilbert_dim, self.local_hilbert_dim),
                param_dtype,
            )
            self.graph_conditional_logits = self.param(
                "graph_conditional_logits",
                zero_init,
                (3, self.local_hilbert_dim, self.local_hilbert_dim),
                param_dtype,
            )

        embedding_input_dim = self.local_hilbert_dim + 1
        if bool(self.one_hot):
            self.embed = nn.Dense(
                int(self.embedding_size),
                use_bias=False,
                param_dtype=param_dtype,
                name="embedding",
            )
        else:
            self.embed = nn.Embed(
                embedding_input_dim,
                int(self.embedding_size),
                param_dtype=param_dtype,
                name="embedding",
            )
        kernel_init = nn.initializers.variance_scaling(
            self.init_variance,
            mode="fan_in",
            distribution="truncated_normal",
        )
        self.base_context = nn.Dense(
            int(self.embedding_size),
            use_bias=True,
            param_dtype=param_dtype,
            kernel_init=kernel_init,
            name="base_context",
        )
        block_cls = (
            RWKV2DGraphCausalBlock if bool(self.graph_context) else RWKV2DCausalBlock
        )
        self.blocks = [
            block_cls(
                layer_num=i,
                num_layers=int(self.num_layers),
                embedding_size=int(self.embedding_size),
                hidden_size=int(self.hidden_size),
                init_variance=self.init_variance,
                param_dtype=param_dtype,
                compute_dtype=self.compute_dtype,
                name=f"block_{i}",
            )
            for i in range(int(self.num_layers))
        ]
        self.neck = nn.Dense(
            int(self.hidden_size),
            use_bias=bool(self.bias),
            param_dtype=param_dtype,
            kernel_init=kernel_init,
            name="neck",
        )
        self.head = nn.Dense(
            self.local_hilbert_dim,
            use_bias=False,
            param_dtype=param_dtype,
            kernel_init=kernel_init,
            name="head",
        )
        if bool(self.flag_phase):
            if phase_mode == "cpm":
                self.cpm_bias = self.param(
                    "CPMBias",
                    nn.initializers.normal(stddev=float(self.cpm_bias_init_std)),
                    (self.local_hilbert_dim,)
                    if bool(self.translation_shared)
                    else (self.n_patches, self.local_hilbert_dim),
                    param_dtype,
                )
                self.cpm_kernel = self.param(
                    "CPMKernel",
                    nn.initializers.normal(stddev=float(self.cpm_weight_init_std)),
                    (self.local_hilbert_dim, int(self.hidden_size))
                    if bool(self.translation_shared)
                    else (
                        self.n_patches,
                        self.local_hilbert_dim,
                        int(self.hidden_size),
                    ),
                    param_dtype,
                )
            else:
                self.phase_head = nn.Dense(
                    self.local_hilbert_dim,
                    use_bias=False,
                    param_dtype=param_dtype,
                    kernel_init=kernel_init,
                    name="phase_head",
                )
            if phase_mode == "legacy_transformer":
                self.phase_attention = [
                    RWKVPhaseTransformerBlock(
                        embedding_size=int(self.embedding_size),
                        heads=int(self.num_heads),
                        init_variance=self.init_variance,
                        param_dtype=param_dtype,
                        compute_dtype=self.compute_dtype,
                        name=f"phase_attention_{i}",
                    )
                    for i in range(int(self.num_layers_phase))
                ]
        if bool(self.conditional_phase):
            if not bool(self.flag_phase):
                raise ValueError(
                    "2D RWKV conditional_phase requires phase output to be enabled."
                )
            zero_init = nn.initializers.zeros
            self.position_phase_bias = self.param(
                "position_phase_bias",
                zero_init,
                (self.local_hilbert_dim,)
                if bool(self.translation_shared)
                else (self.n_patches, self.local_hilbert_dim),
                param_dtype,
            )
            self.left_conditional_phase = self.param(
                "left_conditional_phase",
                zero_init,
                (self.local_hilbert_dim, self.local_hilbert_dim),
                param_dtype,
            )
            self.up_conditional_phase = self.param(
                "up_conditional_phase",
                zero_init,
                (self.local_hilbert_dim, self.local_hilbert_dim),
                param_dtype,
            )
            self.graph_conditional_phase = self.param(
                "graph_conditional_phase",
                zero_init,
                (3, self.local_hilbert_dim, self.local_hilbert_dim),
                param_dtype,
            )

    def _patch_indices(self, s):
        s = jnp.asarray(s, dtype=jnp.int32).reshape(int(self.L))
        s = s[self.site_permutation_array].reshape(
            int(self.n_patches), int(self.patch_size)
        )
        return jnp.sum(s * self.index_weights[None, :], axis=-1)

    def _fixed_sign_phase(self, s):
        param_dtype = _param_dtype(self.param_dtype)
        compute_dtype = _compute_dtype(self.compute_dtype, param_dtype)
        if self.sign_rule == "none":
            return jnp.asarray(0.0, dtype=compute_dtype)
        s_ordered = jnp.asarray(s, dtype=jnp.int32).reshape(int(self.L))
        s_ordered = s_ordered[self.site_permutation_array].reshape(
            int(self.n_patches), int(self.patch_size)
        )
        exponent = jnp.sum(s_ordered[:, 1])
        return jnp.pi * jnp.asarray(exponent % 2, dtype=compute_dtype)

    def _token_embedding(self, tokens):
        param_dtype = _param_dtype(self.param_dtype)
        compute_dtype = _compute_dtype(self.compute_dtype, param_dtype)
        tokens = jnp.asarray(tokens, dtype=jnp.int32)
        if bool(self.one_hot):
            x = jax.nn.one_hot(tokens, self.local_hilbert_dim + 1, axis=-1)
            return jnp.asarray(self.embed(x), dtype=compute_dtype)
        return jnp.asarray(self.embed(tokens), dtype=compute_dtype)

    def _base_inputs(self, patch_indices):
        param_dtype = _param_dtype(self.param_dtype)
        compute_dtype = _compute_dtype(self.compute_dtype, param_dtype)
        tokens = jnp.asarray(patch_indices, dtype=jnp.int32).reshape(
            int(self.n_patches)
        )
        null = jnp.full((1,), int(self.null_token), dtype=jnp.int32)
        padded = jnp.concatenate((tokens, null), axis=0)
        embeddings = self._token_embedding(padded)
        token_embeddings = embeddings[: int(self.n_patches)]
        null_embedding = embeddings[int(self.n_patches)]

        def site_context(p):
            row = p // int(self.grid_Lx)
            col = p - row * int(self.grid_Lx)
            left_tok = jax.lax.cond(
                col > 0,
                lambda _: token_embeddings[p - 1],
                lambda _: null_embedding,
                operand=None,
            )
            up_tok = jax.lax.cond(
                row > 0,
                lambda _: token_embeddings[p - int(self.grid_Lx)],
                lambda _: null_embedding,
                operand=None,
            )
            scan_tok = jax.lax.cond(
                p > 0,
                lambda _: token_embeddings[p - 1],
                lambda _: null_embedding,
                operand=None,
            )
            return self.base_context(
                jnp.concatenate((left_tok, up_tok, scan_tok), axis=-1)
            )

        out = jax.vmap(site_context)(jnp.arange(int(self.n_patches), dtype=jnp.int32))
        out = jnp.asarray(out, dtype=compute_dtype)
        if bool(self.position_embedding):
            out = out + jnp.asarray(self.token_position_embedding, dtype=compute_dtype)
        return out

    def _conditional_skip_logits(self, patch_indices):
        param_dtype = _param_dtype(self.param_dtype)
        compute_dtype = _compute_dtype(self.compute_dtype, param_dtype)
        logits = jnp.zeros(
            (int(self.n_patches), int(self.local_hilbert_dim)), dtype=compute_dtype
        )
        if not bool(self.conditional_logits):
            return logits
        tokens = jnp.asarray(patch_indices, dtype=jnp.int32).reshape(
            int(self.n_patches)
        )
        position_logit_bias = jnp.asarray(self.position_logit_bias, dtype=compute_dtype)
        logits = (
            logits + position_logit_bias[None, :]
            if bool(self.translation_shared)
            else logits + position_logit_bias
        )
        left_table = jnp.asarray(self.left_conditional_logits, dtype=compute_dtype)
        up_table = jnp.asarray(self.up_conditional_logits, dtype=compute_dtype)
        graph_table = jnp.asarray(self.graph_conditional_logits, dtype=compute_dtype)
        for p in range(int(self.n_patches)):
            row = p // int(self.grid_Lx)
            col = p - row * int(self.grid_Lx)
            if col > 0:
                logits = logits.at[p].add(left_table[tokens[p - 1]])
            if row > 0:
                logits = logits.at[p].add(up_table[tokens[p - int(self.grid_Lx)]])
        if bool(self.graph_context):
            for src, dst, kind in zip(
                self.graph_edge_src, self.graph_edge_dst, self.graph_edge_type
            ):
                src = int(src)
                dst = int(dst)
                if src < dst:
                    logits = logits.at[dst].add(graph_table[int(kind), tokens[src]])
        return logits

    def _conditional_skip_phase(self, patch_indices):
        param_dtype = _param_dtype(self.param_dtype)
        compute_dtype = _compute_dtype(self.compute_dtype, param_dtype)
        phase = jnp.zeros((int(self.n_patches),), dtype=compute_dtype)
        if not bool(self.conditional_phase):
            return phase
        tokens = jnp.asarray(patch_indices, dtype=jnp.int32).reshape(
            int(self.n_patches)
        )
        positions = jnp.arange(int(self.n_patches))
        position_phase_bias = jnp.asarray(self.position_phase_bias, dtype=compute_dtype)
        if bool(self.translation_shared):
            phase = phase + position_phase_bias[tokens]
        else:
            phase = phase + position_phase_bias[positions, tokens]
        left_table = jnp.asarray(self.left_conditional_phase, dtype=compute_dtype)
        up_table = jnp.asarray(self.up_conditional_phase, dtype=compute_dtype)
        graph_table = jnp.asarray(self.graph_conditional_phase, dtype=compute_dtype)
        for p in range(int(self.n_patches)):
            row = p // int(self.grid_Lx)
            col = p - row * int(self.grid_Lx)
            if col > 0:
                phase = phase.at[p].add(left_table[tokens[p - 1], tokens[p]])
            if row > 0:
                phase = phase.at[p].add(
                    up_table[tokens[p - int(self.grid_Lx)], tokens[p]]
                )
        if bool(self.graph_context):
            for src, dst, kind in zip(
                self.graph_edge_src, self.graph_edge_dst, self.graph_edge_type
            ):
                src = int(src)
                dst = int(dst)
                if src < dst:
                    phase = phase.at[dst].add(
                        graph_table[int(kind), tokens[src], tokens[dst]]
                    )
        return phase

    def _graph_context_by_type(self, post_states, p):
        param_dtype = _param_dtype(self.param_dtype)
        compute_dtype = _compute_dtype(self.compute_dtype, param_dtype)
        if not bool(self.graph_context):
            return jnp.zeros((3, int(self.embedding_size)), dtype=compute_dtype)
        edge_src = jnp.asarray(self.graph_edge_src, dtype=jnp.int32)
        edge_dst = jnp.asarray(self.graph_edge_dst, dtype=jnp.int32)
        edge_type = jnp.asarray(self.graph_edge_type, dtype=jnp.int32)
        src_h = post_states[edge_src]
        mask = (
            (edge_dst == jnp.asarray(p, dtype=jnp.int32))
            & (edge_src < jnp.asarray(p, dtype=jnp.int32))
        ).astype(compute_dtype)
        type_one_hot = jax.nn.one_hot(edge_type, 3, dtype=compute_dtype)
        weighted_types = type_one_hot * mask[:, None]
        messages = jnp.einsum("et,ed->td", weighted_types, src_h)
        counts = jnp.sum(weighted_types, axis=0)
        return messages / jnp.maximum(
            counts[:, None], jnp.asarray(1.0, dtype=compute_dtype)
        )

    def _apply_causal_layer(self, inputs, block):
        param_dtype = _param_dtype(self.param_dtype)
        compute_dtype = _compute_dtype(self.compute_dtype, param_dtype)
        inputs = jnp.asarray(inputs, dtype=compute_dtype)
        zero = jnp.zeros((int(self.embedding_size),), dtype=compute_dtype)
        up_grid = [zero for _ in range(int(self.grid_Lx))]
        post_states = jnp.zeros_like(inputs)
        outputs = []
        for row in range(int(self.grid_Ly)):
            left_h = zero
            for col in range(int(self.grid_Lx)):
                p = row * int(self.grid_Lx) + col
                left_context = left_h if col > 0 else zero
                up_h = up_grid[col] if row > 0 else zero
                if bool(self.graph_context):
                    graph_context = self._graph_context_by_type(post_states, p)
                    h = block(inputs[p], left_context, up_h, graph_context)
                else:
                    h = block(inputs[p], left_context, up_h)
                outputs.append(h)
                left_h = h
                up_grid[col] = h
                post_states = post_states.at[p].set(h)
        return jnp.stack(outputs, axis=0)

    def _causal_features(self, patch_indices):
        param_dtype = _param_dtype(self.param_dtype)
        compute_dtype = _compute_dtype(self.compute_dtype, param_dtype)
        y = self._base_inputs(patch_indices)
        for block in self.blocks:
            y = self._apply_causal_layer(y, block)
        return jnp.asarray(nn.gelu(self.neck(y)), dtype=compute_dtype)

    def _probability_logits(self, logits):
        param_dtype = _param_dtype(self.param_dtype)
        compute_dtype = _compute_dtype(self.compute_dtype, param_dtype)
        logits = jnp.asarray(logits, dtype=compute_dtype)
        logits = logits / jnp.asarray(float(self.temperature), dtype=compute_dtype)
        if float(self.logit_clip) > 0.0:
            clip = jnp.asarray(float(self.logit_clip), dtype=compute_dtype)
            logits = clip * jnp.tanh(logits / clip)
        return logits - jnp.max(logits, axis=-1, keepdims=True)

    def _fixed_sector_mask(self, patch_indices):
        if int(self.fixed_n_up) < 0:
            return jnp.ones(
                (int(self.n_patches), int(self.local_hilbert_dim)), dtype=bool
            )
        tokens = jnp.asarray(patch_indices, dtype=jnp.int32).reshape(
            int(self.n_patches)
        )
        selected_counts = jnp.asarray(self.patch_bit_counts, dtype=jnp.int32)[tokens]
        prefix_counts = jnp.concatenate(
            (
                jnp.zeros((1,), dtype=jnp.int32),
                jnp.cumsum(selected_counts, axis=0)[:-1],
            ),
            axis=0,
        )
        token_counts = jnp.asarray(self.patch_bit_counts, dtype=jnp.int32)
        positions = jnp.arange(int(self.n_patches), dtype=jnp.int32)
        remaining_after = int(self.L) - (positions + 1) * int(self.patch_size)
        proposed_counts = prefix_counts[:, None] + token_counts[None, :]
        target = jnp.asarray(int(self.fixed_n_up), dtype=jnp.int32)
        return (proposed_counts <= target) & (
            (proposed_counts + remaining_after[:, None]) >= target
        )

    def _apply_fixed_sector_mask(self, logits, patch_indices):
        if int(self.fixed_n_up) < 0:
            return logits
        mask = self._fixed_sector_mask(patch_indices)
        param_dtype = _param_dtype(self.param_dtype)
        compute_dtype = _compute_dtype(self.compute_dtype, param_dtype)
        invalid = jnp.asarray(jnp.finfo(compute_dtype).min, dtype=compute_dtype)
        return jnp.where(mask, logits, invalid)

    def _fixed_sector_valid(self, patch_indices):
        if int(self.fixed_n_up) < 0:
            return jnp.asarray(True)
        tokens = jnp.asarray(patch_indices, dtype=jnp.int32).reshape(
            int(self.n_patches)
        )
        selected_counts = jnp.asarray(self.patch_bit_counts, dtype=jnp.int32)[tokens]
        return jnp.sum(selected_counts) == jnp.asarray(
            int(self.fixed_n_up), dtype=jnp.int32
        )

    def _phase_logits(self, patch_indices, features, y):
        param_dtype = _param_dtype(self.param_dtype)
        compute_dtype = _compute_dtype(self.compute_dtype, param_dtype)
        phase_mode = str(self.phase_mode).lower().replace("-", "_")
        if phase_mode == "cpm":
            positions = jnp.arange(int(self.n_patches))
            cpm_bias = jnp.asarray(self.cpm_bias, dtype=compute_dtype)
            cpm_kernel = jnp.asarray(self.cpm_kernel, dtype=compute_dtype)
            if bool(self.translation_shared):
                selected_bias = cpm_bias[patch_indices]
                selected_kernel = cpm_kernel[patch_indices, :]
            else:
                selected_bias = cpm_bias[positions, patch_indices]
                selected_kernel = cpm_kernel[positions, patch_indices, :]
            global_feature = features[-1]
            return selected_bias + jnp.einsum(
                "pk,k->p", selected_kernel, global_feature
            )
        if phase_mode == "shared_neck":
            phase = jnp.asarray(self.phase_head(features), dtype=compute_dtype)
            return phase - jnp.mean(phase, axis=-1, keepdims=True)
        phase = y
        for block in self.phase_attention:
            phase = block(phase)
        return jnp.asarray(self.phase_head(phase), dtype=compute_dtype)

    def __call__(self, s):
        param_dtype = _param_dtype(self.param_dtype)
        compute_dtype = _compute_dtype(self.compute_dtype, param_dtype)
        patch_indices = self._patch_indices(s)
        y = self._base_inputs(patch_indices)
        for block in self.blocks:
            y = self._apply_causal_layer(y, block)
        features = jnp.asarray(nn.gelu(self.neck(y)), dtype=compute_dtype)
        logits = self._probability_logits(
            self.head(features) + self._conditional_skip_logits(patch_indices)
        )
        logits = self._apply_fixed_sector_mask(logits, patch_indices)
        if bool(self.linear_output):
            probs = nn.elu(logits) + 1.0 + 1e-8
            log_probs = jnp.log(probs / jnp.sum(probs, axis=-1, keepdims=True))
        else:
            log_probs = jax.nn.log_softmax(logits, axis=-1)
        log_amp = jnp.take_along_axis(log_probs, patch_indices[:, None], axis=-1).sum(
            axis=-2
        ).squeeze(-1) * jnp.asarray(float(self.log_prob_factor), dtype=compute_dtype)
        log_amp = jnp.where(
            self._fixed_sector_valid(patch_indices),
            log_amp,
            jnp.asarray(-jnp.inf, dtype=compute_dtype),
        )
        fixed_sign_phase = self._fixed_sign_phase(s)
        if not bool(self.flag_phase):
            if self.sign_rule == "none":
                return log_amp.astype(_out_real_dtype())
            return log_amp.astype(_out_cpx_dtype()) + 1j * fixed_sign_phase.astype(
                _out_cpx_dtype()
            )

        phase = self._phase_logits(patch_indices, features, y)
        if str(self.phase_mode).lower().replace("-", "_") != "cpm":
            phase = (
                jnp.take_along_axis(phase, patch_indices[:, None], axis=-1)
                .sum(axis=-2)
                .squeeze(-1)
            )
        else:
            phase = jnp.sum(phase)
        phase = phase + jnp.sum(self._conditional_skip_phase(patch_indices))
        phase = jnp.asarray(float(self.phase_scale), dtype=compute_dtype) * phase
        phase = phase + fixed_sign_phase
        return log_amp.astype(_out_cpx_dtype()) + 1j * phase.astype(_out_cpx_dtype())

    def sample(self, key):
        keys = random.split(key, int(self.n_patches))
        axis_marker = jnp.asarray(key[0], dtype=jnp.int32) * jnp.asarray(
            0, dtype=jnp.int32
        )
        tokens = jnp.zeros((int(self.n_patches),), dtype=jnp.int32) + axis_marker

        def step(tokens_state, p):
            features = self._causal_features(tokens_state)
            logits = self._apply_fixed_sector_mask(
                self._probability_logits(
                    self.head(features) + self._conditional_skip_logits(tokens_state)
                ),
                tokens_state,
            )[p]
            choice = random.categorical(keys[p], jnp.real(logits).reshape(-1)).astype(
                jnp.int32
            )
            tokens_state = tokens_state.at[p].set(choice)
            return tokens_state, choice

        tokens, _ = jax.lax.scan(
            step, tokens, jnp.arange(int(self.n_patches), dtype=jnp.int32)
        )
        ordered_state = self.patch_states[tokens].reshape(int(self.L))
        return (
            jnp.zeros_like(ordered_state)
            .at[self.site_permutation_array]
            .set(ordered_state)
            .astype(global_defs.DT_SAMPLES)
        )

    def sample_batch(self, num_samples: int, key):
        return jax.vmap(self.sample)(random.split(key, int(num_samples)))


class LogRWKV2DGraphAutoregressiveJVMC(LogRWKV2DAutoregressiveJVMC):
    graph_context: bool = True


RWKV2D = LogRWKV2DAutoregressiveJVMC
