from __future__ import annotations

from functools import partial
from itertools import product, repeat
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import random
from jax.nn import log_softmax

from jVMC_exp import global_defs


def _param_dtype(param_dtype):
    return jnp.dtype(global_defs.DT_PARAMS_REAL if param_dtype is None else param_dtype)


def _compute_dtype(compute_dtype, param_dtype):
    return jnp.dtype(param_dtype if compute_dtype is None else compute_dtype)


def _out_cpx_dtype():
    return jnp.dtype(global_defs.DT_OUT_CPX)


def _init_to(value, dtype):
    def init(key, shape=None, dtype_arg=None):
        del key, shape, dtype_arg
        return jnp.asarray(value, dtype=dtype)

    return init


def _empty_rwkv_state(num_heads: int, embedding_size: int, dtype):
    zeros_head = jnp.zeros((num_heads, embedding_size), dtype=dtype)
    zeros = jnp.zeros((embedding_size,), dtype=dtype)
    minus_inf = jnp.full((num_heads, embedding_size), -jnp.inf, dtype=dtype)
    return (zeros, zeros_head, zeros_head, minus_inf), zeros


class MultiHeadTimeMixing(nn.Module):
    layer_depth: int
    num_layers: int
    embedding_size: int
    num_heads: int
    param_dtype: Any = None
    compute_dtype: Any = None
    init_variance: float = 0.1

    def setup(self):
        param_dtype = _param_dtype(self.param_dtype)
        compute_dtype = _compute_dtype(self.compute_dtype, param_dtype)
        layer_span = max(int(self.num_layers) - 1, 1)
        ratio_0_to_1 = float(self.layer_depth) / float(layer_span)
        ratio_1_to_almost_0 = 1.0 - (
            float(self.layer_depth) / float(max(int(self.num_layers), 1))
        )
        zigzag = 0.5 * (jnp.arange(1, int(self.embedding_size) + 1) % 3 - 1)
        time_first = jnp.full((int(self.embedding_size),), jnp.log(0.3)) + zigzag
        time_first = jnp.repeat(time_first[None, :], int(self.num_heads), axis=0)
        self.time_first = self.param("time_first", _init_to(time_first, param_dtype))

        h = jnp.arange(int(self.embedding_size))
        h = jnp.repeat(h[None, :], int(self.num_heads), axis=0)
        embedding_span = max(int(self.embedding_size) - 1, 1)
        time_decay = -5.0 + 8.0 * (h / embedding_span) ** (0.7 + 1.3 * ratio_0_to_1)
        time_decay = (
            time_decay
            - jnp.arange(int(self.num_heads)).reshape(int(self.num_heads), 1) * 0.5
        )
        self.time_decay = self.param("time_decay", _init_to(time_decay, param_dtype))

        x = jnp.arange(int(self.embedding_size)) / jnp.asarray(
            int(self.embedding_size), dtype=compute_dtype
        )
        self.time_mix_k = self.param(
            "time_mix_k", _init_to(jnp.power(x, ratio_1_to_almost_0), param_dtype)
        )
        self.time_mix_v = self.param(
            "time_mix_v",
            _init_to(
                jnp.power(x, ratio_1_to_almost_0) + 0.3 * ratio_0_to_1, param_dtype
            ),
        )
        self.time_mix_r = self.param(
            "time_mix_r", _init_to(jnp.power(x, 0.5 * ratio_1_to_almost_0), param_dtype)
        )

        kernel_init = nn.initializers.variance_scaling(
            self.init_variance,
            mode="fan_in",
            distribution="truncated_normal",
        )
        dense_kwargs = {
            "use_bias": False,
            "param_dtype": param_dtype,
            "kernel_init": kernel_init,
        }
        self.key = nn.Dense(
            int(self.num_heads) * int(self.embedding_size), name="key", **dense_kwargs
        )
        self.value = nn.Dense(
            int(self.num_heads) * int(self.embedding_size), name="value", **dense_kwargs
        )
        self.receptance = nn.Dense(
            int(self.num_heads) * int(self.embedding_size),
            name="receptance",
            **dense_kwargs,
        )
        self.output = nn.Dense(int(self.embedding_size), name="output", **dense_kwargs)
        self.head_collapse = self.param(
            "head_collapse",
            _init_to(jnp.ones((int(self.num_heads),)), param_dtype),
        )

    def __call__(self, x, time_mix_state):
        param_dtype = _param_dtype(self.param_dtype)
        compute_dtype = _compute_dtype(self.compute_dtype, param_dtype)
        sx, aa, bb, pp = time_mix_state
        x = jnp.asarray(x, dtype=compute_dtype)
        sx = jnp.concatenate(
            (jnp.asarray(sx, dtype=compute_dtype)[None, :], x[:-1, :]), axis=0
        )
        time_mix_k = jnp.asarray(self.time_mix_k, dtype=compute_dtype)
        time_mix_v = jnp.asarray(self.time_mix_v, dtype=compute_dtype)
        time_mix_r = jnp.asarray(self.time_mix_r, dtype=compute_dtype)
        kx = x * time_mix_k + sx * (1.0 - time_mix_k)
        vx = x * time_mix_v + sx * (1.0 - time_mix_v)
        rx = x * time_mix_r + sx * (1.0 - time_mix_r)

        r = nn.sigmoid(self.receptance(rx)).reshape(
            rx.shape[0], int(self.num_heads), int(self.embedding_size)
        )
        k = self.key(kx).reshape(
            rx.shape[0], int(self.num_heads), int(self.embedding_size)
        )
        v = self.value(vx).reshape(
            rx.shape[0], int(self.num_heads), int(self.embedding_size)
        )
        r = jnp.asarray(r, dtype=compute_dtype)
        k = jnp.asarray(k, dtype=compute_dtype)
        v = jnp.asarray(v, dtype=compute_dtype)
        time_first = jnp.asarray(self.time_first, dtype=compute_dtype)
        time_decay = jnp.asarray(self.time_decay, dtype=compute_dtype)
        aa = jnp.asarray(aa, dtype=compute_dtype) + jnp.zeros_like(k[0])
        bb = jnp.asarray(bb, dtype=compute_dtype) + jnp.zeros_like(k[0])
        pp = jnp.asarray(pp, dtype=compute_dtype) + jnp.zeros_like(k[0])

        def step(state, kv):
            aa_state, bb_state, pp_state = state
            kk, vv = kv
            ww = time_first + kk
            q = jnp.maximum(pp_state, ww)
            e1 = jnp.exp(pp_state - q)
            e2 = jnp.exp(ww - q)
            out = (e1 * aa_state + e2 * vv) / (e1 * bb_state + e2)

            ww_next = pp_state - jnp.exp(time_decay)
            q_next = jnp.maximum(ww_next, kk)
            e1_next = jnp.exp(ww_next - q_next)
            e2_next = jnp.exp(kk - q_next)
            aa_next = e1_next * aa_state + e2_next * vv
            bb_next = e1_next * bb_state + e2_next
            return (aa_next, bb_next, q_next), out.astype(compute_dtype)

        (aa, bb, pp), wkv = jax.lax.scan(step, (aa, bb, pp), (k, v))
        head_collapse = nn.tanh(
            jnp.asarray(self.head_collapse, dtype=compute_dtype)
        ).reshape(int(self.num_heads), 1)
        wkv = wkv * head_collapse
        r = r * head_collapse
        out = x + self.output(
            jnp.sum(r * wkv, axis=-2)
            / jnp.asarray(int(self.num_heads), dtype=compute_dtype)
        )
        return jnp.asarray(out, dtype=compute_dtype), (x[-1, :], aa, bb, pp)


class RWKVBlock(nn.Module):
    layer_num: int
    num_layers: int
    embedding_size: int
    hidden_size: int
    num_heads: int
    param_dtype: Any = None
    compute_dtype: Any = None
    init_variance: float = 0.1

    def setup(self):
        self.time_mix = MultiHeadTimeMixing(
            layer_depth=int(self.layer_num),
            num_layers=int(self.num_layers),
            embedding_size=int(self.embedding_size),
            num_heads=int(self.num_heads),
            param_dtype=self.param_dtype,
            compute_dtype=self.compute_dtype,
            init_variance=self.init_variance,
            name="time_mix",
        )

    def __call__(self, x, block_state):
        param_dtype = _param_dtype(self.param_dtype)
        compute_dtype = _compute_dtype(self.compute_dtype, param_dtype)
        if block_state is None:
            block_state = _empty_rwkv_state(
                int(self.num_heads), int(self.embedding_size), compute_dtype
            )
        time_mix_state, channel_mix_state = block_state
        x, time_mix_state = self.time_mix(x, time_mix_state)
        return x, (time_mix_state, channel_mix_state)


class RWKVCPM(nn.Module):
    """One-dimensional autoregressive RWKV log-amplitude network with CPM phase."""

    L: int
    l_hil_dim: int = 2
    patch_size: int = 1
    hidden_size: int = 16
    num_heads: int = 1
    num_layers: int = 2
    embedding_size: int = 16
    log_prob_factor: float = 0.5
    temperature: float = 1.0
    init_variance: float = 0.1
    cpm_bias_init_std: float = 3.141592653589793
    cpm_weight_init_std: float = 0.1
    param_dtype: Any = None
    compute_dtype: Any = None

    def setup(self):
        param_dtype = _param_dtype(self.param_dtype)
        if int(self.L) <= 0:
            raise ValueError("RWKV system size must be positive.")
        if int(self.l_hil_dim) <= 1:
            raise ValueError("RWKV local Hilbert dimension must be at least 2.")
        if int(self.patch_size) <= 0 or int(self.L) % int(self.patch_size) != 0:
            raise ValueError("RWKV patch_size must be positive and divide L.")
        if int(self.num_layers) < 1:
            raise ValueError("RWKV requires at least one layer.")
        if int(self.embedding_size) <= 0 or int(self.hidden_size) <= 0:
            raise ValueError("RWKV embedding_size and hidden_size must be positive.")
        if int(self.num_heads) <= 0:
            raise ValueError("RWKV num_heads must be positive.")
        if float(self.temperature) <= 0.0:
            raise ValueError("RWKV sampling temperature must be positive.")

        patch_states = list(
            product(range(int(self.l_hil_dim)), repeat=int(self.patch_size))
        )
        self.patch_states = jnp.asarray(patch_states, dtype=jnp.int32)
        self.local_hil_dim = int(self.l_hil_dim) ** int(self.patch_size)
        self.local_hilbert_dim = self.local_hil_dim
        self.PL = int(self.L) // int(self.patch_size)
        self.n_patches = self.PL
        self.index_weights = jnp.asarray(
            int(self.l_hil_dim) ** jnp.arange(int(self.patch_size) - 1, -1, -1),
            dtype=jnp.int32,
        )

        self.embed = nn.Embed(
            self.local_hil_dim,
            int(self.embedding_size),
            param_dtype=param_dtype,
            name="embedding",
        )
        self.blocks = [
            RWKVBlock(
                layer_num=i,
                num_layers=int(self.num_layers),
                embedding_size=int(self.embedding_size),
                hidden_size=int(self.hidden_size),
                num_heads=int(self.num_heads),
                param_dtype=param_dtype,
                compute_dtype=self.compute_dtype,
                init_variance=self.init_variance,
                name=f"block_{i}",
            )
            for i in range(int(self.num_layers))
        ]
        dense_init = nn.initializers.variance_scaling(
            self.init_variance, mode="fan_in", distribution="truncated_normal"
        )
        self.neck = nn.Dense(
            int(self.hidden_size),
            use_bias=False,
            param_dtype=param_dtype,
            kernel_init=dense_init,
            name="neck",
        )
        self.head = nn.Dense(
            self.local_hil_dim,
            use_bias=False,
            param_dtype=param_dtype,
            kernel_init=dense_init,
            name="head",
        )
        self.CPMBias = self.param(
            "CPMBias",
            nn.initializers.normal(stddev=float(self.cpm_bias_init_std)),
            (self.PL, self.local_hil_dim),
            param_dtype,
        )
        self.CPMKernel = self.param(
            "CPMKernel",
            nn.initializers.normal(stddev=float(self.cpm_weight_init_std)),
            (self.PL, self.local_hil_dim, int(self.hidden_size)),
            param_dtype,
        )

    def _tokens(self, s):
        s = jnp.asarray(s, dtype=jnp.int32).reshape(int(self.PL), int(self.patch_size))
        return jnp.sum(s * self.index_weights[None, :], axis=-1)

    def _features_from_tokens(self, tokens, block_states=None, output_state=False):
        param_dtype = _param_dtype(self.param_dtype)
        compute_dtype = _compute_dtype(self.compute_dtype, param_dtype)
        y = jnp.asarray(
            self.embed(jnp.asarray(tokens, dtype=jnp.int32)), dtype=compute_dtype
        )
        next_states = []
        if block_states is None:
            block_states = repeat(None)
        for block, state in zip(self.blocks, block_states):
            y, new_state = block(y, state)
            if output_state:
                next_states.append(new_state)
        return jnp.asarray(nn.gelu(self.neck(y)), dtype=compute_dtype), next_states

    def _cpm_phase_logits(self, tokens, features):
        param_dtype = _param_dtype(self.param_dtype)
        compute_dtype = _compute_dtype(self.compute_dtype, param_dtype)
        positions = jnp.arange(int(self.PL))
        selected_bias = jnp.asarray(self.CPMBias, dtype=compute_dtype)[
            positions, tokens
        ]
        selected_kernel = jnp.asarray(self.CPMKernel, dtype=compute_dtype)[
            positions, tokens, :
        ]
        global_feature = jnp.asarray(features, dtype=compute_dtype)[-1]
        return selected_bias + jnp.einsum("pk,k->p", selected_kernel, global_feature)

    def __call__(self, s, block_states=None, output_state: bool = False):
        param_dtype = _param_dtype(self.param_dtype)
        compute_dtype = _compute_dtype(self.compute_dtype, param_dtype)
        if output_state:
            tokens_in = jnp.asarray(s, dtype=jnp.int32).reshape(-1)
            features, next_states = self._features_from_tokens(
                tokens_in, block_states, output_state=True
            )
            logits = jnp.asarray(self.head(features), dtype=compute_dtype)
            logits = logits - jnp.max(logits, axis=-1, keepdims=True)
            return logits, next_states

        tokens = self._tokens(s)
        tokens_in = jnp.pad(tokens[:-1], (1, 0), mode="constant", constant_values=0)
        features, _ = self._features_from_tokens(tokens_in)
        logits = jnp.asarray(self.head(features), dtype=compute_dtype)
        logits = logits - jnp.max(logits, axis=-1, keepdims=True)
        log_probs = log_softmax(logits, axis=-1) * jnp.asarray(
            float(self.log_prob_factor), dtype=compute_dtype
        )
        amp = jnp.take_along_axis(log_probs, tokens[:, None], axis=-1).sum()
        phase = self._cpm_phase_logits(tokens, features).sum()
        return amp.astype(_out_cpx_dtype()) + 1j * phase.astype(_out_cpx_dtype())

    def sample(self, key):
        keys = random.split(key, int(self.PL))
        axis_marker = jnp.asarray(key[0], dtype=jnp.int32) * jnp.asarray(
            0, dtype=jnp.int32
        )
        tokens = jnp.zeros((int(self.PL),), dtype=jnp.int32) + axis_marker

        def step(tokens_state, p):
            tokens_in = jnp.pad(
                tokens_state[:-1], (1, 0), mode="constant", constant_values=0
            )
            logits, _ = self(tokens_in, block_states=None, output_state=True)
            logits = logits[p]
            choice = random.categorical(
                keys[p], jnp.real(logits).reshape(-1) / float(self.temperature)
            )
            choice = choice.astype(jnp.int32)
            return tokens_state.at[p].set(choice), choice

        tokens, _ = jax.lax.scan(
            step, tokens, jnp.arange(int(self.PL), dtype=jnp.int32)
        )
        return self.patch_states[tokens].reshape(-1).astype(global_defs.DT_SAMPLES)

    def sample_batch(self, num_samples: int, key):
        return jax.vmap(self.sample)(random.split(key, int(num_samples)))

    @partial(nn.scan, variable_broadcast="params", split_rngs={"params": False})
    def _scan_sample(self, carry, key):
        logits, next_states = self(carry[0], block_states=carry[1], output_state=True)
        choice = random.categorical(
            key, jnp.real(logits.ravel()) / float(self.temperature)
        )
        choice = choice.astype(jnp.int32)
        return (jnp.expand_dims(choice, 0), next_states), choice


class ParticleConservingAutoregressive(nn.Module):
    """Fixed-particle-sector wrapper for token autoregressive networks."""

    net: nn.Module
    Q: int

    def setup(self):
        self.is_particle = True
        self.L = int(self.net.L)
        self.l_hil_dim = int(self.net.l_hil_dim)
        self.patch_size = int(self.net.patch_size)
        self.local_hil_dim = int(self.l_hil_dim) ** int(self.patch_size)
        self.local_hilbert_dim = self.local_hil_dim
        self.patch_states = jnp.asarray(
            list(product(range(int(self.l_hil_dim)), repeat=int(self.patch_size))),
            dtype=jnp.int32,
        )
        self.patch_particles = jnp.sum(self.patch_states, axis=1).astype(jnp.int32)
        self.PL = int(self.L) // int(self.patch_size)
        self.max_particle_patch = (self.l_hil_dim - 1) * self.patch_size
        particles = self.patch_particles
        max_particle_p1 = self.max_particle_patch + 1
        self.must_mask = 2 * jnp.asarray(
            [particles < j for j in range(max_particle_p1)]
        )
        self.can_mask = jnp.flip(self.must_mask, axis=0)
        self.max_particles_after = jnp.pad(
            self.max_particle_patch * jnp.arange(1, self.PL + 1)[::-1], (0, 1)
        )[1:]
        self.log_prob_factor = float(self.net.log_prob_factor)

    def _additive_mask(self, invalid, logits):
        dtype = jnp.asarray(logits).dtype
        invalid_value = jnp.asarray(jnp.finfo(dtype).min, dtype=dtype)
        return jnp.where(invalid, invalid_value, jnp.asarray(0.0, dtype=dtype))

    def _sector_valid(self, tokens):
        tokens = jnp.asarray(tokens, dtype=jnp.int32).reshape(int(self.PL))
        particles = jnp.asarray(self.patch_particles, dtype=jnp.int32)[tokens]
        return jnp.sum(particles) == jnp.asarray(int(self.Q), dtype=jnp.int32)

    def __call__(self, *args, **kwargs):
        if kwargs.get("output_state", False):
            if "cumsum" not in kwargs or "position" not in kwargs:
                return self.net(*args, **kwargs)
            cumsum_left = int(self.Q) - kwargs.pop("cumsum")
            position = kwargs.pop("position")
            must_give = jnp.maximum(cumsum_left - self.max_particles_after[position], 0)
            can_give = jnp.minimum(cumsum_left, self.max_particle_patch)
            particles = self.patch_particles
            invalid = (particles < must_give) | (particles > can_give)
            logits, state = self.net(*args, **kwargs)
            additive_mask = self._additive_mask(invalid, logits)
            return log_softmax(logits + additive_mask), state

        s = jnp.asarray(args[0], dtype=jnp.int32)
        tokens = self.net._tokens(s)
        tokens_in = jnp.pad(tokens[:-1], (1, 0), mode="constant", constant_values=0)
        token_particles = self.patch_states[tokens_in].sum(axis=1).astype(jnp.int32)
        cumsum = jnp.cumsum(token_particles)
        logits, _ = self.net(tokens_in, output_state=True)
        cumsum_left = int(self.Q) - cumsum
        must_give = jnp.maximum(cumsum_left - self.max_particles_after, 0)
        can_give = jnp.minimum(cumsum_left, self.max_particle_patch)
        particles = self.patch_particles
        invalid = (particles[None, :] < must_give[:, None]) | (
            particles[None, :] > can_give[:, None]
        )
        additive_mask = self._additive_mask(invalid, logits)
        log_probs = log_softmax(logits + additive_mask, axis=-1) * self.log_prob_factor
        amp = jnp.take_along_axis(log_probs, tokens[:, None], axis=-1).sum()
        features, _ = self.net._features_from_tokens(tokens_in)
        phase = self.net._cpm_phase_logits(tokens, features).sum()
        out = amp.astype(_out_cpx_dtype()) + 1j * phase.astype(_out_cpx_dtype())
        zero_amp = jnp.asarray(-jnp.inf + 0.0j, dtype=_out_cpx_dtype())
        return jnp.where(self._sector_valid(tokens), out, zero_amp)

    def sample(self, key):
        keys = random.split(key, int(self.PL))
        axis_marker = jnp.asarray(key[0], dtype=jnp.int32) * jnp.asarray(
            0, dtype=jnp.int32
        )
        tokens = jnp.zeros((int(self.PL),), dtype=jnp.int32) + axis_marker
        cumsum = axis_marker

        def step(carry, p):
            tokens_state, cumsum_state = carry
            tokens_in = jnp.pad(
                tokens_state[:-1], (1, 0), mode="constant", constant_values=0
            )
            logits, _ = self(
                tokens_in,
                block_states=None,
                output_state=True,
                cumsum=cumsum_state,
                position=p,
            )
            choice = random.categorical(
                keys[p], jnp.real(logits[p]).reshape(-1)
            ).astype(jnp.int32)
            tokens_state = tokens_state.at[p].set(choice)
            cumsum_state = cumsum_state + self.patch_states[choice].sum().astype(
                jnp.int32
            )
            return (tokens_state, cumsum_state), choice

        (tokens, _), _ = jax.lax.scan(
            step, (tokens, cumsum), jnp.arange(int(self.PL), dtype=jnp.int32)
        )
        return self.patch_states[tokens].reshape(-1).astype(global_defs.DT_SAMPLES)

    def sample_batch(self, num_samples: int, key):
        return jax.vmap(self.sample)(random.split(key, int(num_samples)))

    @partial(nn.scan, variable_broadcast="params", split_rngs={"params": False})
    def _scan_sample(self, carry, key):
        logits, next_states = self(
            carry[0],
            block_states=carry[1],
            output_state=True,
            cumsum=carry[2],
            position=key[1],
        )
        choice = random.categorical(key[0], jnp.real(logits.ravel())).astype(jnp.int32)
        cumsum = carry[2] + self.patch_states[choice].sum().astype(jnp.int32)
        return (jnp.expand_dims(choice, 0), next_states, cumsum), choice


@jax.jit
def _sort_gumbel(sample, logits, gumbel, states):
    num_samples = sample.shape[0]
    local_dim = sample.shape[1]
    length = sample.shape[2]
    indexes = jnp.argsort(-gumbel, axis=None)
    indexes_states = (indexes // local_dim)[:num_samples]
    sample = sample.reshape(-1, length)[indexes].reshape(local_dim, num_samples, length)
    sample = jnp.swapaxes(sample, 0, 1)
    logits = logits.ravel()[indexes].reshape(local_dim, num_samples).T
    gumbel = gumbel.ravel()[indexes].reshape(local_dim, num_samples).T
    leaves, treedef = jax.tree_util.tree_flatten(states)
    states = jax.tree_util.tree_unflatten(
        treedef, [leaf[indexes_states] for leaf in leaves]
    )
    return sample, logits, gumbel, states


class GumbelWithoutReplacement(nn.Module):
    """Gumbel top-k sampler wrapper for autoregressive RWKV networks."""

    net: nn.Module

    def setup(self):
        self.is_gumbel = True
        base_net = self.net.net if hasattr(self.net, "net") else self.net
        self.is_particle_net = hasattr(self.net, "Q") and hasattr(self.net, "net")
        self.L = int(base_net.L)
        self.l_hil_dim = int(base_net.l_hil_dim)
        self.patch_size = int(base_net.patch_size)
        self.sequence_length = int(self.L) // int(self.patch_size)
        self.local_hil_dim = int(self.l_hil_dim) ** int(self.patch_size)
        self.local_hilbert_dim = self.local_hil_dim
        self.patch_states = jnp.asarray(
            list(product(range(int(self.l_hil_dim)), repeat=int(self.patch_size))),
            dtype=jnp.int32,
        )
        self.patch_particles = jnp.sum(self.patch_states, axis=1).astype(jnp.int32)
        self.max_particle_patch = (self.l_hil_dim - 1) * self.patch_size
        self.max_particles_after = jnp.pad(
            self.max_particle_patch * jnp.arange(1, self.sequence_length + 1)[::-1],
            (0, 1),
        )[1:]
        self.Q = int(self.net.Q) if self.is_particle_net else -1
        self.log_prob_factor = float(base_net.log_prob_factor)

    def __call__(self, *args, **kwargs):
        return self.net(*args, **kwargs)

    def sample(self, key):
        return self.net.sample(key)

    def sample_batch(self, num_samples: int, key):
        return self.net.sample_batch(num_samples, key)

    def _decode_samples(self, samples):
        return self.patch_states[samples].reshape(samples.shape[:-1] + (-1,))

    def _conditional_token_log_probs(self, prefix_tokens, cumsum, position):
        tokens_in = jnp.pad(
            prefix_tokens[:-1], (1, 0), mode="constant", constant_values=0
        )
        if bool(self.is_particle_net):
            logits, _ = self.net.net(tokens_in, output_state=True)
            cumsum_left = self.Q - cumsum
            must_give = jnp.maximum(cumsum_left - self.max_particles_after[position], 0)
            can_give = jnp.minimum(cumsum_left, self.max_particle_patch)
            particles = self.patch_particles
            invalid = (particles < must_give) | (particles > can_give)
            additive_mask = jnp.where(invalid, -jnp.inf, 0.0)
            logits = logits[position] + additive_mask
        else:
            logits, _ = self.net(tokens_in, output_state=True)
            logits = logits[position]
        return log_softmax(jnp.real(logits), axis=-1)

    def sample_weighted(self, num_samples: int, key):
        domain_size = self.local_hil_dim**self.sequence_length
        if int(num_samples) >= domain_size:
            raise RuntimeError(
                "number of samples must be smaller than the finite Gumbel domain"
            )

        num_samples = int(num_samples)
        samples = jnp.zeros((1, self.sequence_length), dtype=jnp.int32)
        log_probs = jnp.zeros((1,), dtype=global_defs.DT_OUT_REAL)
        cumsum = jnp.zeros((1,), dtype=jnp.int32)
        temperature = jnp.asarray(
            getattr(self.net, "temperature", 1.0), dtype=global_defs.DT_OUT_REAL
        )
        token_choices = jnp.arange(self.local_hil_dim, dtype=jnp.int32)
        token_particles = self.patch_particles

        for position in range(self.sequence_length):
            key, subkey = random.split(key)
            cond = jax.vmap(
                lambda prefix, used: self._conditional_token_log_probs(
                    prefix, used, position
                )
            )(samples, cumsum)
            beam = samples.shape[0]
            expanded = jnp.repeat(samples, self.local_hil_dim, axis=0)
            expanded = expanded.at[jnp.arange(beam * self.local_hil_dim), position].set(
                jnp.tile(token_choices, beam)
            )
            expanded_log_probs = jnp.repeat(log_probs, self.local_hil_dim) + cond.reshape(
                -1
            )
            expanded_cumsum = jnp.repeat(cumsum, self.local_hil_dim) + jnp.tile(
                token_particles, beam
            )
            gumbel = random.gumbel(
                subkey, shape=expanded_log_probs.shape, dtype=expanded_log_probs.dtype
            )
            score = expanded_log_probs / temperature + gumbel
            k = min(num_samples, int(score.shape[0]))
            top = jnp.argsort(-score)[:k]
            samples = expanded[top]
            log_probs = expanded_log_probs[top]
            cumsum = expanded_cumsum[top]

        kappa = jnp.where(log_probs.shape[0] > 1, log_probs[1], log_probs[0])
        weights = jax.nn.softmax(log_probs)
        decoded = self._decode_samples(samples)
        return decoded, log_probs * self.log_prob_factor, weights, kappa

    def _gumbel_step(self, sample, logits, gumbel, key, states, cumsum, position):
        base = sample[0]
        candidates = jnp.tile(base[None, :], (self.local_hil_dim, 1))
        candidate_tokens = jnp.arange(self.local_hil_dim, dtype=base.dtype)
        sample = candidates.at[jnp.arange(self.local_hil_dim), position].set(
            candidate_tokens
        )
        call_kwargs = {"block_states": states, "output_state": True}
        if bool(self.is_particle_net):
            call_kwargs["cumsum"] = cumsum
            call_kwargs["position"] = position
        logit_new, next_states = self(sample[:, position], **call_kwargs)
        temperature = getattr(self.net, "temperature", 1.0)
        logit_new = log_softmax(jnp.real(logit_new) / temperature, axis=-1)
        logit_new = logits[0] + logit_new
        gumbel_new = logit_new + random.gumbel(
            key[0], shape=(self.local_hil_dim,), dtype=logit_new.dtype
        )
        z = jnp.nanmax(gumbel_new)
        gumbel_new = jnp.nan_to_num(
            -jnp.log(jnp.exp(-gumbel[0]) - jnp.exp(-z) + jnp.exp(-gumbel_new)),
            nan=-jnp.inf,
        )
        return sample, logit_new, gumbel_new, next_states

    @partial(nn.scan, variable_broadcast="params", split_rngs={"params": False})
    def _scan_gumbel(self, carry, key):
        sample, logits, gumbel, states, cumsum = carry
        position = key[1]
        keys = jnp.expand_dims(random.split(key[0], sample.shape[0]), -2)
        step = partial(self._gumbel_step, position=position)
        sample, logits, gumbel, states = jax.vmap(step)(
            sample, logits, gumbel, keys, states, cumsum
        )
        sorted_carry = _sort_gumbel(sample, logits, gumbel, states)
        chosen_path = sorted_carry[0][:, 0, :]
        prefix_mask = (jnp.arange(self.sequence_length) <= position).astype(jnp.int64)
        token_particles = jnp.sum(self.net.patch_states[chosen_path], axis=-1)
        cumsum = jnp.sum(token_particles * prefix_mask[None, :], axis=-1)
        return (*sorted_carry, cumsum), None
