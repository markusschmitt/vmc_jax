# Networks and variational states

```python
import jax
import jax.numpy as jnp
import flax.linen as nn
import jVMC_exp
```

## The contract of a network

* A Flax `linen` module (or an `nnx.Module` class, see below) whose `__call__` takes **one**
  configuration of shape `sample_shape` and returns a **scalar**: log ψ(s), complex in general.
  Return real log ψ only for a positive wave function. With `NQS(..., logarithmic=False)` the
  network returns ψ(s) instead.
* Spins arrive as integers 0/1. Map them yourself if you want ±1 (`2 * s - 1`).
* Parameters live in the `params` collection, all with one dtype. For complex parameters use
  `param_dtype=jVMC_exp.global_defs.DT_PARAMS_CPX`. Flax's `dtype=` sets only the computation
  dtype, not the parameter dtype.
* Do not hard-code `dtype=` in layers if you want mixed precision (see below).
* Optional methods: `sample(key)` makes the network a direct (autoregressive) sampler;
  `eval_ratio(s, sp)` returns ψ(s')/ψ(s) cheaply for Metropolis; `eval_real(s)` returns Re log ψ.

```python
class MyRBM(nn.Module):
    num_hidden: int = 8

    @nn.compact
    def __call__(self, s):
        h = nn.Dense(
            self.num_hidden,
            param_dtype=jVMC_exp.global_defs.DT_PARAMS_CPX,
            kernel_init=jVMC_exp.nets.initializers.cplx_init,
        )(2 * s.ravel() - 1)
        return jnp.sum(jnp.log(jnp.cosh(h)))

L = 6
psi = jVMC_exp.vqs.NQS(MyRBM(num_hidden=8), sample_shape=L, batch_size=2**L, seed=1234)
print(psi.num_parameters, psi.real_params, psi.holomorphic)   # holomorphicity is detected automatically
configs = jax.random.bernoulli(jax.random.PRNGKey(0), shape=(5, L)).astype(jnp.int32)
log_psi = psi(configs)                                         # shape (5,)
```

## `NQS`

`jVMC_exp.vqs.NQS(net, sample_shape, batch_size=None, batch_size_per_device=None, logarithmic=True,
seed=None, orbit=None, symmetry_average="exp", sampler_dtype=None, eval_dtype=None, grad_dtype=None,
nnx_init=None)`. Give exactly one of `batch_size` and `batch_size_per_device`.

| Member | Meaning |
|---|---|
| `psi(s)` | log ψ for `s` of shape `(N, *sample_shape)` |
| `psi.gradients(s)` | Jacobian ∂ log ψ/∂θₖ, shape `(N, len(parameters_flat))`; `lazy_gradients(s)` yields it in batches |
| `psi.parameters` | parameter tree `{"params": ...}`; settable with a tree **or** a flat array |
| `psi.parameters_flat` | flat **real** array; complex parameters appear as real parts followed by imaginary parts, per leaf |
| `psi.num_parameters`, `real_params`, `holomorphic`, `is_generator` | introspection |
| `psi.frozen_parameters = ["Dense_0"]` | keep these parameter subtrees fixed during optimization |
| `psi.to_array(basis)` | normalized state vector on a basis (small systems) |

Two networks for amplitude and phase: `NQS((amplitude_net, phase_net), ...)` returns
log ψ = amplitude + i·phase, with real parameters.

`flax.nnx` networks: pass the **class** and its constructor arguments. The `rngs` argument is
handled internally:

```python
from flax import nnx

class RBMNNX(nnx.Module):
    def __init__(self, in_features, num_hidden, *, rngs: nnx.Rngs):
        self.layer = nnx.Linear(in_features, num_hidden, param_dtype=jVMC_exp.global_defs.DT_PARAMS_CPX,
                                kernel_init=jVMC_exp.nets.initializers.cplx_init, rngs=rngs)

    def __call__(self, s):
        return jnp.sum(jnp.log(jnp.cosh(self.layer(2 * s.ravel() - 1))))

psi_nnx = jVMC_exp.vqs.NQS(RBMNNX, L, batch_size=2**L, seed=0, nnx_init=dict(in_features=L, num_hidden=4))
```

## Built-in networks (`jVMC_exp.nets`)

| Network | Notes |
|---|---|
| `RBM(num_hidden, bias)`, `CpxRBM(num_hidden, bias)` | real / complex restricted Boltzmann machine |
| `CNN(F, channels, strides, act_fun, bias, periodic_boundary)`, `CpxCNN(...)` | convolutional; `F` is a tuple of filter sizes, one entry per spatial dimension |
| `FFN(layers, bias, act_fun)` | feed-forward |
| `RNN1DGeneral(L, hidden_size, depth, cell="RNN"/"LSTM"/"GRU", ...)`, `RNN2DGeneral(...)` | autoregressive RNNs |
| `RWKVCPM(L, l_hil_dim, patch_size, hidden_size, num_heads, num_layers, embedding_size)` | 1D autoregressive RWKV; `ParticleConservingAutoregressive(net, Q)` restricts to Q particles |
| `RWKV2D` (= `LogRWKV2DAutoregressiveJVMC`)`(L, grid_Lx, grid_Ly, ...)` | 2D autoregressive RWKV |
| `ViterittiViT` (= `LogViterittiSpatialViTJVMC`)`(Lx, Ly, patch_size, layers, embed_dim, heads, symmetry_average="none"/"d4", ...)` | vision transformer of Viteritti, Rende and Becca; `sample_shape=(Ly, Lx)` |
| `HiddenPfaffianViT(Lx, Ly, symmetry, particles_per_spin, ...)` | spinful fermions: pairing wave function × ViT |
| `FullConfigurationState(L, d)` | one parameter per basis state, for loading exact vectors |
| `TwoNets((net_a, net_b))` | what `NQS` builds from a tuple |

Transformer and RWKV networks take `param_dtype` and `compute_dtype`. Check each class's
docstring for the full list of fields.

Autoregressive networks provide `sample()`. `MCSampler(psi, None, ...)` then draws exact,
independent samples with equal weights; no proposer or thermalization is needed:

```python
net_ar = jVMC_exp.nets.RWKVCPM(L=8, patch_size=2, hidden_size=8, embedding_size=8, num_layers=1)
psi_ar = jVMC_exp.vqs.NQS(net_ar, 8, batch_size=16, seed=31)
sampler_ar = jVMC_exp.sampler.MCSampler(psi_ar, None, key=32, num_samples=64)
s, log_psi_ar, w = sampler_ar.sample()

psi_q = jVMC_exp.vqs.NQS(jVMC_exp.nets.ParticleConservingAutoregressive(net=net_ar, Q=4), 8, batch_size=16, seed=31)
s_q, _, _ = jVMC_exp.sampler.MCSampler(psi_q, None, key=1, num_samples=16).sample()
assert (s_q.sum(axis=-1) == 4).all()
```

## Symmetry projection

`jVMC_exp.symmetry` projects a network onto a symmetry sector,
ψ_χ(s) = (d_χ/|G|) Σ_g χ*_g η_g(s) ψ(g⁻¹s). Projectors come from
`jVMC_exp.symmetry.lattice_symmetries`:

* `square_translation_symmetry(Lx, Ly, particle_type, q=(0, 0))` (lattice momentum `q`)
* `square_d4_symmetry(Lx, Ly, particle_type)`, `rectangle_d2_symmetry(...)`, `point_group_symmetry(...)`
* `chain_reflection_symmetry(L, particle_type)`
* `spin_flip_symmetry(Lx, Ly, particle_type)`, `particle_hole_symmetry(Lx, Ly, particle_type)`

`particle_type` is `"spin"`, `"spinless_fermion"` or `"spinful_fermion"`; fermionic signs are
included. For a chain use `Lx=L, Ly=1`. `projector * projector` composes commuting projectors,
and `projector * net` returns a `ProjectedOrbitNet`. Passing `NQS(net, ..., orbit=projector)` is
equivalent.

```python
from jVMC_exp.symmetry import lattice_symmetries as sym

Lx = Ly = 4
D4 = sym.square_d4_symmetry(Lx, Ly, particle_type="spin")
T0 = sym.square_translation_symmetry(Lx, Ly, particle_type="spin", q=(0, 0))
net_sym = (D4 * T0) * jVMC_exp.nets.RBM(num_hidden=4, bias=True)
psi_sym = jVMC_exp.vqs.NQS(net_sym, Lx * Ly, batch_size=16, seed=7)
```

`symmetry_average`: `"exp"` (default) is the exact coherent projection and works for any
character and fermionic signs. `"log"` and `"sep"` ignore the characters, so they only work for
the trivial sector. Networks with `sample()` accept only `"sep"` (or a custom callable), with
unit characters. Custom groups: `jVMC_exp.symmetry.SymmetryProjector(Lx, Ly, particle_type,
perm=..., character=...)`, where each row of `perm` is a permutation of the site indices.

## Mixed precision

`NQS(..., sampler_dtype=jnp.float32, eval_dtype=jnp.float32, grad_dtype=jnp.float32)` (complex
dtypes for complex parameters) runs sampling, evaluation and gradients at lower precision. The
outputs are still cast to `global_defs.DT_OUT_REAL`/`DT_OUT_CPX`, so the statistics stay in
float64. The network must not fix `dtype=` in its layers. The default parameter dtype comes from
`JVMC_DT_PARAMS_REAL`/`JVMC_DT_PARAMS_CPX` (environment variables, read at import).
