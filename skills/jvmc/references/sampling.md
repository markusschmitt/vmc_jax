# Sampling and statistics

```python
import numpy as np
import jax.numpy as jnp
import jVMC
import jVMC.operator.discrete as op

L = 8
psi = jVMC.vqs.NQS(jVMC.nets.CpxRBM(num_hidden=8), L, batch_size=2**L, seed=0)
H = sum(-op.SigmaZ(i) * op.SigmaZ((i + 1) % L) - op.SigmaX(i) for i in range(L))
```

## Samplers (`jVMC.sampler`)

| Sampler | Use |
|---|---|
| `ExactSampler(psi, l_dim=2)` | Enumerates all `l_dim**N` configurations, weighted by \|ψ\|². No noise; for small systems, tests and benchmarks. `.basis` is the ordered basis. |
| `MCSampler(psi, proposer, key=..., num_chains=..., num_samples=..., thermalization_sweeps=10, sweep_steps=None, init_state=None, mu=2)` | Metropolis–Hastings on discrete configurations. With `proposer=None`, direct sampling from a network with `sample()`. |
| `MCSamplerCont(psi, proposer, ...)` | Same, for continuous positions. |
| `CutoffSampler(psi, proposer, eps, ...)` | MCMC from a distribution with a floor `eps`, reweighted to stay unbiased; reduces the bias from rarely visited configurations. Not for networks with `eval_ratio`. |

* `sweep_steps` is the number of proposals between kept samples (default: the number of sites).
  `thermalization_sweeps` sweeps are discarded at every `sample()` call.
* `num_chains` and `num_samples` are rounded up to multiples of the number of devices. Use many
  chains on GPUs (hundreds to thousands); the best value depends on the problem.
* `mu < 2` samples |ψ|^μ and reweights, i.e. importance sampling. The weights are then not
  uniform.
* `sampler.acceptance_ratio` checks the Metropolis acceptance; very low values mean bad mixing.

Proposers (`jVMC.propose`, also `jVMC.sampler.propose`):

| Proposer | Move |
|---|---|
| `SpinFlip()` | flip one random site |
| `SpinFlipZ2()` | single flips plus occasional global spin flips, for ferromagnetic/Z2-symmetric phases |
| `SpinFlipZeroMag()` | exchange a 0 and a 1 (stays in the zero-magnetization sector); needs an `init_state` in that sector |
| `RWM(geometry)` | Gaussian random walk of all particles, step size tuned during thermalization |
| `MALA(geometry, apply_fun)` | Langevin move along ∇ log \|ψ\| |

A custom proposer subclasses `jVMC.propose.AbstractProposer` and implements
`__call__(key, s, arg) -> (s_new, log_proposal_ratio)`; it must be `jax.jit`-compatible.

```python
n_samples = 2**10
mc = jVMC.sampler.MCSampler(psi, jVMC.propose.SpinFlip(), key=1,
                                num_chains=n_samples // 4, num_samples=n_samples, thermalization_sweeps=20)
s, log_psi, weights = mc.sample()            # s: (num_samples, L) int32; weights sum to 1

zero_mag = jnp.array([[0, 1] * (L // 2)] * 16, dtype=jnp.int32)
mc_u1 = jVMC.sampler.MCSampler(psi, jVMC.propose.SpinFlipZeroMag(), key=2,
                                   num_chains=16, num_samples=64, init_state=zero_mag)
```

## Measuring

`sampler(operator, **kwargs)` returns a `jVMC.stats.SampledObs` evaluated on the **stored**
samples of the last `sample()` call. For `MCSampler`, `resample=True` forces new samples, and
`num_samples=n` draws `n` samples for this call only.

```python
E = mc(H, resample=True)
print(E.mean.real.item(), E.error_of_mean.item(), E.var.item())
tau = E.get_autocorrelation_time(mc.num_chains)    # needs many samples per chain to be meaningful
r_hat = E.get_R_hat(mc.num_chains)                 # Gelman–Rubin; close to 1 means converged chains

E_big = mc(H, num_samples=2**12)                     # a larger, one-off measurement

obs = jVMC.util.measure({"energy": H, "Z0": op.SigmaZ(0)}, mc)
print(obs["energy"]["mean"], obs["energy"]["MC_error"])
```

`SampledObs(observations, weights)` holds the data. Its members are `.mean`, `.var`,
`.error_of_mean` (arrays with one entry per observable column), `.get_covar(other)`,
`.transform(fn)`, `.get_subset(...)` and `.select(idx)`. `LazySampledObs` does the same for data
that arrive in batches (the batched Jacobian).

For the exact sampler, refresh after changing parameters: `exact.sample(); exact(H)`.

## Custom samplers

To restrict an exact sum to a symmetry sector, subclass `jVMC.sampler.base.AbstractSampler`.
Implement `sample()`, which sets `self._samples`, `self._log_psi` and `self._weights` and returns
them, and `__call__(observable, **kw)`, which returns
`SampledObs(observable.get_O_loc(self.samples, self.psi, log_psi_s=self.log_psi, **kw), self.weights)`.
The full example is `FixedSpinfulSectorSampler` in
`examples/hidden_pfaffian_vit_ground_state_search.ipynb`.
