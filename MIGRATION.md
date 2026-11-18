# Migrating from jVMC 1.x to 2.0

jVMC 2.0 is an almost complete rewrite. The ideas are the same (a network, a sampler, operators
and an optimizer), but every component now sits behind its own small interface, MPI is replaced
by JAX sharding, and the whole API follows PEP 8 naming. Code written for jVMC 1.x does not run
on 2.0 without changes. This guide lists what changed and how to port a script.

If you use an AI coding agent, the skill in [`skills/jvmc/`](https://github.com/computational-quantum-science/jvmc-2.0/blob/master/skills/jvmc/SKILL.md) gives it the
2.0 API, including the table of old names below.

## Overview

| Topic | 1.x | 2.0 |
|---|---|---|
| Naming | camelCase (`numSamples`, `batchSize`) | snake_case (`num_samples`, `batch_size`) everywhere |
| Parallelism | MPI through `mpi4py` + `pmap` over local devices | JAX sharding over all devices; `jax.distributed` for several processes. No MPI. |
| Array shapes | leading `(devices, batch, ...)` dimensions | flat `(N, *sample_shape)` arrays, sharded along the first axis |
| Sample shape | given to the sampler | given to `NQS` |
| Operators | `BranchFreeOperator` filled with `add(scal_opstr(...))`, then `compile()` | algebraic expressions: `H += a * op.SigmaZ(i) * op.SigmaZ(j)` |
| Optimization | manual loop over `stepper.step(...)` with a `TDVP` object | optimizer objects with `ground_state_search()` and `time_evolution()` |
| Ground state vs dynamics | one `TDVP` class, switched by `rhsPrefactor` and `makeReal` | `SR`, `MinSR` or `Adam` for ground states, `TDVP` for real time |
| Linear solver | options of `TDVP` (`svdTol`, `snrTol`, `pinvTol`, ...) | separate solver objects in `jVMC_exp.solver` |
| Loss | the energy, hard-wired | objective functions in `jVMC_exp.objective_function` (`Observable`, `Infidelity`, `Estimator`, ...) |
| Precision | `global_defs.tReal`, `tCpx` | environment variables `JVMC_DT_*` and `global_defs.update()`; mixed precision per `NQS` |

## Porting a ground-state search

jVMC 1.x:

```python
import jax
jax.config.update("jax_enable_x64", True)
import jax.random as random
import jVMC

psi = jVMC.vqs.NQS(jVMC.nets.CpxRBM(numHidden=8, bias=False), seed=1234)

hamiltonian = jVMC.operator.BranchFreeOperator()
for l in range(L):
    hamiltonian.add(jVMC.operator.scal_opstr(-1., (jVMC.operator.Sz(l), jVMC.operator.Sz((l + 1) % L))))
    hamiltonian.add(jVMC.operator.scal_opstr(g, (jVMC.operator.Sx(l), )))

sampler = jVMC.sampler.MCSampler(psi, (L,), random.PRNGKey(4321),
                                 updateProposer=jVMC.sampler.propose_spin_flip_Z2,
                                 numChains=100, sweepSteps=L, numSamples=5000, thermalizationSweeps=25)
tdvpEquation = jVMC.util.tdvp.TDVP(sampler, rhsPrefactor=1., svdTol=1e-8, diagonalShift=10, makeReal='real')
stepper = jVMC.util.stepper.Euler(timeStep=1e-2)

for n in range(300):
    dp, _ = stepper.step(0, tdvpEquation, psi.get_parameters(), hamiltonian=hamiltonian, psi=psi, numSamples=None)
    psi.set_parameters(dp)
    print(n, tdvpEquation.ElocMean0 / L, tdvpEquation.ElocVar0 / L)
```

jVMC 2.0:

```python
import jVMC_exp                                   # enables float64 in JAX
import jVMC_exp.operator.discrete as op

psi = jVMC_exp.vqs.NQS(jVMC_exp.nets.CpxRBM(num_hidden=8, bias=False), sample_shape=L,
                       batch_size=5000, seed=1234)

hamiltonian = 0
for l in range(L):
    hamiltonian += -op.SigmaZ(l) * op.SigmaZ((l + 1) % L) + g * op.SigmaX(l)

sampler = jVMC_exp.sampler.MCSampler(psi, jVMC_exp.propose.SpinFlipZ2(), key=4321,
                                     num_chains=100, sweep_steps=L, num_samples=5000, thermalization_sweeps=25)
opt = jVMC_exp.optimizer.SR(sampler, psi, diagonal_shift=0.0, diagonal_scale=10,
                            solver=jVMC_exp.solver.Pinv(pinv_cutoff=1e-8))
out = opt.ground_state_search(300, jVMC_exp.objective_function.Observable(hamiltonian),
                              jVMC_exp.stepper.Euler(1e-2))
energy, variance = out["energy"]["mean"], out["energy"]["variance"]
```

For real-time evolution, replace `SR` by
`jVMC_exp.optimizer.TDVP(sampler, psi, make_real=True, solver=jVMC_exp.solver.PinvSNR(...))` and
`ground_state_search(steps, ...)` by `time_evolution(t_max, ...)`, usually with an adaptive
stepper such as `jVMC_exp.stepper.RK45`. See `examples/unitary_time_evolution.ipynb`.

## Behaviour changes to check

These are changes in meaning or defaults; code using them still runs but does something different.

* **`diagonalShift` of the 1.x `TDVP` was relative** (S → S + ρ·diag(S)). In 2.0 that is
  `diagonal_scale`, while `diagonal_shift` adds a constant (S → S + λ·1). The 1.x `MinSR`
  `diagonalShift` was already a constant and maps to `diagonal_shift`.
* **SNR regularization is off by default.** The 1.x `TDVP` used `snrTol=2`; `PinvSNR` has
  `snr_tol=0`. Pass `PinvSNR(snr_tol=2)` to keep the old behaviour.
* **`NQS` has no default seed.** 1.x used `seed=1234`; 2.0 draws a random seed when `seed=None`.
  Pass a seed for reproducible runs.
* **`NQS` needs a batch size.** 1.x defaulted to `batchSize=1000`; 2.0 requires `batch_size` or
  `batch_size_per_device`, which must be divisible by the number of devices.
* **Sampler defaults.** `num_chains` defaults to 32 (was 1), and `sweep_steps` to the number of
  sites (was 10). The numbers of samples and chains are rounded up to multiples of the number
  of devices.
* **Stored samples.** `sampler(op)` evaluates on the samples of the last `sample()` call, including
  after an optimization. Pass `resample=True` to measure the current state.
* **`OutputManager`** takes `path=None` (in memory) and defaults to `append=True` (was
  `append=False`, which overwrote the file).
* **`SampledObs.mean`, `.var`** are properties instead of methods; `covar` is `get_covar`,
  `subset` is `get_subset`.
* **Importing the library enables `jax_enable_x64`** for the whole process.

## Name changes

### Variational state (`jVMC.vqs` → `jVMC_exp.vqs`)

| 1.x | 2.0 |
|---|---|
| `NQS(net, logarithmic=True, batchSize=1000, seed=1234, orbit=None, avgFun=...)` | `NQS(net, sample_shape, batch_size=..., logarithmic=True, seed=None, orbit=None, symmetry_average="exp")` |
| `psi.get_parameters()` | `psi.parameters_flat` |
| `psi.set_parameters(p)` | `psi.parameters = p` (flat array or tree) |
| `psi.update_parameters(dp)` | `psi.parameters = psi.parameters_flat + dp` |
| `psi.gradients(s)` (with device/batch dimensions) | `psi.gradients(s)` with `s` of shape `(N, *sample_shape)` |
| `psi.is_generator` | `psi.is_generator` (property) |

### Networks (`jVMC.nets` → `jVMC_exp.nets`)

All fields are renamed to snake_case: `numHidden` → `num_hidden`, `actFun` → `act_fun`,
`firstLayerBias` → `first_layer_bias`, `periodicBoundary` → `periodic_boundary`, `hiddenSize` →
`hidden_size`, `inputDim` → `input_dim`, `initScale` → `init_scale`, `logProbFactor` →
`log_prob_factor`, `realValuedOutput` → `real_valued_output`, `realValuedParams` →
`real_valued_params`, `initFun` → `init_fun`.

| 1.x | 2.0 |
|---|---|
| `CpxRBM_Nospinflip` | `CpxRBMNoSpinFlip` |
| `SymNet`, `CpxCNNSym`, `sym_wrapper.avgFun_Coefficients_Exp/Log/Sep` | `jVMC_exp.symmetry.ProjectedOrbitNet`, `projector * net`, `symmetry_average="exp"/"log"/"sep"` |
| `jVMC.util.util.init_net(descr, dims, seed)` | build the network and `NQS` directly |

New in 2.0: `RWKVCPM`, `RWKV2D`, `ViterittiViT`, `HiddenPfaffianViT`, `FullConfigurationState`,
`ParticleConservingAutoregressive`, support for `flax.nnx` networks (`NQS(..., nnx_init=...)`).

### Operators (`jVMC.operator` → `jVMC_exp.operator.discrete`, `.continuous`)

| 1.x | 2.0 |
|---|---|
| `BranchFreeOperator(lDim=2)`, `.add(...)`, `.compile()` | sums and products of operators; no compilation step |
| `scal_opstr(a, (A, B))` | `a * A * B` |
| `Sx(i)`, `Sy(i)`, `Sz(i)` | `SigmaX(i)`, `SigmaY(i)`, `SigmaZ(i)` |
| `Sp(i)`, `Sm(i)` | `SigmaPlus(i)`, `SigmaMinus(i)` |
| `Id(idx, lDim)` | `IdentityOperator(l_dim, idx)` |
| `number(i)`, `creation(i)`, `annihilation(i)` | `Number(i, spin=None)`, `Creation(i, spin=None)`, `Annihilation(i, spin=None)`; `spin="up"/"down"` for spinful fermions |
| time-dependent prefactor functions in `scal_opstr` | callables `f(t, **kwargs)` as coefficients |
| `op.get_s_primes(s)`, `op.get_O_loc(logPsiS, logPsiSP)` | `op.get_conn_elements(s, batch_size)`, `op.get_O_loc(s, psi)`, or `sampler(op)` |
| `jVMC.operator.povm` | removed |

New in 2.0: `to_dense(basis)`, `to_sparse(basis)`, continuous-space operators
(`TotalKineticOperator`, `PotentialOperator`, `CoulombInteraction`, ...) with
`jVMC_exp.geometry`, and `jVMC_exp.util.matrix_to_jvmc_operator`.

### Sampling (`jVMC.sampler` → `jVMC_exp.sampler`, `jVMC_exp.propose`)

| 1.x | 2.0 |
|---|---|
| `MCSampler(net, sampleShape, key, updateProposer, numChains, updateProposerArg, numSamples, thermalizationSweeps, sweepSteps, initState, mu, logProbFactor)` | `MCSampler(psi, update_proposer, key, num_chains, num_samples, thermalization_sweeps, sweep_steps, init_state, mu, log_prob_factor)` |
| `ExactSampler(net, sampleShape, lDim, logProbFactor)` | `ExactSampler(psi, l_dim, log_prob_factor)` |
| `propose_spin_flip`, `propose_spin_flip_Z2`, `propose_spin_flip_zeroMag`, `propose_POVM_outcome` | `propose.SpinFlip()`, `SpinFlipZ2()`, `SpinFlipZeroMag()`, `POVMOutcome()` |
| `sampler.sample(numSamples=...)` | `sampler.sample(num_samples=...)` |
| `set_number_of_samples(n)`, `get_last_number_of_samples()` | `sampler.num_samples` |
| `set_random_key(k)` | `sampler.key = k` |
| `ExactSampler.get_basis()` | `ExactSampler.basis` |
| `key=random.PRNGKey(n)` | `key=n` or a `PRNGKey` |

New in 2.0: `MCSamplerCont` with the `RWM` and `MALA` proposers, `CutoffSampler`, and
`sampler(op)`, which returns a `SampledObs`.

### Optimization (`jVMC.util.tdvp`, `jVMC.util.minsr`, `jVMC.util.stepper` → `jVMC_exp.optimizer`, `.solver`, `.stepper`)

| 1.x | 2.0 |
|---|---|
| `TDVP(sampler, rhsPrefactor=1., makeReal='real', diagonalShift=ρ, ...)` (ground state) | `optimizer.SR(sampler, psi, diagonal_scale=ρ, solver=...)` |
| `TDVP(sampler, rhsPrefactor=1.j, makeReal=..., ...)` (dynamics) | `optimizer.TDVP(sampler, psi, make_real=..., solver=...)` |
| `svdTol`, `pinvTol`, `pinvCutoff`, `snrTol`, `diagonalizeOnDevice` | `solver.Pinv(pinv_cutoff=...)` or `solver.PinvSNR(pinv_tol=..., pinv_cutoff=..., snr_tol=..., diagonalization_mode="device"/"host"/"distributed")` |
| `crossValidation=True` | `TDVP(..., use_cross_validation=True)` |
| `MinSR(sampler, pinvTol, diagonalShift)` | `optimizer.MinSR(sampler, psi, diagonal_shift=..., solver=solver.Pinv(...))` |
| `tdvpEquation.get_energy_mean()`, `ElocMean0`, `get_metadata()` | the dictionaries returned by `ground_state_search` / `time_evolution` (`save_meta_data=True` for metadata) |
| `Euler(timeStep)`, `Heun(timeStep)`, `AdaptiveHeun(timeStep, tol, maxStep)` | `Euler(time_step)`, `Heun(time_step)`, `AdaptiveHeun(time_step, rtol, atol, min_step, max_step)` |
| manual `stepper.step(...)` loop | `opt.ground_state_search(steps, objective, stepper)`, `opt.time_evolution(t_max, objective, stepper)` |
| `jVMC.util.util.ground_state_search(psi, ham, tdvpEquation, sampler, ...)` | `opt.ground_state_search(...)` |
| `jVMC.util.measure(observables, psi, sampler)` | `jVMC_exp.util.measure(observables, sampler)`, or `observables=` of the run methods |

New in 2.0: `RK23`, `RK45`, the `Adam` optimizer, the `CG` solver, objective functions
(`Infidelity`, `Estimator`, `ParametricObservable`), learning-rate and regularization schedules
(callables, e.g. `optax` schedules), callbacks, and the batched Jacobian
(`batched_jacobian=True`).

### Statistics, parallelism and output

| 1.x | 2.0 |
|---|---|
| `jVMC.mpi_wrapper.global_mean(x, p)`, `global_variance`, `global_covariance`, `global_sum` | `jVMC_exp.stats.SampledObs(x, p).mean`, `.var`, `.get_covar()` |
| `mpi_wrapper.distribute_sampling(...)`, `global_defs.set_pmap_devices(...)` | automatic; all visible devices are used |
| running with `mpirun -n k` | one process per GPU with `JVMC_USE_DISTRIBUTED=true` (e.g. under `srun`) |
| `SampledObs.mean()`, `.var()`, `.covar()`, `.subset()` | `.mean`, `.var`, `.get_covar()`, `.get_subset()` |
| `global_defs.tReal`, `global_defs.tCpx` | `global_defs.DT_PARAMS_REAL`, `DT_PARAMS_CPX`, `DT_OUT_REAL`, `DT_OUT_CPX`, set with `JVMC_DT_*` or `global_defs.update()` |
| `OutputManager(dataFileName, group="/", append=False)` | `jVMC_exp.util.OutputManager(path=None, group="/", append=True)` |
| `jVMC.util.symmetries.get_orbit_1D(L, ...)`, `get_orbit_2D_square(L, ...)` | `jVMC_exp.symmetry.lattice_symmetries.chain_reflection_symmetry`, `square_translation_symmetry`, `square_d4_symmetry`, `spin_flip_symmetry`, ... (composed with `*`) |

## Removed

* `jVMC.mpi_wrapper` and the dependence on `mpi4py`.
* POVM operators and the Lindblad (open-system) workflow (`jVMC.operator.povm`).
* `jVMC.util.util.init_net` and the dictionary-based network construction.
* Python < 3.11.
