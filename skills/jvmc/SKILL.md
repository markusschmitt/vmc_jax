---
name: jvmc
description: Write, run, debug and review Python code that uses jVMC 2.0 (imported as jVMC), a JAX/Flax library for variational Monte Carlo with neural quantum states. Use it for ground-state searches (SR, MinSR, Adam), real-time dynamics with TDVP, infidelity optimization, building spin, fermion or continuous-space Hamiltonians, custom or symmetric or autoregressive networks, sampling and error bars, out-of-memory problems, and multi-GPU or multi-node runs. Also use it when porting jVMC 1.x code (camelCase API, mpi_wrapper, BranchFreeOperator) to 2.0.
---

# jVMC 2.0

jVMC is a variational Monte Carlo (VMC) library for neural quantum states (NQS), built on JAX and
Flax. The library is imported as `jVMC`.

**Your training data probably contains jVMC 1.x, which has a different API.** Old code uses
camelCase names (`numHidden`, `batchSize`, `numSamples`), `jVMC.mpi_wrapper`,
`jVMC.util.tdvp.TDVP`, `BranchFreeOperator`/`scal_opstr`, `psi.get_parameters()` and manual
`stepper.step(...)` loops. **None of these exist in 2.0.** Write code only with the names on this
page and in `references/`. If unsure, check `jVMC.__version__` (2.x) and read the
docstring of the class (`help(jVMC.optimizer.SR)`); the docstrings are complete and accurate.
The section [Names from jVMC 1.x](#names-from-jvmc-1x) maps the old names.

## The pipeline

Every calculation is the same chain of objects. Each one is independent and replaceable:

| Step | Object | Typical choice |
|---|---|---|
| 1. Network: one configuration → log ψ(s) | `jVMC.nets.*` or any Flax module | `CpxRBM`, `CNN`, `ViterittiViT`, `RWKVCPM`, custom |
| 2. Variational state | `jVMC.vqs.NQS(net, sample_shape, batch_size=..., seed=...)` | |
| 3. Sampler | `jVMC.sampler.ExactSampler` (small systems) / `MCSampler` + proposer from `jVMC.propose` | |
| 4. Operator | `jVMC.operator.discrete` (spins, fermions) / `.continuous` (particles in space) | |
| 5. Objective function | `jVMC.objective_function.Observable(H)` | `Infidelity`, `Estimator` |
| 6. Optimizer | `jVMC.optimizer.SR` / `MinSR` / `Adam` / `TDVP` | |
| 7. Solver of the linear system | `jVMC.solver.Pinv` / `PinvSNR` / `CG` | |
| 8. Stepper (integrator; its time step is the learning rate) | `jVMC.stepper.Euler` / `Heun` / `AdaptiveHeun` / `RK23` / `RK45` | |
| 9. Run | `opt.ground_state_search(steps, objective, stepper)` / `opt.time_evolution(t_max, objective, stepper)` | |

## Ground-state search: complete example

Transverse-field Ising chain, $H = -\sum_l \sigma^z_l\sigma^z_{l+1} + g\sum_l \sigma^x_l$, periodic.

```python
import numpy as np
import optax
import jVMC
import jVMC.operator.discrete as op

L, g = 10, -0.7
n_steps = 100
n_samples = 2**11

# Hamiltonian: operators combine with +, * and scalar coefficients
H = 0
for l in range(L):
    H += -op.SigmaZ(l) * op.SigmaZ((l + 1) % L) + g * op.SigmaX(l)
M_x = sum(op.SigmaX(l) for l in range(L)) / L

# Network and variational state
net = jVMC.nets.CpxRBM(num_hidden=2 * L, bias=False)
psi = jVMC.vqs.NQS(net, sample_shape=L, batch_size=n_samples, seed=123)

# Metropolis sampler; SpinFlipZ2 adds occasional global spin flips
sampler = jVMC.sampler.MCSampler(
    psi,
    jVMC.propose.SpinFlipZ2(),
    key=123,
    num_chains=n_samples // 4,
    num_samples=n_samples,
    thermalization_sweeps=25,
)

# Stochastic reconfiguration; the Euler time step is the learning rate (float or optax schedule)
lr = optax.linear_schedule(5e-2, 1e-3, transition_steps=int(0.7 * n_steps), transition_begin=int(0.3 * n_steps))
stepper = jVMC.stepper.Euler(lr)
loss = jVMC.objective_function.Observable(H)
opt = jVMC.optimizer.SR(sampler, psi, diagonal_shift=1e-3, solver=jVMC.solver.Pinv())

out = opt.ground_state_search(n_steps, loss, stepper, observables={"M_x": M_x})

energy = np.array(out["energy"]["mean"])        # one entry per step; also "variance", "MC_error"
magnetization = np.array(out["M_x"]["mean"])
steps = np.array(out["times"])

# Final estimate with fresh samples (the sampler still holds the samples of the last step)
E = sampler(H, resample=True)
print(f"E/L = {E.mean.real.item() / L:.6f} +- {E.error_of_mean.item() / L:.6f}")
```

To check against exact diagonalization on a small system, use the exact sampler and the sparse
matrix of the operator. The basis is in the order of `ExactSampler.basis`:

```python
import scipy.sparse.linalg as sla

exact = jVMC.sampler.ExactSampler(psi)          # enumerates all 2**L configurations
E_vmc = exact(H).mean.real.item()                    # exact expectation value in psi
E_0 = sla.eigsh(H.to_sparse(exact.basis), k=1, which="SA")[0][0]
print(f"relative error {abs((E_vmc - E_0) / E_0):.1e}")
```

## Real-time evolution (TDVP): what changes

Replace the optimizer, the stepper and the method that is called. Use more samples than for a
ground-state search and no diagonal shift (it biases the dynamics):

```python
L_t = 6
H_t = 0
for l in range(L_t):
    H_t += -op.SigmaZ(l) * op.SigmaZ((l + 1) % L_t) - 0.7 * op.SigmaX(l) + 0.1 * op.SigmaZ(l)

psi_t = jVMC.vqs.NQS(jVMC.nets.CpxRBM(num_hidden=10, bias=True), L_t, batch_size=2**L_t, seed=1)
sampler_t = jVMC.sampler.ExactSampler(psi_t)     # MCSampler for larger systems

opt_t = jVMC.optimizer.TDVP(sampler_t, psi_t, make_real=True, solver=jVMC.solver.PinvSNR(pinv_tol=1e-8))
stepper_t = jVMC.stepper.RK45(time_step=1e-3, rtol=0.0, atol=1e-6)   # adaptive

out_t, meta = opt_t.time_evolution(
    0.5,                                             # t_max
    jVMC.objective_function.Observable(H_t),
    stepper_t,
    observables={"X": sum(op.SigmaX(l) for l in range(L_t)) / L_t},
    norm_function=jVMC.util.s_norm_fn(opt_t),   # step-size error measured as a distance between states
    save_meta_data=True,
)
times = np.array(out_t["times"])
tdvp_error = np.array(meta["tdvp_error"])            # r^2; r^2 * Var(H) growing = results not trustworthy
```

The energy and its variance are conserved by exact unitary dynamics, so their drift is a cheap
sanity check. Details: `references/optimization.md`.

## Rules that prevent most mistakes

1. **A network evaluates one configuration.** Its `__call__` takes an array of shape
   `sample_shape` and returns a scalar log ψ(s) (complex, or real for a positive state). `NQS`
   vectorizes, batches and shards it. Calling `psi(s)` takes `s` of shape `(N, *sample_shape)` and
   returns shape `(N,)`. There are no device/batch leading dimensions as in 1.x.
2. **Configurations.** Spins are `int32` 0/1, where local state 0 has σᶻ = +1. Spinful fermions use
   0 = empty, 1 = ↑, 2 = ↓, 3 = ↑↓ (local dimension 4; use `ExactSampler(psi, l_dim=4)`). 2D
   lattices use `sample_shape=(Ly, Lx)`, and operators act on the flattened index `y * Lx + x`.
   Continuous systems use `float64` positions of shape `(n_particles * n_dim,)`.
3. **Samplers keep their last samples.** `sampler(op)` evaluates on the stored samples, which are
   not refreshed when `psi.parameters` changes. After an optimization use
   `sampler(op, resample=True)` (MC) or call `sampler.sample()` before `sampler(op)` (exact).
4. **Results are `SampledObs`.** Use `.mean`, `.var` and `.error_of_mean`. They are arrays of shape
   `(1,)` for a scalar observable, so use `.mean.real.item()`. MCMC diagnostics:
   `.get_autocorrelation_time(sampler.num_chains)` and `.get_R_hat(sampler.num_chains)`. These
   need many samples per chain.
5. **`batch_size` is the memory knob**, not a statistics knob. It must be divisible by the number
   of devices. Start with `batch_size = num_samples` and halve it on out-of-memory errors. For
   very large networks also pass `batched_jacobian=True` to the objective function. Sample and
   chain counts are rounded up to multiples of the device count, so a run can return more
   samples than requested.
6. **Regularization naming.** `diagonal_shift` adds a constant to the diagonal of S (or of the
   tangent kernel for MinSR). `diagonal_scale` multiplies the diagonal of S. Both accept a
   callable `step -> value`. The 1.x TDVP `diagonalShift` corresponds to `diagonal_scale`.
7. **Which optimizer.** Use `SR` when parameters ≲ samples, and `MinSR` when parameters ≫ samples
   (transformers, large RNNs). `Adam` takes no stepper: call
   `Adam(sampler, psi, learning_rate=...).ground_state_search(steps, loss)`. `TDVP` is for real
   time only (its `ground_state_search` raises), and `SR` is for imaginary time only.
8. **Solvers.** `Pinv(pinv_cutoff=...)` is a hard eigenvalue cutoff. `PinvSNR` (the default for
   SR/TDVP) uses a smooth adaptive cutoff; its `snr_tol > 0` needs `batched_jacobian=False` and
   does not apply to `MinSR`. `CG` is matrix-free and iterative.
9. **Global settings go before the import.** Use environment variables: `JVMC_DT_PARAMS_REAL`,
   `JVMC_DT_PARAMS_CPX`, `JVMC_DT_OUT_REAL`, `JVMC_DT_OUT_CPX`, `JVMC_BATCH_LOOP`,
   `JVMC_USE_DISTRIBUTED`. Alternatively call `jVMC.global_defs.update(...)` after the
   import but before creating `NQS`. Importing the library turns on `jax_enable_x64` for the
   whole process.
10. **Validate on a small system first.** Use `ExactSampler` and `op.to_sparse(sampler.basis)` (or
    `.to_dense`) to compare with exact diagonalization. Then switch to `MCSampler` and scale up.
    Report energies with their error bars, and use the energy variance as a convergence measure
    that does not need a reference.

## Where to look next

| Task | Read |
|---|---|
| Hamiltonians: spins, fermions, time-dependent couplings, custom operators, continuous space | `references/operators.md` |
| Built-in networks, custom Flax/nnx networks, symmetry projection, autoregressive models, mixed precision | `references/networks.md` |
| Samplers, proposers, measuring observables, statistics | `references/sampling.md` |
| Objective functions, optimizers, solvers, steppers, callbacks, checkpoints, TDVP details, infidelity | `references/optimization.md` |
| Memory, batch size, Markov chains, multiple GPUs and nodes | `references/scaling.md` |

The example notebooks in the repository (`examples/` at
https://github.com/computational-quantum-science/jvmc-2.0) are complete, tested workflows:
`ground_state_search`, `unitary_time_evolution`, `custom_net`, `fermions`,
`infidelity_optimization`, `projected_time_evolution` (ptVMC), `2dho_ground_state_search` (continuous space),
`viteritti_vit_ground_state_search`, `rwkv_ground_state_search`, `rwkv2d_ground_state_search`
and `hidden_pfaffian_vit_ground_state_search`.

## Names from jVMC 1.x

| 1.x | 2.0 |
|---|---|
| `jVMC.mpi_wrapper.global_mean(x, p)` and the other MPI helpers | `SampledObs(x, p).mean`; the statistics in `jVMC.stats`. There is no MPI; multi-node runs use `jax.distributed` (`references/scaling.md`) |
| `NQS(net, batchSize=..., seed=...)` | `NQS(net, sample_shape, batch_size=..., seed=...)` |
| `psi.get_parameters()` / `set_parameters(p)` / `update_parameters(dp)` | `psi.parameters_flat` / `psi.parameters = p` / `psi.parameters = psi.parameters_flat + dp` |
| `MCSampler(psi, (L,), key, updateProposer=jVMC.sampler.propose_spin_flip, numChains=..., numSamples=..., thermalizationSweeps=..., sweepSteps=...)` | `MCSampler(psi, jVMC.propose.SpinFlip(), key=..., num_chains=..., num_samples=..., thermalization_sweeps=..., sweep_steps=...)` |
| `propose_spin_flip_Z2`, `propose_spin_flip_zeroMag` | `jVMC.propose.SpinFlipZ2()`, `SpinFlipZeroMag()` |
| `ExactSampler(psi, (L,), lDim=2)` | `ExactSampler(psi, l_dim=2)` |
| `BranchFreeOperator()` + `.add(scal_opstr(a, (Sz(i), Sz(j))))` | `H += a * op.SigmaZ(i) * op.SigmaZ(j)` |
| `Sx, Sy, Sz, Sp, Sm, Id, number, creation, annihilation` | `SigmaX, SigmaY, SigmaZ, SigmaPlus, SigmaMinus, IdentityOperator, Number, Creation, Annihilation` |
| `op.get_s_primes(s)` + `op.get_O_loc(logPsi, logPsiSP)` | `op.get_O_loc(s, psi)`, or simply `sampler(op)` |
| `jVMC.util.tdvp.TDVP(sampler, rhsPrefactor=1., diagonalShift=..., makeReal='real')` for ground states | `jVMC.optimizer.SR(sampler, psi, diagonal_scale=...)` |
| `jVMC.util.tdvp.TDVP(sampler, rhsPrefactor=1.j, ...)` for dynamics | `jVMC.optimizer.TDVP(sampler, psi, make_real=True, solver=...)` |
| `svdTol`, `snrTol`, `pinvTol`, `pinvCutoff`, `diagonalizeOnDevice` of TDVP | the solver: `PinvSNR(snr_tol=..., pinv_tol=..., pinv_cutoff=..., diagonalization_mode="device"/"host"/"distributed")` |
| `jVMC.util.minsr.MinSR(sampler, diagonalShift=...)` | `jVMC.optimizer.MinSR(sampler, psi, diagonal_shift=...)` |
| `jVMC.util.stepper.Euler(timeStep=...)`, `AdaptiveHeun(tol=...)` | `jVMC.stepper.Euler(time_step)`, `AdaptiveHeun(time_step, rtol=..., atol=...)` |
| loop of `stepper.step(0, tdvpEquation, psi.get_parameters(), hamiltonian=H, psi=psi)` | `opt.ground_state_search(steps, Observable(H), stepper)` / `opt.time_evolution(...)` |
| `jVMC.util.measure(obs, psi, sampler)` | `jVMC.util.measure(obs, sampler)`, or `observables=` of the run methods |
| `jVMC.util.symmetries.get_orbit_1D(...)`, `NQS(..., orbit=..., avgFun=...)`, `nets.SymNet` | `jVMC.symmetry.lattice_symmetries.*` → `SymmetryProjector`; `projector * net` or `NQS(..., orbit=projector, symmetry_average="exp")` |
| net fields `numHidden`, `actFun`, `hiddenSize`, `inputDim`, `logProbFactor`, `periodicBoundary`, ... | `num_hidden`, `act_fun`, `hidden_size`, `input_dim`, `log_prob_factor`, `periodic_boundary`, ... |
| `global_defs.tReal`, `tCpx`, `set_pmap_devices` | `global_defs.DT_PARAMS_REAL`, `DT_PARAMS_CPX`, ... (set by environment variables); devices are found automatically |
| `operator.povm`, Lindblad examples | not available in 2.0 |
