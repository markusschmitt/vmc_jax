# Objective functions, optimizers, solvers, steppers

```python
import numpy as np
import jax.numpy as jnp
import optax
import jVMC
import jVMC.operator.discrete as op

L = 6
H = sum(-op.SigmaZ(i) * op.SigmaZ((i + 1) % L) - 0.8 * op.SigmaX(i) for i in range(L))
psi = jVMC.vqs.NQS(jVMC.nets.CpxRBM(num_hidden=12, bias=True), L, batch_size=2**L, seed=3)
sampler = jVMC.sampler.ExactSampler(psi)
```

## How a step works

An optimizer is the right-hand side of dθ/dt = f(θ), and a stepper integrates it. Each
evaluation of f samples, calls `objective.value_and_grad(sampler)` (local losses, Jacobian Γ,
force F), and then `optimizer.get_update(...)`. For SR this solves (S + regularization) θ̇ = −F
with the solver. `ground_state_search(steps, ...)` repeats the step `steps` times;
`time_evolution(t_max, ...)` repeats it until `t_max`.

With multi-stage steppers (Heun, RK), `resample_stepper=True` (the default) draws new samples at
every stage. With `False`, the samples of the first stage are reused and reweighted, which is
cheaper and less noisy in the adaptive error estimate.

## Objective functions (`jVMC.objective_function`)

| Class | Loss |
|---|---|
| `Observable(H, batched_jacobian=False)` | ⟨H⟩: energy minimization, or the generator of TDVP dynamics |
| `Estimator(f, batched_jacobian=False)` | ⟨f(parameters, s)⟩ for a real function, e.g. a regularizer; `f` is called batched (returns `N` values) and on single configurations (returns a scalar) |
| `ParametricObservable(operator, estimator_fn)` | an operator whose matrix elements depend explicitly on θ |
| `Infidelity(reference_sampler, operator=None, conjugated_operator=None, control_variate=None/"simple"/"adaptive")` | 1 − \|⟨ψ\|Ô\|χ⟩\|²/(⟨ψ\|ψ⟩⟨χ\|χ⟩); fits ψ to Ô\|χ⟩, the building block of projected time evolution (ptVMC) |

`batched_jacobian=True` never builds the full N_samples × N_params Jacobian. It streams batches
of `psi.batch_size` rows instead, which bounds memory for large networks at a modest cost in
speed. It is incompatible with `PinvSNR(snr_tol>0)`.

`objective(sampler)` evaluates the losses on the stored samples as a `SampledObs`, e.g. to
measure an infidelity.

## Optimizers (`jVMC.optimizer`)

| Class | Use | Key arguments |
|---|---|---|
| `SR(sampler, psi, diagonal_shift=1e-3, diagonal_scale=0., solver=PinvSNR())` | ground states (imaginary-time / natural gradient); best when parameters ≲ samples | `diagonal_shift`, `diagonal_scale`: float or `step -> value` |
| `MinSR(sampler, psi, diagonal_shift=1e-3, solver=Pinv())` | same update solved in sample space; best when parameters ≫ samples | use `Pinv` or `CG`, not `PinvSNR(snr_tol>0)` |
| `Adam(sampler, psi, learning_rate=1e-3, b1, b2, eps)` | plain first-order optimization | `ground_state_search(steps, objective)`, **no stepper** |
| `TDVP(sampler, psi, make_real, diagonal_shift=0., solver=PinvSNR(), store_qgt=True, compute_tdvp_error=True)` | real-time dynamics | `make_real=True` (McLachlan) is the usual choice for non-holomorphic networks; holomorphic networks use S θ̇ = −iF automatically |

Every optimizer accepts `output_manager=jVMC.util.OutputManager(...)` (see below). `SR` and
`TDVP` with `store_qgt=True` expose the last quantum geometric tensor as `opt.qgt`.

A custom optimizer subclasses `jVMC.optimizer.base.AbstractOptimizer` and implements only
`get_update(objective_function_output)`.

## Solvers (`jVMC.solver`)

| Class | Behaviour |
|---|---|
| `Pinv(pinv_cutoff=1e-14, diagonalization_mode="device")` | pseudo-inverse via `eigh`, hard relative cutoff |
| `PinvSNR(snr_tol=0, pinv_tol=1e-14, pinv_cutoff=1e-8, ...)` | smooth cutoff, adapted per solve until the residual is below `pinv_tol`; `snr_tol>0` also drops directions with a low signal-to-noise ratio (SR/TDVP only, needs `batched_jacobian=False`) |
| `CG(tol=1e-5, maxiter=None)` | conjugate gradient, matrix-free |

`diagonalization_mode`: `"device"` (default), `"host"` (NumPy on the CPU, saves GPU memory) or
`"distributed"` (`jaxmg`, splits the matrix over all GPUs; needs one process per GPU, see
`scaling.md`).

## Steppers (`jVMC.stepper`)

| Class | Use |
|---|---|
| `Euler(time_step)` | ground states: `time_step` is the learning rate, a float or a callable `step -> value` such as an `optax` schedule |
| `Heun(time_step)` | second order, fixed step |
| `AdaptiveHeun`, `RK23`, `RK45` `(time_step, rtol, atol, min_step=None, max_step=1)` | adaptive steps for dynamics; pass `norm_function=jVMC.util.s_norm_fn(opt)` to the run method to measure the error in the metric S |

## Running, recording, checkpointing

```python
out_m = jVMC.util.OutputManager()     # in memory; OutputManager("run.h5", group="run_0", append=False) writes HDF5

calls = []
def checkpoint():                           # callbacks take no arguments: close over what you need
    calls.append(1)
    if len(calls) % 25 == 0:
        out_m.write_parameters(len(calls), psi.parameters)

opt = jVMC.optimizer.SR(sampler, psi, diagonal_shift=lambda n: max(1e-2 * 0.95**n, 1e-4),
                            solver=jVMC.solver.Pinv(pinv_cutoff=1e-10), output_manager=out_m)
stepper = jVMC.stepper.Euler(optax.exponential_decay(5e-2, transition_steps=50, decay_rate=0.5))
out, meta = opt.ground_state_search(
    50, jVMC.objective_function.Observable(H), stepper,
    observables={"Z0": op.SigmaZ(0), "XX": (op.SigmaX(0) * op.SigmaX(1), {})},   # operator or (operator, kwargs)
    callback=[checkpoint],
    save_meta_data=True,
)
energies = np.array(out["energy"]["mean"])   # out = output_manager.data["observables"]
```

* `out["times"]` holds the step numbers (or times); `out[name]` holds `"mean"`, `"variance"` and
  `"MC_error"` per step. The loss is always stored under `"energy"`, even for `Infidelity`.
* Observables are measured on the samples of each step, **before** the update. The sampler is
  left holding those samples, so measure the final state with `sampler(H, resample=True)` (MC) or
  `sampler.sample(); sampler(H)` (exact).
* `observables` also accepts callables that return a `SampledObs`.
* Load checkpoints with `jVMC.util.InputManager(path).load_parameters(psi.parameters)`.
* `ground_state_search` can be called repeatedly to continue a run, e.g. in blocks until
  `out["energy"]["variance"]` drops below a target.

## Real-time evolution with TDVP

* The network must be able to represent the dynamics. Holomorphic networks (e.g. `CpxRBM`) use
  S θ̇ = −iF. Real-parameter networks use `make_real=True` (McLachlan: Re S θ̇ = Im F) or
  `False` (Lagrangian).
* Regularize with the solver (`PinvSNR(pinv_tol=..., snr_tol=...)`), not with
  `diagonal_shift`/`diagonal_scale`, which change the tracked trajectory.
* Use many more samples than for ground states, and an adaptive stepper with
  `norm_function=jVMC.util.s_norm_fn(opt)`.
* Diagnostics: `save_meta_data=True` returns `meta["tdvp_error"]` (r²). r²·Var(H) is the squared
  distance between the variational and the exact update, so if it grows, stop trusting the
  results. Energy and variance must stay constant for a time-independent H.
* Time-dependent Hamiltonians: give coefficients as callables `f(t, **kwargs)` (see
  `operators.md`); the current time is passed automatically.

```python
opt_t = jVMC.optimizer.TDVP(sampler, psi, make_real=True, solver=jVMC.solver.PinvSNR(pinv_tol=1e-8))
out_t, meta_t = opt_t.time_evolution(
    0.2, jVMC.objective_function.Observable(H), jVMC.stepper.Heun(1e-2), save_meta_data=True,
)
```

## Infidelity minimization (fitting a target state)

```python
chi = jVMC.vqs.NQS(jVMC.nets.CpxRBM(num_hidden=4), L, batch_size=2**L, seed=11)   # reference |chi>
U, U_dag = jVMC.util.matrix_to_jvmc_operator(np.eye(4)[[0, 2, 1, 3]].astype(complex), (0, 1), return_conjugate=True)
loss_inf = jVMC.objective_function.Infidelity(
    reference_sampler=jVMC.sampler.ExactSampler(chi), operator=U, conjugated_operator=U_dag,
)
opt_inf = jVMC.optimizer.SR(sampler, psi)
out_inf = opt_inf.ground_state_search(20, loss_inf, jVMC.stepper.Euler(1e-1))   # minimizes 1 - |<psi|U|chi>|^2
```

With Monte Carlo samplers for both states, use `control_variate="adaptive"` to reduce the
variance. Use independent sampler keys for ψ and χ.
