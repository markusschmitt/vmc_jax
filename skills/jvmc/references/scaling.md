# Memory, speed and multiple devices

## Memory: the knobs, in order

1. **`NQS(..., batch_size=...)`** sets how many configurations are evaluated at once. Start with
   `batch_size = num_samples`, and halve it on out-of-memory errors. Speed plateaus well before
   `batch_size = num_samples`, so a smaller value often costs little.
2. **`batched_jacobian=True`** on the objective function (`Observable(H, batched_jacobian=True)`).
   The N_samples × N_params Jacobian is then streamed in batches and never stored whole. It is
   essential for many samples. It is incompatible with `PinvSNR(snr_tol>0)`.
3. **The linear system.** S is N_params², and the MinSR kernel is N_samples². Choose the
   optimizer whose matrix is smaller. If it still does not fit, use
   `solver=Pinv(diagonalization_mode="host")` (CPU) or `"distributed"` (several GPUs, below).
4. **Precision.** Set `NQS(..., sampler_dtype=..., eval_dtype=..., grad_dtype=...)` (see
   `networks.md`), or the environment variables `JVMC_DT_PARAMS_REAL=f32` and
   `JVMC_DT_OUT_REAL=f32` before the import.
5. **`JVMC_BATCH_LOOP`** (environment variable before the import, or
   `jVMC_exp.global_defs.update(batch_loop=...)`). `"python"` (the default) compiles once per
   batch size, whatever the number of samples. `"compiled"` fuses the loop and is faster for cheap
   batches, but it recompiles, and caches another executable, for every new number of samples. Use
   it only when the number of samples is fixed.

## Speed

* **Markov chains.** More parallel chains speed up sampling until the device saturates, after
  which it slows down again, since each chain pays its own thermalization. Hundreds to a few
  thousand chains per GPU is typical; measure.
* `sweep_steps` defaults to the number of sites. `thermalization_sweeps` is paid at every
  `sample()` call, because chains continue from their last state.
* Timings of sampling, objective and solve are printed after every run (`opt.output_manager`).

## Multiple GPUs on one node

Nothing to do: all visible devices form a one-dimensional mesh at import, and samples, network
evaluations, gradients and the S matrix are sharded across it. Requirements: `batch_size` must
be divisible by the number of devices. Sample and chain counts are rounded up to multiples of
it. Limit the devices with `CUDA_VISIBLE_DEVICES`.

## Several processes / nodes

Run one process per GPU (e.g. `srun` with one task per GPU under SLURM) and set
`JVMC_USE_DISTRIBUTED=true` in the environment. The library then calls
`jax.distributed.initialize()` at import, and JAX detects the SLURM layout automatically. Where
it cannot, call `jax.distributed.initialize(coordinator_address=..., num_processes=...,
process_id=...)` yourself **before** importing the library. Only process 0 prints progress and
writes HDF5 files.

```bash
export JVMC_USE_DISTRIBUTED=true
srun --ntasks-per-node=4 --gpus-per-task=1 python run.py
```

Distributed diagonalization (`Pinv(diagonalization_mode="distributed")` or `PinvSNR(...)`) needs
`jaxmg`, which `pip install "jVMC[cuda12]"` or `"jVMC[cuda13]"` installs. It requires one
process per GPU even on a single node. It pays off for large matrices (≳ 10⁴ parameters for SR);
for small ones it is slower than `"device"`.

## Installation reminders

* CPU: `pip install jVMC`. GPU: `pip install "jVMC[cuda12]"` or `"jVMC[cuda13]"`. On clusters
  with CUDA modules: `"jVMC[cuda12-local]"`.
* Python ≥ 3.11. Importing the library enables float64 in JAX for the whole process.
