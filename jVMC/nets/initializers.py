from functools import partial

import jax
import jax.numpy as jnp

from jVMC import global_defs


# TODO: this should be removed as 99% of the functions in this file
def init_fn_args(**kwargs):
    """Return Flax module kwargs with matching compute and parameter dtype."""
    if "dtype" in kwargs and "param_dtype" not in kwargs:
        kwargs["param_dtype"] = kwargs["dtype"]
    for name, value in tuple(kwargs.items()):
        if (
            name.endswith("init")
            and isinstance(value, partial)
            and "dtype" in (value.keywords or {})
        ):
            kwargs[name] = _drop_flax_initializer_dtype(value)
    return kwargs


def _drop_flax_initializer_dtype(init):
    """Adapt dtype-bound JAX initializers to Flax' ``(key, shape, dtype)`` call."""

    def wrapped(rng, shape, dtype=None):
        return init(rng, shape)

    return wrapped


def _canonical_real_dtype(dtype=None):
    dtype = jnp.dtype(global_defs.DT_PARAMS_REAL if dtype is None else dtype)
    if dtype == jnp.dtype(jnp.float16):
        return jnp.float16
    if dtype == jnp.dtype(jnp.bfloat16):
        return jnp.bfloat16
    if dtype == jnp.dtype(jnp.float32):
        return jnp.float32
    if dtype == jnp.dtype(jnp.float64):
        return jnp.float64
    raise TypeError(
        f"Expected real parameter dtype float16, bfloat16, float32, or float64, got {dtype}."
    )


def _canonical_complex_dtype(dtype=None):
    dtype = jnp.dtype(global_defs.DT_PARAMS_CPX if dtype is None else dtype)
    if dtype == jnp.dtype(jnp.complex64):
        return jnp.complex64
    if dtype == jnp.dtype(jnp.complex128):
        return jnp.complex128
    if dtype == jnp.dtype(jnp.float32):
        return jnp.complex64
    if dtype == jnp.dtype(jnp.float64):
        return jnp.complex128
    raise TypeError(
        f"Expected complex parameter dtype complex64 or complex128, got {dtype}."
    )


def _real_dtype_for_complex(dtype):
    dtype = _canonical_complex_dtype(dtype)
    return jnp.float32 if dtype == jnp.complex64 else jnp.float64


def real_uniform(scale=1e-2, dtype=None):
    dtype = _canonical_real_dtype(dtype)
    return jax.nn.initializers.uniform(scale=scale, dtype=dtype)


def real_variance_scaling(
    scale=1.0, mode="fan_avg", distribution="uniform", dtype=None
):
    dtype = _canonical_real_dtype(dtype)
    return jax.nn.initializers.variance_scaling(
        scale=scale, mode=mode, distribution=distribution, dtype=dtype
    )


def complex_uniform(scale=1e-2, dtype=None):
    param_dtype = _canonical_complex_dtype(dtype)
    unif = jax.nn.initializers.uniform(scale=scale)

    def init(rng, shape, dtype=None):
        out_dtype = _canonical_complex_dtype(dtype or param_dtype)
        component_dtype = _real_dtype_for_complex(out_dtype)
        rng1, rng2 = jax.random.split(rng)
        return (
            unif(rng1, shape, dtype=component_dtype)
            + 1j * unif(rng2, shape, dtype=component_dtype)
        ).astype(out_dtype)

    return init


def complex_variance_scaling(
    scale=1.0, mode="fan_avg", distribution="uniform", dtype=None
):
    param_dtype = _canonical_complex_dtype(dtype)
    real_init = jax.nn.initializers.variance_scaling(
        scale=scale / 2.0,
        mode=mode,
        distribution=distribution,
    )

    def init(rng, shape, dtype=None):
        out_dtype = _canonical_complex_dtype(dtype or param_dtype)
        component_dtype = _real_dtype_for_complex(out_dtype)
        rng1, rng2 = jax.random.split(rng)
        return (
            real_init(rng1, shape, dtype=component_dtype)
            + 1j * real_init(rng2, shape, dtype=component_dtype)
        ).astype(out_dtype)

    return init


def cplx_init(rng, shape, dtype=None):
    """Backward-compatible complex uniform initializer honoring ``dtype``."""
    dtype = _canonical_complex_dtype(dtype)
    rng1, rng2 = jax.random.split(rng)
    unif = jax.nn.initializers.uniform()
    real_dtype = _real_dtype_for_complex(dtype)
    return (
        unif(rng1, shape, dtype=real_dtype) + 1j * unif(rng2, shape, dtype=real_dtype)
    ).astype(dtype)


def cplx_variance_scaling(rng, shape, dtype=None):
    """Backward-compatible complex variance-scaling initializer."""
    return complex_variance_scaling(dtype=dtype)(rng, shape, dtype)
