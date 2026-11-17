"""
Global configuration of the library: data types, parallelism and batching.

Importing the library enables the 64-bit mode of JAX (``jax_enable_x64``) for the whole Python
process.

The settings are read from these environment variables when the library is imported. Their values
are not case sensitive:

* ``JVMC_DT_PARAMS_REAL``: ``DT_PARAMS_REAL``, one of ``f64`` (default), ``f32``, ``f16``, ``bf16``.
* ``JVMC_DT_PARAMS_CPX``: ``DT_PARAMS_CPX``, one of ``c128`` (default), ``c64``.
* ``JVMC_DT_OUT_REAL``: ``DT_OUT_REAL``, one of ``f64`` (default), ``f32``, ``f16``, ``bf16``.
* ``JVMC_DT_OUT_CPX``: ``DT_OUT_CPX``, one of ``c128`` (default), ``c64``.
* ``JVMC_USE_DISTRIBUTED``: ``USE_DISTRIBUTED``, ``true`` or ``false`` (default). If ``true``,
  ``jax.distributed.initialize()`` is called when the library is imported.
* ``JVMC_BATCH_LOOP``: ``BATCH_LOOP``, ``python`` (default) or ``compiled``.
* ``JVMC_MATMUL_CHUNK_BYTES``: ``MATMUL_CHUNK_BYTES``, an integer. The default is ``2**30``.

The function ``update()`` changes these settings afterwards, except for ``USE_DISTRIBUTED``, which
can only be set through its environment variable. It takes data types as objects, e.g.,
``jnp.float32``, while the environment variables take the names above, whose number is the size in
bits. Read the settings through the module, e.g., as ``global_defs.DT_OUT_REAL``, to see the current
values. A variational state keeps the output data type it was created with, so the data types should
be set before creating it.
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import os

#: Allowed real data types, by the names used in the environment variables.
allowed_dtypes_real = {
    "f64": jnp.float64,
    "f32": jnp.float32,
    "f16": jnp.float16,
    "bf16": jnp.bfloat16
}
#: Allowed complex data types, by the names used in the environment variables.
allowed_dtypes_cpx = {
    "c128": jnp.complex128,
    "c64": jnp.complex64
}

def _read_dtype(dtype_str: str, real: bool):
    """Return the data type called ``dtype_str``, which has to be real if ``real``, and complex otherwise."""
    dtype_str = dtype_str.lower()
    dtype_real = allowed_dtypes_real.get(dtype_str)
    dtype_cpx = allowed_dtypes_cpx.get(dtype_str)

    if real:
        if dtype_real is None:
            if dtype_cpx is not None:
                raise ValueError("Trying to set a real variable with a complex dtype.")

            allowed = ", ".join(k for k in allowed_dtypes_real.keys())
            raise ValueError(
                f"Allowed real dtypes are {allowed}. "
                f"Got {dtype_str}."
            )
        return jnp.dtype(dtype_real)

    if dtype_cpx is None:
        if dtype_real is not None:
            raise ValueError("Trying to set a complex variable with a real dtype.")

        allowed = ", ".join(k for k in allowed_dtypes_cpx.keys())
        raise ValueError(
            f"Allowed complex dtypes are {allowed}. "
            f"Got {dtype_str}."
        )
    return jnp.dtype(dtype_cpx)

def _check_dtype(dtype, real: bool, label: str):
    """Return ``dtype`` as a ``jnp.dtype``, after checking that it is an allowed real or complex data type."""
    if isinstance(dtype, str):
        raise TypeError(f"{label} has to be a data type, e.g., jnp.float32. Got the string '{dtype}'.")

    dtype = jnp.dtype(dtype)
    allowed = [jnp.dtype(d) for d in (allowed_dtypes_real if real else allowed_dtypes_cpx).values()]
    if dtype not in allowed:
        raise TypeError(
            f"Allowed dtypes for {label} are {', '.join(str(d) for d in allowed)}. "
            f"Got {dtype}."
        )

    return dtype

#: Data type of continuous particle configurations.
DT_SAMPLES_CONT = jnp.float64
#: Data type of discrete computational basis configurations.
DT_SAMPLES = jnp.int32
#: Default data type of real network parameters.
DT_PARAMS_REAL = _read_dtype(os.environ.get("JVMC_DT_PARAMS_REAL", "f64"), True)
#: Default data type of complex network parameters.
DT_PARAMS_CPX = _read_dtype(os.environ.get("JVMC_DT_PARAMS_CPX", "c128"), False)
#: Data type of real coefficients and gradients returned by a variational state, and of its flat parameters.
DT_OUT_REAL = _read_dtype(os.environ.get("JVMC_DT_OUT_REAL", "f64"), True)
#: Data type of complex coefficients and gradients returned by a variational state.
DT_OUT_CPX = _read_dtype(os.environ.get("JVMC_DT_OUT_CPX", "c128"), False)
#: Data type of real operator matrix elements. It follows ``DT_OUT_REAL``.
DT_OPERATORS_REAL = DT_OUT_REAL
#: Data type of complex operator matrix elements. It follows ``DT_OUT_CPX``.
DT_OPERATORS_CPX = DT_OUT_CPX

#: Whether the library runs on several JAX processes. It can only be set through ``JVMC_USE_DISTRIBUTED``.
USE_DISTRIBUTED = os.environ.get('JVMC_USE_DISTRIBUTED', 'false').lower() == 'true'

#: Allowed values of ``BATCH_LOOP``.
BATCH_LOOPS = ("python", "compiled")

def _read_batch_loop(batch_loop: str):
    """Return ``batch_loop`` in lower case, after checking that it is one of ``BATCH_LOOPS``."""
    batch_loop = batch_loop.lower()
    if batch_loop not in BATCH_LOOPS:
        raise ValueError(
            f"Allowed batch loops are {', '.join(BATCH_LOOPS)}. "
            f"Got {batch_loop}."
        )

    return batch_loop

#: Where the loop over the batches of a sharded call runs (see ``jVMC_exp.sharding_config.sharded``):
#: ``"python"`` makes one dispatch per batch, and the network is compiled once per batch size,
#: whatever the number of samples. ``"compiled"`` makes one dispatch per call, which is faster when
#: batches are cheap, and everything, the network included, is compiled again for every new number
#: of samples. It is read every time a sharded method is called, so it can be changed at any time
#: with ``update()``.
BATCH_LOOP = _read_batch_loop(os.environ.get("JVMC_BATCH_LOOP", "python"))

def _read_matmul_chunk_bytes(chunk_bytes):
    """Return ``chunk_bytes`` as an ``int``, after checking that it is not negative."""
    chunk_bytes = int(chunk_bytes)
    if chunk_bytes < 0:
        raise ValueError(f"The matmul chunk size has to be non-negative, got {chunk_bytes}.")

    return chunk_bytes

#: Largest chunk, in bytes, of the Jacobian that the products building the S matrix of SR/TDVP and
#: the tangent kernel of MinSR handle at once (0: the whole Jacobian at once). When it compiles a
#: matrix product, XLA autotunes it on its own copies of the operands, as large as the operands.
#: It is read when a product is compiled, and it affects the memory use, not the result.
MATMUL_CHUNK_BYTES = _read_matmul_chunk_bytes(os.environ.get("JVMC_MATMUL_CHUNK_BYTES", 2**30))

def update(
    *,
    dt_params_real: None | jnp.dtype = None,
    dt_params_cpx: None | jnp.dtype = None,
    dt_out_real: None | jnp.dtype = None,
    dt_out_cpx: None | jnp.dtype = None,
    batch_loop: None | str = None,
    matmul_chunk_bytes: None | int = None,
):
    """
    Update the global configuration. Arguments that are ``None`` leave their setting unchanged.

    Args:
        * ``dt_params_real``: New value of ``DT_PARAMS_REAL``, one of the data types in \
          ``allowed_dtypes_real``, e.g., ``jnp.float32``.
        * ``dt_params_cpx``: New value of ``DT_PARAMS_CPX``, one of the data types in \
          ``allowed_dtypes_cpx``, e.g., ``jnp.complex64``.
        * ``dt_out_real``: New value of ``DT_OUT_REAL`` and ``DT_OPERATORS_REAL``, one of the data \
          types in ``allowed_dtypes_real``.
        * ``dt_out_cpx``: New value of ``DT_OUT_CPX`` and ``DT_OPERATORS_CPX``, one of the data \
          types in ``allowed_dtypes_cpx``.
        * ``batch_loop``: New value of ``BATCH_LOOP``, one of ``BATCH_LOOPS``.
        * ``matmul_chunk_bytes``: New value of ``MATMUL_CHUNK_BYTES``, a non-negative integer.

    Example::

        jVMC_exp.global_defs.update(
            dt_out_real=jnp.float32,
            dt_out_cpx=jnp.complex64,
            batch_loop="compiled",
        )
    """
    global DT_PARAMS_REAL
    global DT_PARAMS_CPX
    global DT_OUT_REAL
    global DT_OUT_CPX
    global BATCH_LOOP
    global MATMUL_CHUNK_BYTES
    global DT_OPERATORS_REAL
    global DT_OPERATORS_CPX

    if dt_params_real is not None:
        DT_PARAMS_REAL = _check_dtype(dt_params_real, True, "dt_params_real")

    if dt_params_cpx is not None:
        DT_PARAMS_CPX = _check_dtype(dt_params_cpx, False, "dt_params_cpx")

    if dt_out_real is not None:
        DT_OUT_REAL = _check_dtype(dt_out_real, True, "dt_out_real")
        DT_OPERATORS_REAL = DT_OUT_REAL

    if dt_out_cpx is not None:
        DT_OUT_CPX = _check_dtype(dt_out_cpx, False, "dt_out_cpx")
        DT_OPERATORS_CPX = DT_OUT_CPX

    if batch_loop is not None:
        BATCH_LOOP = _read_batch_loop(batch_loop)

    if matmul_chunk_bytes is not None:
        MATMUL_CHUNK_BYTES = _read_matmul_chunk_bytes(matmul_chunk_bytes)