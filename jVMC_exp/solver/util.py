import jax
import jax.numpy as jnp
import warnings
import numpy as np
from typing import Literal
import math

try:
    import jaxmg
except ImportError:
    jaxmg = None

from jVMC_exp.global_defs import USE_DISTRIBUTED
from jVMC_exp.sharding_config import (
    MESH, MESH_2D,
    DEVICE_SHARDING, DEVICE_SHARDING_2D
)

def _distributed_warning(A, mode):
    if mode.lower() == "distributed":
        if jaxmg is None:
            if jax.process_index() == 0:
                warnings.warn(
                    "mode='distributed' requires the 'jaxmg' package, which is not installed "
                    "(install it via a CUDA extra, e.g. `pip install jVMC_exp[cuda13]`). "
                    "Falling back to mode='device'."
                )
            mode = "device"

        if jax.default_backend() != "gpu":
            if jax.process_index() == 0:
                warnings.warn(
                    "mode='distributed' requires a real GPU backend, "
                    f"got jax.default_backend()={jax.default_backend()}. "
                    "Falling back to mode='device'."
                )
            mode = "device"

        if A.shape[0] % MESH.shape["devices"] != 0:
            if jax.process_index() == 0:
                warnings.warn(
                    "mode=distributed requires the given matrix to "
                    "have shape divisible by the number of devices."
                    f"Got shape={A.shape}, n_devices={MESH.shape['devices']}."
                    "Falling back to mode='device'."
                )
            mode = "device"

        if MESH.shape["devices"] == 1:
            if jax.process_index() == 0:
                warnings.warn(
                    "Got mode=distributed but there is only one device. "
                    "Falling back to mode='device'."
                )
            mode = "device"

        if not USE_DISTRIBUTED:
            if jax.process_index() == 0:
                warnings.warn(
                    "mode='distributed' requires JVMC_USE_DISTRIBUTED=true and "
                    "one process per GPU, since jaxmg's cuSOLVERMp backend "
                    "needs rank-per-GPU execution "
                    "(e.g. launch with `srun --ntasks=<n_gpus> --gpus-per-task=1`). "
                    "Falling back to mode='device'."
                )
            mode = "device"

    return mode

def _eigh_numpy(S):
    e, V = np.linalg.eigh(np.array(S))

    return jnp.array(e), jnp.array(V)

def smooth_cutoff_fn(x, c, exp=6):
    """
    Smooth step :math:`r(x)=1/(1+(c/x)^\\mathrm{exp})` from 0 to 1, with :math:`r(c)=1/2`.

    It is close to 0 for :math:`x\\ll c`, and to 1 for :math:`x\\gg c`.

    Args:
        * ``x``: Non-negative array.
        * ``c``: Position of the step.
        * ``exp``: Steepness of the step.

    Returns:
        :math:`r(x)`, with the shape of ``x``.
    """
    return 1 / (1 + (c / x)**exp)

def diagonalize(
        A,
        pad_size: int = 0,
        mode: Literal["device", "distributed", "host"] = "device",
        T_A: int | None = None
    ):
    """
    Eigenvalues and eigenvectors of a Hermitian matrix.

    If ``pad_size`` is nonzero, the last ``pad_size`` rows and columns of :math:`A` have to be \
    zero padding. The constant :math:`c=2\\|A\\|_F+1`, larger than every eigenvalue of \
    :math:`A`, is added to their diagonal elements before the diagonalization, so that the \
    ``pad_size`` eigenvalues of the padding are the last ones, with eigenvectors that vanish \
    outside of the padding. They are not removed from the result: when :math:`V` is sharded \
    across the devices, slicing it along the sharded axis would gather it on every device.

    The backends are

    * ``"device"``: ``jax.numpy.linalg.eigh`` on the devices of :math:`A`. If it raises \
      ``ValueError``, ``numpy.linalg.eigh`` is used instead, with a ``RuntimeWarning``.
    * ``"host"``: ``numpy.linalg.eigh`` on the host, which gathers :math:`A`.
    * ``"distributed"``: ``jaxmg.syevd`` on :math:`A` sharded in blocks over ``MESH_2D`` (see \
      ``jVMC_exp.sharding_config``), with :math:`V` sharded along its rows afterwards. It needs \
      the ``jaxmg`` package, a GPU backend, more than one device, a matrix size divisible by the \
      number of devices, and ``JVMC_USE_DISTRIBUTED=true`` with one process per GPU. If one of \
      these is missing, it warns and falls back to ``"device"``.

    Args:
        * ``A``: Hermitian matrix.
        * ``pad_size``: Number of rows and columns of padding at the end of :math:`A`.
        * ``mode``: Backend, ``"device"``, ``"host"`` or ``"distributed"``, not case sensitive.
        * ``T_A`` [optional]: Tile size of ``jaxmg.syevd``, for ``mode="distributed"``. The \
          default is the greatest common divisor of the size of :math:`A` divided by the number \
          of rows and by the number of columns of ``MESH_2D``.

    Returns:
        ``ev, V``: The eigenvalues in ascending order, the ``pad_size`` eigenvalues of the \
        padding included, and the matrix :math:`V` whose columns are the eigenvectors.

    Raises ``ValueError`` for an unknown ``mode``.
    """
    mode = _distributed_warning(A, mode)

    if pad_size !=0:
        # Frobenius norm >= spectral radius for Hermitian A
        # Used here to recognize the fake eigenvalues given by padding
        c = 2 * jnp.linalg.norm(A) + 1
        N = A.shape[0] - pad_size
        pad_idx = jnp.arange(N, A.shape[0])
        A = A.at[pad_idx, pad_idx].add(c)

    if mode.lower() == "device":
        try:
            ev, V = jnp.linalg.eigh(A)
        except ValueError:
            if jax.process_index() == 0:
                warnings.warn(
                    "jax.numpy.linalg.eigh raised an exception. Falling back to " 
                    "numpy.linalg.eigh for diagonalization.", RuntimeWarning
                )
            ev, V = _eigh_numpy(A)

    elif mode.lower() == "host":
        ev, V = _eigh_numpy(A)

    elif mode.lower() == "distributed":
        A = jax.device_put(A, DEVICE_SHARDING_2D, donate=True)

        if T_A is None:
            p_rows, p_cols = MESH_2D.shape["row"], MESH_2D.shape["col"]
            T_A = math.gcd((A.shape[0]) // p_rows, (A.shape[0]) // p_cols)

        ev, V = jaxmg.syevd(A, T_A, MESH_2D)

        V = jax.device_put(V, DEVICE_SHARDING)

    else:
        raise ValueError(
            f"The available modes are 'device', 'host' or 'distributed' got {mode}."
        )

    return ev, V