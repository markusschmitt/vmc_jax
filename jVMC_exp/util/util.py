"""
Measurements, the conversion of dense matrices into operators, the norm of the quantum geometric
tensor for the adaptive steppers and the parameter layouts of the optimizers.
"""
from __future__ import annotations
import jax.numpy as jnp
import collections
from typing import Dict, Union, Any, TYPE_CHECKING
import numpy as np
from functools import partial
import jax

from jVMC_exp import global_defs

if TYPE_CHECKING:
    from jVMC_exp.sampler.base import AbstractMCSampler
    from jVMC_exp.optimizer import TDVP

#: An operator, or a callable returning a ``SampledObs``, with the keyword arguments to measure it
#: with, e.g. ``(hamiltonian, {"t": 0.1})``.
OperatorWithKwargs = tuple[Any, dict[str, Any]]
#: A value of the ``observables`` dictionary of ``measure()``: an operator, a callable returning a
#: ``SampledObs``, or an ``OperatorWithKwargs``.
ObservableEntry = Union[Any, OperatorWithKwargs]
    
def has_callable_attr(cls, attr: str):
    """
    Whether ``cls`` has a callable attribute ``attr``, e.g. whether a network has a ``sample`` \
    method.
    """
    return callable(getattr(cls, attr, None))

def remove_double(arr, params_shape):
    """
    Keep the first half of the block of every leaf of a flat array.

    For a holomorphic network, ``NQS.gradients()`` holds for every leaf of the parameter tree a \
    block of ``2 * size`` entries: the derivatives with respect to the real parts of the \
    parameters, followed by those with respect to the imaginary parts, which are ``1j`` times \
    the former. The first half of every block is the complex derivative.

    Args:
        * ``arr``: Flat array with a block of ``2 * size`` entries per leaf.
        * ``params_shape``: ``NQS.paramShapes``, the ``(size, shape)`` of every leaf.

    Returns:
        Flat array with the first ``size`` entries of every block.
    """
    new_arr = []
    start = 0
    for s in params_shape:
        new_arr.append(arr[start:start + s[0]])
        start += 2 * s[0]
    
    return jnp.concatenate(new_arr)

def make_real_array(arr, params_shape):
    """
    Real layout of a flat complex array, the layout of ``NQS.parameters_flat``: for every leaf of \
    the parameter tree, the real parts of its ``size`` entries followed by their imaginary parts.

    Args:
        * ``arr``: Complex flat array with ``size`` entries per leaf.
        * ``params_shape``: ``NQS.paramShapes``, the ``(size, shape)`` of every leaf.

    Returns:
        Real flat array with ``2 * size`` entries per leaf. ``make_cmplx_array()`` is the inverse.
    """
    new_arr = []
    start = 0
    for s in params_shape:
        cmplx_arr = arr[start:start + s[0]]
        new_arr.append(jnp.real(cmplx_arr))
        new_arr.append(jnp.imag(cmplx_arr))
        start += s[0]
    
    return jnp.concatenate(new_arr)

def make_cmplx_array(arr, params_shape):
    """
    Inverse of ``make_real_array()``: the complex flat array whose real and imaginary parts are \
    the two halves of the block of every leaf.

    Args:
        * ``arr``: Real flat array with ``2 * size`` entries per leaf.
        * ``params_shape``: ``NQS.paramShapes``, the ``(size, shape)`` of every leaf.

    Returns:
        Complex flat array with ``size`` entries per leaf.
    """
    new_arr = []
    start = 0
    for s in params_shape:
        new_arr.append((arr[start:start + s[0]] + 1.j * arr[start + s[0]:start + 2 * s[0]]))
        start += 2 * s[0]
    
    return jnp.concatenate(new_arr)

def get_iterable(x):
    """``x`` if it is iterable, otherwise the tuple ``(x,)``."""
    if isinstance(x, collections.abc.Iterable):
        return x
    else:
        return (x,)

def measure(
    observables: Dict[str, ObservableEntry],
    sampler: AbstractMCSampler, 
    num_samples: int = None
):
    """
    Measure observables with a sampler.

    Args:
        * ``observables``: Dictionary from names to operators, to callables without arguments \
          that return a ``jVMC_exp.stats.SampledObs``, or to tuples ``(operator, kwargs)`` or \
          ``(callable, kwargs)``. The keyword arguments are passed to ``sampler(operator, ...)``, \
          e.g. the time ``t`` of a time-dependent operator, or to the callable.
        * ``sampler``: Sampler of the variational state. The operators are measured on its \
          current samples, see ``sampler.__call__()``.
        * ``num_samples`` [optional]: Number of samples, passed to ``sampler(operator, ...)``.

    Returns:
        Dictionary with, for every name, a dictionary with the real parts of the mean \
        (``"mean"``), of the variance (``"variance"``) and of the statistical error of the mean \
        (``"MC_error"``), with the axes of length one removed.

    Example::

        result = measure({"energy": hamiltonian, "X0": (op.SigmaX(0), {"t": 0.1})}, sampler)
        result["energy"]["mean"], result["energy"]["MC_error"]
    """
    # Local import to avoid circular import issues.
    from jVMC_exp.operator.base import AbstractOperator

    result = {}
    for name, op in observables.items():
        kwargs = {}
        if isinstance(op, tuple):
            if len(op) != 2:
                raise ValueError("If an operator takes kwargs it has to be passed as tuple: (operator, dict_of_kwargs)")
            kwargs = op[1]
            op = op[0]

        if isinstance(op, AbstractOperator):
            Oloc = sampler(op, num_samples=num_samples, **kwargs)
        else:
            Oloc = op(**kwargs)
        
        result[name] = {}
        result[name]["mean"] = jnp.real(Oloc.mean).squeeze()
        result[name]["variance"] = jnp.real(Oloc.var).squeeze()
        result[name]["MC_error"] = jnp.real(Oloc.error_of_mean).squeeze()

    return result

def matrix_to_jvmc_operator(
    matrix: np.ndarray,
    sites,
    *,
    return_conjugate: bool = False,
):
    """
    Convert a dense one- or two-site matrix into a discrete operator of ``jVMC_exp``.

    The matrix is expanded in the products of ``IdentityOperator``, ``SigmaX``, ``SigmaY`` and \
    ``SigmaZ`` on ``sites``: 4 terms for one site, 16 for two sites. The expectation values \
    measured with the returned operator, by ``sampler(operator)`` or in ``Infidelity``, are \
    those of ``matrix``. Rows and columns of ``matrix`` are numbered by the local \
    configurations, starting with the local state 0; for two sites the index is \
    ``2 * s[sites[0]] + s[sites[1]]``.

    Args:
        * ``matrix``: Dense matrix, of shape ``(2, 2)`` for one site or ``(4, 4)`` for two sites.
        * ``sites``: Site, or pair of sites.
        * ``return_conjugate``: If ``True``, also return the operator of the conjugate \
          transpose ``matrix.conj().T``.

    Returns:
        The operator, or ``(operator, conjugate_operator)`` with ``return_conjugate=True``.

    Raises ``ValueError`` if ``sites`` has neither one nor two sites, if ``matrix`` is not \
    square, or if its shape does not match the number of sites.
    """
    # Local import to avoid circular import issues.
    import jVMC_exp.operator.discrete.branch_free as op

    local_basis = [
        np.eye(2, dtype=global_defs.DT_OPERATORS_CPX),
        np.array([[0.0, 1.0], [1.0, 0.0]], dtype=global_defs.DT_OPERATORS_CPX),
        np.array([[0.0, 1.0j], [-1.0j, 0.0]], dtype=global_defs.DT_OPERATORS_CPX),
        np.array([[1.0, 0.0], [0.0, -1.0]], dtype=global_defs.DT_OPERATORS_CPX),
    ]
    jvmc_basis = [
        lambda i: op.IdentityOperator(2, i),
        lambda i: op.SigmaX(i),
        lambda i: op.SigmaY(i),
        lambda i: op.SigmaZ(i),
    ]

    if isinstance(sites, (int, np.integer)):
        sites = (int(sites),)
    else:
        sites = tuple(sites)

    if len(sites) not in (1, 2):
        raise ValueError("sites must be an int or a sequence of length 2.")

    matrix = np.asarray(matrix, dtype=global_defs.DT_OPERATORS_CPX)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square.")

    if len(sites) == 1 and matrix.shape != (2, 2):
        raise ValueError("Single-site operators require a 2x2 matrix.")
    if len(sites) == 2 and matrix.shape != (4, 4):
        raise ValueError("Two-site operators require a 4x4 matrix.")

    def build_from(mat):
        if len(sites) == 1:
            jvmc_op = 0
            for idx, basis in enumerate(local_basis):
                coeff = 0.5 * complex(np.trace(basis.conj().T @ mat))
                jvmc_op += coeff * jvmc_basis[idx](sites[0])
            return jvmc_op

        jvmc_op = 0
        for i, left_basis in enumerate(local_basis):
            for j, right_basis in enumerate(local_basis):
                basis = np.kron(left_basis, right_basis)
                coeff = 0.25 * complex(np.trace(basis.conj().T @ mat))
                jvmc_op += coeff * (jvmc_basis[i](sites[0]) * jvmc_basis[j](sites[1]))
        return jvmc_op

    if return_conjugate:
        return build_from(matrix), build_from(matrix.conj().T)
    return build_from(matrix)

def s_norm_fn(opt: TDVP):
    """
    Norm induced by the quantum geometric tensor, :math:`\\|v\\|_S = \\sqrt{|v^\\dagger S v|}`, \
    for the ``normFunction`` of the adaptive steppers.

    ``v`` is in the layout of ``psi.parameters_flat``; for a holomorphic network it is converted \
    to the complex parameters first. :math:`S` is ``opt.qgt``, the quantum geometric tensor of the \
    last evaluation of ``opt``, i.e. of the last substep of the stepper. For a real ``v``, \
    :math:`\\|v\\|_S^2` is the variance of :math:`\\sum_k v_k\\partial_k\\ln\\psi`, with the \
    derivatives with respect to the real parameters.

    Args:
        * ``opt``: Optimizer derived from ``Evolution``, e.g. ``TDVP``, created with \
          ``store_qgt=True``. Raises ``ValueError`` otherwise.

    Returns:
        The norm, a function of ``v``. It raises ``RuntimeError`` if ``opt`` has not been \
        evaluated yet.

    Example::

        opt = jVMC_exp.optimizer.TDVP(sampler, psi, make_real=False)
        opt.time_evolution(
            t_max, jVMC_exp.objective_function.Observable(H),
            jVMC_exp.stepper.RK23(timeStep=1e-3), normFunction=jVMC_exp.util.s_norm_fn(opt)
        )
    """
    if not opt.store_qgt:
        raise ValueError(
            "The S-metric norm was requested, but the optimizer was not constructed "
            "with store_qgt=True. Either construct the optimizer with store_qgt=True, "
            "or use a different norm function."
        )

    pad_size = opt._params_pad_size
    make_cmplx_array_fn = jax.jit(partial(make_cmplx_array, params_shape=opt.psi.paramShapes))

    @jax.jit
    def _mat_vec(v, S):
        Sv = jnp.dot(S, jnp.pad(v, (0, pad_size)))

        return Sv[:-pad_size] if pad_size else Sv

    @jax.jit
    def _norm_from_sv(v, Sv):
        return jnp.sqrt(jnp.abs(jnp.real(jnp.vdot(v, Sv))))

    def _norm_fn(v):
        S = opt.qgt
        if S is None:
            raise RuntimeError(
                "The S-metric norm was called before the right hand side of the TDVP "
                "equation was evaluated, so the quantum geometric tensor is not available "
                "yet. Call the norm only after the first evaluation of f."
            )

        if opt.psi.holomorphic:
            v = make_cmplx_array_fn(v)
        Sv = S(v) if callable(S) else _mat_vec(v, S)

        return _norm_from_sv(v, Sv)

    return _norm_fn