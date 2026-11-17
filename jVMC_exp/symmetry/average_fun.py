import jax.numpy as jnp
from typing import Literal, Callable

def avg_coefficients_exp(log_amplitudes, weights):
    """
    Logarithm of the weighted mean of the coefficients, \
    :math:`\\ln\\big(\\frac{1}{n}\\sum_g w_g e^{l_g}\\big)`, computed with a shift by the largest \
    real part. Coefficients with a real part of :math:`-\\infty` are zero; if all of them are \
    zero, or if the terms cancel, the result is :math:`-\\infty`.

    Args:
        * ``log_amplitudes``: Logarithmic coefficients :math:`l_g`, of shape ``(n,)``.
        * ``weights``: Weights :math:`w_g`, of shape ``(n,)``.

    Returns:
        Complex scalar.
    """
    log_amplitudes = jnp.asarray(log_amplitudes, dtype=jnp.complex128)
    weights = jnp.asarray(weights, dtype=jnp.complex128)
    finite = jnp.isfinite(jnp.real(log_amplitudes))
    shift = jnp.max(jnp.where(finite, jnp.real(log_amplitudes), -jnp.inf))
    shift = jnp.where(jnp.any(finite), shift, 0.0)
    safe_log_amplitudes = jnp.where(finite, log_amplitudes, -jnp.inf + 0.0j)
    average = jnp.mean(weights * jnp.exp(safe_log_amplitudes - shift))

    return jnp.where(
        jnp.any(finite),
        shift + jnp.log(average),
        jnp.asarray(-jnp.inf + 0.0j, dtype=jnp.complex128),
    )

def avg_coefficients_log(log_amplitudes, weights):
    """
    Mean of the logarithmic coefficients, :math:`\\frac{1}{n}\\sum_g l_g`. The weights are not used.

    Args:
        * ``log_amplitudes``: Logarithmic coefficients :math:`l_g`, of shape ``(n,)``.
        * ``weights``: Not used.

    Returns:
        Complex scalar.
    """
    del weights
    return jnp.mean(jnp.asarray(log_amplitudes, dtype=jnp.complex128))

def avg_coefficients_sep(log_amplitudes, weights):
    """
    Separate averages of the probabilities and of the phases, \
    :math:`\\frac12\\ln\\big(\\frac{1}{n}\\sum_g e^{2\\,\\mathrm{Re}\\,l_g}\\big)+i\\arg\\sum_g e^{i\\,\\mathrm{Im}\\,l_g}`. \
    The weights are not used.

    Args:
        * ``log_amplitudes``: Logarithmic coefficients :math:`l_g`, of shape ``(n,)``.
        * ``weights``: Not used.

    Returns:
        Complex scalar.
    """
    del weights
    log_amplitudes = jnp.asarray(log_amplitudes, dtype=jnp.complex128)
    re = jnp.real(log_amplitudes)
    im = jnp.imag(log_amplitudes)

    return 0.5 * jnp.log(jnp.mean(jnp.exp(2 * re))) + 1j * jnp.angle(jnp.mean(jnp.exp(1j * im)))

#: Built-in averages, by name.
average_coefficients_table = {
    "exp": avg_coefficients_exp,
    "log": avg_coefficients_log,
    "sep": avg_coefficients_sep
}
#: Names of the built-in averages.
BuiltinSymmetryAverage = Literal["exp", "log", "sep"]
#: A built-in average, or a callable ``f(log_amplitudes, weights)``.
SymmetryAverage = BuiltinSymmetryAverage | Callable

def average_coefficients(log_amplitudes, weights, average: SymmetryAverage):
    """
    Combine the logarithmic coefficients of the transformed configurations into one, as in \
    ``ProjectedOrbitNet``.

    Args:
        * ``log_amplitudes``: Logarithmic coefficients, of shape ``(nsymm,)``.
        * ``weights``: Weights of the projector, of shape ``(nsymm,)``.
        * ``average``: ``"exp"`` (``avg_coefficients_exp()``), ``"log"`` \
          (``avg_coefficients_log()``), ``"sep"`` (``avg_coefficients_sep()``), or a callable \
          ``f(log_amplitudes, weights)``.

    Returns:
        The combined logarithmic coefficient, as a ``complex128`` scalar. Raises ``ValueError`` \
        for an unknown name.
    """
    if callable(average):
        return jnp.asarray(
            average(log_amplitudes, weights),
            dtype=jnp.complex128,
        )

    fn = average_coefficients_table.get(average)
    if fn is None:
        allowed = ", ".join(k for k in average_coefficients_table.keys())
        raise ValueError(
            f"symmetry_average must be one of {allowed}, or a callable."
        )

    return fn(log_amplitudes, weights)