import flax.linen as nn
import jax.numpy as jnp


def square(x):
    return x**2


def poly6(x):
    x = x**2
    return ((0.022222222 * x - 0.083333333) * x + 0.5) * x


def poly5(x):
    xsq = x**2
    return ((0.133333333 * xsq - 0.333333333) * xsq + 1.0) * x


def log_cosh(x):
    two = jnp.asarray(2, dtype=x.real.dtype)
    half_log_two = jnp.log(two)
    sgn_x = -two * jnp.signbit(x.real) + 1
    sgn_x = sgn_x.astype(x.dtype)
    x = x * sgn_x

    return x + jnp.log1p(jnp.exp(-two * x)) - half_log_two


activation_functions = {
    "square": square,
    "poly5": poly5,
    "poly6": poly6,
    "elu": nn.elu,
    "relu": nn.relu,
    "tanh": jnp.tanh,
}
