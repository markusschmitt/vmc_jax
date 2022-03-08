import jax
import jax.numpy as jnp
import flax.linen as nn


def square(x):
    return x**2


def poly6(x):
    x = x**2
    return ((0.022222222 * x - 0.083333333) * x + 0.5) * x


def poly5(x):
    xsq = x**2
    return ((0.133333333 * xsq - 0.333333333) * xsq + 1.) * x

def log_cosh(x):
    sgn_x = -2 * jnp.signbit(x.real) + 1
    x = x * sgn_x
    return x + jnp.log1p(jnp.exp(-2.0 * x)) - jnp.log(2.0)


activationFunctions = {
    "square": square,
    "poly5": poly5,
    "poly6": poly6,
    "elu": nn.elu,
    "relu": nn.relu,
    "tanh": jnp.tanh
}
