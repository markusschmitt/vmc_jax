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


def logCosh(x):
    return jnp.log(jnp.cosh(x))


activationFunctions = {
    "square": square,
    "poly5": poly5,
    "poly6": poly6,
    "elu": nn.elu,
    "relu": nn.relu,
    "tanh": jnp.tanh
}
