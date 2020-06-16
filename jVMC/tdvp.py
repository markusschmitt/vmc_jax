import jax
import jax.numpy as jnp
import numpy as np

def get_tdvp_equation(Eloc, gradients):
    energyMean = jnp.mean(Eloc)
    gradientMean = jnp.mean(gradients, axis=0)

    F = Eloc.dot(gradients.conj()) / Eloc.shape[0] - energyMean * jnp.conj(gradientMean)

    S = jnp.matmul(jnp.conj(jnp.transpose(gradients)), gradients)

    return S, F
