import jax
import jax.numpy as jnp
import numpy as np

def get_F(Eloc, gradients):
    energyMean = jnp.mean(Eloc)
    gradientMean = jnp.mean(gradients, axis=0)
    
    return Eloc.dot(gradients.conj()) / Eloc.shape[0] - energyMean * jnp.conj(gradientMean)


def get_tdvp_equation(Eloc, gradients, rhsPrefactor=1.j):
    energyMean = jnp.mean(Eloc)
    gradientMean = jnp.mean(gradients, axis=0)

    F = Eloc.dot(gradients.conj()) / Eloc.shape[0] - energyMean * jnp.conj(gradientMean)

    S = jnp.matmul(jnp.conj(jnp.transpose(gradients)), gradients) \
        - jnp.outer(jnp.conj(gradientMean), gradientMean)

    return S, -rhsPrefactor * F


def get_sr_equation(Eloc, gradients):

    return get_tdvp_equation(Eloc, gradients, rhsPrefactor=1.)


def solve(Eloc, gradients, solver, makeReal='imag'):

    S, F = get_tdvp_equation(Eloc, gradients)

    if makeReal == 'imag':
        S = 0.5 * (S - jnp.conj(S))
        F = 0.5 * (F - jnp.conj(F))
    else:
        S = jnp.real(S)
        F = jnp.real(F)

    return jnp.real( solver(S,F) )
