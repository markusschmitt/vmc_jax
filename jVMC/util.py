import sys
# Find jVMC package
sys.path.append(sys.path[0]+"/..")

import jax
from jax.config import config
config.update("jax_enable_x64", True)

import jax.numpy as jnp

import jVMC.stepper as jVMCstepper

def measure(ops, psi, sampler, numSamples=0):
    
    # Get sample
    sampleConfigs, sampleLogPsi, p =  sampler.sample( psi, numSamples )

    result = []
    
    for op in ops:

        sampleOffdConfigs, matEls = op.get_s_primes(sampleConfigs)
        sampleLogPsiOffd = psi(sampleOffdConfigs)
        Oloc = op.get_O_loc(sampleLogPsi,sampleLogPsiOffd)

        if p is not None:
            result.append( jnp.dot(p, Oloc) )
        else:
            result.append( [jnp.mean(Oloc), jnp.var(Oloc) / jnp.sqrt(numSamples)] )

    return jnp.real(jnp.array(result))


def ground_state_search(psi, ham, tdvpEquation, sampler, numSteps=200, stepSize=1e-2, observables=None):

    delta = tdvpEquation.diagonalShift

    stepper = jVMCstepper.Euler(timeStep=stepSize)

    n=0
    if observables is not None:
        obs = measure(observables, psi, sampler)
        print("{0:d} {1:.6f} {2:.6f} {3:.6f}".format(n, obs[0], obs[1], obs[2]))
    while n<numSteps:
        stepperArgs = {'hamiltonian': ham, 'psi': psi, 'numSamples': None}
        dp, _ = stepper.step(0, tdvpEquation, psi.get_parameters(), stepperArgs)
        psi.set_parameters(dp)
        n += 1

        if observables is not None:
            obs = measure(observables, psi, sampler)
            print("{0:d} {1:.6f} {2:.6f} {3:.6f}".format(n, obs[0], obs[1], obs[2]))

        delta=0.95*delta
        tdvpEquation.set_diagonal_shift(delta)


