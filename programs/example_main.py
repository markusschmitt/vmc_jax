import sys
# Find jVMC package
sys.path.append(sys.path[0]+"/..")

import jax
from jax.config import config
config.update("jax_enable_x64", True)

import jax.random as random
import flax.nn as nn
import jax.numpy as jnp

import jVMC
import jVMC.stepper as jVMCstepper
import jVMC.nets as nets
from jVMC.vqs import NQS
import jVMC.operator as op
import jVMC.sampler as sampler
import jVMC.tdvp as tdvp

def measure(ops, psi, sampler, numSamples):
    
    # Get sample
    sampleConfigs, sampleLogPsi =  sampler.sample( psi, numSamples )
    print(sampleConfigs)

    result = []
    for op in ops:
        sampleOffdConfigs, matEls = op.get_s_primes(sampleConfigs)
        sampleLogPsiOffd = psi(sampleOffdConfigs)
        Oloc = op.get_O_loc(sampleLogPsi,sampleLogPsiOffd)

        result.append( [jnp.mean(Oloc), jnp.var(Oloc) / jnp.sqrt(numSamples)] )

    return jnp.real(jnp.array(result))


L=4
J=-1.0
hx=-3

numSamples=500

# Set up variational wave function
rbm = nets.CpxRBM.partial(L=L,numHidden=2,bias=False)
_, params = rbm.init_by_shape(random.PRNGKey(0),[(1,L)])
rbmModel = nn.Model(rbm,params)
psi = NQS(rbmModel)

# Set up hamiltonian
hamiltonian = op.Operator()
for l in range(L):
    hamiltonian.add( op.scal_opstr( J, ( op.Sz(l), op.Sz((l+1)%L) ) ) )
    hamiltonian.add( op.scal_opstr( hx, ( op.Sx(l), ) ) )

# Set up observables
observables = [hamiltonian, op.Operator(), op.Operator()]
for l in range(L):
    observables[1].add( ( op.Sx(l), ) )
    observables[2].add( ( op.Sz(l), op.Sz((l+1)%L) ) )

# Set up sampler
mcSampler = sampler.Sampler(random.PRNGKey(123), sampler.propose_spin_flip, [L], numChains=10)

tdvpEquation = jVMC.tdvp.TDVP(mcSampler, snrTol=1)

stepper = jVMCstepper.Euler(timeStep=1e-4)

t=0
obs = measure(observables, psi, mcSampler, numSamples)
print(t, obs[0], obs[1], obs[2])
while t<1:
    stepperArgs = {'hamiltonian': hamiltonian, 'psi': psi, 'numSamples': numSamples}
    dp, dt = stepper.step(0, tdvpEquation, psi.get_parameters(), stepperArgs)
    psi.update_parameters(dp)
    t+=dt
    obs = measure(observables, psi, mcSampler, numSamples)

    print(t, obs[0], obs[1], obs[2])
