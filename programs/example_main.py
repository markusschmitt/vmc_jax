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
import jVMC.solver as solver

L=4
J=-1.0
hx=-0.3

numSamples=30

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

# Set up sampler
mcSampler = sampler.Sampler(random.PRNGKey(123), sampler.propose_spin_flip, [L], numChains=5)

# Set up solver
eigenSolver = solver.EigenSolver()

tdvpEquation = jVMC.tdvp.TDVP(mcSampler, snrTol=1)

stepper = jVMCstepper.Euler()

stepperArgs = {'hamiltonian': hamiltonian, 'psi': psi, 'numSamples': numSamples}
dp, dt = stepper.step(0, tdvpEquation, psi.get_parameters(), stepperArgs)
psi.update_parameters(dp)
