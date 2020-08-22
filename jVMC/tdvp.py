import jax
import jax.numpy as jnp
import numpy as np

import jVMC.mpi_wrapper as mpi

def realFun(x):
    return jnp.real(x)

def imagFun(x):
    return 0.5 * ( x - jnp.conj(x) )


class TDVP:

    def __init__(self, sampler, snrTol=2, svdTol=1e-14, makeReal='imag', rhsPrefactor=1.j, diagonalShift=0.):

        self.sampler = sampler
        self.snrTol = snrTol
        self.svdTol = svdTol
        self.diagonalShift = diagonalShift
        self.rhsPrefactor = rhsPrefactor

        self.makeReal = realFun
        if makeReal == 'imag':
            self.makeReal = imagFun


    def set_diagonal_shift(self, delta):
        self.diagonalShift = delta

    def get_tdvp_equation(self, Eloc, gradients, p=None):
        
        if p is None:
            # Need MPI
            Eloc -= jnp.mean(Eloc)
            gradients -= jnp.mean(gradients, axis=0)
            
            EOdata = -self.rhsPrefactor * jnp.multiply(Eloc[:,None], jnp.conj(gradients))
            EOdata = self.makeReal( EOdata )

            # Need MPI
            F = jnp.mean(EOdata, axis=0)
            S = self.makeReal( jnp.matmul(jnp.conj(jnp.transpose(gradients)), gradients) ) / Eloc.shape[0]

        else:
            # Need MPI
            Eloc -= mpi.global_sum( jnp.array([jnp.dot(p, Eloc)]) )
            gradients -= mpi.global_sum( jnp.expand_dims(jnp.dot(p,gradients), axis=0) )

            EOdata = -self.rhsPrefactor * jnp.multiply((p*Eloc)[:,None], jnp.conj(gradients))
            EOdata = self.makeReal( EOdata )
            
            # Need MPI
            #F = jnp.sum(EOdata, axis=0)
            F = mpi.global_sum(EOdata)
            #S = self.makeReal( jnp.matmul(jnp.conj(jnp.transpose(gradients)), jnp.matmul(jnp.diag(p), gradients) ) )
            S = self.makeReal(
                    mpi.global_sum( 
                        jnp.expand_dims(
                            jnp.matmul(jnp.conj(jnp.transpose(gradients)), jnp.matmul(jnp.diag(p), gradients) ),
                            axis=0
                        )
                    )
                )

            #work with complex matrix
            #np = gradients.shape[1]//2
            #EOdata = EOdata[:,:np]
            #F = jnp.sum(EOdata, axis=0)
            #S = jnp.matmul(jnp.conj(jnp.transpose(gradients[:,:np])), jnp.matmul(jnp.diag(p), gradients[:,:np]) )

        if self.diagonalShift > 1e-10:
            S = S + jnp.diag(self.diagonalShift * jnp.diag(S))

        return S, F, EOdata


    def get_sr_equation(self, Eloc, gradients):

        return get_tdvp_equation(Eloc, gradients, rhsPrefactor=1.)
    
    
    def transform_to_eigenbasis(self, S, F, EOdata):
        
        self.ev, self.V = jnp.linalg.eigh(S)
        self.VtF = jnp.dot(jnp.transpose(jnp.conj(self.V)),F)

        EOdata = jnp.matmul(EOdata, jnp.conj(self.V))
        # Need MPI
        # self.rhoVar = jnp.var(EOdata, axis=0)
        self.rhoVar = mpi.global_variance(EOdata)

        self.snr = jnp.sqrt(EOdata.shape[0] / (self.rhoVar / (jnp.conj(self.VtF) * self.VtF)  - 1.))


    def solve(self, Eloc, gradients, p=None):

        # Get TDVP equation from MC data
        S, F, Fdata = self.get_tdvp_equation(Eloc, gradients, p)

        # Transform TDVP equation to eigenbasis
        self.transform_to_eigenbasis(S,F,Fdata)

        # Discard eigenvalues below numerical precision
        self.invEv = jnp.where(jnp.abs(self.ev / self.ev[-1]) > self.svdTol, 1./self.ev, 0.)

        if p is None:
            # Construct a soft cutoff based on the SNR
            regularizer = 1. / (1. + (self.snrTol / self.snr)**6 )
        else:
            regularizer = jnp.ones(len(self.invEv))

        update = jnp.real( jnp.dot( self.V, (self.invEv * regularizer * self.VtF) ) )
        return update

        #work with complex matrix
        #update = jnp.dot( self.V, (self.invEv * (regularizer * self.VtF)) )
        #return jnp.concatenate((jnp.real(update), jnp.imag(update)))


    def __call__(self, netParameters, t, rhsArgs):

        tmpParameters = rhsArgs['psi'].get_parameters()
        rhsArgs['psi'].set_parameters(netParameters)
        
        # Get sample
        myNumSamples = 0
        if rhsArgs['numSamples'] is not None:
            myNumSamples = mpi.distribute_sampling(rhsArgs['numSamples'])
        sampleConfigs, sampleLogPsi, p =  self.sampler.sample( rhsArgs['psi'], myNumSamples )

        # Evaluate local energy
        sampleOffdConfigs, matEls = rhsArgs['hamiltonian'].get_s_primes(sampleConfigs)
        sampleLogPsiOffd = rhsArgs['psi'](sampleOffdConfigs)
        Eloc = rhsArgs['hamiltonian'].get_O_loc(sampleLogPsi,sampleLogPsiOffd)

        # Evaluate gradients
        sampleGradients = rhsArgs['psi'].gradients(sampleConfigs)

        update=self.solve(Eloc, sampleGradients, p)
        
        rhsArgs['psi'].set_parameters(tmpParameters)

        return update


