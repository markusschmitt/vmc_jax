import jax
import jax.numpy as jnp
import numpy as np

import jVMC.mpi_wrapper as mpi

from functools import partial

def realFun(x):
    return jnp.real(x)

def imagFun(x):
    return 0.5 * ( x - jnp.conj(x) )


@partial(jax.pmap, in_axes=(0,None))
def subtract_helper(x,y):
    return x-y

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
            
        ElocMean = mpi.global_mean(Eloc, p)
        Eloc = subtract_helper(Eloc, ElocMean)

        gradientsMean = mpi.global_mean(gradients, p)
        gradients = subtract_helper(gradients, gradientsMean)

        S = self.makeReal( mpi.global_covariance(gradients, p) )
        
        if p is None:
            
            # consider defining separate pmap'd function for this
            EOdata = jax.pmap(lambda a, f, Eloc, grad: a(-f * jnp.multiply(Eloc[:,None], jnp.conj(grad))), 
                                in_axes=(None, None, 0, 0, 0), static_broadcasted_argnums=(0,1))(
                                    self.makeReal, self.rhsPrefactor, Eloc, gradients
                                )

            F = mpi.global_mean(EOdata)

        else:

            # consider defining separate pmap'd function for this
            EOdata = jax.pmap(lambda a, f, p, Eloc, grad: a(-f * jnp.multiply((p*Eloc)[:,None], jnp.conj(grad))), 
                                in_axes=(None, None, 0, 0, 0), static_broadcasted_argnums=(0,1))(
                                    self.makeReal, self.rhsPrefactor, p, Eloc, gradients
                                )
            
            F = mpi.global_sum(EOdata)

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

        # consider defining separate pmap'd function for this
        EOdata = jax.pmap(lambda eo, v: jnp.matmul(eo, jnp.conj(v)), in_axes=(0,None))(EOdata, self.V)
        self.rhoVar = mpi.global_variance(EOdata)

        self.snr = jnp.sqrt( jnp.abs( mpi.globNumSamples / (self.rhoVar / (jnp.conj(self.VtF) * self.VtF)  - 1.) ) )


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


    def __call__(self, netParameters, t, **rhsArgs):

        tmpParameters = rhsArgs['psi'].get_parameters()
        rhsArgs['psi'].set_parameters(netParameters)
        
        outp = None
        if "outp" in rhsArgs:
            outp = rhsArgs["outp"]

        def start_timing(outp, name):
            if outp is not None:
                outp.start_timing(name)

        def stop_timing(outp, name):
            if outp is not None:
                outp.stop_timing(name)

        # Get sample
        start_timing(outp, "sampling")
        sampleConfigs, sampleLogPsi, p =  self.sampler.sample( rhsArgs['psi'], rhsArgs['numSamples'] )
        stop_timing(outp, "sampling")

        # Evaluate local energy
        start_timing(outp, "compute Eloc")
        sampleOffdConfigs, matEls = rhsArgs['hamiltonian'].get_s_primes(sampleConfigs)
        start_timing(outp, "evaluate off-diagonal")
        sampleLogPsiOffd = rhsArgs['psi'](sampleOffdConfigs)
        stop_timing(outp, "evaluate off-diagonal")
        Eloc = rhsArgs['hamiltonian'].get_O_loc(sampleLogPsi,sampleLogPsiOffd)
        stop_timing(outp, "compute Eloc")

        # Evaluate gradients
        start_timing(outp, "compute gradients")
        sampleGradients = rhsArgs['psi'].gradients(sampleConfigs)
        stop_timing(outp, "compute gradients")

        start_timing(outp, "solve TDVP eqn.")
        update=self.solve(Eloc, sampleGradients, p)
        stop_timing(outp, "solve TDVP eqn.")
        
        rhsArgs['psi'].set_parameters(tmpParameters)

        return update


