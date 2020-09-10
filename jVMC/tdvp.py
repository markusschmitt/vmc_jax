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

    
    def _get_tdvp_error(self, update):

        return jnp.abs(1. + jnp.real(update.dot(self.S0.dot(update)) - 2. * jnp.real( update.dot(self.F0) ) ) / self.ElocVar0)
    

    def get_residuals(self):

        return self.tdvpError, self.solverResidual
    

    def get_snr(self):

        return self.snr0
    

    def get_spectrum(self):

        return self.ev0


    def get_energy_variance(self):

        return self.ElocVar0


    def get_energy_mean(self):

        return jnp.real(self.ElocMean0)


    def get_tdvp_equation(self, Eloc, gradients, p=None):
            
        self.ElocMean = mpi.global_mean(Eloc, p)
        self.ElocVar = jnp.real(mpi.global_variance(Eloc, p))
        Eloc = subtract_helper(Eloc, self.ElocMean)

        gradientsMean = mpi.global_mean(gradients, p)
        gradients = subtract_helper(gradients, gradientsMean)
        
        if p is None:
            
            # consider defining separate pmap'd function for this
            EOdata = jax.pmap(lambda f, Eloc, grad: -f * jnp.multiply(Eloc[:,None], jnp.conj(grad)), 
                                in_axes=(None, 0, 0, 0), static_broadcasted_argnums=(0))(
                                    self.rhsPrefactor, Eloc, gradients
                                )
        
            self.F0 = mpi.global_mean(EOdata)

        else:

            # consider defining separate pmap'd function for this
            EOdata = jax.pmap(lambda f, p, Eloc, grad: -f * jnp.multiply((p*Eloc)[:,None], jnp.conj(grad)), 
                                in_axes=(None, 0, 0, 0), static_broadcasted_argnums=(0))(
                                    self.rhsPrefactor, p, Eloc, gradients
                                )
            
            self.F0 = mpi.global_sum(EOdata)

            #work with complex matrix
            #np = gradients.shape[1]//2
            #EOdata = EOdata[:,:np]
            #F = jnp.sum(EOdata, axis=0)
            #S = jnp.matmul(jnp.conj(jnp.transpose(gradients[:,:np])), jnp.matmul(jnp.diag(p), gradients[:,:np]) )

        F = self.makeReal( self.F0 )

        self.S0 = mpi.global_covariance(gradients, p)
        S = self.makeReal( self.S0 )

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

        return update, jnp.linalg.norm(S.dot(update)-F)/jnp.linalg.norm(F)

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
        update, solverResidual = self.solve(Eloc, sampleGradients, p)
        stop_timing(outp, "solve TDVP eqn.")
        
        rhsArgs['psi'].set_parameters(tmpParameters)
        
        if "intStep" in rhsArgs:
            if rhsArgs["intStep"] == 0:
                self.ElocMean0 = self.ElocMean
                self.ElocVar0 = self.ElocVar
                self.tdvpError = self._get_tdvp_error(update)
                self.solverResidual = solverResidual
                self.snr0 = self.snr
                self.ev0 = self.ev

        return update


