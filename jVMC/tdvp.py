import jax
import jax.numpy as jnp
import numpy as np

def realFun(x):
    return jnp.real(x)

def imagFun(x):
    return 0.5 * ( x - jnp.conj(x) )

class TDVP:

    def __init__(self, sampler, snrTol=2, makeReal='imag', rhsPrefactor=1.j):

        self.sampler = sampler
        self.snrTol = snrTol
        self.rhsPrefactor = rhsPrefactor

        self.makeReal = realFun
        if makeReal == 'imag':
            self.makeReal = imagFun


    def get_tdvp_equation(self, Eloc, gradients):
        
        Eloc -= jnp.mean(Eloc)
        gradients -= jnp.mean(gradients, axis=0)
        
        def eoFun(carry, xs):
            return carry, xs[0] * xs[1]
        _, EOdata = jax.lax.scan(eoFun, [None], (Eloc, jnp.conjugate(gradients)))
        EOdata = self.makeReal( -self.rhsPrefactor * EOdata )

        F = jnp.mean(EOdata, axis=0)
        S = self.makeReal( jnp.matmul(jnp.conj(jnp.transpose(gradients)), gradients) )

        return S, F, EOdata


    def get_sr_equation(self, Eloc, gradients):

        return get_tdvp_equation(Eloc, gradients, rhsPrefactor=1.)
    
    
    def transform_to_eigenbasis(self, S, F, EOdata):
        
        self.ev, self.V = jnp.linalg.eigh(S)
        self.VtF = jnp.dot(jnp.transpose(self.V),F)

        EOdata = jnp.matmul(EOdata, jnp.conj(self.V))
        self.rhoVar = jnp.var(EOdata, axis=0)

        self.snr = jnp.sqrt(EOdata.shape[0] / (self.rhoVar / (jnp.conj(self.VtF) * self.VtF)  - 1.))


    def solve(self, Eloc, gradients):

        # Get TDVP equation from MC data
        S, F, Fdata = self.get_tdvp_equation(Eloc, gradients)

        # Transform TDVP equation to eigenbasis
        self.transform_to_eigenbasis(S,F,Fdata)

        # Discard eigenvalues below numerical precision
        self.invEv = jnp.where(jnp.abs(self.ev / self.ev[-1]) > 1e-14, 1./self.ev, 0.)

        # Construct a soft cutoff based on the SNR
        regularizer = 1. / (1. + (self.snrTol / self.snr)**6 )

        print(jnp.dot( self.V, (self.invEv * regularizer * self.VtF) ))
        return jnp.real( jnp.dot( self.V, (self.invEv * regularizer * self.VtF) ) )


    def __call__(self, netParameters, t, rhsArgs):
        
        # Get sample
        sampleConfigs, sampleLogPsi, p =  self.sampler.sample( rhsArgs['psi'], rhsArgs['numSamples'] )

        # Evaluate local energy
        sampleOffdConfigs, matEls = rhsArgs['hamiltonian'].get_s_primes(sampleConfigs)
        sampleLogPsiOffd = rhsArgs['psi'](sampleOffdConfigs)
        Eloc = rhsArgs['hamiltonian'].get_O_loc(sampleLogPsi,sampleLogPsiOffd)

        # Evaluate gradients
        sampleGradients = rhsArgs['psi'].gradients(sampleConfigs)

        return self.solve(Eloc, sampleGradients)


