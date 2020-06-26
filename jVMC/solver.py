import jax
import jax.numpy as jnp

class EigenSolver:

    def __init__(self, snrTol=2):

        self.snrTol=snrTol


    def __call__(self, A, b, Eloc, gradients):

        self.get_eq_in_eigenbasis(A,b)

        self.Eloc = Eloc
        self.O = gradients
        self.get_empirical_snr()

        self.invEv = 1. / self.ev

        return self.V.dot(self.invEv * self.Vtb)


    def get_empirical_snr(self):

        energyMean = jnp.mean(self.Eloc)
        QMean = jnp.dot( jnp.mean(self.O, axis=0), self.V )

        Qbar = jnp.matmul(self.O, self.V) - QMean
        QbarSq = jnp.conjugate(Qbar) * Qbar

        Ebar = self.Eloc - energyMean
        EbarSq = jnp.conjugate(Ebar) * Ebar
        print("SNR=")
        print(Ebar)

        self.SNR = jnp.matmul( EbarSq, QbarSq ) / self.Eloc.shape[0]

        self.SNR /= jnp.conjugate(self.Vtb) * self.Vtb

        self.SNR = jnp.sqrt( self.Eloc.shape[0] / (self.SNR - 1.) )


    def get_eq_in_eigenbasis(self, A, b):
        
        self.ev, self.V = jnp.linalg.eigh(A)
        self.Vtb = jnp.dot(jnp.transpose(self.V),b)
