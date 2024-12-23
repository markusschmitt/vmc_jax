import unittest

import jax
jax.config.update("jax_enable_x64", True)

import jax.random as random
import jax.numpy as jnp

import numpy as np

import jVMC
import jVMC.operator as op
import jVMC.sampler
import jVMC.nets as nets
from jVMC.vqs import NQS
import jVMC.global_defs as global_defs

import flax.linen as nn
class Target(nn.Module):
  """Target wave function, returns a vector with the same dimension as the Hilbert space

    Initialization arguments:
        * ``L``: System size
        * ``d``: local Hilbert space dimension
        * ``delta``: small number to avoid log(0)

    """
  L: int
  d: float = 2.00
  delta: float = 1e-15

  @nn.compact
  def __call__(self, s):
    kernel = self.param('kernel',
                        nn.initializers.constant(1),
                        (int(self.d**self.L)))
    # return amplitude for state s
    idx = ((self.d**jnp.arange(self.L)).dot(s[::-1])).astype(int) # NOTE that the state is reversed to account for different bit conventions used in openfermion
    return jnp.log(abs(kernel[idx]+self.delta)) + 1.j*jnp.angle(kernel[idx]) 


def get_shape(shape):
    return (global_defs.device_count(),) + shape


class TestOperator(unittest.TestCase):

    def test_nonzeros(self):

        L = 4
        lDim = 2
        key = random.PRNGKey(3)
        s = random.randint(key, (24, L), 0, 2, dtype=np.int32).reshape(get_shape((-1, L)))

        h = op.BranchFreeOperator()

        h += 2. * op.Sp(0)
        h += 2. * op.Sp(1)
        h += 2. * op.Sp(2)

        sp, matEl = h.get_s_primes(s)

        logPsi = jnp.ones(s.shape[:-1])
        logPsiSP = jnp.ones(sp.shape[:-1])

        tmp = h.get_O_loc_unbatched(logPsi, logPsiSP)

        self.assertTrue(jnp.sum(jnp.abs(tmp - 2. * jnp.sum(-(s[..., :3] - 1), axis=-1))) < 1e-7)

    def test_op_with_arguments(self):

        L = 4
        key = random.PRNGKey(3)
        s = random.randint(key, (24, L), 0, 2, dtype=np.int32).reshape(get_shape((-1, L)))

        h = op.BranchFreeOperator()

        def f(t):
            return 2.0 * t

        h.add(op.scal_opstr(f, (op.Sp(0),)))
        h.add(op.scal_opstr(f, (op.Sp(1),)))
        h.add(op.scal_opstr(f, (op.Sp(2),)))

        for t in [0.5, 2, 13.9]:
            sp, matEl = h.get_s_primes(s, t)

            logPsi = jnp.ones(s.shape[:-1])
            logPsiSP = jnp.ones(sp.shape[:-1])

            tmp = h.get_O_loc_unbatched(logPsi, logPsiSP)

            self.assertTrue(jnp.sum(jnp.abs(tmp - f(t) * jnp.sum(-(s[..., :3] - 1), axis=-1))) < 1e-7)

    def test_op_2d(self):

        L = 4
        key = random.PRNGKey(3)
        s = random.randint(key, (24, L, L), 0, 2, dtype=np.int32).reshape(get_shape((-1, L, L)))

        h = op.BranchFreeOperator()

        h += 0.3 * op.Sp(0)
        h += 1.1 * op.Sp(1)
        h += 0.15 * op.Sp(4)

        sp, matEl = h.get_s_primes(s)

        self.assertTrue(sp.shape[2:] == (L,L))


    def test_batched_Oloc(self):

        L = 4

        h = op.BranchFreeOperator()
        for i in range(L):
            h.add(op.scal_opstr(2., (op.Sx(i),)))
            h.add(op.scal_opstr(2., (op.Sy(i), op.Sz((i + 1) % L))))

        rbm = nets.CpxRBM(numHidden=2, bias=False)
        psi = NQS(rbm)

        mcSampler = jVMC.sampler.MCSampler(psi, (L,), random.PRNGKey(0), updateProposer=jVMC.sampler.propose_spin_flip, numChains=1)

        numSamples = 100
        s, logPsi, _ = mcSampler.sample(numSamples=numSamples)

        sp, matEl = h.get_s_primes(s)
        logPsiSp = psi(sp)
        Oloc1 = h.get_O_loc_unbatched(logPsi, logPsiSp)

        batchSize = 13
        Oloc2 = h.get_O_loc_batched(s, psi, logPsi, batchSize)

        self.assertTrue(jnp.abs(jnp.sum(Oloc1) - jnp.sum(Oloc2)) < 1e-5)

    def test_batched_Oloc2(self):
        L = 4

        hamilton_unbatched = op.BranchFreeOperator()
        hamilton_batched = op.BranchFreeOperator(ElocBatchSize=13)
        for i in range(L):
            hamilton_unbatched.add(op.scal_opstr(2., (op.Sx(i),)))
            hamilton_unbatched.add(op.scal_opstr(2., (op.Sy(i), op.Sz((i + 1) % L))))
            hamilton_batched.add(op.scal_opstr(2., (op.Sx(i),)))
            hamilton_batched.add(op.scal_opstr(2., (op.Sy(i), op.Sz((i + 1) % L))))

        rbm = nets.CpxRBM(numHidden=2, bias=False)
        psi = NQS(rbm)

        mcSampler = jVMC.sampler.MCSampler(psi, (L,), random.PRNGKey(0), updateProposer=jVMC.sampler.propose_spin_flip,
                                           numChains=1)

        numSamples = 100
        s, logPsi, _ = mcSampler.sample(numSamples=numSamples)

        Oloc1 = hamilton_unbatched.get_O_loc(s, psi, logPsi)

        Oloc2 = hamilton_batched.get_O_loc(s, psi, logPsi)

        self.assertTrue(jnp.abs(jnp.sum(Oloc1) - jnp.sum(Oloc2)) < 1e-5)

    def test_td_prefactor(self):

        hamiltonian = op.BranchFreeOperator()
        hamiltonian.add((op.Sz(0),))
        hamiltonian.add((op.Sz(1),))
        # hamiltonian.add((op.Sx(0),op.Sx(1)))
        hamiltonian.add(op.scal_opstr(0.1, (op.Sx(0), op.Sx(1))))

        hamiltonian.compile()

    def test_fermionic_operators(self):
        L = 2

        rbm = nets.CpxRBM(numHidden=2, bias=True)
        psi = NQS(rbm)

        sampler = jVMC.sampler.ExactSampler(psi, (L,))

        def commutator(i,j):
            Comm = op.BranchFreeOperator()
            # Comm.add(op.scal_opstr( 1., (op.annihilation(i), op.creation(j), ) ) )
            # Comm.add(op.scal_opstr( 1., (op.creation(j), op.annihilation(i), ) ) )
            Comm.add(op.scal_opstr( 1., (op.creation(j), op.annihilation(i)) ) )
            Comm.add(op.scal_opstr( 1., (op.annihilation(i), op.creation(j)) ) )
            return Comm

        observalbes_dict = {
                            "same_site": [commutator(0,0),commutator(1,1)], 
                            "distinct_site": [commutator(0,1),commutator(1,0)]
                            }
        out_dict = jVMC.util.util.measure(observalbes_dict, psi, sampler)

        self.assertTrue(
            jnp.allclose(
                jnp.concatenate(
                    (out_dict["same_site"]['mean'],
                     out_dict["distinct_site"]['mean'])),
                        jnp.array([1.,1.,0.,0.]),
                        rtol=1e-15)
            )
        
        self.assertTrue(
            jnp.allclose(
                jnp.concatenate(
                    (out_dict["same_site"]['variance'],
                     out_dict["distinct_site"]['variance'])),
                        jnp.array([0.,0.,0.,0.]),
                        rtol=1e-15)
            )
        

        t = - 1.0 # hopping
        mu = -2.0 # chemical potential
        V = 4.0 # interaction
        L = 4   # number of sites
        flavour = 2 # number of flavours
        flavourL = flavour*L # number of spins times sites

        # initalize the Hamitonian
        hamiltonian = op.BranchFreeOperator()
        # impurity definitions
        site1UP = 0
        site1DO = flavourL-1#//flavour
        # loop over the 1d lattice
        for i in range(0,flavourL//flavour):
            # interaction
            hamiltonian.add(op.scal_opstr( V, ( op.number(site1UP + i) , op.number(site1DO - i) ) ) )
            # chemical potential
            hamiltonian.add(op.scal_opstr(mu , ( op.number(site1UP + i) ,) ) )
            hamiltonian.add(op.scal_opstr(mu , ( op.number(site1DO - i) ,) ) )
            if i == flavourL//flavour-1:
                continue
            # up chain hopping
            hamiltonian.add(op.scal_opstr( t, ( op.creation(site1UP + i + 1), op.annihilation(site1UP + i) ,  ) ) )
            hamiltonian.add(op.scal_opstr( t, ( op.creation(site1UP + i), op.annihilation(site1UP + i + 1) ,  ) ) )
            # down chain hopping
            hamiltonian.add(op.scal_opstr( t, ( op.creation(site1DO - i - 1), op.annihilation(site1DO - i) ,  ) ) )
            hamiltonian.add(op.scal_opstr( t, ( op.creation(site1DO - i), op.annihilation(site1DO - i - 1) ,  ) ) )

        b = np.loadtxt("tests/data_ref/fermion_ref.txt", dtype=np.complex128)
        chi_model = Target(L=flavourL, d=2)
        chi = NQS(chi_model)
        chi(jnp.array(jnp.ones((1, 1, flavourL))))
        chi.set_parameters(b)
        chiSampler = jVMC.sampler.ExactSampler(chi, (flavourL,))
        s, logPsi, p = chiSampler.sample()
        sPrime, _ = hamiltonian.get_s_primes(s)
        Oloc = hamiltonian.get_O_loc(s, chi, logPsi)
        Omean = jVMC.mpi_wrapper.global_mean(Oloc,p)

        self.assertTrue(jnp.allclose(Omean, -9.95314531))
        
    def test_opstr(self):
        op1 = op.Sz(3)
        op2 = op.Sx(5)

        opstr1 = 13. * op1 * op2
        opstr2 = 1.j * opstr1 * op1

        self.assertTrue(jnp.allclose(opstr2[0](), 13.j))
        for o in opstr2[1:]:
            self.assertTrue(isinstance(o, (op.LocalOp, dict)))


if __name__ == "__main__":
    unittest.main()
