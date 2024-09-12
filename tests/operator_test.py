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


def get_shape(shape):
    return (global_defs.device_count(),) + shape


class TestOperator(unittest.TestCase):

    def test_nonzeros(self):

        L = 4
        lDim = 2
        key = random.PRNGKey(3)
        s = random.randint(key, (24, L), 0, 2, dtype=np.int32).reshape(get_shape((-1, L)))

        h = op.BranchFreeOperator()

        # h.add(op.scal_opstr(2., (op.Sp(0),)))
        # h.add(op.scal_opstr(2., (op.Sp(1),)))
        # h.add(op.scal_opstr(2., (op.Sp(2),)))
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
            Comm.add(op.scal_opstr( 1., (op.annihilation(i), op.creation(j), ) ) )
            Comm.add(op.scal_opstr( 1., (op.creation(j), op.annihilation(i), ) ) )
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


if __name__ == "__main__":
    unittest.main()
