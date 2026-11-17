import unittest
import numpy as np
import jax.numpy as jnp

import jVMC_exp
from jVMC_exp.stats import SampledObs

# Functions T of the right hand side b = T(F) and of the matrix A = T_A(S) of Evolution
TRANSFORMATIONS = {
    "holomorphic": (lambda x: -1j * x, lambda x: x),
    "make_real=True": (lambda x: jnp.real(-1j * x), jnp.real),
    "make_real=False": (lambda x: 1j * jnp.imag(-1j * x), lambda x: 1j * jnp.imag(x)),
}

class TestPinvSNR(unittest.TestCase):
    """
    The empiric signal-to-noise ratio of mode i is sqrt(N |(V^H b)_i|^2 / var_i), with the
    variance over the samples of (V^H b_n)_i, b_n = T(conj(O_n - <O>) (E_n - <E>)).
    """
    def test_empiric_snr_matches_reference(self):
        rng = np.random.default_rng(0)
        N, P = 600, 5
        mix = rng.standard_normal((P, P)) + 1j * rng.standard_normal((P, P))
        O = (rng.standard_normal((N, P)) + 1j * rng.standard_normal((N, P))) @ mix
        E = rng.standard_normal(N) + 1j * rng.standard_normal(N) + 0.3 * O.sum(axis=1)

        for name, (rhs_trans, lhs_trans) in TRANSFORMATIONS.items():
            for pad_size in (0, 2):
                with self.subTest(name, pad_size=pad_size):
                    grad_log_psi, o_loc = SampledObs(jnp.array(O)), SampledObs(jnp.array(E))
                    A = lhs_trans(grad_log_psi.get_covar()) + 1e-3 * jnp.eye(P)
                    b = rhs_trans(grad_log_psi.get_covar(o_loc))

                    ev, V = np.linalg.eigh(np.array(A))
                    dO, dE = O - O.mean(axis=0), E - E.mean()
                    rho = np.array(rhs_trans(jnp.array(np.conj(dO) * dE[:, None]))) @ np.conj(V)
                    expected = np.sqrt(N * np.abs(rho.mean(axis=0)) ** 2 / (rho.var(axis=0) + 1e-14))

                    _, info = jVMC_exp.solver.PinvSNR(snr_tol=1)(
                        jnp.pad(A, ((0, pad_size), (0, pad_size))), b,
                        grad_log_psi=grad_log_psi, o_loc=o_loc,
                        transformation=rhs_trans, pad_size=pad_size
                    )

                    self.assertTrue(np.allclose(np.array(info["snr"]), expected, rtol=1e-8))
                    # The logarithmic derivatives are left as they were
                    self.assertTrue(np.allclose(np.array(grad_log_psi.observations)[:N], O, rtol=1e-12))

if __name__ == "__main__":
    unittest.main()
