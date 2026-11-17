import unittest
import jax
import jax.numpy as jnp
from math import isclose
import flax.linen as nn

import jVMC_exp
import jVMC_exp.nets as nets
from jVMC_exp.vqs import NQS
from jVMC_exp.symmetry.lattice_symetries import square_translation_symmetry
from jVMC_exp.sharding_config import MESH, DEVICE_SPEC
from jVMC_exp import global_defs
import jVMC_exp.nets.activation_functions as act_funs

def _general_test_grad(model, num_samples, L, test_class: unittest.TestCase):
    psi = NQS(model, L, num_samples)

    s = jnp.zeros((num_samples, L), dtype=global_defs.DT_SAMPLES)
    s = s.at[0, 1].set(1)
    s = s.at[2, 2].set(1)

    psi0 = psi(s)
    G = psi.gradients(s)
    delta = 1e-6
    params = psi.parameters
    for j in range(G.shape[-1]):
        u = jnp.zeros(G.shape[-1], dtype=global_defs.DT_PARAMS_REAL).at[j].set(1)
        psi.parameters = delta * u + psi.parameters_flat
        psi1 = psi(s)
        psi.parameters = params

        # Finite difference gradients
        Gfd = (psi1 - psi0) / delta
        with test_class.subTest(i=j):
            test_class.assertTrue(jnp.max(jnp.abs(Gfd - G[..., j])) < 1e-2)

class CpxRBM_nonHolomorphic(nn.Module):
    """
    Restricted Boltzmann machine with complex parameters.

    Initialization arguments:
        * ``s``: Computational basis configuration.
        * ``numHidden``: Number of hidden units.
        * ``bias``: ``Boolean`` indicating whether to use bias.

    """
    numHidden: int = 2
    bias: bool = False

    @nn.compact
    def __call__(self, s):

        layer = nn.Dense(
            self.numHidden, use_bias=self.bias,
            **jVMC_exp.nets.initializers.init_fn_args(
                kernel_init=jVMC_exp.nets.initializers.cplx_init,
                bias_init=jax.nn.initializers.zeros,
                param_dtype=global_defs.DT_PARAMS_CPX
            )
        )

        out = layer(2 * s.ravel() - 1)
        out = out + jnp.real(out) * 1e-2
        return jnp.sum(act_funs.log_cosh(out))

class Simple_nonHolomorphic(nn.Module):
    """
    Restricted Boltzmann machine with complex parameters.

    Initialization arguments:
        * ``s``: Computational basis configuration.
        * ``numHidden``: Number of hidden units.
        * ``bias``: ``Boolean`` indicating whether to use bias.

    """
    numHidden: int = 2
    bias: bool = False

    @nn.compact
    def __call__(self, s):
        z = self.param('z', jVMC_exp.nets.initializers.cplx_init, (1,), global_defs.DT_PARAMS_CPX)
    
        return jnp.sum(jnp.conj(z))

class MatrixMultiplication_NonHolomorphic(nn.Module):
    holo: bool = False

    @nn.compact
    def __call__(self, s):
        layer1 = nn.Dense(1, use_bias=False, **jVMC_exp.nets.initializers.init_fn_args(dtype=global_defs.DT_PARAMS_CPX))
        out = layer1(2 * s.ravel() - 1)
        if not self.holo:
            out = out + 1e-1 * jnp.real(out)
    
        return jnp.sum(out)
    
class TestGradients(unittest.TestCase):
    def test_automatic_holomorphicity_recognition(self):
        for k in range(10):
            L = 3
            num_samples = 4

            rbm = nets.CpxRBM(numHidden=2**k, bias=True)
            orbit = square_translation_symmetry(L, 1, "spin")
            psiC = NQS(rbm, L, num_samples, orbit=orbit)

            self.assertTrue(psiC.holomorphic)

    def test_gradients_cpx(self):
        L = 3
        num_samples = 4

        rbm = nets.CpxRBM(numHidden=2, bias=True)
        orbit = square_translation_symmetry(L, 1, "spin")
        model = orbit * rbm

        _general_test_grad(model, num_samples, L, self)

    def test_gradients_real(self):
        L = 3
        num_samples = 4

        rbmModel1 = nets.RBM(numHidden=2, bias=True)
        rbmModel2 = nets.RBM(numHidden=3, bias=True)

        _general_test_grad((rbmModel1, rbmModel2), num_samples, L, self)

    def test_gradients_nonholomorphic(self): 
        L = 3
        num_samples = 4
        model = nets.RNN1DGeneral(L=L)

        _general_test_grad(model, num_samples, L, self)
    
    def test_gradients_complex_nonholomorphic(self):
        L = 3
        num_samples = 4
        model = Simple_nonHolomorphic()

        _general_test_grad(model, num_samples, L, self)

        model = CpxRBM_nonHolomorphic()

        _general_test_grad(model, num_samples, L, self)

        model = MatrixMultiplication_NonHolomorphic(holo=False)

        psi = NQS(model, 1, num_samples)
        s= jnp.zeros((num_samples, 1), dtype=global_defs.DT_SAMPLES)
        G = psi.gradients(s)
        ref = jnp.array([-1.1+0.j, -1.1+0.j, -1.1+0.j, -1.1+0.j, -0.-1.j, -0.-1.j, -0.-1.j, -0.-1.j])
        self.assertTrue(jnp.allclose(G.T.ravel(), ref))

    def test_lazy_gradients_matches_gradients(self):
        """
        lazy_gradients (batched_jacobian's backend) must reproduce the
        dense gradients(), batch by batch. 4 samples per device and batch over
        10 samples per device is deliberately uneven ([4, 4, 2] per device) to
        hit the padding/trimming logic. On more than one device a batch does
        not consist of consecutive samples, so the dense gradients are split
        with the layout of the lazy iterable.
        """
        L = 3
        batch_size = 4 * jax.device_count()

        rbm = nets.CpxRBM(numHidden=3, bias=True)
        psi = NQS(rbm, L, batch_size, seed=42)

        # The second number of samples is not divisible by the number of devices (if there
        # are several): the samples are padded and the batches include the padding samples
        for num_samples in (10 * jax.device_count(), 10 * jax.device_count() + 1):
            with self.subTest(num_samples=num_samples):
                key = jax.random.PRNGKey(0)
                s = jax.random.randint(key, (num_samples, L), 0, 2).astype(global_defs.DT_SAMPLES)

                dense = psi.gradients(s)
                lazy = psi.lazy_gradients(s)
                pieces = list(lazy)

                self.assertEqual(len(pieces), lazy.layout.n_batches)
                in_sample_order = jnp.concatenate(pieces)[lazy.layout.batch_positions()]
                self.assertTrue(jnp.allclose(in_sample_order[:num_samples], dense))
                if num_samples == lazy.layout.num_samples:
                    for piece, expected in zip(pieces, lazy.layout.split(dense)):
                        self.assertEqual(piece.shape, expected.shape)
                        self.assertTrue(jnp.allclose(piece, expected))

    def test_gradient_dict(self):
        L = 3
        num_samples = 4
        net = jVMC_exp.nets.CpxRBM(numHidden=8, bias=False)

        psi = jVMC_exp.vqs.NQS(net, L, num_samples, seed=1234)
        s = jnp.zeros((num_samples, L))
        g1 = psi.gradients(s)
        g2 = psi.gradients_dict(s)["Dense_0"]["kernel"]

        self.assertTrue(isclose(jnp.linalg.norm(g1 - g2), 0.0))

class TestEvaluation(unittest.TestCase):
    def test_mixed_precision_nqs_boundary(self):
        L = 3
        num_samples = 4
        model = nets.CpxRBM(numHidden=2, bias=True)
        psi = NQS(
            model, 
            L, 
            num_samples, 
            seed=1234, 
            sampler_dtype=jnp.complex64,
            eval_dtype=jnp.complex64,
            grad_dtype=jnp.complex64
        )

        s = jnp.zeros((num_samples, L), dtype=global_defs.DT_SAMPLES)
        out = psi(s)
        gradients = psi.gradients(s)
        self.assertEqual(out.dtype, global_defs.DT_OUT_CPX)
        self.assertEqual(gradients.dtype, global_defs.DT_OUT_CPX)

        psi.parameters = psi.parameters_flat + jnp.asarray(1e-4, dtype=jnp.float64)
        param_leaves = jax.tree_util.tree_leaves(psi.parameters["params"])
        self.assertTrue(all(p.dtype == global_defs.DT_PARAMS_CPX for p in param_leaves))

        ratio_psi = NQS(
            nets.CpxRBM_ratio(numHidden=2, bias=True), 
            L, 
            num_samples, 
            seed=1234, 
            sampler_dtype=jnp.complex64,
            eval_dtype=jnp.complex64,
            grad_dtype=jnp.complex64
        )
        sp = s.at[:, 0].set(1)
        ratio = ratio_psi.call_ratio(s, sp)
        self.assertEqual(ratio.dtype, global_defs.DT_OUT_CPX)

if __name__ == "__main__":
    unittest.main()
