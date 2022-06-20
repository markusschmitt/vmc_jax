import unittest

import jVMC
import jVMC.operator as op
import jax
import jax.numpy as jnp


class TestPOVM(unittest.TestCase):
    def test_matrix_to_povm(self):
        unity = jnp.eye(2)
        zero_matrix = jnp.zeros((2, 2))

        system_data = {"dim": "1D", "L": 2}
        povm = jVMC.operator.POVM(system_data)

        self.assertTrue(jnp.isclose(op.matrix_to_povm(unity, povm.M, povm.T_inv, mode='observable'),
                                    jnp.ones(4)).all())
        self.assertTrue(jnp.isclose(op.matrix_to_povm(unity, povm.M, povm.T_inv, mode='unitary'),
                                    jnp.zeros(4)).all())
        self.assertTrue(jnp.isclose(op.matrix_to_povm(unity, povm.M, povm.T_inv, mode='dissipative'),
                                    jnp.zeros(4)).all())

        self.assertTrue(jnp.isclose(op.matrix_to_povm(zero_matrix, povm.M, povm.T_inv, mode='observable'),
                                    jnp.zeros(4)).all())
        self.assertTrue(jnp.isclose(op.matrix_to_povm(zero_matrix, povm.M, povm.T_inv, mode='unitary'),
                                    jnp.zeros(4)).all())
        self.assertTrue(jnp.isclose(op.matrix_to_povm(zero_matrix, povm.M, povm.T_inv, mode='dissipative'),
                                    jnp.zeros(4)).all())

        self.assertRaises(ValueError, op.matrix_to_povm, zero_matrix, povm.M, povm.T_inv, mode='wrong_mode')

    def test_adding_operator(self):
        unity = jnp.eye(2)
        zero_matrix = jnp.zeros((2, 2))

        system_data = {"dim": "1D", "L": 2}
        povm = jVMC.operator.POVM(system_data)

        unity_povm = op.matrix_to_povm(unity, povm.M, povm.T_inv, mode='unitary')
        zeros_povm =  op.matrix_to_povm(zero_matrix, povm.M, povm.T_inv, mode='dissipative')

        self.assertFalse("unity" in povm.operators.keys())
        self.assertFalse("zero" in povm.operators.keys())

        povm.add_unitary("unity", unity_povm)
        povm.add_dissipator("zero", zeros_povm)

        self.assertTrue("unity" in povm.operators.keys())
        self.assertTrue("zero" in povm.operators.keys())

        self.assertRaises(ValueError, povm.add_unitary, "zero", op.matrix_to_povm(zero_matrix, povm.M,
                                                                                  povm.T_inv, mode='unitary'))
        self.assertRaises(ValueError, povm.add_dissipator, "unity", op.matrix_to_povm(unity, povm.M,
                                                                                      povm.T_inv, mode='dissipative'))

    def test_time_evolution(self):
        # This tests the time evolution of a sample system and compares it with the analytical solution
        def copy_dict(a):
            b = {}
            for key, value in a.items():
                if type(value) == type(a):
                    b[key] = copy_dict(value)
                else:
                    b[key] = value
            return b

        L = 2
        Tmax = 2
        dt = 1E-3

        sample_shape = (L,)
        psi = jVMC.util.util.init_net({"gradient_batch_size": 5000, "net1":
            {"type": "RNN",
             "translation": True,
             "parameters": {"inputDim": 4,
                            "realValuedOutput": True,
                            "realValuedParams": True,
                            "logProbFactor": 1, "hiddenSize": 1, "L": L, "depth": 1}}},
                                      sample_shape, 1234)

        system_data = {"dim": "1D", "L": L}
        povm = op.POVM(system_data)

        Lindbladian = op.POVMOperator(povm)
        for l in range(L):
            Lindbladian.add({"name": "X", "strength": 3.0, "sites": (l,)})
            Lindbladian.add({"name": "dephasing", "strength": 1.0, "sites": (l,)})

        # Set initial state
        prob_dist = jVMC.operator.povm.get_1_particle_distributions("y_up", Lindbladian.povm)
        prob_dist /= prob_dist[0]
        biases = jnp.log(prob_dist[1:])
        params = copy_dict(psi._param_unflatten(psi.get_parameters()))

        params["outputDense"]["bias"] = biases
        params["outputDense"]["kernel"] = 1e-15 * params["outputDense"]["kernel"]
        params = jnp.concatenate([p.ravel()
                                  for p in jax.tree_util.tree_flatten(params)[0]])
        psi.set_parameters(params)

        sampler = jVMC.sampler.ExactSampler(psi, (L,), lDim=4, logProbFactor=1)

        tdvpEquation = jVMC.util.tdvp.TDVP(sampler, rhsPrefactor=-1.,
                                           svdTol=1e-6, diagonalShift=0, makeReal='real', crossValidation=False)

        stepper = jVMC.util.stepper.Euler(timeStep=dt)  # ODE integrator

        res = {"X": [], "Y": [], "Z": []}

        t = 0
        times = jnp.linspace(0, Tmax, int(Tmax / dt))
        for i in range(int(Tmax / dt)):
            result = jVMC.operator.povm.measure_povm(Lindbladian.povm, sampler)
            for dim in ["X", "Y", "Z"]:
                res[dim].append(result[dim]["mean"])

            dp, _ = stepper.step(0, tdvpEquation, psi.get_parameters(), hamiltonian=Lindbladian, psi=psi)
            t += dt
            psi.set_parameters(dp)

        # Analytical solution
        w = jnp.sqrt(35)
        Sx_avg = jnp.zeros_like(times)
        Sy_avg = (w*jnp.cos(w*times)-jnp.sin(w*times))/w*jnp.exp(-times)
        Sz_avg = 6/w*jnp.sin(w*times)*jnp.exp(-times)

        self.assertTrue(jnp.allclose(Sx_avg, jnp.asarray(res["X"]), atol=1e-2))
        self.assertTrue(jnp.allclose(Sy_avg, jnp.asarray(res["Y"]), atol=1e-2))
        self.assertTrue(jnp.allclose(Sz_avg, jnp.asarray(res["Z"]), atol=1e-2))


if __name__ == '__main__':
    unittest.main()
