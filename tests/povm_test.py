import unittest

import jVMC
import jVMC.operator as op
import jax
import jax.numpy as jnp


class TestPOVM(unittest.TestCase):
    def prepare_net(self, L, dt, hiddenSize=1, depth=1, cell="RNN"):
        def copy_dict(a):
            b = {}
            for key, value in a.items():
                if type(value) == type(a):
                    b[key] = copy_dict(value)
                else:
                    b[key] = value
            return b

        sample_shape = (L,)

        self.psi = jVMC.util.util.init_net({"batch_size": 200, "net1":
                                            {"type": "RNN",
                                             "translation": {"use": True, "factor": 1},
                                             "parameters": {"inputDim": 4,
                                                            "realValuedOutput": True,
                                                            "realValuedParams": True,
                                                            "logProbFactor": 1, "hiddenSize": hiddenSize, "L": L, "depth": depth, "cell": cell}}},
                                           sample_shape, 1234)

        system_data = {"dim": "1D", "L": L}
        self.povm = op.POVM(system_data)

        prob_dist = jVMC.operator.povm.get_1_particle_distributions("y_up", self.povm)
        prob_dist /= prob_dist[0]
        biases = jnp.log(prob_dist[1:])
        params = copy_dict(self.psi._param_unflatten(self.psi.get_parameters()))

        params["net"]["outputDense"]["bias"] = biases
        params["net"]["outputDense"]["kernel"] = 1e-15 * params["net"]["outputDense"]["kernel"]
        params = jnp.concatenate([p.ravel()
                                  for p in jax.tree_util.tree_flatten(params)[0]])
        self.psi.set_parameters(params)

        self.sampler = jVMC.sampler.ExactSampler(self.psi, (L,), lDim=4, logProbFactor=1)

        self.tdvpEquation = jVMC.util.tdvp.TDVP(self.sampler, rhsPrefactor=-1.,
                                                pinvTol=0.0, pinvCutoff=1e-6, diagonalShift=0, makeReal='real', crossValidation=False)

        #self.stepper = jVMC.util.stepper.Euler(timeStep=dt)  # ODE integrator
        self.stepper = jVMC.util.stepper.Heun(timeStep=dt)  # ODE integrator

    def test_matrix_to_povm(self):
        unity = jnp.eye(2)
        zero_matrix = jnp.zeros((2, 2))

        system_data = {"dim": "1D", "L": 2}
        povm = jVMC.operator.POVM(system_data)

        self.assertTrue(jnp.isclose(op.matrix_to_povm(unity, povm.M, povm.T_inv, mode='observable'),
                                    jnp.ones(4)).all())
        self.assertTrue(jnp.isclose(op.matrix_to_povm(unity, povm.M, povm.T_inv, mode='unitary'),
                                    jnp.zeros((4, 4))).all())
        self.assertTrue(jnp.isclose(op.matrix_to_povm(unity, povm.M, povm.T_inv, mode='dissipative'),
                                    jnp.zeros((4, 4))).all())
        self.assertTrue(jnp.isclose(op.matrix_to_povm(unity, povm.M, povm.T_inv, mode='imaginary'),
                                    -2*jnp.eye(4)).all())

        self.assertTrue(jnp.isclose(op.matrix_to_povm(zero_matrix, povm.M, povm.T_inv, mode='observable'),
                                    jnp.zeros(4)).all())
        self.assertTrue(jnp.isclose(op.matrix_to_povm(zero_matrix, povm.M, povm.T_inv, mode='unitary'),
                                    jnp.zeros((4, 4))).all())
        self.assertTrue(jnp.isclose(op.matrix_to_povm(zero_matrix, povm.M, povm.T_inv, mode='dissipative'),
                                    jnp.zeros((4, 4))).all())
        self.assertTrue(jnp.isclose(op.matrix_to_povm(zero_matrix, povm.M, povm.T_inv, mode='imaginary'),
                                    jnp.zeros((4, 4))).all())

        self.assertRaises(ValueError, op.matrix_to_povm, zero_matrix, povm.M, povm.T_inv, mode='wrong_mode')

    def test_adding_operator(self):
        unity = jnp.eye(2)
        zero_matrix = jnp.zeros((2, 2))

        system_data = {"dim": "1D", "L": 2}
        povm = jVMC.operator.POVM(system_data)

        unity_povm = op.matrix_to_povm(unity, povm.M, povm.T_inv, mode='unitary')
        zeros_povm = op.matrix_to_povm(zero_matrix, povm.M, povm.T_inv, mode='dissipative')

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

    def test_time_evolution_one_site(self):
        # This tests the time evolution of a sample system and compares it with the analytical solution

        L = 3
        Tmax = 0.5
        dt = 2E-3

        self.prepare_net(L, dt, hiddenSize=1, depth=1)

        Lindbladian = op.POVMOperator(self.povm)
        for l in range(L):
            Lindbladian.add({"name": "X", "strength": 3.0, "sites": (l,)})
            Lindbladian.add({"name": "dephasing", "strength": 1.0, "sites": (l,)})

        res = {"X": [], "Y": [], "Z": []}

        times = []
        t=0.
        while t<Tmax:

            result = jVMC.operator.povm.measure_povm(Lindbladian.povm, self.sampler)
            for dim in ["X", "Y", "Z"]:
                res[dim].append(result[dim]["mean"])
            times.append(t)
            if t>0.005:
                self.stepper.set_dt(3e-2)

            dp, stepSize = self.stepper.step(0, self.tdvpEquation, self.psi.get_parameters(), hamiltonian=Lindbladian,
                                      psi=self.psi)
            
            t += stepSize

            self.psi.set_parameters(dp)

        times = jnp.array(times)

        # Analytical solution
        w = jnp.sqrt(35)
        Sx_avg = jnp.zeros_like(times)
        Sy_avg = (w * jnp.cos(w * times) - jnp.sin(w * times)) / w * jnp.exp(-times)
        Sz_avg = 6 / w * jnp.sin(w * times) * jnp.exp(-times)

        print(Sz_avg-jnp.asarray(res["Z"]))
        self.assertTrue(jnp.allclose(Sx_avg, jnp.asarray(res["X"]), atol=1e-2))
        self.assertTrue(jnp.allclose(Sy_avg, jnp.asarray(res["Y"]), atol=1e-2))
        self.assertTrue(jnp.allclose(Sz_avg, jnp.asarray(res["Z"]), atol=1e-2))

    def test_time_evolution_two_site(self):
        # This tests the time evolution of a sample system and compares it with the analytical solution

        L = 3
        Tmax = 0.25
        dt = 2E-3

        self.prepare_net(L, dt, hiddenSize=3, depth=1)

        sx = op.get_paulis()[0]
        XX_ = jnp.kron(sx, sx)
        M_2_body = jnp.array(
            [[jnp.kron(self.povm.M[i], self.povm.M[j]) for j in range(4)] for i in range(4)]).reshape(16, 4, 4)
        T_inv_2_body = jnp.kron(self.povm.T_inv, self.povm.T_inv)

        self.povm.add_dissipator("XX_", op.matrix_to_povm(XX_, M_2_body, T_inv_2_body, mode="dissipative"))

        Lindbladian = op.POVMOperator(self.povm)
        Lindbladian.add({"name": "XX_", "strength": 1.0, "sites": (0, 1)})
        Lindbladian.add({"name": "Z", "strength": 3.0, "sites": (0,)})
        Lindbladian.add({"name": "Z", "strength": 3.0, "sites": (1,)})
        Lindbladian.add({"name": "Z", "strength": 3.0, "sites": (2,)})

        res = {"X": [], "Y": [], "Z": []}

        times = []
        t=0.
        while t<Tmax:

            result = jVMC.operator.povm.measure_povm(Lindbladian.povm, self.sampler)
            for dim in ["X", "Y", "Z"]:
                res[dim].append(result[dim]["mean"])
            times.append(t)
            if t>0.005:
                self.stepper.set_dt(2.5e-2)

            dp, stepSize = self.stepper.step(0, self.tdvpEquation, self.psi.get_parameters(), hamiltonian=Lindbladian,
                                      psi=self.psi)
            
            t += stepSize

            self.psi.set_parameters(dp)

        times = jnp.array(times)

        # Analytical solution
        w = jnp.sqrt(35)
        Sx_avg = -jnp.sin(6 * times) / 3 - 4 / w * jnp.sin(w * times) * jnp.exp(-times)
        Sy_avg = jnp.cos(6 * times) / 3 + (2 / 3 * jnp.cos(w * times) - 2 / 3 / w * jnp.sin(w * times)) * jnp.exp(-times)
        Sz_avg = jnp.zeros_like(times)

        self.assertTrue(jnp.allclose(Sx_avg, jnp.asarray(res["X"]), atol=1e-2))
        self.assertTrue(jnp.allclose(Sy_avg, jnp.asarray(res["Y"]), atol=1e-2))
        self.assertTrue(jnp.allclose(Sz_avg, jnp.asarray(res["Z"]), atol=1e-2))

    def test_time_evolution_three_site(self):
        # This tests the time evolution of a sample system and compares it with the analytical solution

        L = 3
        Tmax = 0.2
        dt = 5E-4

        self.prepare_net(L, dt, hiddenSize=3, depth=1)

        sx = op.get_paulis()[0]
        XXX = jnp.kron(jnp.kron(sx, sx), sx)
        M_3_body = jnp.array(
            [[[jnp.kron(jnp.kron(self.povm.M[i], self.povm.M[j]), self.povm.M[k]) for j in range(4)] for i in range(4)]
             for k in range(4)]).reshape(64, 8, 8)
        T_inv_3_body = jnp.kron(jnp.kron(self.povm.T_inv, self.povm.T_inv), self.povm.T_inv)

        self.povm.add_dissipator("XXX", op.matrix_to_povm(XXX, M_3_body, T_inv_3_body, mode="dissipative"))

        Lindbladian = op.POVMOperator(self.povm)
        Lindbladian.add({"name": "XXX", "strength": 1.0, "sites": (0, 1, 2)})
        Lindbladian.add({"name": "Z", "strength": 3.0, "sites": (0,)})
        Lindbladian.add({"name": "Z", "strength": 3.0, "sites": (1,)})
        Lindbladian.add({"name": "Z", "strength": 3.0, "sites": (2,)})

        res = {"X": [], "Y": [], "Z": []}

        times = []
        t=0.
        while t<Tmax:

            result = jVMC.operator.povm.measure_povm(Lindbladian.povm, self.sampler)
            for dim in ["X", "Y", "Z"]:
                res[dim].append(result[dim]["mean"])
            times.append(t)
            if t>0.003:
                self.stepper.set_dt(1e-2)

            dp, stepSize = self.stepper.step(0, self.tdvpEquation, self.psi.get_parameters(), hamiltonian=Lindbladian,
                                      psi=self.psi)
            
            t += stepSize

            self.psi.set_parameters(dp)

        times = jnp.array(times)

        # Analytical solution
        w = jnp.sqrt(35)
        Sx_avg = -6 * jnp.sin(w * times) * jnp.exp(-times) / w
        Sy_avg = jnp.cos(w * times) * jnp.exp(-times) - jnp.sin(w * times) * jnp.exp(-times) / w
        Sz_avg = jnp.zeros_like(times)

        print(Sx_avg - jnp.asarray(res["X"]))
        self.assertTrue(jnp.allclose(Sx_avg, jnp.asarray(res["X"]), atol=1e-2))
        self.assertTrue(jnp.allclose(Sy_avg, jnp.asarray(res["Y"]), atol=1e-2))
        self.assertTrue(jnp.allclose(Sz_avg, jnp.asarray(res["Z"]), atol=1e-2))

    def test_ground_state_search(self):
        L = 2
        dt = 1E-2

        self.prepare_net(L, dt)

        sz = op.get_paulis()[2]
        self.povm.add_imaginary("imag_Z", op.matrix_to_povm(sz, self.povm.M, self.povm.T_inv, mode='imag'))

        Lindbladian = op.POVMOperator(self.povm)
        Lindbladian.add({"name": "imag_Z", "strength": 4., "sites": (0, )})
        Lindbladian.add({"name": "imag_Z", "strength": 4., "sites": (1, )})

        def measure_energy(confs, probs):
            return jnp.sum(jVMC.mpi_wrapper.global_mean(self.povm.observables["Z"][confs], probs))

        for i in range(40):
            dp, _ = self.stepper.step(0, self.tdvpEquation, self.psi.get_parameters(), hamiltonian=Lindbladian,
                                      psi=self.psi)
            self.psi.set_parameters(dp)

        confs, _, probs = self.sampler.sample()
        self.assertTrue(jnp.allclose(measure_energy(confs, probs), -2, atol=1e-2))


if __name__ == '__main__':
    unittest.main()
