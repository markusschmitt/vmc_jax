import unittest

import jVMC
import jVMC.operator as op
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


if __name__ == '__main__':
    unittest.main()
