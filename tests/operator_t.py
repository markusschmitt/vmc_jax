import sys
# Find jVMC package
sys.path.append(sys.path[0] + "/..")

import unittest

import jax
from jax.config import config
config.update("jax_enable_x64", True)

import jax.random as random
import flax.nn as nn
import jax.numpy as jnp

import numpy as np

import jVMC
import jVMC.operator as op
import jVMC.global_defs as global_defs


def get_shape(shape):
    if global_defs.usePmap:
        return (global_defs.device_count(),) + shape
    return shape


class TestOperator(unittest.TestCase):

    def test_nonzeros(self):

        L = 4
        lDim = 2
        key = random.PRNGKey(3)
        s = random.randint(key, (24, L), 0, 2, dtype=np.int32).reshape(get_shape((-1, L)))

        h = op.BranchFreeOperator()

        h.add(op.scal_opstr(2., (op.Sp(0),)))
        h.add(op.scal_opstr(2., (op.Sp(1),)))
        h.add(op.scal_opstr(2., (op.Sp(2),)))

        sp, matEl = h.get_s_primes(s)

        logPsi = jnp.ones(s.shape[:-1])
        logPsiSP = jnp.ones(sp.shape[:-1])

        tmp = h.get_O_loc(logPsi, logPsiSP)

        self.assertTrue(jnp.sum(jnp.abs(tmp - 2. * jnp.sum(-(s[..., :3] - 1), axis=-1))) < 1e-7)

    def test_povm(self):
        povm = op.POVM()
        povm_Operator = op.POVMOperator()

        # Full test case that compares P_dot obtained from density matrix equation to the one obtained with the custom operator
        N = 4
        base = jnp.array([[[[[i, j, k, l] for l in range(4)] for k in range(4)] for j in range(4)] for i in range(4)]).reshape(4**N, 4)

        # build all Ms
        Ms = np.empty(shape=((4,) * N)).tolist()
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    for l in range(4):
                        Ms[i][j][k][l] = np.kron(povm.M[i], np.kron(povm.M[j], np.kron(povm.M[k], povm.M[l])))

        # build single particle operators
        sigmas = op.get_paulis()
        singleParticleOps = {}
        for site_id in range(N):
            site_dict = {}
            for dim, dimname in zip(range(3), ["X", "Y", "Z"]):
                op_temp = 1
                for site_id2 in range(N):
                    if site_id == site_id2:
                        op_temp = np.kron(op_temp, sigmas[dim])
                    else:
                        op_temp = np.kron(op_temp, np.eye(2))
                site_dict[dimname] = op_temp
            singleParticleOps[site_id] = site_dict

        # build random 2D Lindbladian
        J = np.random.random(size=(N, 2, 3))
        h = np.random.random(size=(N, 3))
        gammas = np.random.random(size=(N,))

        # 2x2 lattice, PBC
        neighbour_array = [[1, 2], [0, 3], [0, 3], [1, 2]]
        # 2x2 lattice with no connections to the bottom right spin to form a special case
        neighbour_array = [[1, 2], [0, ], [0, ], []]
        H = 0
        L_list = []
        for site_id in range(N):
            # add two particle interactions
            for idx, neighbour_id in enumerate(neighbour_array[site_id]):
                for dim, dimname in zip(range(3), ["X", "Y", "Z"]):
                    H += J[site_id, idx, dim] * singleParticleOps[site_id][dimname].dot(singleParticleOps[neighbour_id][dimname])
                    povm_Operator.add({"name": dimname + dimname, "strength": J[site_id, idx, dim], "sites": (site_id, neighbour_id)})
            # add single particle interactions
            for dim, dimname in zip(range(3), ["X", "Y", "Z"]):
                H += h[site_id, dim] * singleParticleOps[site_id][dimname]
                povm_Operator.add({"name": dimname, "strength": h[site_id, dim], "sites": (site_id,)})
            L_list.append(np.sqrt(gammas[site_id]) * (singleParticleOps[site_id]["X"] - 1j * singleParticleOps[site_id]["Y"]) / 2)
            povm_Operator.add({"name": "decaydown", "strength": gammas[site_id], "sites": (site_id,)})

        povm_Operator.compile()
        # define rho as a random density matrix
        X = np.random.random(size=(2**N, 2**N)) + 1.j * np.random.random(size=(2**N, 2**N))
        rho = X.dot(np.conj(X).T)
        rho /= np.trace(rho)

        def return_rho_dot(rho, H, L_list):
            rho_dot = 0
            rho_dot += -1.j * (H.dot(rho) - rho.dot(H))

            for L in L_list:
                rho_dot += L.dot(rho.dot(np.conj(L).T)) - 0.5 * np.conj(L).T.dot(L.dot(rho)) - 0.5 * rho.dot(np.conj(L).T.dot(L))

            return rho_dot

        rho_dot = return_rho_dot(rho, H, L_list)
        P_dot = np.zeros(4**N)
        P = np.zeros(4**N)
        P_dot_fromOp = np.zeros(4**N)

        for idx, M in enumerate(np.reshape(Ms, (4**N, 2**N, 2**N))):
            P_dot[idx] = np.real(np.trace(rho_dot.dot(M)))
            P[idx] = np.real(np.trace(rho.dot(M)))

        P = P.reshape((4,) * N)

        for idx in range(4**N):
            s = jnp.array([[base[idx]]])
            offdConfigs, offdmatEls = povm_Operator.get_s_primes(s)
            for offdConfig, offdmatEl in zip(offdConfigs[0], offdmatEls[0][0]):
                P_dot_fromOp[idx] += offdmatEl * P[offdConfig[0], offdConfig[1], offdConfig[2], offdConfig[3]]

        self.assertTrue(np.abs(np.trace(rho) - 1) < 1e-12)
        self.assertTrue(np.abs(np.trace(rho_dot)) < 1e-12)
        self.assertTrue(np.linalg.norm(rho_dot - np.conj(rho_dot).T) < 1e-12)
        self.assertTrue(np.abs(np.sum(P_dot)) < 1e-12)
        self.assertTrue(np.abs(np.sum(P_dot_fromOp)) < 1e-12)
        self.assertTrue(np.linalg.norm(P_dot - P_dot_fromOp) < 1e-12)


if __name__ == "__main__":
    unittest.main()
