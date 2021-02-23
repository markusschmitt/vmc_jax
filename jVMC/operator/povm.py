import jax
from jax import jit, vmap, grad, partial
import jax.numpy as jnp
import numpy as np

import sys
# Find jVMC package
sys.path.append(sys.path[0] + "/../..")

import jVMC.global_defs as global_defs
from . import Operator

import functools

opDtype = global_defs.tReal


def get_paulis():
    """Returns the Pauli matrices
    """
    return jnp.array([[[0.0, 1.0], [1.0, 0.0]],
                      [[0.0, -1.0j], [1.0j, 0.0]],
                      [[1, 0], [0, -1]]])


def get_M(theta, phi, name):
    """Returns 4 POVM measurement operators

    Args:
        * ``theta``: angle theta on the Bloch - sphere
        * ``phi``: angle phi on the Bloch - sphere
        * ``ǹame``: specifier of the POVM

    Returns:
        jnp.array with the leading axis giving the different POVM-Measurement operators

    """

    if name == 'SIC':
        s = jnp.array([[0.0, 0.0, 1.0],
                       [2.0 * jnp.sqrt(2.0) / 3.0, 0.0, -1.0 / 3.0],
                       [- jnp.sqrt(2.0) / 3.0, jnp.sqrt(2.0 / 3.0), -1.0 / 3.0],
                       [- jnp.sqrt(2.0) / 3.0, - jnp.sqrt(2.0 / 3.0), -1.0 / 3.0]])
    else:
        raise ValueError('The requrested POVM is not implemented.')

    # rotate s
    t_cos = np.cos(theta)
    t_sin = jnp.sin(theta)
    p_cos = jnp.cos(phi)
    p_sin = jnp.sin(phi)

    rotator_theta = jnp.array([[t_cos, 0.0, t_sin],
                               [0.0, 1.0, 0.0],
                               [-t_sin, 0.0, t_cos]])
    rotator_phi = jnp.array([[p_cos, -p_sin, 0.0],
                             [p_sin, p_cos, 0.0],
                             [0.0, 0.0, 1.0]])
    rotator = jnp.dot(jnp.transpose(rotator_theta), jnp.transpose(rotator_phi))

    s = jnp.dot(s, rotator)
    M = jnp.array(jnp.eye(2) + jnp.einsum('ak, kij -> aij', s, get_paulis()), dtype=global_defs.tCpx) / 4
    return M


def get_dissipators(M, T_inv):
    """Get the dissipation operators in the POVM-formalism

    Args:
        * ``M``: POVM-Measurement operators
        * ``T_inv``: Inverse POVM-Overlap matrix

    Returns:
        Dictionary with common (unscaled) 1-body dissipation channels
    """

    sigmas = get_paulis()
    dissipators_DM = {"decayup": (sigmas[0] + 1.j * sigmas[1]) / 2, "decaydown": (sigmas[0] - 1.j * sigmas[1]) / 2, "dephasing": sigmas[2]}
    dissipators_POVM = {}

    for key, value in dissipators_DM.items():
        dissipators_POVM[key] = jnp.array((jnp.einsum('ij, bc, cjk, kl, ali -> ab', value, T_inv, M, jnp.conj(value).T, M)
                                           - 0.5 * jnp.einsum('ij, jk, bc, ckl, ali -> ab', jnp.conj(value).T, value, T_inv, M, M)
                                           - 0.5 * jnp.einsum('ij, jk, akl, bc, cli -> ab', jnp.conj(value).T, value, M, T_inv, M)), dtype=opDtype)
    return dissipators_POVM


def get_unitaries(M, T_inv):
    """Get common 1- and 2-body unitary operators in the POVM formalism

    Args:
        * ``M``: POVM-Measurement operators
        * ``T_inv``: Inverse POVM-Overlap matrix

    Returns:
        Dictionary with common 1- and 2-body unitary interactions
    """

    sigmas = get_paulis()
    unitaries_DM = {"X": sigmas[0], "Y": sigmas[1], "Z": sigmas[2], "XX": jnp.kron(sigmas[0], sigmas[0]), "YY": jnp.kron(sigmas[1], sigmas[1]), "ZZ": jnp.kron(sigmas[2], sigmas[2])}
    unitaries_POVM = {}

    M_2Body = jnp.array([[jnp.kron(M[i], M[j]) for j in range(4)] for i in range(4)]).reshape(16, 4, 4)
    T_inv_2Body = jnp.kron(T_inv, T_inv)

    for key, value in unitaries_DM.items():
        if len(key) == 1:
            unitaries_POVM[key] = jnp.array((- 1.j * jnp.einsum('ij, bc, cjk, aki -> ab', value, T_inv, M, M)
                                             + 1.j * jnp.einsum('ij, ajk, bc, cki -> ab', value, M, T_inv, M)), dtype=opDtype)
        else:
            unitaries_POVM[key] = jnp.array((- 1.j * jnp.einsum('ij, bc, cjk, aki -> ab', value, T_inv_2Body, M_2Body, M_2Body)
                                             + 1.j * jnp.einsum('ij, ajk, bc, cki -> ab', value, M_2Body, T_inv_2Body, M_2Body)), dtype=opDtype)
    return unitaries_POVM


def get_observables(M, T_inv):
    """Get X, Y and Z observables in the POVM-formalism

    Args:
        * ``M``: POVM-Measurement operators
        * ``T_inv``: Inverse POVM-Overlap matrix

    Returns:
        Dictionary giving the X, Y and Z magnetization
    """
    sigmas = get_paulis()
    observables_DM = {"X": sigmas[0], "Y": sigmas[1], "Z": sigmas[2]}
    observables_POVM = {}
    for key, value in observables_DM.items():
        observables_POVM[key] = jnp.array(jnp.einsum('ab, bij, ji -> a', T_inv, M, value), dtype=opDtype)
    return observables_POVM


class POVM:
    """This class provides POVM - operators and related matrices

    Initializer arguments:

        * ``theta``: angle theta on the Bloch - sphere
        * ``phi``: angle phi on the Bloch - sphere
        * ``ǹame``: specifier of the POVM
    """

    def __init__(self, theta=0, phi=0, name='SIC'):
        """Initialize ``POVM``
        """
        self.theta = theta
        self.phi = phi
        self.name = name
        self.set_standard_povm_operators()

    def set_standard_povm_operators(self):
        """Obtain all relevant matrices
        """
        self.M = get_M(self.theta, self.phi, self.name)
        self.T = jnp.einsum('aij, bji -> ab', self.M, self.M)
        self.T_inv = jnp.linalg.inv(self.T)
        self.dissipators = get_dissipators(self.M, self.T_inv)
        self.unitaries = get_unitaries(self.M, self.T_inv)
        self.operators = {**self.unitaries, **self.dissipators}
        self.observables = get_observables(self.M, self.T_inv)


class POVMOperator(Operator):
    """This class provides functionality to compute operator matrix elements

    Initializer arguments:

        * ``lDim``: Dimension of local Hilbert space.
    """

    def __init__(self, povm=None, ldim=4):
        """Initialize ``Operator``.
        """
        if povm == None:
            self.povm = POVM()
        else:
            self.povm = povm
        self.ops = []
        self.ldim = ldim
        super().__init__()

    def add(self, opDescr):
        """Add another operator to the operator

        Args:

        * ``opDescr``: Operator string to be added to the operator.

        """
        self.ops.append(opDescr)
        self.compiled = False

    def _get_s_primes(self, s, stateCouplings, matEls, siteCouplings):
        def apply_on_singleSiteCoupling(s, stateCouplings, matEls, siteCoupling):
            stateIndices = tuple(s[siteCoupling])
            OffdConfig = jnp.vstack([s] * 16)
            OffdConfig = jax.ops.index_update(OffdConfig, jax.ops.index[:, siteCoupling], stateCouplings[stateIndices].reshape(-1, 2))
            return OffdConfig.reshape((4, 4, -1)), matEls[stateIndices]

        OffdConfigs, matEls = jax.vmap(apply_on_singleSiteCoupling, in_axes=(None, None, 0, 0))(s, stateCouplings, matEls, siteCouplings)
        return OffdConfigs.reshape((-1, OffdConfigs.shape[-1])), matEls.reshape(-1)

    def compile(self):
        """Compiles an operator mapping function
        """
        self.siteCouplings = []
        self.matEls = []
        self.idxbase = jnp.array([[[i, j] for j in range(4)] for i in range(4)])
        self.stateCouplings = jnp.array([[self.idxbase for j in range(4)] for i in range(4)])

        # find the highest index of the (many) local Hilbert spaces
        self.max_site = max([max(op["sites"]) for op in self.ops])

        # loop over all local Hilbert spaces
        for idx in range(self.max_site + 1):
            # sort interactions that involve the current local Hilbert space in the 0-th index into one- and two-body interactions
            ops_oneBody = [op for op in self.ops if op["sites"][0] == idx and len(op["sites"]) == 1]
            ops_twoBody = [op for op in self.ops if op["sites"][0] == idx and len(op["sites"]) == 2]

            # find all the local Hilbert spaces that are coupled to the one we currently selected ("idx")
            neighbour_indices = set([op["sites"][1] for op in ops_twoBody])
            if len(ops_twoBody) == 0 and len(ops_oneBody) > 0:
                # the current local Hilbert space is not listed in the 0-th index of any of the two body interactions, but 1-body interactions still need to respected
                # create artificial coupling to a second spin with a unity operator acting on spin #2
                neighbour_op_comp = jnp.zeros((16, 16), dtype=opDtype)
                for op_oneBody in ops_oneBody:
                    neighbour_op_comp += op_oneBody["strength"] * jnp.kron(self.povm.operators[op_oneBody["name"]], jnp.eye(4, dtype=opDtype))

                self.matEls.append(neighbour_op_comp.reshape((4,) * 4))
                self.siteCouplings.append([idx, (idx + 1) % self.max_site])
            else:
                for neighbour_idx in neighbour_indices:
                    # get all the operators that include the neighbour index
                    neighbour_ops = [op for op in ops_twoBody if op["sites"][1] == neighbour_idx]

                    # obtain 16x16 matrices for these operators and add the single particle  operators with weight 1 / # of neighbours
                    neighbour_op_comp = jnp.zeros((16, 16), dtype=opDtype)
                    for neighbour_op in neighbour_ops:
                        neighbour_op_comp += neighbour_op["strength"] * self.povm.operators[neighbour_op["name"]]
                    for op_oneBody in ops_oneBody:
                        neighbour_op_comp += op_oneBody["strength"] * jnp.kron(self.povm.operators[op_oneBody["name"]], jnp.eye(4, dtype=opDtype)) / len(neighbour_indices)

                    # for the computed operator (16x16) obtain a representation in which a matrix element and the connected indices are stored
                    self.matEls.append(neighbour_op_comp.reshape((4,) * 4))
                    self.siteCouplings.append([idx, neighbour_idx])

        self.siteCouplings = jnp.array(self.siteCouplings)
        self.matEls = jnp.array(self.matEls)
        return functools.partial(self._get_s_primes, stateCouplings=self.stateCouplings, matEls=self.matEls, siteCouplings=self.siteCouplings)


if __name__ == "__main__":
    povm = POVM()
    povm_Operator = POVMOperator()

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
    sigmas = get_paulis()
    singleParticleOps = {}
    for site_id in range(N):
        site_dict = {}
        for dim, dimname in zip(range(3), ["X", "Y", "Z"]):
            op = 1
            for site_id2 in range(N):
                if site_id == site_id2:
                    op = np.kron(op, sigmas[dim])
                else:
                    op = np.kron(op, np.eye(2))
            site_dict[dimname] = op
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
        L_list.append(np.sqrt(gammas[site_id]) * singleParticleOps[site_id]["Z"])
        povm_Operator.add({"name": "dephasing", "strength": gammas[site_id], "sites": (site_id,)})

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
        P_dot[idx] = np.trace(rho_dot.dot(M))
        P[idx] = np.trace(rho.dot(M))

    P = P.reshape((4,) * N)

    for idx in range(4**N):
        s = jnp.array([[base[idx]]])
        offdConfigs, offdmatEls = povm_Operator.get_s_primes(s)
        for offdConfig, offdmatEl in zip(offdConfigs[0], offdmatEls[0][0]):
            P_dot_fromOp[idx] += offdmatEl * P[offdConfig[0], offdConfig[1], offdConfig[2], offdConfig[3]]

    print(np.trace(rho))
    print(np.trace(rho_dot))
    print(np.linalg.norm(rho_dot - np.conj(rho_dot).T))
    print(np.sum(P_dot))
    print(np.linalg.norm(P_dot - P_dot_fromOp))
