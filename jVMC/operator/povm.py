import jax
from jax import jit, vmap, grad#, partial
import jax.numpy as jnp
import numpy as np

import jVMC.global_defs as global_defs
import jVMC.mpi_wrapper as mpi
from jVMC.operator import Operator

import functools
import itertools

opDtype = global_defs.tReal


def measure_povm(povm, sampler, sampleConfigs=None, probs=None, observables=None):
    """For a set of sampled configurations, compute the associated expectation values
    for a given set of observables. If none is provided, magnetizations and correlations for X, Y and Z are computed.

    Args:
        * ``povm``: the povm that holds the jitted evaluation function
        * ``sampler``: the sampler used for the computation of expectation values
        * ``sampleConfigs``: optional, if configurations have already been generated
        * ``probs``: optional, the associated probabilities
        * ``observables``: optional, observables for which expectation values are computed
    """
    if sampleConfigs == None:
        # sampleConfigs, sampleLogPsi, probs = sampler.sample(p, numSamples)
        sampleConfigs, sampleLogPsi, probs = sampler.sample()

    if observables == None:
        observables = povm.observables
    result = {}

    for name, ops in observables.items():
        results = povm.evaluate_observable(ops, sampleConfigs)
        result[name] = {}
        if probs is not None:
            result[name]["mean"] = jnp.array(mpi.global_mean(results[0], probs))
            result[name]["variance"] = jnp.array(mpi.global_variance(results[0], probs))
            result[name]["MC_error"] = jnp.array(0)

        else:
            result[name]["mean"] = jnp.array(mpi.global_mean(results[0]))
            result[name]["variance"] = jnp.array(mpi.global_variance(results[0]))
            result[name]["MC_error"] = jnp.array(result[name]["variance"] / jnp.sqrt(sampler.get_last_number_of_samples()))

        for key, value in results[1].items():
            result_name = name + "_corr_L" + str(key)
            result[result_name] = {}
            if probs is not None:
                result[result_name]["mean"] = jnp.array(mpi.global_mean(value, probs) - result[name]["mean"]**2)
                result[result_name]["variance"] = jnp.array(mpi.global_variance(value, probs))
                result[result_name]["MC_error"] = jnp.array(0.)
            else:
                result[result_name]["mean"] = jnp.array(mpi.global_mean(value) - result[name]["mean"]**2)
                result[result_name]["variance"] = jnp.array(mpi.global_variance(value))
                result[result_name]["MC_error"] = jnp.array(result[result_name]["variance"] / jnp.sqrt(sampler.get_last_number_of_samples()))

    return result


def get_1_particle_distributions(state, povm):
    """compute 1 particle POVM-representations, mainly used for defining initial states.

    Args:
        * ``state``: the desired state on the bloch-sphere, set together from [``"x"``, ``"y"``, ``"z"``] + ``"_"`` + [``"up"``, ``"down"``]
        * ``povm``: the povm for which the distribution is desired
    """
    M = povm.M
    probs = jnp.zeros(4)
    if state == "z_up":
        s = jnp.array([1, 0])
    elif state == "z_down":
        s = jnp.array([0, 1])
    elif state == "x_up":
        s = jnp.array([1, 1]) / jnp.sqrt(2)
    elif state == "x_down":
        s = jnp.array([1, -1]) / jnp.sqrt(2)
    elif state == "y_up":
        s = jnp.array([1, 1.j]) / jnp.sqrt(2)
    elif state == "y_down":
        s = jnp.array([1, -1.j]) / jnp.sqrt(2)
    else:
        raise ValueError("The desired state is not recognized.")
    for (idx, ele) in enumerate(M):
        probs = probs.at[idx].set(jnp.real(jnp.dot(jnp.conj(jnp.transpose(s)), jnp.dot(ele, s))))
    return probs


def get_paulis():
    """
    Returns the Pauli matrices.
    """
    return jnp.array([[[0.0, 1.0], [1.0, 0.0]],
                      [[0.0, -1.0j], [1.0j, 0.0]],
                      [[1, 0], [0, -1]]])


def get_M(theta, phi, name):
    """Returns 4 POVM measurement operators.

    Args:
        * ``theta``: angle theta on the Bloch - sphere
        * ``phi``: angle phi on the Bloch - sphere
        * ``name``: specifier of the POVM

    Returns:
        jnp.array with the leading axis giving the different POVM-Measurement operators.

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


def matrix_to_povm(A, M, T_inv, mode='unitary', dtype=opDtype):
    """Get operator from a matrix representation in POVM-formalism for the Lindblad equation or an observable.


    In unitary mode this function implements
        :math:`\Omega^{ab} = -i T^{-1 bc}  \mathrm{Tr}(A [M^c, M^a])`
    in dissipative mode
        :math:`\Omega^{ab} = T^{-1 bc} \mathrm{Tr}(A M^c A^\dagger M^a- 1/2 A^\dagger A \{M^c, M^a\})`
    and in observable mode
        :math:`O^a = T^{-1 ab} \mathrm{Tr}(M^b A)`
    where the Einstein sum convention is assumed.

    Args:
        * ``A``: Matrix representation of desired operator
        * ``M``: POVM-Measurement operators
        * ``T_inv``: Inverse POVM-Overlap matrix
        * ``mode``: String specifying the conversion mode. Possible values are 'unitary' ('uni'), 'dissipative' ('dis')
                    and 'observable' ('obs')

    Returns:
        jax.numpy.ndarray
    """
    if mode in ['unitary', 'uni']:
        return jnp.array(jnp.real(- 1.j * jnp.einsum('ij, bc, cjk, aki -> ab', A, T_inv, M, M)
                                  + 1.j * jnp.einsum('ij, ajk, bc, cki -> ab', A, M, T_inv, M)), dtype=dtype)
    elif mode in ['dissipative', 'dis']:
        return jnp.array(jnp.real(jnp.einsum('ij, bc, cjk, kl, ali -> ab', A, T_inv, M, jnp.conj(A).T, M)
                                  - 0.5 * jnp.einsum('ij, jk, bc, ckl, ali -> ab', jnp.conj(A).T, A, T_inv, M, M)
                                  - 0.5 * jnp.einsum('ij, jk, akl, bc, cli -> ab', jnp.conj(A).T, A, M, T_inv, M)),
                         dtype=dtype)
    elif mode in ['observable', 'obs']:
        return jnp.array(jnp.real(jnp.einsum('ab, bij, ji -> a', T_inv, M, A)), dtype=dtype)
    else:
        raise ValueError("Unknown mode string given. Allowed modes are 'unitary' ('uni'), 'dissipative' ('dis') "
                         "and 'observable' ('obs').")


def get_dissipators(M, T_inv):
    """Get the dissipation operators in the POVM-formalism.

    Args:
        * ``M``: POVM-Measurement operators
        * ``T_inv``: Inverse POVM-Overlap matrix

    Returns:
        Dictionary with common (unscaled) 1-body dissipation channels.
    """

    sigmas = get_paulis()
    dissipators_DM = {"decayup": (sigmas[0] + 1.j * sigmas[1]) / 2, "decaydown": (sigmas[0] - 1.j * sigmas[1]) / 2, "dephasing": sigmas[2]}
    dissipators_POVM = {}

    for key, value in dissipators_DM.items():
        dissipators_POVM[key] = matrix_to_povm(value, M, T_inv, mode='dissipative')
    return dissipators_POVM


def get_unitaries(M, T_inv):
    """Get common 1- and 2-body unitary operators in the POVM formalism.

    Args:
        * ``M``: POVM-Measurement operators
        * ``T_inv``: Inverse POVM-Overlap matrix

    Returns:
        Dictionary with common 1- and 2-body unitary interactions.
    """

    sigmas = get_paulis()
    unitaries_DM = {"X": sigmas[0], "Y": sigmas[1], "Z": sigmas[2], "XX": jnp.kron(sigmas[0], sigmas[0]), "YY": jnp.kron(sigmas[1], sigmas[1]), "ZZ": jnp.kron(sigmas[2], sigmas[2])}
    unitaries_POVM = {}

    M_2Body = jnp.array([[jnp.kron(M[i], M[j]) for j in range(4)] for i in range(4)]).reshape(16, 4, 4)
    T_inv_2Body = jnp.kron(T_inv, T_inv)

    for key, value in unitaries_DM.items():
        if len(key) == 1:
            unitaries_POVM[key] = matrix_to_povm(value, M, T_inv, mode='unitary')
        else:
            unitaries_POVM[key] = matrix_to_povm(value, M_2Body, T_inv_2Body, mode='unitary')
    return unitaries_POVM


def get_observables(M, T_inv):
    """Get X, Y and Z observables in the POVM-formalism.

    Args:
        * ``M``: POVM-Measurement operators
        * ``T_inv``: Inverse POVM-Overlap matrix

    Returns:
        Dictionary giving the X, Y and Z magnetization.
    """
    sigmas = get_paulis()
    observables_DM = {"X": sigmas[0], "Y": sigmas[1], "Z": sigmas[2]}
    observables_POVM = {}
    for key, value in observables_DM.items():
        observables_POVM[key] = matrix_to_povm(value, M, T_inv, mode='observable')
    return observables_POVM


class POVM():
    """This class provides POVM - operators and related matrices.

    Initializer arguments:
        * ``system_data``: dictionary with lattice dimension ``"dim"`` (either ``"1D"`` or ``"2D"``) and length ``"L"`` specifying the computation of correlators.
        * ``theta``: angle theta on the Bloch - sphere
        * ``phi``: angle phi on the Bloch - sphere
        * ``name``: specifier of the POVM
    """

    def __init__(self, system_data, theta=0, phi=0, name='SIC'):
        self.system_data = system_data
        self.theta = theta
        self.phi = phi
        self.name = name
        self.set_standard_povm_operators()

        # pmap'd member functions
        self._evaluate_mean_magnetization_pmapd = global_defs.pmap_for_my_devices(jax.vmap(lambda ops, idx: jnp.sum(
            self._evaluate_observable(ops, idx)) / idx.shape[0], in_axes=(None, 0)), in_axes=(None, 0))
        self._evaluate_magnetization_pmapd = global_defs.pmap_for_my_devices(jax.vmap(lambda ops, idx:
                                                                                      self._evaluate_observable(ops, idx), in_axes=(None, 0)), in_axes=(None, 0))
        self._evaluate_correlators_pmapd = global_defs.pmap_for_my_devices(jax.vmap(lambda resPerSamplePerSpin, corrLen: resPerSamplePerSpin *
                                                                                    jnp.roll(resPerSamplePerSpin, corrLen, axis=0), in_axes=(0, None)), in_axes=(0, None))
        self._spin_average_pmapd = global_defs.pmap_for_my_devices(jax.vmap(lambda obsPerSamplePerSpin: jnp.mean(obsPerSamplePerSpin, axis=-1), in_axes=(0,)), in_axes=(0,))

    def set_standard_povm_operators(self):
        """
        Obtain matrices required for dynamics and observables.
        """
        self.M = get_M(self.theta, self.phi, self.name)
        self.T = jnp.einsum('aij, bji -> ab', self.M, self.M)
        self.T_inv = jnp.linalg.inv(self.T)
        self.dissipators = get_dissipators(self.M, self.T_inv)
        self.unitaries = get_unitaries(self.M, self.T_inv)
        self._update_operators()  # This creates self.operators
        self.observables = get_observables(self.M, self.T_inv)

    def _update_operators(self):
        self.operators = {**self.unitaries, **self.dissipators}

    def _check_name_availabilty(self, name):
        """
        Raises ValueError if ``name`` is already used by a unitary or dissipator.
        """
        if name in self.unitaries.keys():
            raise ValueError("There already exists a unitary with name " + name + "!")
        if name in self.dissipators.keys():
            raise ValueError("There already exists a dissipator with name " + name + "!")
        if name in self.operators.keys():
            raise ValueError("There already exists an operator with name " + name + ", that has not been added"
                                                                                    " using the appropriate methods!")

    def add_unitary(self, name, omega):
        self._check_name_availabilty(name)
        self.unitaries[name] = omega
        self._update_operators()

    def add_dissipator(self, name, omega):
        self._check_name_availabilty(name)
        self.dissipators[name] = omega
        self._update_operators()

    #@partial(jax.vmap, in_axes=(None, None, 0))
    @functools.partial(jax.vmap, in_axes=(None, None, 0))
    def _evaluate_observable(self, obs, idx):
        return obs[idx]

    def evaluate_observable(self, operator, states):
        """
        Obtain X, Y and Z magnetizations and their correlators up to the specified length in ``system data["L"]``.
        Note that this function assumes translational invariance and averages the results obtained for single spins.
        """

        if self.system_data["dim"] == "1D":
            resPerSamplePerSpin = self._evaluate_magnetization_pmapd(operator, states)

            correlators = {}
            for corrLen in np.arange(self.system_data["L"]):
                corrPerSamplePerSpin = self._evaluate_correlators_pmapd(resPerSamplePerSpin, corrLen)
                correlators[corrLen] = self._spin_average_pmapd(corrPerSamplePerSpin)
            return self._spin_average_pmapd(resPerSamplePerSpin), correlators
        else:
            resPerSamplePerSpin = self._evaluate_magnetization_pmapd(operator, states.reshape(1, -1, self.system_data["L"]**2)).reshape(1, -1, self.system_data["L"], self.system_data["L"])
            correlators = {}
            for corrLen in np.arange(self.system_data["L"]):
                corrPerSamplePerSpin = self._evaluate_correlators_pmapd(resPerSamplePerSpin, corrLen)
                correlators[corrLen] = self._spin_average_pmapd(corrPerSamplePerSpin.reshape(1, -1, self.system_data["L"]**2))
            return self._spin_average_pmapd(resPerSamplePerSpin.reshape(1, -1, self.system_data["L"]**2)), correlators


class POVMOperator(Operator):
    """This class provides functionality to compute operator matrix elements.

    Initializer arguments:

        * ``povm``: An instance of the POVM-class.
        * ``lDim``: Dimension of local Hilbert space.
    """

    def __init__(self, povm, ldim=4):
        """Initialize ``Operator``.
        """
        self.povm = povm
        self.ldim = ldim
        self.ops = []
        super().__init__()

    def add(self, opDescr):
        """Add another operator to the operator.

        Args:
            * ``opDescr``: Operator dictionary to be added to the operator.
        """
        id = len(self.ops) + 1
        opDescr["id"] = id
        self.ops.append(opDescr)
        self.compiled = False

    def _get_s_primes(self, s, *args):
        def apply_on_singleSiteCoupling(s, stateCouplings, matEls, siteCoupling, max_int_size):
            stateIndices = tuple(s[siteCoupling])
            OffdConfig = jnp.vstack([s] * 4**max_int_size)
            OffdConfig = OffdConfig.at[:, siteCoupling].set(stateCouplings[stateIndices].reshape(4**max_int_size, max_int_size))
            return OffdConfig.reshape((4,) * max_int_size + (-1,)), matEls[stateIndices]

        OffdConfigs, matEls = jax.vmap(apply_on_singleSiteCoupling, in_axes=(None, None, 0, 0, None))(s.reshape(-1),
                                                                                                      self.stateCouplings,
                                                                                                      self.matEls,
                                                                                                      self.siteCouplings,
                                                                                                      self.max_int_size)
        return OffdConfigs.reshape((-1,) + s.shape), matEls.reshape(-1)

    def compile(self):
        """Compiles an operator mapping function from the previously added dictionaries.
        """
        self.siteCouplings = []
        self.matEls = []

        # Get maximum interaction size (max_int_size)
        self.max_int_size = max([len(op["sites"]) for op in self.ops])

        self.idxbase = jnp.array(list(itertools.product([0, 1, 2, 3],
                                                        repeat=self.max_int_size))).reshape((4,)*self.max_int_size+(self.max_int_size,))
        self.stateCouplings = jnp.tile(self.idxbase, (4,)*self.max_int_size + (1,)*self.max_int_size + (1,))

        # Find the highest index of the (many) local Hilbert spaces
        self.max_site = max([max(op["sites"]) for op in self.ops])

        # loop over all local Hilbert spaces
        for idx in range(self.max_site + 1):
            # Sort interactions that involve the current local Hilbert space in the 0-th index according to number of
            # sites
            ops_ordered = [[op for op in self.ops if op["sites"][0] == idx and len(op["sites"]) == (n + 1)]
                           for n in range(self.max_int_size)]

            for n in range(self.max_int_size - 1, -1, -1):
                all_indices = set(op["sites"] for op in ops_ordered[n])
                for i, indices in enumerate(all_indices):
                    # Search for operators acting on the same indices, also operators acting on fewer sites are
                    # accounted for here
                    ops_same_indices = [[op for op in ops_ordered[_n] if op["sites"] == indices[:_n+1]]
                                        for _n in range(n+1)]
                    used_op_ids = [[op["id"] for op in ops_same_indices[_n]] for _n in range(n+1)]

                    # Add contribution of all operators in ops_same_indices, if necessary multiply unity interaction
                    # to additional sites to make them `max_int_size`-body interactions
                    neighbour_op_comp = jnp.zeros((4**self.max_int_size, 4**self.max_int_size), dtype=opDtype)
                    for j, ops in enumerate(ops_same_indices):
                        for op in ops:
                            op_matrix = self.povm.operators[op["name"]]
                            for _ in range(self.max_int_size - j - 1):
                                op_matrix = jnp.kron(op_matrix, jnp.eye(4, dtype=opDtype))
                            neighbour_op_comp += op["strength"] * op_matrix

                    # Avoid counting operators multiple times
                    ops_ordered = [[op for op in ops_ordered[k] if op["id"] not in used_op_ids[k]]
                                   for k in range(self.max_int_size)]

                    while len(indices) < self.max_int_size:
                        empty_idx = (indices[-1] + 1) % (self.max_site + 1)
                        while empty_idx in indices:
                            empty_idx = (empty_idx + 1) % (self.max_site + 1)
                        indices += (empty_idx,)
                    self.matEls.append(neighbour_op_comp.reshape((4,) * 2 * self.max_int_size))
                    self.siteCouplings.append(indices)

        self.siteCouplings = jnp.array(self.siteCouplings)
        self.matEls = jnp.array(self.matEls)
        return self._get_s_primes
