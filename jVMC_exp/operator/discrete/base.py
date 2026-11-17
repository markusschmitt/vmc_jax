from __future__ import annotations
from abc import abstractmethod
import jax
import jax.numpy as jnp
from scipy.sparse import coo_matrix

from jVMC_exp.vqs import NQS
from jVMC_exp.sharding_config import sharded, MESH
from jVMC_exp.operator.base import AbstractOperator

class Operator(AbstractOperator):
    """
    Base class of the operators acting on discrete computational basis configurations.

    Operators are combined with ``+``, ``-``, ``*`` and ``/``. A product of two operators applies \
    the right factor first, as for matrices. Operators can be multiplied by numbers, or by \
    callables that accept arbitrary keyword arguments, e.g., ``lambda **kw: jnp.cos(kw["t"])``. \
    These are evaluated with the keyword arguments given to ``get_O_loc()``, \
    ``get_conn_elements()`` or ``to_sparse()``, which allows time-dependent prefactors. Adding a \
    number is not supported: add a multiple of ``IdentityOperator`` instead. ``sum()`` of a list of \
    operators works.

    The operator is compiled into arrays when it is first used.

    Args:
        * ``ldim``: Local Hilbert space dimension.
    """
    def __init__(self, ldim):
        self._ldim = ldim
        self._is_compiled = False
        self._scale = 1

    @property
    def ldim(self):
        """Local Hilbert space dimension of an elementary operator, ``None`` for combined operators."""
        return self._ldim
    
    def __add__(self, other) -> Operator:
        if isinstance(other, (int, float, complex)):
            # TODO: Since I don't know the total dim of the Hilbert space this is not doable
            raise NotImplementedError 
        elif isinstance(other, Operator):
            return self._create_composite(self, other, 'sum')
        else:
            return NotImplemented
        
    def __radd__(self, other) -> Operator:
        if isinstance(other, (int, float, complex)):
            if other == 0:
                return self
            # TODO: Same as previous todo
            raise NotImplementedError
        else:
            return NotImplemented
        
    def __neg__(self) -> Operator:
        return self._create_scaled(self, -1)
    
    def __sub__(self, other) -> Operator:
        if isinstance(other, (int, float, complex)):
            # TODO: Same as previous todo
            raise NotImplementedError
        elif isinstance(other, Operator):
            return self._create_composite(self, -other, 'sum')
        else:
            return NotImplemented
        
    def __rsub__(self, other) -> Operator:
        if isinstance(other, (int, float, complex)):
            # TODO: Same as previous todo
            raise NotImplementedError
        else:
            return NotImplemented
        
    def __mul__(self, other) -> Operator:
        if isinstance(other, (int, float, complex)) or callable(other):
            return self._create_scaled(self, other)
        elif isinstance(other, Operator):
            return self._create_composite(self, other, 'mul')
        else:
            return NotImplemented
        
    def __rmul__(self, other) -> Operator:
        if isinstance(other, (int, float, complex)) or callable(other):
            return self._create_scaled(self, other)
        else:
            return NotImplemented
        
    def __truediv__(self, other) -> Operator:
        if isinstance(other, (int, float, complex)):
            if other == 0:
                raise ZeroDivisionError("Division of Operator by zero.")
            return self._create_scaled(self, 1 / other)
        else:
            return NotImplemented

    def get_O_loc(self, s, psi: NQS, *, logPsiS=None, **kwargs): 
        """
        Compute the local estimator of the operator, see ``AbstractOperator.get_O_loc()``.

        The connected configurations are evaluated with ``psi.eval_ratio()`` if the network \
        provides it, and from :math:`\\ln\\psi` otherwise. At most ``psi.batchSize`` network \
        evaluations run at once.
        """
        logPsiS = psi(s) if logPsiS is None else logPsiS

        if not self._is_compiled:
            self._compile()

        # Bound the number of simultaneous network evaluations by psi.batchSize,
        # i.e. by B = psi.batchSize / n_devices on each device:
        #   - B >= n_conn: all connected configurations of a sample fit at once,
        #     so each device processes B // n_conn samples per call.
        #   - B < n_conn: each device processes a single sample per call and
        #     evaluates its connected configurations in chunks of B (lax.map).
        evals_per_device = psi.batchSize // MESH.size
        chunk_size = max(1, min(self.n_conn, evals_per_device))
        batch_size = MESH.size * (evals_per_device // chunk_size)

        return self._get_O_loc(
            s,
            logPsiS,
            parameters=psi.eval_parameters,
            psi=psi,
            chunk_size=chunk_size,
            batch_size=batch_size,
            **kwargs
        )

    def get_conn_elements(self, s, batch_size, **kwargs):
        """
        Return the connected configurations and matrix elements of each sample in ``s``.

        The first entry along the connected axis is ``s`` itself, carrying the sum of
        all diagonal matrix elements; the remaining ones are the non-diagonal connections.

        Args:
            * ``s``: Configurations, of shape ``(N, *sampleShape)``.
            * ``batch_size``: Number of configurations processed at once, a multiple of the \
              number of devices.
            * ``**kwargs``: Keyword arguments of the callable prefactors.

        Returns:
            ``s_primes, mat_els``: The connected configurations, of shape \
            ``(N, n_conn + 1, *sampleShape)``, and the matrix elements, of shape \
            ``(N, n_conn + 1)``.
        """
        if not self._is_compiled:
            self._compile()

        return self._get_conn_elements_sh(s, batch_size=batch_size, **kwargs)

    def to_dense(self, basis, *, zero_tolerance=0.0, **op_kwargs):
        """
        Return the dense matrix representation of this operator in the supplied ordered
        computational basis, with the arguments of ``to_sparse()``.
        """
        return self.to_sparse(basis, zero_tolerance=zero_tolerance, **op_kwargs).todense()

    def to_sparse(self, basis, *, zero_tolerance=0.0, **op_kwargs):
        """
        Return the SciPy sparse matrix representation of this operator
        in the supplied ordered computational basis.

        The entry in row ``i`` and column ``j`` is the matrix element \
        :math:`\\langle b_i|\\hat O|b_j\\rangle` between the basis configurations ``b_i`` and ``b_j``.

        Args:
            * ``basis``: Ordered computational basis with shape ``(num_states, *sampleShape)``. \
              The basis must be closed under every nonzero action of the operator. Zero-weight \
              padded connections returned by the operator are ignored.
            * ``zero_tolerance``: Connections with absolute matrix element less than or equal \
              to this value are discarded.
            * ``**op_kwargs``: Keyword arguments of the callable prefactors, forwarded to \
              ``get_conn_elements()``.

        Returns:
            Sparse matrix representation of the operator, as a ``scipy.sparse.csr_matrix``.
        """
        if zero_tolerance < 0:
            raise ValueError(
                "zero_tolerance must be nonnegative. "
                f"Got {zero_tolerance}."
            )

        basis_array = jnp.asarray(basis)
        if basis_array.ndim < 2:
            raise ValueError(
                "basis must have shape (num_states, *sample_shape). "
                f"Got {basis_array.shape}."
            )

        dim = basis_array.shape[0]
        if dim == 0:
            raise ValueError(
                "Cannot construct a matrix from an empty basis. "
                f"Got {basis_array.shape}."
            )

        # SciPy and the Python lookup dictionary operate on CPU arrays
        flat_basis = jax.device_get(basis_array).reshape(dim, -1)
        basis_index = {
            tuple(configuration.tolist()): index
            for index, configuration in enumerate(flat_basis)
        }
        if len(basis_index) != dim:
            raise ValueError(
                "The supplied basis contains duplicate configurations."
            )

        n_devices = jax.device_count()
        s_primes, mat_els = self.get_conn_elements(
            basis_array,
            ((dim + n_devices - 1) // n_devices) * n_devices,
            **op_kwargs,
        )
        s_primes_host = jax.device_get(s_primes)
        mat_els_host = jax.device_get(mat_els)
        flat_s_primes = s_primes_host.reshape(-1, flat_basis.shape[1])
        flat_mat_els = mat_els_host.reshape(-1)

        if flat_s_primes.shape[0] != flat_mat_els.shape[0]:
            raise RuntimeError(
                "Incompatible get_conn_elements output shapes: "
                f"s_primes={s_primes_host.shape}, "
                f"mat_els={mat_els_host.shape}."
            )

        if flat_mat_els.size % dim != 0:
            raise RuntimeError(
                "The number of returned matrix elements is not "
                "divisible by the basis dimension: "
                f"{flat_mat_els.size} versus {dim}."
            )

        n_conn = flat_mat_els.size // dim

        all_cols = jax.device_get(
            jnp.repeat(jnp.arange(dim), n_conn,)
        )

        rows = []
        cols = []
        data = []
        missing_connections = []
        for connected_configuration, matrix_element, col in zip(
            flat_s_primes,
            flat_mat_els,
            all_cols,
        ):
            if abs(matrix_element) <= zero_tolerance:
                continue

            configuration_key = tuple(connected_configuration.tolist())
            row = basis_index.get(configuration_key)

            if row is None:
                missing_connections.append(
                    (
                        int(col),
                        tuple(flat_basis[col].tolist()),
                        configuration_key,
                        matrix_element,
                    )
                )
                continue

            rows.append(row)
            cols.append(int(col))
            data.append(matrix_element)

        if missing_connections:
            raise ValueError(
                "The operator has nonzero connections outside "
                "the supplied basis."
            )

        matrix = coo_matrix(
            (
                jax.device_get(jnp.asarray(data)),
                (
                    jax.device_get(jnp.asarray(rows)),
                    jax.device_get(jnp.asarray(cols))
                ),
            ),
            shape=(dim, dim),
        ).tocsr()

        matrix.sum_duplicates()
        matrix.eliminate_zeros()

        return matrix

    @sharded(static_kwarg_names=("psi", "chunk_size"))
    def _get_O_loc(
        self, s, log_psi_s, *, 
        parameters, psi: NQS, chunk_size, batch_size, **kwargs
    ):
        s_p, mat_els, mat_el_diag = self._get_conn_elements(s, kwargs)

        # Purely diagonal operator: no network evaluation needed
        if s_p.shape[0] == 0:
            return mat_el_diag

        if psi.eval_ratio:
            psi_ratio = jax.lax.map(
                lambda x: psi.apply_fun(parameters, s, x, method=psi.net.eval_ratio),
                s_p, batch_size=chunk_size
            ).astype(psi.out_dtype)
        else:
            log_psi_s_p = jax.lax.map(
                lambda x: psi.apply_fun(parameters, x), s_p, batch_size=chunk_size
            ).astype(psi.out_dtype)
            psi_ratio = jnp.exp(log_psi_s_p - log_psi_s)

        return mat_el_diag + jnp.sum(psi_ratio * mat_els)

    @sharded()
    def _get_conn_elements_sh(self, s, *, batch_size, **kwargs):
        s_p, mat_els, mat_el_diag = self._get_conn_elements(s, kwargs)

        return (
            jnp.concatenate([s[None], s_p], axis=0),
            jnp.concatenate([mat_el_diag[None], mat_els], axis=0)
        )

    @property
    @abstractmethod
    def n_conn(self):
        """
        Number of non-diagonal connected configurations generated per sample.
        """
        pass

    @abstractmethod
    def _compile(self):
        """
        Compile the operator into JAX arrays for efficient computation.
        """
        pass

    @abstractmethod
    def _get_conn_elements(self, s, kwargs):
        """
        Compute the connected configurations and corresponding matrix elements
        generated by the action of an operator on a given input configuration.

        This method must return:
        (i) all non-diagonal connected configurations with shape (NumConnectedElements, SampleShape),
        (ii) their associated matrix elements with shape (NumConnectedElements,), and
        (iii) the total diagonal contribution.

        Parameters
        ----------
        s : Single input configuration 

        kwargs : Auxiliary parameters required to compute operator prefactors.

        Returns
        -------
        s_p_non_diag : Array of configurations connected to `s`.

        mat_els_non_diag : One-dimensional array of complex matrix elements corresponding
            to each configuration in `s_p_non_diag`.

        mat_els_diag : Scalar equal to the sum of all diagonal matrix elements.
        """
        pass
    
    @classmethod
    @abstractmethod
    def _create_composite(cls, O_1, O_2, label):
        """Return the sum (``label="sum"``) or product (``label="mul"``) of two operators."""
        pass

    @classmethod
    @abstractmethod
    def _create_scaled(cls, O, scalar):
        """Return the operator ``O`` multiplied by ``scalar``."""
        pass