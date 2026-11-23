from functools import cached_property, partial

import jax
import jax.numpy as jnp

from jVMC import global_defs
from jVMC.operator.base import AbstractOperator
from jVMC.sampler.base import AbstractSampler
from jVMC.sharding_config import (
    DEVICE_SHARDING,
    DEVICE_SPEC,
    MESH,
    distribute,
)
from jVMC.stats import SampledObs
from jVMC.vqs import NQS


class ExactSampler(AbstractSampler):
    """
    Class for full enumeration of basis states.

    This class generates a full basis of the many-body Hilbert space. Thereby, it \
    allows to exactly perform sums over the full Hilbert space instead of stochastic \
    sampling.

    The result of ``sample()`` is stored and is not refreshed when ``psi.parameters`` changes: \
    call ``sample()`` again after updating the parameters.

    .. warning::
        The full basis cannot be sharded evenly unless its size is divisible by the number of \
        devices. Otherwise ``sample()`` raises a ``ValueError``.

    Initialization arguments:
        * ``psi``: Variational state defining the probability distribution.
        * ``l_dim``: Local Hilbert space dimension.
        * ``log_prob_factor``: Factor for the log-probabilities, equivalent to the exponent for the \
          probability distribution. For pure wave functions this should be 0.5, and 1.0 for POVMs.
    """

    def __init__(self, psi: NQS, l_dim=2, log_prob_factor=0.5):
        super().__init__(psi)

        self._l_dim = l_dim
        self._log_prob_factor = log_prob_factor
        self._last_norm = 0.0
        self.num_samples = self.num_states

        self.get_probabilities = jax.jit(
            lambda log_psi, last_norm: jnp.exp(
                jnp.real(log_psi - last_norm) / self.log_prob_factor
            )
        )

    @property
    def num_sites(self):
        """Number of degrees of freedom of a configuration."""
        return jnp.prod(jnp.asarray(self.sample_shape))

    @property
    def num_states(self):
        """Dimension of the Hilbert space, :math:`\\text{l_dim}^{\\text{num\\_sites}}`."""
        return self.l_dim**self.num_sites

    @property
    def l_dim(self):
        """Local Hilbert space dimension."""
        return self._l_dim

    @property
    def log_prob_factor(self):
        """Factor for the log-probabilities."""
        return self._log_prob_factor

    @cached_property
    def basis(self):
        """
        All basis configurations, of shape ``(num_states, *sample_shape)``.

        Configurations are in counting order: the ``i``-th one holds the digits of ``i`` in base \
        ``l_dim``, most significant digit first.
        """
        adjusted_dof = distribute(self.num_states)
        int_repr = jax.device_put(
            jnp.arange(adjusted_dof, dtype=global_defs.DT_SAMPLES), DEVICE_SHARDING
        )

        def get_basis(int_repr, n_sites):
            def make_state(int_repr, n_sites):
                def scan_fun(c, x):
                    loc_state = c % self.l_dim
                    c = (c - loc_state) // self.l_dim

                    return c, loc_state

                _, state = jax.lax.scan(scan_fun, int_repr, jnp.arange(n_sites))

                return state[::-1].reshape(self.psi.sample_shape)

            basis = jax.vmap(make_state, in_axes=(0, None))(int_repr, n_sites)

            return basis

        return jax.jit(
            jax.shard_map(
                partial(get_basis, n_sites=self.num_sites),
                mesh=MESH,
                in_specs=DEVICE_SPEC,
                out_specs=DEVICE_SPEC,
            )
        )(int_repr)[: self.num_states]

    def __call__(self, observable: AbstractOperator, **obs_kwargs) -> SampledObs:
        """
        Evaluate an observable exactly on the full basis.

        This uses the result of the last ``sample()`` call, or calls it if there is none.

        Args:
            * ``observable``: Operator to evaluate.
            * ``**obs_kwargs``: Additional keyword arguments for ``observable.get_O_loc()``.

        Returns:
            The local estimator of ``observable`` as a ``jVMC.stats.SampledObs``, together \
            with the probabilities of the basis configurations.
        """
        raw_data = observable.get_O_loc(
            self.samples, self.psi, log_psi_s=self.log_psi, **obs_kwargs
        )

        return SampledObs(raw_data, self.weights)

    def sample(self, num_samples=None):
        """
        Return all computational basis states.

        Sampling is automatically distributed across processes and available devices.

        Args:
            * ``num_samples``: Dummy argument to provide identical interface as the ``MCSampler`` class.

        Returns:
            ``configs, log_psi, p``: All computational basis configurations, corresponding \
            logarithmic wave function coefficients, and normalized probabilities \
            :math:`|\\psi(s)|^{1/\\text{log_prob_factor}}`. The result is also stored in ``samples``, \
            ``log_psi`` and ``weights``.
        """

        log_psi = self.psi(self.basis)
        p = self.get_probabilities(log_psi, self._last_norm)
        norm = jnp.sum(p)
        p = p / norm
        self._last_norm += self.log_prob_factor * jnp.log(norm)

        self._samples = self.basis
        self._log_psi = log_psi
        self._weights = p

        return self.basis, log_psi, p
