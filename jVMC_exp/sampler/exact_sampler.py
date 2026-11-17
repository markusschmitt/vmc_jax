import jax
import jax.numpy as jnp
from functools import partial, cached_property

from jVMC_exp.vqs import NQS
from jVMC_exp.sharding_config import (
    MESH, DEVICE_SPEC, REPLICATED_SPEC, DEVICE_SHARDING, 
    distribute
)
from jVMC_exp.operator.base import AbstractOperator
from jVMC_exp.stats import SampledObs
from jVMC_exp import global_defs
from jVMC_exp.sampler.base import AbstractSampler

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
        * ``lDim``: Local Hilbert space dimension.
        * ``logProbFactor``: Factor for the log-probabilities, equivalent to the exponent for the \
          probability distribution. For pure wave functions this should be 0.5, and 1.0 for POVMs.
    """

    def __init__(self, psi: NQS, lDim=2, logProbFactor=0.5):
        super().__init__(psi)

        self._lDim = lDim
        self._logProbFactor = logProbFactor
        self._lastNorm = 0.
        self.numSamples = self.num_states

        self.get_probabilities = jax.jit(
            lambda logPsi, lastNorm : jnp.exp(
                jnp.real(logPsi - lastNorm) / self.logProbFactor
            )
        )

    @property
    def num_sites(self):
        """Number of degrees of freedom of a configuration."""
        return jnp.prod(jnp.asarray(self.sampleShape))

    @property
    def num_states(self):
        """Dimension of the Hilbert space, :math:`\\text{lDim}^{\\text{num\\_sites}}`."""
        return self.lDim ** self.num_sites

    @property
    def lDim(self):
        """Local Hilbert space dimension."""
        return self._lDim

    @property
    def logProbFactor(self):
        """Factor for the log-probabilities."""
        return self._logProbFactor

    @cached_property
    def basis(self):
        """
        All basis configurations, of shape ``(num_states, *sampleShape)``.

        Configurations are in counting order: the ``i``-th one holds the digits of ``i`` in base \
        ``lDim``, most significant digit first.
        """
        adjusted_dof = distribute(self.num_states)
        int_repr = jax.device_put(jnp.arange(adjusted_dof, dtype=global_defs.DT_SAMPLES), DEVICE_SHARDING)

        def get_basis(int_repr, n_sites):
            def make_state(int_repr, n_sites):
                def scan_fun(c, x):
                    locState = c % self.lDim
                    c = (c - locState) // self.lDim
                    
                    return c, locState
                _, state = jax.lax.scan(scan_fun, int_repr, jnp.arange(n_sites))

                return state[::-1].reshape(self.psi.sampleShape)
            basis = jax.vmap(make_state, in_axes=(0, None))(int_repr, n_sites)

            return basis

        return jax.jit(
            jax.shard_map(partial(get_basis, n_sites=self.num_sites), mesh=MESH, in_specs=DEVICE_SPEC, out_specs=DEVICE_SPEC)
        )(int_repr)[:self.num_states]
    
    def __call__(self, observable: AbstractOperator, **obs_kwargs) -> SampledObs:
        """
        Evaluate an observable exactly on the full basis.

        This uses the result of the last ``sample()`` call, or calls it if there is none.

        Args:
            * ``observable``: Operator to evaluate.
            * ``**obs_kwargs``: Additional keyword arguments for ``observable.get_O_loc()``.

        Returns:
            The local estimator of ``observable`` as a ``jVMC_exp.stats.SampledObs``, together \
            with the probabilities of the basis configurations.
        """
        raw_data = observable.get_O_loc(self.samples, self.psi, logPsiS=self.logPsi, **obs_kwargs)

        return SampledObs(raw_data, self.weights)
    
    def sample(self, numSamples=None):
        """
        Return all computational basis states.

        Sampling is automatically distributed across processes and available devices.

        Args:
            * ``numSamples``: Dummy argument to provide identical interface as the ``MCSampler`` class.

        Returns:
            ``configs, logPsi, p``: All computational basis configurations, corresponding \
            logarithmic wave function coefficients, and normalized probabilities \
            :math:`|\\psi(s)|^{1/\\text{logProbFactor}}`. The result is also stored in ``samples``, \
            ``logPsi`` and ``weights``.
        """

        logPsi = self.psi(self.basis)
        p = self.get_probabilities(logPsi, self._lastNorm)
        norm = jnp.sum(p)
        p = p / norm
        self._lastNorm += self.logProbFactor * jnp.log(norm)

        self._samples = self.basis
        self._logPsi = logPsi
        self._weights = p

        return self.basis, logPsi, p