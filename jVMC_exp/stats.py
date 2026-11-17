from __future__ import annotations
import jax
import jax.numpy as jnp
from functools import partial, cached_property

from jVMC_exp.sharding_config import DEVICE_SHARDING, MESH, SizedIterable, pad_to_devices

@jax.jit
def _get_mean(data, weights):
    return jnp.tensordot(weights, data, axes=(0, 0))

@jax.jit
def _get_var(norm_data):
    return jnp.sum(jnp.abs(norm_data)**2, axis=0)

@jax.jit
def _get_error_of_mean(var, weights):
    return jnp.sqrt(var * jnp.sum(weights ** 2))

@jax.jit
def _center(data, mean):
    return data - mean

@jax.jit
def _normalize(data, weights, mean):
    return jnp.einsum("i, i... -> i...", jnp.sqrt(weights), data - mean)

@jax.jit
def _inv_normalize(data, inv_weights, mean, mask):
    raw = jnp.einsum("i, i... -> i...", jnp.sqrt(inv_weights), data) + mean
    return jnp.einsum("i, i... -> i...", mask, raw)

@jax.jit
def _get_weighted_covar(data, weights, other_normalized):
    """
    Covariance of ``data`` with the normalized observations of another observable, without
    copies of ``data``: the weights and the complex conjugate are applied to the other
    observable, and a real ``data`` is never converted to complex.
    """
    v = jnp.einsum("i, i... -> i...", jnp.sqrt(weights), other_normalized)
    if jnp.iscomplexobj(data):
        return jnp.conj(jnp.tensordot(data, jnp.conj(v), axes=(0, 0))).squeeze()
    if jnp.iscomplexobj(v):
        k = v.shape[-1]
        covar = jnp.tensordot(data, jnp.concatenate([v.real, v.imag], axis=-1), axes=(0, 0))
        return (covar[..., :k] + 1j * covar[..., k:]).squeeze()

    return jnp.tensordot(data, v, axes=(0, 0)).squeeze()

@jax.jit
def _mean_and_weighted_no_center(data, weights):
    mean = jnp.tensordot(weights, data, axes=(0, 0))
    weighted = jnp.einsum("i, i... -> i...", jnp.sqrt(weights), data)
    
    return mean, weighted

@jax.jit
def _get_covar(norm_data_1, norm_data_2):
    return jnp.tensordot(jnp.conj(norm_data_1), norm_data_2, axes=(0, 0)).squeeze()

@jax.jit
def _get_self_covar(norm_data):
    return jnp.tensordot(jnp.conj(norm_data), norm_data, axes=(0, 0)).squeeze()

@jax.jit
@jax.vmap
def _get_covar_per_sample(centered_data_1, centered_data_2):
    return jnp.outer(jnp.conj(centered_data_1), centered_data_2)

@jax.jit(static_argnums=(1,))
def _apply_and_project(data, apply_fn, projection):
    return apply_fn(data) @ projection.T

@jax.jit(static_argnums=(1, 2))
@partial(jax.vmap, in_axes=(0, None, None))
def _get_autocorrelation_time(x, c, dim):
    fft = jnp.abs(jnp.fft.fft(x - jnp.mean(x), n=dim))
    correlation = jnp.fft.ifft(fft ** 2)[:x.size].real
    correlation = correlation / correlation[0]
    tau = 2 * jnp.cumsum(correlation) - 1

    m = jnp.arange(len(tau)) < c * tau
    window = jax.lax.cond(jnp.any(m), lambda: jnp.argmin(m), lambda: len(tau) - 1)

    return tau[window]

class SampledObs():
    """
    Sample statistics of an observable.

    Holds the observations :math:`O_n` of an observable on :math:`N` samples, together with \
    normalized weights :math:`w_n`, and estimates averages over the distribution of the samples:

    * mean: :math:`\\langle O\\rangle=\\sum_n w_n O_n`,
    * variance: :math:`\\mathrm{Var}(O)=\\sum_n w_n|O_n-\\langle O\\rangle|^2`,
    * covariance: \
      :math:`\\mathrm{Cov}(O,P)=\\sum_n w_n(O_n-\\langle O\\rangle)^*(P_n-\\langle P\\rangle)`.

    Samplers return the local estimator of an observable in this form, see \
    ``jVMC_exp.sampler.base.AbstractSampler.__call__``. An observation can be a scalar or an \
    array, e.g. the gradient :math:`\\partial_{\\theta_k}\\ln\\psi(s_n)` with respect to all the \
    parameters, and the statistics are computed for each of its components.

    The observations are sharded across the devices along the sample axis. If :math:`N` is not a \
    multiple of the number of devices, rows of zeros with weight zero are appended to the \
    observations and to the weights, which are then longer than :math:`N`. These padding rows do \
    not contribute to any estimate.

    To save memory, a single array holds the observations. It is converted in place into the \
    representation that a computation needs: the observations :math:`O_n`, the centered \
    observations :math:`O_n-\\langle O\\rangle`, or the normalized observations \
    :math:`\\sqrt{w_n}(O_n-\\langle O\\rangle)`. Every conversion is a pass over the data. \
    Converting the normalized observations back is exact up to rounding errors, except for the \
    samples with weight zero, which are zero in every representation from then on.

    Args:
        * ``observations``: Observations :math:`O_n`, an array with a leading sample axis of \
          length :math:`N`. An array of shape ``(N,)`` is stored with shape ``(N, 1)``.
        * ``weights``: Weights :math:`w_n`, an array of shape ``(N,)``. They are normalized to \
          :math:`\\sum_n w_n=1`. If ``None``, every sample gets the weight :math:`1/N`.
    """
    def __init__(self, observations, weights=None):
        if len(observations.shape) == 1:
            self._num_obs = 1
            observations = observations.reshape((-1, 1))
        else:
            self._num_obs = jnp.prod(jnp.array(observations.shape[1:]))
        self._num_samples = observations.shape[0]
        num_devices = MESH.shape["devices"]

        if weights is None:
            weights = jnp.full(self._num_samples, 1.0 / self._num_samples)
        elif weights.shape != (self._num_samples,):
            raise ValueError(f"Weights must have shape ({self._num_samples},), got {weights.shape}")
        weights /= jnp.sum(weights)

        remainder = self._num_samples % num_devices
        if remainder != 0:
            num_pad = num_devices - remainder
            pad = ((0, num_pad),) + ((0, 0),) * (observations.ndim - 1)
            observations = jnp.pad(observations, pad, mode='constant')
            weights = jnp.pad(weights, (0, num_pad), constant_values=0)
        inv_weights = jnp.where(weights != 0, 1 / weights, weights)
        zero_mask = jnp.where(weights != 0, 1, 0)

        self._weights = jax.device_put(weights, DEVICE_SHARDING)
        self._inv_weights = jax.device_put(inv_weights, DEVICE_SHARDING)
        self._zero_mask = jax.device_put(zero_mask, DEVICE_SHARDING)
        self._observations = jax.device_put(observations, DEVICE_SHARDING)

        self._state = "o"
        self._mean = _get_mean(self._observations, self.weights)

    def __repr__(self):
        return self.__str__()
    
    def __str__(self):
        """
        ``mean ± error_of_mean (Var = var)`` for an observation with one component, and the \
        number of components otherwise.
        """
        if self._num_obs == 1:
            return f"{self.mean.item():.4e} ± {self.error_of_mean.item():.4e} (Var = {self.var.item():.4e})"
        else:
            return f"SampledObs with {self._num_obs} features"

    @property
    def observations(self):
        """
        Observations :math:`O_n`, an array of shape ``(N_pad, ...)``, where ``N_pad`` is :math:`N` \
        rounded up to a multiple of the number of devices.
        """
        if self._state == "c":
            self._observations = _normalize(self._observations, self._zero_mask, -self.mean)
        elif self._state == "n":
            self._observations = _inv_normalize(
                self._observations, self._inv_weights, self.mean, self._zero_mask
            )
        self._state = "o"

        return self._observations

    @property
    def _centered_obs(self):
        """Centered observations :math:`O_n-\\langle O\\rangle`, zero for samples with weight zero."""
        if self._state == "o":
            self._observations = _normalize(self._observations, self._zero_mask, self.mean)
        elif self._state == "n":
            self._observations = _normalize(self._observations, self._inv_weights, 0)
        self._state = "c"

        return self._observations
    
    @property
    def _normalized_obs(self):
        """Normalized observations :math:`\\sqrt{w_n}(O_n-\\langle O\\rangle)`."""
        if self._state == "o":
            self._observations = _normalize(self._observations, self.weights, self.mean)
        elif self._state == "c":
            self._observations = _normalize(self._observations, self.weights, 0)
        self._state = "n"

        return self._observations
    
    @property
    def weights(self):
        """Normalized weights :math:`w_n`, of shape ``(N_pad,)``, zero for the padding rows."""
        return self._weights

    @property
    def effective_num_samples(self):
        """
        Kish effective sample size :math:`1/\\sum_n w_n^2`.

        Equals the number of samples for uniform weights and is the quantity \
        that replaces it in Monte-Carlo error estimates when the samples carry \
        importance weights (e.g. from ``CutoffSampler`` or ``mu != 2``).
        """
        return 1.0 / jnp.sum(self.weights ** 2)

    @property
    def mean(self):
        """Mean :math:`\\langle O\\rangle=\\sum_n w_n O_n`, with the shape of one observation."""
        return self._mean

    @property
    def var(self):
        """
        Variance :math:`\\mathrm{Var}(O)=\\sum_n w_n|O_n-\\langle O\\rangle|^2` of each component, \
        with the shape of one observation. It is real, and the sum is not rescaled by \
        :math:`N/(N-1)`.
        """
        return _get_var(self._normalized_obs)

    @property
    def error_of_mean(self):
        """
        Statistical error of the mean, :math:`\\sqrt{\\mathrm{Var}(O)\\sum_n w_n^2}`, for each \
        component.

        For uniform weights this is :math:`\\sqrt{\\mathrm{Var}(O)/N}`. The estimate treats the \
        samples as independent: for samples of Markov chains, the error is larger by a factor of \
        about :math:`\\sqrt{\\tau}`, with the integrated autocorrelation time :math:`\\tau` of \
        ``get_autocorrelation_time()``.
        """
        return _get_error_of_mean(self.var, self.weights)

    def get_covar(self, other: SampledObs | None = None):
        """
        Covariance of the observations with those of ``other``.

        Computes \
        :math:`\\mathrm{Cov}(O_k,P_l)=\\sum_n w_n(O_{n,k}-\\langle O_k\\rangle)^*(P_{n,l}-\\langle P_l\\rangle)` \
        for all the components :math:`k` of :math:`O` and :math:`l` of :math:`P`, with :math:`P=O` \
        if ``other`` is ``None``. For the gradients :math:`O_{n,k}=\\partial_{\\theta_k}\\ln\\psi(s_n)`, \
        for example, ``get_covar()`` is the quantum geometric tensor, and ``get_covar(energy)`` the \
        force, for the local energy ``energy``.

        Args:
            * ``other`` [optional]: Another ``SampledObs`` on the same samples, with the same \
              weights.

        Returns:
            An array of shape ``(*shape_O, *shape_P)``, where ``shape_O`` and ``shape_P`` are the \
            shapes of one observation, with the axes of length one removed.
        """
        if other is None:
            return _get_self_covar(self._normalized_obs)
        
        return _get_weighted_covar(self.observations, self.weights, other._normalized_obs)

    def get_covar_obs(self, other: SampledObs | None = None) -> SampledObs:
        """
        Covariance with ``other`` as a ``SampledObs``, e.g. to estimate its statistical error.

        The observations of the result are the products \
        :math:`(O_{n,k}-\\langle O_k\\rangle)^*(P_{n,l}-\\langle P_l\\rangle)` of each sample, with \
        the weights of this ``SampledObs``. Its ``mean`` is ``get_covar(other)``, up to axes of \
        length one, and its ``error_of_mean`` is the statistical error of the covariance.

        Args:
            * ``other`` [optional]: Another ``SampledObs`` on the same samples, with the same \
              weights. If ``None``, :math:`P=O`.

        Returns:
            A ``SampledObs`` whose observations have shape ``(K, L)``, for ``K`` components of \
            :math:`O` and ``L`` components of :math:`P`, or ``(K,)`` if ``L`` is one.
        """
        other_num_obs = self._num_obs if other is None else other._num_obs
        if other is None:
            centered = self._centered_obs
            covar_per_sample = _get_covar_per_sample(centered, centered)
        else:
            covar_per_sample = _get_covar_per_sample(self._centered_obs, other._centered_obs)
        if other_num_obs == 1:
            covar_per_sample = covar_per_sample[..., 0]

        return SampledObs(covar_per_sample, self.weights)
    
    def get_R_hat(self, n_chains):
        """
        Gelman-Rubin convergence diagnostic :math:`\\hat R` of Markov chains.

        The samples are split into ``n_chains`` chains of :math:`n` consecutive samples each, the \
        order in which the MCMC samplers store the samples of their chains. The variances within \
        the chains (normalized by :math:`n-1`) are averaged into :math:`W`, and the variance of the \
        means of the chains (normalized by ``n_chains - 1``) is :math:`B/n`. Then

        .. math::

            \\hat R=\\sqrt{\\frac{(n-1)W/n+B/n}{W}}.

        If the chains have converged to the same distribution, :math:`\\hat R` is close to one. \
        Convergence is typically assumed when :math:`\\hat R<1.01`. Every sample counts with the \
        same weight: the weights are not used.

        Args:
            * ``n_chains``: Number of chains, e.g. ``sampler.numChains``. The number of samples \
              :math:`N` must be a multiple of it.

        Returns:
            :math:`\\hat R` as a scalar. Raises ``NotImplementedError`` if an observation has more \
            than one component.
        """
        if self._num_obs != 1:
            raise NotImplementedError

        chain_length = self._num_samples // n_chains
        chain_obs = SampledObs(
            self.observations[:self._num_samples].reshape((n_chains, chain_length)).T
        )

        B = jnp.var(chain_obs.mean, ddof=1)
        W = jnp.mean(chain_obs.var * chain_length / (chain_length - 1))
        
        return jnp.sqrt(((chain_length - 1) / chain_length * W + B) / W)
    
    def get_autocorrelation_time(self, n_chains, c=5) -> SampledObs:
        """
        Integrated autocorrelation time :math:`\\tau` of each Markov chain.

        The samples are split into chains as in ``get_R_hat()``. For each chain, the normalized \
        autocorrelation function :math:`\\rho(t)` is computed with a fast Fourier transform, and \
        :math:`\\tau(M)=1+2\\sum_{t=1}^{M}\\rho(t)` is evaluated at the smallest window :math:`M` \
        with :math:`M\\geq c\\,\\tau(M)` (automatic windowing of Sokal, as in ``emcee``). The \
        statistical error of the mean of correlated samples is larger than ``error_of_mean`` by a \
        factor of about :math:`\\sqrt{\\tau}`. Every sample counts with the same weight: the \
        weights are not used.

        Args:
            * ``n_chains``: Number of chains, e.g. ``sampler.numChains``. The number of samples \
              :math:`N` must be a multiple of it.
            * ``c``: Window parameter :math:`c`.

        Returns:
            A ``SampledObs`` with the autocorrelation time of each chain, whose ``mean`` is the \
            average over the chains. Raises ``NotImplementedError`` if an observation has more \
            than one component.
        """
        if self._num_obs != 1:
            raise NotImplementedError

        chain_length = self._num_samples // n_chains
        obs = self.observations[:self._num_samples].reshape((n_chains, chain_length))
        z = obs - jnp.mean(obs, axis=-1)[:, None]
        dim = 2 * 2 ** int(jnp.ceil(jnp.log2(chain_length)))
        
        return SampledObs(_get_autocorrelation_time(z, c, dim))
    
    def transform(self, element_wise_fn=lambda x: x, linear_map=None) -> SampledObs:
        """
        Transform the observations in place, :math:`O_n\\to L\\,f(O_n)`, and update the mean.

        The function :math:`f` is compiled with ``jax.jit``. It gets the whole array \
        ``observations``, padding rows included, and must return an array with the same leading \
        axis, transforming every sample independently of the others, e.g. ``jax.vmap(g)`` or \
        ``lambda x: x[:, :k]``. JAX reuses the compiled code only when the same function object \
        is passed again, so a function defined once compiles once, while a new ``lambda`` at every \
        call compiles at every call.

        Args:
            * ``element_wise_fn``: The function :math:`f`, the identity by default.
            * ``linear_map`` [optional]: Matrix :math:`L` of shape ``(K', K)``, applied to the \
              transformed observation :math:`f(O_n)` of every sample, which then has ``K`` \
              components.
        """
        if linear_map is not None:
            self._observations = _apply_and_project(self.observations, element_wise_fn, linear_map)
        else:
            self._observations = jax.jit(element_wise_fn)(self.observations)
        self._mean = _get_mean(self._observations, self.weights)

    def select(self, idx):
        """
        Components ``idx`` of the observations, as a new ``SampledObs`` with the same weights.

        Args:
            * ``idx``: Index along the first axis of an observation: an integer, a slice or an \
              array of indices.
        """
        return SampledObs(self.observations[:, idx], self.weights)
    
    def get_subset(self, start=None, end=None, step=None) -> SampledObs:
        """
        Samples ``observations[start:end:step]``, as a new ``SampledObs`` with the weights \
        normalized on the subset.

        The indices refer to ``observations`` and ``weights``, which end with the padding rows.

        Args:
            * ``start``: First sample of the subset.
            * ``end``: End of the subset, excluded.
            * ``step``: Step between the samples of the subset.
        """
        sl = slice(start, end, step)
        new_weights = self.weights[sl]

        return SampledObs(self.observations[sl], new_weights / jnp.sum(new_weights))

class LazySampledObs():
    """
    Sample statistics of observations that are computed one batch at a time.

    The counterpart of ``SampledObs`` for observations that are too large to be held in memory \
    at once, such as the gradients of a large network, which ``NQS.lazy_gradients()`` computes in \
    batches (objective functions with ``batched_jacobian=True``). The estimates are defined as in \
    ``SampledObs``. Each computation iterates over the batches, which computes them again; \
    ``mean`` and ``var`` are cached until ``observations`` is replaced.

    Args:
        * ``observations``: ``jVMC_exp.sharding_config.SizedIterable`` of the batches of the \
          observations, with a ``layout``, as returned by ``sharded(..., yield_iter=True)``. \
          The layout defines which samples make up each batch, and splits ``weights``, or any \
          other array with one entry per sample, into batches that line up with those of the \
          observations. On more than one device a batch does not consist of consecutive \
          samples, see ``jVMC_exp.sharding_config.BatchLayout``.
        * ``weights``: Weights :math:`w_n`, an array of shape ``(N,)``, normalized to \
          :math:`\\sum_n w_n=1`. Like the observations, they are padded with zeros to a multiple \
          of the number of devices.

    Raises ``ValueError`` if ``weights`` is ``None``, if ``observations`` has no layout, or if the \
    number of weights or of batches does not match the layout.
    """
    def __init__(self, observations: SizedIterable, weights):
        if weights is None:
            raise ValueError("LazySampledObs require weights to be an array")
        if not isinstance(observations, SizedIterable) or observations.layout is None:
            raise ValueError("Observations must be a SizedIterable with a BatchLayout")
        self._layout = observations.layout
        
        # Like the observations, the weights are padded to a multiple of the number of devices.
        # The padding samples get zero weight and do not contribute to any estimate.
        num_padded = len(pad_to_devices(weights))
        if num_padded != self._layout.num_samples or len(observations) != self._layout.n_batches:
            raise ValueError(
                f"Got {len(weights)} weights and {len(observations)} batches of observations, "
                f"but the layout describes {self._layout.num_samples} samples "
                f"in {self._layout.n_batches} batches."
            )
        weights /= jnp.sum(weights)
        self._num_samples = len(weights)
        self._batch_size = observations.batch_size

        self._full_weights = jax.device_put(pad_to_devices(weights), DEVICE_SHARDING)
        self._weights = self._layout.split(self._full_weights)
        self.observations = observations

    @property
    def observations(self):
        """
        ``SizedIterable`` of the batches of the observations. Assigning a new one clears the \
        cached ``mean`` and ``var``.
        """
        return self._observations

    @observations.setter
    def observations(self, value):
        if not isinstance(value, SizedIterable):
            raise ValueError(
                "Observations must be an instance jVMC_exp.sharding_config.SizedIterable"
        )

        self._observations = value

        # Clear cached properties
        for name, attr in type(self).__dict__.items():
            if isinstance(attr, cached_property):
                self.__dict__.pop(name, None)
    
    @property
    def layout(self):
        """``jVMC_exp.sharding_config.BatchLayout`` of the batches."""
        return self._layout

    @property
    def weights(self):
        """
        Normalized weights :math:`w_n` of all the samples, of shape ``(N_pad,)``, zero for the \
        padding rows.
        """
        return self._full_weights

    @cached_property
    def mean(self):
        """Mean :math:`\\langle O\\rangle=\\sum_n w_n O_n`, with the shape of one observation."""
        mean = 0
        for batch, weights in zip(self._observations, self._weights):
            mean += _get_mean(batch, weights)
        
        return mean

    @cached_property
    def var(self):
        """
        Variance :math:`\\mathrm{Var}(O)=\\sum_n w_n|O_n|^2-|\\langle O\\rangle|^2` of each \
        component, computed in a single pass over the batches, with the axes of length one removed.
        """
        sq_sum = 0
        mean = 0
        for batch, weights in zip(self._observations, self._weights):
            sq_sum += jnp.tensordot(weights, jnp.abs(batch) ** 2, axes=(0, 0))
            mean += _get_mean(batch, weights)

        return (sq_sum - jnp.abs(mean) ** 2).squeeze()
    
    @property
    def error_of_mean(self):
        """
        Statistical error of the mean, :math:`\\sqrt{\\mathrm{Var}(O)\\sum_n w_n^2}`, as in \
        ``SampledObs``.
        """
        return _get_error_of_mean(self.var, self.weights)

    def get_covar(self, other: SampledObs | LazySampledObs | None = None):
        """
        Covariance of the observations with those of ``other``, defined as in \
        ``SampledObs.get_covar()``.

        With ``other=None`` or a ``LazySampledObs``, it is computed in a single pass over the \
        batches as :math:`\\sum_n w_n O_{n,k}^*P_{n,l}-\\langle O_k\\rangle^*\\langle P_l\\rangle`.

        Args:
            * ``other`` [optional]: A ``SampledObs`` or a ``LazySampledObs`` on the same samples, \
              with the same weights. A ``LazySampledObs`` must have the same ``layout``, or a \
              ``ValueError`` is raised. If ``None``, :math:`P=O`. Any other type raises \
              ``NotImplementedError``.

        Returns:
            An array of shape ``(*shape_O, *shape_P)``, where ``shape_O`` and ``shape_P`` are the \
            shapes of one observation, with the axes of length one removed.
        """
        covar = 0
        if other is None:
            mean = 0
            for batch, weights in zip(self._observations, self._weights):
                batch_mean, batch = _mean_and_weighted_no_center(batch, weights)
                mean += batch_mean
                covar += _get_covar(batch, batch)

            return (covar - jnp.tensordot(jnp.conj(mean), mean, axes=0)).squeeze()

        elif isinstance(other, SampledObs):
            normalized_obs_other = self._layout.split(other._normalized_obs)

            for batch, weights, batch_other in zip(self._observations, self._weights, normalized_obs_other):
                covar += _get_weighted_covar(batch, weights, batch_other)

            return covar.squeeze()

        elif isinstance(other, LazySampledObs):
            if other._layout != self._layout:
                raise ValueError(
                    f"Both LazySampledObs must be batched with the same layout, "
                    f"got {self._layout} and {other._layout}"
                )
            mean_1 = 0
            mean_2 = 0
            for batch_1, weights_1, batch_2, weights_2 in zip(self._observations, self._weights, other._observations, other._weights):
                m1, batch_1 = _mean_and_weighted_no_center(batch_1, weights_1)
                m2, batch_2 = _mean_and_weighted_no_center(batch_2, weights_2)
                mean_1 += m1
                mean_2 += m2
                covar += _get_covar(batch_1, batch_2)

            return (covar - jnp.tensordot(jnp.conj(mean_1), mean_2, axes=0)).squeeze()
        
        else:
            raise NotImplementedError(
                "Can only compute the variance with a SampledObs or a LazySampledObs, "
                f"got {other}"
            )

    def transform(self, element_wise_fn=lambda x: x, linear_map=None) -> LazySampledObs:
        """
        Transform the observations, :math:`O_n\\to f(O_n)`.

        The function :math:`f` is compiled with ``jax.jit`` and applied to every batch whenever \
        the observations are iterated, after the transformations applied before. It must return \
        an array with the same leading axis, transforming every sample independently of the \
        others, as in ``SampledObs.transform()``. The cached ``mean`` and ``var`` are cleared.

        Args:
            * ``element_wise_fn``: The function :math:`f`, the identity by default.
            * ``linear_map``: Not supported: a linear map raises ``NotImplementedError``, since it \
              needs all the observations at once.
        """
        if linear_map is not None:
            raise NotImplementedError(
                "A linear map can't be applied without materializing all the observables"
            )

        jitted_fn = jax.jit(element_wise_fn)
        iterable = self._observations
        self.observations = SizedIterable(
            lambda: map(jitted_fn, iterable), 
            iterable.n_iterations, 
            iterable.batch_size, 
            iterable.layout
        )
