from abc import ABC, abstractmethod
from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import random

from jVMC import global_defs
from jVMC.operator.base import AbstractOperator
from jVMC.sampler.propose import AbstractProposer
from jVMC.sharding_config import (
    DEVICE_SHARDING,
    DEVICE_SPEC,
    MESH,
    REPLICATED_SPEC,
    broadcast_split_key,
    distribute,
)
from jVMC.stats import SampledObs
from jVMC.util.key_gen import format_key
from jVMC.util.util import has_callable_attr
from jVMC.vqs import NQS


class AbstractSampler(ABC):
    """
    Interface shared by all samplers.

    A sampler is bound to a variational state and keeps the result of its last ``sample()`` call
    in ``samples``, ``log_psi`` and ``weights``. Reading one of these before the first call
    triggers it. The stored values are not refreshed when ``psi.parameters`` changes: call
    ``sample()`` again after updating the parameters.

    Args:
        * ``psi``: Variational state to sample from.
    """

    def __init__(self, psi: NQS):
        self._psi = psi
        self._samples = None
        self._log_psi = None
        self._weights = None
        self.num_samples = None

    @property
    def psi(self):
        """Variational state the sampler is bound to."""
        return self._psi

    @property
    def sample_shape(self):
        """Shape of a single configuration."""
        return self.psi.sample_shape

    @property
    def log_psi(self):
        """:math:`\\ln\\psi(s)` of the stored samples."""
        if self._samples is None:
            self.sample()

        return self._log_psi

    @property
    def weights(self):
        """Normalized weights of the stored samples."""
        if self._samples is None:
            self.sample()

        return self._weights

    @property
    def samples(self):
        """Configurations returned by the last ``sample()`` call."""
        if self._samples is None:
            self.sample()

        return self._samples

    @abstractmethod
    def __call__(self, observable: AbstractOperator, **obs_kwargs) -> SampledObs:
        """
        Evaluate an observable on the samples of the sampler.

        Args:
            * ``observable``: Operator to evaluate.
            * ``**obs_kwargs``: Additional keyword arguments for ``observable.get_O_loc()``.

        Returns:
            The local estimator of ``observable`` together with the sample weights.
        """

    @abstractmethod
    def sample(self, num_samples=None) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Sample configurations from the distribution defined by the network.

        Args:
            * ``num_samples``: Number of samples to generate in this call. If ``None``, the \
              default number of samples is used.

        Returns:
            ``configs, log_psi, p``: The sampled configurations, their logarithmic wave function \
            coefficients and their normalized weights.
        """


class AbstractMCSampler(AbstractSampler):
    """
    Base class of the samplers that draw configurations stochastically.

    This abstract class provides functionality to sample states from \
    the distribution 

        :math:`p_{\\mu}(s)=\\frac{|\\psi(s)|^{\\mu}}{\\sum_s|\\psi(s)|^{\\mu}}`.

    For :math:`\\mu=2` this corresponds to sampling from the Born distribution. \
    :math:`0\\leq\\mu<2` can be used to perform importance sampling \
    (see `[arXiv:2108.08631] <https://arxiv.org/abs/2108.08631>`_). The returned weights are \
    proportional to :math:`|\\psi(s)|^{1/\\text{log_prob_factor}-\\mu}`, so that weighted averages \
    are taken over the distribution :math:`|\\psi(s)|^{1/\\text{log_prob_factor}}`.

    If the network of ``psi`` provides a ``sample()`` method, configurations are drawn directly \
    from it and all get the same weight. In that case the network alone sets the distribution: \
    ``mu``, ``log_prob_factor`` and all the MCMC arguments have no effect on the samples. \
    Otherwise, Markov chains are run with the proposals of ``update_proposer``.

    Sampling is automatically distributed across processes and locally available devices. The \
    numbers of chains and samples are rounded up where this is needed, so a call can return \
    more samples than requested.

    Initializer arguments:
        * ``psi``: Variational state defining the probability distribution.
        * ``update_proposer``: An instance of ``jVMC.sampler.propose.AbstractProposer`` to \
          propose updates for the MCMC algorithm. Can be ``None`` only if ``psi`` samples directly.
        * ``key``: An instance of ``jax.random.PRNGKey``, or an ``int`` that will be used as seed \
          to initialize a ``PRNGKey``. If ``None``, a random seed is generated.
        * ``num_chains``: Number of Markov chains, which are run in parallel. It cannot exceed \
          ``num_samples``. It is unused when sampling directly.
        * ``num_samples``: Default number of samples to be returned by the ``sample()`` member \
          function. All chains contribute the same number of samples.
        * ``thermalization_sweeps``: Number of sweeps discarded at the start of every ``sample()`` \
          call. If ``None``, thermalization is skipped.
        * ``sweep_steps``: Number of proposed updates per sweep, i.e., between two recorded samples \
          of a chain. If ``None``, the total number of sites, ``prod(sample_shape)``, is used.
        * ``init_state``: Initial configurations of the chains, of shape \
          ``(n, *sample_shape)``. Surplus configurations are dropped and missing ones are \
          generated randomly. If ``None``, all the configurations are generated randomly.
        * ``mu``: Parameter for the distribution :math:`p_{\\mu}(s)`, see above, in the \
          interval :math:`[0, 2]`. It cannot be changed after construction.
        * ``log_prob_factor``: Factor for the log-probabilities, equivalent to the exponent for the \
          probability distribution. For pure wave functions this should be 0.5, and 1.0 for POVMs. \
          In the POVM case, the ``mu`` parameter must be set to 1.0, to sample the unchanged POVM \
          distribution.
    """

    def __init__(
        self,
        psi: NQS,
        update_proposer: None | AbstractProposer,
        key=None,
        num_chains=32,
        num_samples=128,
        thermalization_sweeps=10,
        sweep_steps=None,
        init_state=None,
        mu=2,
        log_prob_factor=0.5,
    ):
        if (not psi.is_generator) and (
            not isinstance(update_proposer, AbstractProposer)
        ):
            raise RuntimeError(
                "Instantiation of MCSampler: `update_proposer` is `None` and cannot be used for MCMC sampling. "
                "'update_proposer' must be an instance of 'jVMC.propose.AbstractProposer'."
            )
        super().__init__(psi)

        self.log_prob_factor = log_prob_factor
        if mu < 0 or mu > 2:
            raise ValueError("mu must be in the range [0, 2]")
        self._mu = mu
        self._update_proposer = update_proposer

        self.key = key

        if sweep_steps is None:
            sweep_steps = int(np.prod(self.sample_shape))
        self.sweep_steps = sweep_steps
        self.thermalization_sweeps = thermalization_sweeps
        self.num_samples = num_samples
        self.num_chains = num_chains

        self.states = init_state
        self.num_proposed = None
        self.num_accepted = None

        if has_callable_attr(self.psi.net, "eval_real"):

            def log_prob_fun(p, s):
                return self.mu * self.psi.apply_fun(
                    p, s, method=self.psi.net.eval_real
                ).astype(global_defs.DT_OUT_REAL)
        else:

            def log_prob_fun(p, s):
                return self.mu * jnp.real(self.psi.apply_fun(p, s)).astype(
                    global_defs.DT_OUT_REAL
                )

        self._log_prob_fun_jsh = jax.jit(
            jax.shard_map(
                jax.vmap(log_prob_fun, in_axes=(None, 0)),
                mesh=MESH,
                in_specs=(REPLICATED_SPEC, DEVICE_SPEC),
                out_specs=DEVICE_SPEC,
            )
        )

        if self.psi.eval_ratio:

            def get_ratio(log_prob, log_prob_correction, params, state, new_state):
                abs_ratio = jnp.abs(
                    self.psi.apply_fun(
                        params, state, new_state, method=self.psi.net.eval_ratio
                    )
                ).astype(global_defs.DT_OUT_REAL)

                return abs_ratio**self.mu * jnp.exp(log_prob_correction), log_prob
        else:

            def get_ratio(log_prob, log_prob_correction, params, state, new_state):
                new_log_prob = log_prob_fun(params, new_state)

                return jnp.exp(
                    new_log_prob - log_prob + log_prob_correction
                ), new_log_prob

        self._get_ratio = get_ratio

    @property
    def mu(self):
        """Parameter of the distribution :math:`p_{\\mu}(s)`, read-only."""
        return self._mu

    @property
    def thermalization_sweeps(self):
        """Number of sweeps discarded at the start of every ``sample()`` call."""
        return self._thermalization_sweeps

    @thermalization_sweeps.setter
    def thermalization_sweeps(self, value):
        self._thermalization_sweeps = value
        self._get_samples_jsh = {}

    @property
    def sweep_steps(self):
        """Number of proposed updates per sweep."""
        return self._sweep_steps

    @sweep_steps.setter
    def sweep_steps(self, value):
        self._sweep_steps = value
        self._get_samples_jsh = {}

    @property
    def update_proposer(self):
        """Proposer of the MCMC updates."""
        return self._update_proposer

    @property
    def num_chains(self):
        """
        Number of Markov chains.

        Setting it adapts the current chains at the next ``sample()`` call, as described for \
        ``init_state``, and initializes the proposer again.
        """
        return self._num_chains

    @num_chains.setter
    def num_chains(self, value):
        if not self.psi.is_generator and self.num_samples < value:
            raise ValueError(
                f"The provided number of chains {value} is bigger "
                f"than the number of samples {self.num_samples}."
            )

        self._num_chains = value
        self._is_state_initialized = False
        self._get_samples_jsh = {}
        self._randomize_samples_jsh = {}

    @property
    def key(self):
        """PRNG state: one key, or one key per chain once the chains are initialized."""
        return self._key

    @key.setter
    def key(self, value):
        self._key = format_key(value)
        self._is_state_initialized = False

    @property
    def states(self):
        """
        Current configurations of the Markov chains, of shape ``(num_chains, *sample_shape)``.

        This is ``init_state`` until the first ``sample()`` call. A new value is used by the next \
        ``sample()`` call, as described for ``init_state``.
        """
        return self._states

    @states.setter
    def states(self, value):
        if value is not None:
            value = jnp.array(value)
            if value.shape[1:] != self.sample_shape:
                raise ValueError(
                    f"The provided state has the wrong sample shape. "
                    f"Got {value.shape[1:]}, while sample_shape is {self.sample_shape}."
                )

        self._states = value
        self._is_state_initialized = False

    @property
    def sampler_state(self):
        """
        Snapshot of the sampler as a dictionary with ``parameters``, ``states`` and ``weights``.

        Reading it samples if there are no stored samples yet. Setting it restores these three \
        entries, but not ``samples``, ``log_psi`` or ``key``: call ``sample()`` before using \
        the stored samples again.
        """
        return dict(
            parameters=self.psi.parameters, weights=self.weights, states=self.states
        )

    @sampler_state.setter
    def sampler_state(self, value):
        self.psi.parameters = value["parameters"]
        self.states = value["states"]
        self._weights = value["weights"]

    def __call__(
        self,
        observable: AbstractOperator,
        num_samples: int | None = None,
        resample: bool = False,
        **obs_kwargs,
    ) -> SampledObs:
        """
        Evaluate an observable using Monte Carlo samples of the current variational state.

        This method draws (or reuses) samples from the underlying sampler and computes \
        the local estimator of the given observable. Resampling is triggered if no \
        samples are currently stored, if the requested number of samples differs from \
        ``num_samples``, or if ``resample=True``.

        Stored samples and :math:`\\ln\\psi(s)` are reused even if ``psi.parameters`` has \
        changed, so pass ``resample=True`` after updating the parameters. ``num_samples`` applies \
        to one call and does not change ``num_samples``: any other value resamples at every call.

        Args:
            * ``observable``: Operator to evaluate.
            * ``num_samples``: Number of Monte Carlo samples to draw. If ``None``, the stored \
              samples are reused.
            * ``resample``: If ``True``, forces resampling even if stored samples are available.
            * ``**obs_kwargs``: Additional keyword arguments for ``observable.get_O_loc()``.

        Returns:
            The local estimator of ``observable`` as a ``jVMC.stats.SampledObs``, together \
            with the sample weights.
        """
        needs_resample = (
            self.samples is None
            or (num_samples is not None and num_samples != self.num_samples)
            or resample
        )
        if needs_resample:
            self._samples, self._log_psi, self._weights = self.sample(num_samples)
        raw_data = observable.get_O_loc(
            self.samples, self.psi, log_psi_s=self.log_psi, **obs_kwargs
        )

        return SampledObs(raw_data, self.weights)

    def reset(self, key=None):
        """
        Replace the PRNG key and initialize the proposer again at the next ``sample()`` call.

        The chains continue from their current configurations and the stored samples are kept, \
        so this does not reproduce the samples of a new sampler created with the same key.

        Args:
            * ``key``: New key, as for the initializer argument ``key``.
        """
        self.key = key

    @abstractmethod
    def _init_state(self):
        """Initialize the chains, usually through ``_init_state_general``."""

    def _init_state_general(self, initializer, dtype):
        """
        Set up one PRNG key per chain, the chain configurations and the proposer arguments.

        Args:
            * ``initializer``: Called as ``initializer(key, shape, dtype)`` to generate the \
              configurations missing from ``states``.
            * ``dtype``: Data type of the configurations.
        """
        master_key = self.key[0] if self.key.ndim > 1 else self.key
        all_keys = broadcast_split_key(master_key, self.num_chains + 1)
        keys = all_keys[:-1]
        init_state_key = all_keys[-1]
        self._key = jax.device_put(keys, DEVICE_SHARDING)

        if self.states is not None:
            states = self.states.astype(dtype)
            res = self.num_chains - states.shape[0]
            if res > 0:
                pad = initializer(init_state_key, (res,) + self.sample_shape, dtype)
                states = jnp.concat([states, pad])
            elif res < 0:
                states = states[: self.num_chains]
        else:
            states = initializer(
                init_state_key, (self.num_chains,) + self.sample_shape, dtype
            )
        self._states = jax.device_put(states, DEVICE_SHARDING)

        self.update_proposer.init_arg(self.psi, self.num_chains)

        self._is_state_initialized = True

    def _distribute_sampling(self):
        """
        Adjust sample (and, for MCMC sampling, chain) counts to be compatible
        with the device mesh used for JAX sharding.

        Device constraints:
            - For direct/generator sampling: num_samples must be divisible by total_devices
            - For MCMC sampling:
                - The number of chains must be >= total_devices
                - The number of chains must be divisible by total_devices
                - Each chain generates the same number of samples

        The method performs the following adjustments:
            1. Generator sampling: rounds num_samples up to the next multiple of
               total_devices (num_chains is unused in this case).
            2. MCMC sampling:
                a. If num_chains < total_devices: increases num_chains to total_devices
                b. If num_chains is not divisible by total_devices: rounds up to the
                   next multiple of total_devices
                c. Rounds up samples per chain using ceiling division to ensure
                   at least num_samples total samples are generated
        """
        if self.psi.is_generator:
            total_samples = distribute(self.num_samples, "samples")
        else:
            self._num_chains = distribute(self.num_chains, "chains")

            # Use ceiling division to ensure at least num_samples total samples
            self._samples_per_chain = (
                self.num_samples + self.num_chains - 1
            ) // self.num_chains
            total_samples = self._samples_per_chain * self.num_chains

        if total_samples > self.num_samples:
            print(f"INFO: Total samples adjusted: {self.num_samples} -> {total_samples}")
        self.num_samples = total_samples

    def sample(self, num_samples=None):
        """
        Generate random samples from wave function.

        If supported by the network, direct sampling is performed. Otherwise, MCMC is run \
        to generate the desired number of samples: the chains continue from their last \
        configurations, are thermalized again and then record one sample per sweep.

        Sampling is automatically distributed across processes and available \
        devices. In that case the number of samples returned might exceed ``num_samples``.

        Args:
            * ``num_samples``: Number of samples to generate in this call, without changing the \
              default. If ``None``, the default number of samples is used.

        Returns:
            ``configs, log_psi, p``: The sampled configurations, of shape \
            ``(N, *sample_shape)`` with ``N`` at least the requested number of samples, their \
            logarithmic wave function coefficients and their normalized weights. Samples of the \
            same chain are contiguous. The result is also stored in ``samples``, ``log_psi`` and \
            ``weights``.
        """

        if num_samples is not None:
            samples_tmp = self.num_samples
            self.num_samples = num_samples

        self._distribute_sampling()

        if self.psi.is_generator:
            configs, log_psi, p = self._get_samples_gen()
        else:
            configs, log_psi, p = self._get_samples_mcmc()

        if num_samples is not None:
            self.num_samples = samples_tmp

        self._samples = configs
        self._log_psi = log_psi
        self._weights = p

        return configs, log_psi, p

    def _get_samples_gen(self):
        self._key, sample_key = random.split(self.key)
        samples = self.psi.sample(self.num_samples, sample_key)

        return samples, self.psi(samples), jnp.ones(self.num_samples) / self.num_samples

    def _get_samples_mcmc(self):
        if not self._is_state_initialized:
            self._init_state()
        self.update_proposer.update_arg(self.psi)
        self.log_prob = self._log_prob_fun_jsh(self.psi.sampler_parameters, self.states)
        self.num_proposed = jax.device_put(
            jnp.zeros((self.num_chains,), dtype=np.int64), DEVICE_SHARDING
        )
        self.num_accepted = jax.device_put(
            jnp.zeros((self.num_chains,), dtype=np.int64), DEVICE_SHARDING
        )

        num_samples_str = str(self._samples_per_chain)

        # check whether _get_samples is already compiled for given number of samples.
        # partial over num_samples, therm_sweeps and sweep_steps is needed cause these must be static
        # at the level of the shard_map, as they are used in the jax loops.
        if num_samples_str not in self._get_samples_jsh:
            get_samples = partial(
                self._get_samples,
                sweep_function=partial(self._sweep, get_ratio=self._get_ratio),
                update_proposer=self.update_proposer,
                num_samples=self._samples_per_chain,
                therm_sweeps=self.thermalization_sweeps,
                sweep_steps=self.sweep_steps,
                sample_shape=self.sample_shape,
            )

            self._get_samples_jsh[num_samples_str] = jax.jit(
                jax.shard_map(
                    get_samples,
                    mesh=MESH,
                    in_specs=(REPLICATED_SPEC,)
                    + (DEVICE_SPEC,) * 5
                    + (self.update_proposer.arg_in_specs,),
                    out_specs=((DEVICE_SPEC,) * 5, DEVICE_SPEC)
                    + (self.update_proposer.arg_in_specs,),
                )
            )

        (
            (self._states, self.log_prob, self._key, self.num_proposed, self.num_accepted),
            configs,
            self.update_proposer._arg,
        ) = self._get_samples_jsh[num_samples_str](
            self.psi.sampler_parameters,
            self.states,
            self.log_prob,
            self.key,
            self.num_proposed,
            self.num_accepted,
            self.update_proposer._arg,
        )

        coeffs = self.psi(configs)
        p = jnp.exp((1.0 / self.log_prob_factor - self.mu) * jnp.real(coeffs))

        return configs, coeffs, p / jnp.sum(p)

    def _get_samples(
        self,
        params,
        states,
        log_prob,
        key,
        num_proposed,
        num_accepted,
        update_proposer_arg,
        num_samples,
        therm_sweeps,
        sweep_steps,
        update_proposer,
        sweep_function,
        sample_shape,
    ):
        # Thermalize
        if self.thermalization_sweeps is not None:
            if update_proposer._use_custom_thermalization:
                states, log_prob, key, num_proposed, num_accepted, update_proposer_arg = (
                    update_proposer._custom_therm_fun(
                        states,
                        log_prob,
                        key,
                        num_proposed,
                        num_accepted,
                        params,
                        sweep_steps,
                        therm_sweeps,
                        sweep_function,
                        update_proposer_arg,
                    )
                )
            else:
                states, log_prob, key, num_proposed, num_accepted = sweep_function(
                    states,
                    log_prob,
                    key,
                    num_proposed,
                    num_accepted,
                    params,
                    therm_sweeps * sweep_steps,
                    update_proposer,
                    update_proposer_arg,
                )

        # Collect samples
        def scan_fun(c, x):
            states, log_prob, key, num_proposed, num_accepted = sweep_function(
                c[0],
                c[1],
                c[2],
                c[3],
                c[4],
                params,
                sweep_steps,
                update_proposer,
                update_proposer_arg,
            )

            return (states, log_prob, key, num_proposed, num_accepted), states

        meta, configs = jax.lax.scan(
            scan_fun,
            (states, log_prob, key, num_proposed, num_accepted),
            None,
            length=num_samples,
        )

        return (
            meta,
            jnp.swapaxes(configs, 0, 1).reshape((-1,) + sample_shape),
            update_proposer_arg,
        )

    def _sweep(
        self,
        states,
        log_prob,
        key,
        num_proposed,
        num_accepted,
        params,
        num_steps,
        update_proposer,
        update_proposer_arg,
        get_ratio,
    ):
        def perform_mc_update_single_chain(state, log_prob, key_single, proposer_arg):
            # Generate update proposal
            proposer_key, new_key, accept_key = random.split(key_single, 3)
            new_state, log_prob_correction = update_proposer(
                proposer_key, state, proposer_arg
            )

            # Compute acceptance probability
            P, new_log_prob = get_ratio(
                log_prob, log_prob_correction, params, state, new_state
            )

            # Roll dice
            accepted = random.bernoulli(accept_key, P)

            # Perform update if accepted
            new_state = jax.lax.cond(accepted, lambda: new_state, lambda: state)
            new_log_prob = jax.lax.cond(accepted, lambda: new_log_prob, lambda: log_prob)

            return new_state, new_log_prob, new_key, accepted

        def sweep_step(carry, _):
            states, log_prob, keys, num_proposed, num_accepted = carry

            new_states, new_log_prob, new_keys, accepted = jax.vmap(
                perform_mc_update_single_chain,
                in_axes=(0, 0, 0, self.update_proposer.arg_in_axes),
            )(states, log_prob, keys, update_proposer_arg)

            num_proposed = num_proposed + 1
            num_accepted = num_accepted + accepted

            return (new_states, new_log_prob, new_keys, num_proposed, num_accepted), None

        (states, log_prob, key, num_proposed, num_accepted), _ = jax.lax.scan(
            sweep_step,
            (states, log_prob, key, num_proposed, num_accepted),
            None,
            length=num_steps,
        )

        return states, log_prob, key, num_proposed, num_accepted

    def acceptance_ratio(self):
        """
        Get acceptance ratio.

        This is available only after a ``sample()`` call that ran MCMC. It raises a \
        ``RuntimeError`` before that, and when sampling directly.

        Returns:
            Acceptance ratio observed in the last call to ``sample()``, including the \
            thermalization.
        """
        if self.psi.is_generator:
            raise RuntimeError(
                "The acceptance ratio is not defined when sampling directly."
            )
        if self.num_proposed is None:
            raise RuntimeError(
                "The acceptance ratio is not available before calling 'sample()'."
            )

        num_prop = jnp.sum(self.num_proposed)
        if num_prop > 0:
            return jnp.sum(self.num_accepted) / num_prop

        return jnp.array([0.0])
