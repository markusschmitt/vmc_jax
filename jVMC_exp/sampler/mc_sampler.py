import warnings
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import random

from jVMC_exp import global_defs
from jVMC_exp.sampler.base import AbstractMCSampler
from jVMC_exp.sampler.propose import AbstractProposeCont, AbstractProposer
from jVMC_exp.sharding_config import (
    DEVICE_SHARDING,
    DEVICE_SPEC,
    MESH,
    REPLICATED_SPEC,
)
from jVMC_exp.util.util import has_callable_attr
from jVMC_exp.vqs import NQS


class MCSampler(AbstractMCSampler):
    """
    Sampler of discrete computational basis configurations.

    Chains that do not get their initial configuration from ``init_state`` start from random

    See :class:`~jVMC_exp.sampler.base.AbstractMCSampler` for the sampled distribution and the initializer arguments.
    """

    def _init_state(self):
        initializer = lambda key, shape, dtype: jax.random.bernoulli(
            key, 0.5, shape
        ).astype(dtype)

        return self._init_state_general(initializer, global_defs.DT_SAMPLES)


class MCSamplerCont(AbstractMCSampler):
    """
    Sampler of continuous particle configurations.

    Chains that do not get their initial configuration from ``init_state`` start from particle \
    positions drawn uniformly from the geometry of the proposer.

    See :class:`~jVMC_exp.sampler.base.AbstractMCSampler` for the sampled distribution and the initializer arguments. \
    They differ in the following:

    Initializer arguments:
        * ``update_proposer``: An instance of ``jVMC_exp.sampler.propose.AbstractProposeCont``. \
        * ``sweep_steps``: Number of proposed updates per sweep. If ``None``, the number of \
          particles times the number of spatial dimensions is used.
    """

    def __init__(
        self,
        psi: NQS,
        update_proposer: None | AbstractProposeCont,
        key=None,
        num_chains=32,
        num_samples=128,
        thermalization_sweeps=10,
        sweep_steps=None,
        init_state=None,
        mu=2,
        log_prob_factor=0.5,
    ):
        if sweep_steps is None:
            sweep_steps = (
                update_proposer.geometry.n_particles * update_proposer.geometry.n_dim
            )
        super().__init__(
            psi,
            update_proposer,
            key,
            num_chains,
            num_samples,
            thermalization_sweeps,
            sweep_steps,
            init_state,
            mu,
            log_prob_factor,
        )

    def _init_state(self):
        return self._init_state_general(
            self.update_proposer.geometry.uniform_populate, global_defs.DT_SAMPLES_CONT
        )


class CutoffSampler(MCSampler):
    """
    A cutoff-based MCMC sampler class.

    This class provides functionality to sample computational basis states from
    a cutoff version of the distribution

        :math:`p_{\\mu}(s)=\\frac{|\\psi(s)|^{\\mu}}{\\sum_s|\\psi(s)|^{\\mu}}`.

    During the Markov chain update, the log-probability
    :math:`\\mu \\, \\text{Re}(\\log\\psi)` is clipped from below at
    ``cutoff = max_log_psi + log(eps)``, so that the chains sample
    :math:`q(s) \\propto \\max\\{|\\psi(s)|^{\\mu}, e^{\\text{cutoff}}\\}`.
    The returned weight for each sample is multiplied by the importance ratio
    :math:`r(s) = |\\psi(s)|^{\\mu} / \\max\\{|\\psi(s)|^{\\mu}, e^{\\text{cutoff}}\\}`
    so that estimators stay unbiased.

    ``max_log_psi`` is an estimate of :math:`\\mu\\,\\max_s\\text{Re}(\\log\\psi(s))`. Every
    ``sample()`` call replaces it with the maximum over the new samples, which
    moves the cutoff for the next call: the weights of a call are computed with the
    cutoff that was in place before it.

    After sampling, ``ratio`` holds :math:`r(s)` of the last samples and ``renorm``
    holds ``num_samples / sum(ratio)``.

    Sampling is automatically distributed across processes and locally
    available devices via JAX sharding (see ``jVMC_exp.sharding_config``).

    .. warning::
        Networks with an ``eval_ratio`` method are not supported: the
        initializer raises an ``UnboundLocalError``.

    Initializer arguments:
        * ``psi``: Variational state defining the probability distribution.
          Generators are not supported.
        * ``update_proposer``: An instance of
          ``jVMC_exp.sampler.propose.AbstractProposer``, without custom
          thermalization.
        * ``eps``: Cutoff parameter, in the closed interval :math:`[0, 1]`.
          ``0`` disables the cutoff and ``1`` makes the chains sample uniformly.
        * ``key``: PRNG key (``jax.random.PRNGKey`` or seed ``int``).
        * ``num_chains``: Number of Markov chains run in parallel.
        * ``num_samples``: Default number of samples returned by ``sample()``.
        * ``thermalization_sweeps``: Sweeps used for thermalization.
        * ``sweep_steps``: Proposed updates per sweep.
        * ``init_state``: Optional initial chain configurations.
        * ``mu``: Exponent of the sampled distribution :math:`|\\psi|^{\\mu}`.
        * ``log_prob_factor``: Factor for log-probabilities (``0.5`` for pure
          wave functions, ``1.0`` for POVMs).
        * ``max_log_psi``: Optional initial value of
          :math:`\\mu\\,\\max_s\\text{Re}(\\log\\psi(s))`. If ``None``, a
          quick bootstrap MC run is used to estimate it.
        * ``bootstrap_kwargs``: Optional dict of keyword arguments
          forwarded to the bootstrap :class:`~jVMC_exp.sampler.mc_sampler.MCSampler`. Use this to
          reduce the cost of the bootstrap (e.g., smaller
          ``num_samples``) or to override its key. The bootstrap runs
          25 chains by default, so a ``num_samples`` below 25 also
          needs a smaller ``num_chains`` here.
    """

    def __init__(
        self,
        psi: NQS,
        update_proposer: AbstractProposer,
        eps,
        key=None,
        num_chains=32,
        num_samples=128,
        thermalization_sweeps=10,
        sweep_steps=None,
        init_state=None,
        mu=2,
        log_prob_factor=0.5,
        max_log_psi=None,
        bootstrap_kwargs=None,
    ):
        if psi.is_generator:
            raise RuntimeError(
                "CutoffSampler does not support generator (autoregressive) nets; "
                "use 'MCSampler' for those."
            )
        if not isinstance(update_proposer, AbstractProposer):
            raise RuntimeError(
                "'update_proposer' must be an instance of 'jVMC_exp.propose.AbstractProposer'."
            )
        if update_proposer._use_custom_thermalization:
            raise RuntimeError(
                "CutoffSampler does not support proposers with custom "
                "thermalization (e.g. RWM, MALA)."
            )

        super().__init__(
            psi,
            update_proposer,
            key=key,
            num_chains=num_chains,
            num_samples=num_samples,
            thermalization_sweeps=thermalization_sweeps,
            sweep_steps=sweep_steps,
            init_state=init_state,
            mu=mu,
            log_prob_factor=log_prob_factor,
        )

        if max_log_psi is None:
            max_log_psi = self._bootstrap_max_log_psi(bootstrap_kwargs or {})

        self._max_log_psi = jnp.asarray(max_log_psi)
        self.eps = eps
        self.ratio = None
        self.renorm = None

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
            if jax.process_index() == 0:
                warnings.warn(
                    "CutoffSampler does not support nets with `eval_ratio` method; "
                    "using the default log probability function."
                )
        else:

            def get_ratio(
                log_prob, log_prob_correction, params, state, new_state, cutoff
            ):
                new_log_prob = jnp.maximum(log_prob_fun(params, new_state), cutoff)

                return jnp.exp(
                    new_log_prob - log_prob + log_prob_correction
                ), new_log_prob

        self._get_ratio = get_ratio

    @property
    def eps(self):
        """Cutoff parameter. Setting it updates ``cutoff``."""
        return self._eps

    @eps.setter
    def eps(self, value):
        if not (0.0 <= float(value) <= 1.0):
            raise ValueError("`eps` must be in the closed interval [0, 1].")
        self._eps = float(value)
        self._update_cutoff()

    @property
    def max_log_psi(self):
        """
        Estimate of :math:`\\mu\\,\\max_s\\text{Re}(\\log\\psi(s))`. Setting it updates ``cutoff``.
        """
        return self._max_log_psi

    @max_log_psi.setter
    def max_log_psi(self, value):
        self._max_log_psi = jnp.asarray(value)
        self._update_cutoff()

    @property
    def cutoff(self):
        """Lower bound of the log-probability for the next ``sample()`` call."""
        return self._cutoff

    def _update_cutoff(self):
        """Recompute the cutoff from the current `eps` and `max_log_psi`."""
        self._cutoff = self._max_log_psi + jnp.log(self._eps)

    def _bootstrap_max_log_psi(self, bootstrap_kwargs):
        """Run a small MCSampler to obtain an initial estimate of max log psi."""
        defaults = dict(
            key=random.PRNGKey(4321),
            num_chains=25,
            sweep_steps=int(np.prod(self.sample_shape)),
            num_samples=self.num_samples,
            thermalization_sweeps=50,
            mu=self.mu,
            log_prob_factor=self.log_prob_factor,
        )
        defaults.update(bootstrap_kwargs)
        bootstrap = MCSampler(self.psi, update_proposer=self.update_proposer, **defaults)
        _, coeffs, _ = bootstrap.sample()

        return jnp.max(self.mu * jnp.real(coeffs))

    def _get_samples_mcmc(self):
        if not self._is_state_initialized:
            self._init_state()
        self.update_proposer.update_arg(self.psi)
        self.log_prob = jnp.maximum(
            self._log_prob_fun_jsh(self.psi.sampler_parameters, self.states),
            self.cutoff,
        )
        self.num_proposed = jax.device_put(
            jnp.zeros((self.num_chains,), dtype=np.int64), DEVICE_SHARDING
        )
        self.num_accepted = jax.device_put(
            jnp.zeros((self.num_chains,), dtype=np.int64), DEVICE_SHARDING
        )

        num_samples_str = str(self._samples_per_chain)

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
                    in_specs=(REPLICATED_SPEC,) * 2
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
            self.cutoff,
            self.states,
            self.log_prob,
            self.key,
            self.num_proposed,
            self.num_accepted,
            self.update_proposer._arg,
        )

        coeffs = self.psi(configs)

        # Importance sampling weights for cutoff distribution
        log_P_mu = self.mu * jnp.real(coeffs)
        log_ratio = jnp.minimum(log_P_mu - self.cutoff, 0.0)
        log_weights = (1.0 / self.log_prob_factor - self.mu) * jnp.real(coeffs) + log_ratio
        p = jnp.exp(log_weights - jnp.max(log_weights))
        self.ratio = jnp.exp(log_ratio)
        self.renorm = self.num_samples / jnp.sum(self.ratio)
        self.max_log_psi = jnp.max(log_P_mu)

        return configs, coeffs, p / jnp.sum(p)

    def _get_samples(
        self,
        params,
        cutoff,
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
        if therm_sweeps is not None:
            states, log_prob, key, num_proposed, num_accepted = sweep_function(
                states,
                log_prob,
                key,
                num_proposed,
                num_accepted,
                cutoff,
                params,
                therm_sweeps * sweep_steps,
                update_proposer,
                update_proposer_arg,
            )

        def scan_fun(c, x):
            states, log_prob, key, num_proposed, num_accepted = sweep_function(
                c[0],
                c[1],
                c[2],
                c[3],
                c[4],
                cutoff,
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
        cutoff,
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
                log_prob, log_prob_correction, params, state, new_state, cutoff
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
