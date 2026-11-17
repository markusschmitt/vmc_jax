import warnings
import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
from functools import partial
from abc import ABC, abstractmethod
from typing import Tuple

from jVMC_exp.util.key_gen import format_key
from jVMC_exp.util.util import has_callable_attr
from jVMC_exp.vqs import NQS
from jVMC_exp.sharding_config import (
    MESH, DEVICE_SPEC, REPLICATED_SPEC, DEVICE_SHARDING, 
    distribute, broadcast_split_key
)
from jVMC_exp.sampler.propose import AbstractProposer
from jVMC_exp.operator.base import AbstractOperator
from jVMC_exp.stats import SampledObs
from jVMC_exp import global_defs

class AbstractSampler(ABC):
    """
    Interface shared by all samplers.

    A sampler is bound to a variational state and keeps the result of its last ``sample()`` call
    in ``samples``, ``logPsi`` and ``weights``. Reading one of these before the first call
    triggers it. The stored values are not refreshed when ``psi.parameters`` changes: call
    ``sample()`` again after updating the parameters.

    Args:
        * ``psi``: Variational state to sample from.
    """
    def __init__(self, psi: NQS):
        self._psi = psi
        self._samples = None
        self._logPsi = None
        self._weights = None
        self.numSamples = None

    @property
    def psi(self):
        """Variational state the sampler is bound to."""
        return self._psi
    
    @property
    def sampleShape(self):
        """Shape of a single configuration."""
        return self.psi.sampleShape
    
    @property
    def logPsi(self):
        """:math:`\\ln\\psi(s)` of the stored samples."""
        if self._samples is None:
            self.sample()
        
        return self._logPsi
    
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

    # TODO: make this method explicit
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
        pass

    @abstractmethod
    def sample(self, numSamples=None) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Sample configurations from the distribution defined by the network.

        Args:
            * ``numSamples``: Number of samples to generate in this call. If ``None``, the \
              default number of samples is used.

        Returns:
            ``configs, logPsi, p``: The sampled configurations, their logarithmic wave function \
            coefficients and their normalized weights.
        """
        pass

class AbstractMCSampler(AbstractSampler):
    """
    Base class of the samplers that draw configurations stochastically.

    This abstract class provides functionality to sample states from \
    the distribution 

        :math:`p_{\\mu}(s)=\\frac{|\\psi(s)|^{\\mu}}{\\sum_s|\\psi(s)|^{\\mu}}`.

    For :math:`\\mu=2` this corresponds to sampling from the Born distribution. \
    :math:`0\\leq\\mu<2` can be used to perform importance sampling \
    (see `[arXiv:2108.08631] <https://arxiv.org/abs/2108.08631>`_). The returned weights are \
    proportional to :math:`|\\psi(s)|^{1/\\text{logProbFactor}-\\mu}`, so that weighted averages \
    are taken over the distribution :math:`|\\psi(s)|^{1/\\text{logProbFactor}}`.

    If the network of ``psi`` provides a ``sample()`` method, configurations are drawn directly \
    from it and all get the same weight. In that case the network alone sets the distribution: \
    ``mu``, ``logProbFactor`` and all the MCMC arguments have no effect on the samples. \
    Otherwise, Markov chains are run with the proposals of ``updateProposer``.

    Sampling is automatically distributed across processes and locally available devices. The \
    numbers of chains and samples are rounded up where this is needed, so a call can return \
    more samples than requested.

    Initializer arguments:
        * ``psi``: Variational state defining the probability distribution.
        * ``updateProposer``: An instance of ``jVMC_exp.sampler.propose.AbstractProposer`` to \
          propose updates for the MCMC algorithm. Can be ``None`` only if ``psi`` samples directly.
        * ``key``: An instance of ``jax.random.PRNGKey``, or an ``int`` that will be used as seed \
          to initialize a ``PRNGKey``. If ``None``, a random seed is generated.
        * ``numChains``: Number of Markov chains, which are run in parallel. It cannot exceed \
          ``numSamples``. It is unused when sampling directly.
        * ``numSamples``: Default number of samples to be returned by the ``sample()`` member \
          function. All chains contribute the same number of samples.
        * ``thermalizationSweeps``: Number of sweeps discarded at the start of every ``sample()`` \
          call. If ``None``, thermalization is skipped.
        * ``sweepSteps``: Number of proposed updates per sweep, i.e., between two recorded samples \
          of a chain. If ``None``, the total number of sites, ``prod(sampleShape)``, is used.
        * ``initState``: Initial configurations of the chains, of shape \
          ``(n, *sampleShape)``. Surplus configurations are dropped and missing ones are \
          generated randomly. If ``None``, all the configurations are generated randomly.
        * ``mu``: Parameter for the distribution :math:`p_{\\mu}(s)`, see above, in the \
          interval :math:`[0, 2]`. It cannot be changed after construction.
        * ``logProbFactor``: Factor for the log-probabilities, equivalent to the exponent for the \
          probability distribution. For pure wave functions this should be 0.5, and 1.0 for POVMs. \
          In the POVM case, the ``mu`` parameter must be set to 1.0, to sample the unchanged POVM \
          distribution.
    """
    def __init__(
            self, psi: NQS, updateProposer: None | AbstractProposer, key=None, 
            numChains=32, numSamples=128, thermalizationSweeps=10, sweepSteps=None, 
            initState=None, mu=2, logProbFactor=0.5
        ):
        if (not psi.is_generator) and (not isinstance(updateProposer, AbstractProposer)):
            raise RuntimeError(
                "Instantiation of MCSampler: `updateProposer` is `None` and cannot be used for MCMC sampling. "
                "'updateProposer' must be an instance of 'jVMC.propose.AbstractProposer'."
            )
        super().__init__(psi)

        self.logProbFactor = logProbFactor
        if mu < 0 or mu > 2:
            raise ValueError("mu must be in the range [0, 2]")
        self._mu = mu
        self._updateProposer = updateProposer

        self.key = key

        if sweepSteps is None:
            sweepSteps = int(np.prod(self.sampleShape))
        self.sweepSteps = sweepSteps
        self.thermalizationSweeps = thermalizationSweeps
        self.numSamples = numSamples
        self.numChains = numChains

        self.states = initState
        self.numProposed = None
        self.numAccepted = None

        if has_callable_attr(self.psi.net, "eval_real"):
            def log_prob_fun(p, s):
                return (self.mu * self.psi.apply_fun(p, s, method=self.psi.net.eval_real)
                        .astype(global_defs.DT_OUT_REAL))
        else:
            def log_prob_fun(p, s):
                return (self.mu * jnp.real(self.psi.apply_fun(p, s))
                        .astype(global_defs.DT_OUT_REAL))
        self._log_prob_fun_jsh = jax.jit(
            jax.shard_map(
                jax.vmap(log_prob_fun, in_axes=(None, 0)),
                mesh=MESH,
                in_specs=(REPLICATED_SPEC, DEVICE_SPEC),
                out_specs=DEVICE_SPEC
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

                return jnp.exp(new_log_prob - log_prob + log_prob_correction), new_log_prob
        self._get_ratio = get_ratio

    @property
    def mu(self):
        """Parameter of the distribution :math:`p_{\\mu}(s)`, read-only."""
        return self._mu

    @property
    def thermalizationSweeps(self):
        """Number of sweeps discarded at the start of every ``sample()`` call."""
        return self._thermalizationSweeps
    
    @thermalizationSweeps.setter
    def thermalizationSweeps(self, value):
        self._thermalizationSweeps = value
        self._get_samples_jsh = {}
    
    @property
    def sweepSteps(self):
        """Number of proposed updates per sweep."""
        return self._sweepSteps
    
    @sweepSteps.setter
    def sweepSteps(self, value):
        self._sweepSteps = value
        self._get_samples_jsh = {}

    @property
    def updateProposer(self):
        """Proposer of the MCMC updates."""
        return self._updateProposer

    @property
    def numChains(self):
        """
        Number of Markov chains.

        Setting it adapts the current chains at the next ``sample()`` call, as described for \
        ``initState``, and initializes the proposer again.
        """
        return self._numChains
    
    @numChains.setter
    def numChains(self, value):
        if not self.psi.is_generator and self.numSamples < value:
            raise ValueError(
                f"The provided number of chains {value} is bigger "
                f"than the number of samples {self.numSamples}."
            )

        self._numChains = value
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
        Current configurations of the Markov chains, of shape ``(numChains, *sampleShape)``.

        This is ``initState`` until the first ``sample()`` call. A new value is used by the next \
        ``sample()`` call, as described for ``initState``.
        """
        return self._states

    @states.setter
    def states(self, value):
        if value is not None:
            value = jnp.array(value)
            if value.shape[1:] != self.sampleShape:
                raise ValueError(
                    f"The provided state has the wrong sample shape. "
                    f"Got {value.shape[1:]}, while sampleShape is {self.sampleShape}."
                )

        self._states = value
        self._is_state_initialized = False

    @property
    def sampler_state(self):
        """
        Snapshot of the sampler as a dictionary with ``parameters``, ``states`` and ``weights``.

        Reading it samples if there are no stored samples yet. Setting it restores these three \
        entries, but not ``samples``, ``logPsi`` or ``key``: call ``sample()`` before using \
        the stored samples again.
        """
        return dict(
            parameters=self.psi.parameters,
            weights=self.weights,
            states=self.states
        )

    @sampler_state.setter
    def sampler_state(self, value):
        self.psi.parameters = value["parameters"]
        self.states = value["states"]
        self._weights = value["weights"]
    
    def __call__(
            self, 
            observable: AbstractOperator, 
            num_samples: int | None=None,
            resample: bool = False,
            **obs_kwargs 
    ) -> SampledObs:
        """
        Evaluate an observable using Monte Carlo samples of the current variational state.

        This method draws (or reuses) samples from the underlying sampler and computes \
        the local estimator of the given observable. Resampling is triggered if no \
        samples are currently stored, if the requested number of samples differs from \
        ``numSamples``, or if ``resample=True``.

        Stored samples and :math:`\\ln\\psi(s)` are reused even if ``psi.parameters`` has \
        changed, so pass ``resample=True`` after updating the parameters. ``num_samples`` applies \
        to one call and does not change ``numSamples``: any other value resamples at every call.

        Args:
            * ``observable``: Operator to evaluate.
            * ``num_samples``: Number of Monte Carlo samples to draw. If ``None``, the stored \
              samples are reused.
            * ``resample``: If ``True``, forces resampling even if stored samples are available.
            * ``**obs_kwargs``: Additional keyword arguments for ``observable.get_O_loc()``.

        Returns:
            The local estimator of ``observable`` as a ``jVMC_exp.stats.SampledObs``, together \
            with the sample weights.
        """
        needs_resample = (
            self.samples is None
            or (num_samples is not None and num_samples != self.numSamples)
            or resample
        )
        if needs_resample:
            self._samples, self._logPsi, self._weights = self.sample(num_samples)
        raw_data = observable.get_O_loc(self.samples, self.psi, logPsiS=self.logPsi, **obs_kwargs)

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
        pass

    def _init_state_general(self, initializer, dtype):
        """
        Set up one PRNG key per chain, the chain configurations and the proposer arguments.

        Args:
            * ``initializer``: Called as ``initializer(key, shape, dtype)`` to generate the \
              configurations missing from ``states``.
            * ``dtype``: Data type of the configurations.
        """
        master_key = self.key[0] if self.key.ndim > 1 else self.key
        all_keys = broadcast_split_key(master_key, self.numChains + 1)
        keys = all_keys[:-1]
        initStateKey = all_keys[-1]
        self._key = jax.device_put(keys, DEVICE_SHARDING)

        if self.states is not None:
            states = self.states.astype(dtype)
            res = self.numChains - states.shape[0]
            if res > 0:
                pad = initializer(initStateKey, (res,) + self.sampleShape, dtype)
                states = jnp.concat([states, pad])
            elif res < 0:
                states = states[:self.numChains]
        else:
            states = initializer(initStateKey, (self.numChains,) + self.sampleShape, dtype)
        self._states = jax.device_put(states, DEVICE_SHARDING)
        
        self.updateProposer.init_arg(self.psi, self.numChains)

        self._is_state_initialized = True
    
    def _distribute_sampling(self):
        """
        Adjust sample (and, for MCMC sampling, chain) counts to be compatible
        with the device mesh used for JAX sharding.

        Device constraints:
            - For direct/generator sampling: numSamples must be divisible by total_devices
            - For MCMC sampling:
                - The number of chains must be >= total_devices
                - The number of chains must be divisible by total_devices
                - Each chain generates the same number of samples

        The method performs the following adjustments:
            1. Generator sampling: rounds numSamples up to the next multiple of
               total_devices (numChains is unused in this case).
            2. MCMC sampling:
                a. If numChains < total_devices: increases numChains to total_devices
                b. If numChains is not divisible by total_devices: rounds up to the
                   next multiple of total_devices
                c. Rounds up samples per chain using ceiling division to ensure
                   at least numSamples total samples are generated
        """
        if self.psi.is_generator:
            totalSamples = distribute(self.numSamples, 'samples')
        else:
            self._numChains = distribute(self.numChains, 'chains')

            # Use ceiling division to ensure at least numSamples total samples
            self._samplePerChain = (self.numSamples + self.numChains - 1) // self.numChains
            totalSamples = self._samplePerChain * self.numChains

        if totalSamples > self.numSamples:
            print(f"INFO: Total samples adjusted: {self.numSamples} -> {totalSamples}")
        self.numSamples = totalSamples

    def sample(self, numSamples=None):
        """
        Generate random samples from wave function.

        If supported by the network, direct sampling is performed. Otherwise, MCMC is run \
        to generate the desired number of samples: the chains continue from their last \
        configurations, are thermalized again and then record one sample per sweep.

        Sampling is automatically distributed across processes and available \
        devices. In that case the number of samples returned might exceed ``numSamples``.

        Args:
            * ``numSamples``: Number of samples to generate in this call, without changing the \
              default. If ``None``, the default number of samples is used.

        Returns:
            ``configs, logPsi, p``: The sampled configurations, of shape \
            ``(N, *sampleShape)`` with ``N`` at least the requested number of samples, their \
            logarithmic wave function coefficients and their normalized weights. Samples of the \
            same chain are contiguous. The result is also stored in ``samples``, ``logPsi`` and \
            ``weights``.
        """

        if numSamples is not None:
            samples_tmp = self.numSamples
            self.numSamples = numSamples

        self._distribute_sampling()

        if self.psi.is_generator:
            configs, logPsi, p = self._get_samples_gen()
        else:
            configs, logPsi, p = self._get_samples_mcmc()
             
        if numSamples is not None:
            self.numSamples = samples_tmp

        self._samples = configs
        self._logPsi = logPsi
        self._weights = p

        return configs, logPsi, p

    def _get_samples_gen(self):
        self._key, sample_key = random.split(self.key)
        samples = self.psi.sample(self.numSamples, sample_key)
        
        return samples, self.psi(samples), jnp.ones(self.numSamples) / self.numSamples

    def _get_samples_mcmc(self):
        if not self._is_state_initialized:
            self._init_state()
        self.updateProposer.update_arg(self.psi)
        self.logProb = self._log_prob_fun_jsh(self.psi.sampler_parameters, self.states)
        self.numProposed = jax.device_put(jnp.zeros((self.numChains,), dtype=np.int64), DEVICE_SHARDING)
        self.numAccepted = jax.device_put(jnp.zeros((self.numChains,), dtype=np.int64), DEVICE_SHARDING)

        numSamplesStr = str(self._samplePerChain)

        # check whether _get_samples is already compiled for given number of samples.
        # partial over numSamples, thermSweeps and sweepSteps is needed cause these must be static
        # at the level of the shard_map, as they are used in the jax loops.
        if numSamplesStr not in self._get_samples_jsh:
            get_samples = partial(
                self._get_samples, 
                sweepFunction=partial(self._sweep, get_ratio=self._get_ratio), 
                updateProposer=self.updateProposer,
                numSamples=self._samplePerChain,
                thermSweeps=self.thermalizationSweeps,
                sweepSteps=self.sweepSteps,
                sampleShape=self.sampleShape, 
            )

            self._get_samples_jsh[numSamplesStr] = jax.jit(
                jax.shard_map(
                    get_samples,
                    mesh=MESH,
                    in_specs=(REPLICATED_SPEC,) + (DEVICE_SPEC,) * 5 + (self.updateProposer.arg_in_specs,),
                    out_specs=((DEVICE_SPEC,) * 5, DEVICE_SPEC) + (self.updateProposer.arg_in_specs,)
                )
            )

        (self._states, self.logProb, self._key, self.numProposed, self.numAccepted), configs, self.updateProposer._arg =\
            self._get_samples_jsh[numSamplesStr](
                self.psi.sampler_parameters, self.states, self.logProb, self.key, 
                self.numProposed, self.numAccepted, self.updateProposer._arg
            )

        coeffs = self.psi(configs)
        p = jnp.exp((1.0 / self.logProbFactor - self.mu) * jnp.real(coeffs))

        return configs, coeffs, p / jnp.sum(p)

    def _get_samples(    
            self, params, states, logProb, key, numProposed, numAccepted, updateProposerArg,
            numSamples, thermSweeps, sweepSteps, updateProposer, sweepFunction, sampleShape
        ):
        # Thermalize
        if self.thermalizationSweeps is not None:
            if updateProposer._use_custom_thermalization:
                states, logProb, key, numProposed, numAccepted, updateProposerArg = updateProposer._custom_therm_fun(
                    states, logProb, key, numProposed, numAccepted, params, 
                    sweepSteps, thermSweeps, sweepFunction, updateProposerArg
                )
            else:
                states, logProb, key, numProposed, numAccepted = sweepFunction(
                    states, logProb, key, numProposed, numAccepted, params, 
                    thermSweeps * sweepSteps, updateProposer, updateProposerArg
                )

        # Collect samples
        def scan_fun(c, x):
            states, logProb, key, numProposed, numAccepted = sweepFunction(
                c[0], c[1], c[2], c[3], c[4], params, sweepSteps, updateProposer, updateProposerArg
            )

            return (states, logProb, key, numProposed, numAccepted), states

        meta, configs = jax.lax.scan(scan_fun, (states, logProb, key, numProposed, numAccepted), None, length=numSamples)

        return meta, jnp.swapaxes(configs, 0, 1).reshape((-1,) + sampleShape), updateProposerArg

    def _sweep(
            self, states, logProb, key, numProposed, numAccepted, params, 
            numSteps, updateProposer, updateProposerArg, get_ratio
        ):
        def perform_mc_update_single_chain(state, logProb, key_single, ProposerArg):
            # Generate update proposal
            proposerKey, newKey, acceptKey = random.split(key_single, 3)
            newState, log_prob_correction = updateProposer(proposerKey, state, ProposerArg)  
            
            # Compute acceptance probability
            P, newLogProb = get_ratio(logProb, log_prob_correction, params, state, newState)
            
            # Roll dice
            accepted = random.bernoulli(acceptKey, P)
            
            # Perform update if accepted
            newState = jax.lax.cond(accepted, lambda: newState, lambda: state)
            newLogProb = jax.lax.cond(accepted, lambda: newLogProb, lambda: logProb)
            
            return newState, newLogProb, newKey, accepted
        
        def sweep_step(carry, _):
            states, logProb, keys, numProposed, numAccepted = carry

            newStates, newLogProb, newKeys, accepted = jax.vmap(
                perform_mc_update_single_chain,
                in_axes=(0, 0, 0, self.updateProposer.arg_in_axes)
            )(states, logProb, keys, updateProposerArg)

            numProposed = numProposed + 1
            numAccepted = numAccepted + accepted

            return (newStates, newLogProb, newKeys, numProposed, numAccepted), None

        (states, logProb, key, numProposed, numAccepted), _ = jax.lax.scan(
            sweep_step, 
            (states, logProb, key, numProposed, numAccepted),
            None,
            length=numSteps
        )

        return states, logProb, key, numProposed, numAccepted
        
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
            raise RuntimeError("The acceptance ratio is not defined when sampling directly.")
        if self.numProposed is None:
            raise RuntimeError("The acceptance ratio is not available before calling 'sample()'.")

        numProp = jnp.sum(self.numProposed)
        if numProp > 0:
            return jnp.sum(self.numAccepted) / numProp

        return jnp.array([0.])