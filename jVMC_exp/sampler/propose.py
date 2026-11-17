import jax 
import jax.numpy as jnp
from abc import ABC, abstractmethod
import jax.random as random 
from functools import partial

from jVMC_exp.geometry import AbstractGeometry
from jVMC_exp.util.key_gen import format_key
from jVMC_exp.sharding_config import DEVICE_SPEC, REPLICATED_SPEC, DEVICE_SHARDING
from jVMC_exp import global_defs

def shard_array_across_chains(arr, numChains, dtype):
    """Copy ``arr`` for each of the ``numChains`` chains and shard the result across devices."""
    arr = jnp.atleast_1d(jnp.asarray(arr, dtype=dtype))
    sharded_arr = jnp.broadcast_to(arr, (numChains,) + arr.shape)
    return jax.device_put(sharded_arr, DEVICE_SHARDING)

class AbstractProposer(ABC):
    """
    Interface of the proposers of MCMC updates.

    A subclass implements ``__call__``, which acts on the configuration of a single chain and must \
    be compatible with ``jax.vmap`` and ``jax.jit``.

    A proposer that needs additional data stores it in ``self._arg``, a pytree set up in \
    ``init_arg()`` and described by ``arg_in_axes``, and receives it as ``arg`` at each call. As \
    this data lives on the proposer, a proposer with arguments cannot be shared between samplers \
    with different numbers of chains.

    Args:
        * ``use_custom_thermalization``: Whether the sampler thermalizes the chains with \
          ``_custom_therm_fun()`` instead of plain sweeps.
    """
    def __init__(self, use_custom_thermalization=False):
        self._use_custom_thermalization = use_custom_thermalization
        self._arg = None

    @abstractmethod
    def __call__(self, key, s, arg):
        """
        Proposal move.

        Args:
            * ``key``: An instance of ``jax.random.PRNGKey``.
            * ``s``: The configuration of a single chain.
            * ``arg``: Additional arguments of the proposer for this chain. ``None`` if the \
              proposer has none.

        Returns:
            ``s_prime, log_prob_correction``: The new proposed configuration and the correction \
            :math:`\\ln q(s|s')-\\ln q(s'|s)` for non-symmetric proposal moves, with :math:`q(s'|s)` \
            the probability to propose :math:`s'` from :math:`s`. Return 0 as second value if no \
            correction is needed.
        """
        pass

    @property
    def arg_in_axes(self):
        """
        Return a dictionary with the same shape as ``self._arg``,
        which indicates which arguments of the proposer must be vmapped across:
        0 for the ones with one entry per chain, ``None`` for the ones shared by all chains.
        """
        return None

    @property
    def arg_in_specs(self):
        """
        Return a dictionary with the same shape as ``self._arg``,
        which indicates which arguments of the proposer must be sharded across
        different devices. It is derived from ``arg_in_axes``.
        """
        return jax.tree_util.tree_map(
            lambda x: DEVICE_SPEC if x is not None else REPLICATED_SPEC, 
            self.arg_in_axes, 
            is_leaf=lambda x: x is None
        ) 

    def init_arg(self, vqs, numChains):
        """
        Initialize all the arguments of the proposer with the right shape 
        for sharding across devices and chains, and store them in ``self._arg``.
        The sampler calls this whenever it initializes its chains.

        Args:
            * ``vqs``: Variational state of the sampler.
            * ``numChains``: Total number of chains.
        """
        return None

    def update_arg(self, vqs):
        """
        Update the arguments of the proposer that depend on the network.
        The sampler calls this at the start of every ``sample()`` call.

        Args:
            * ``vqs``: Variational state of the sampler.
        """
        pass
    
    def _custom_therm_fun(self, states, logAccProb, key, numProposed, numAccepted, 
                          params, sweepSteps, thermSweeps, sweepFunction, arg):
        """
        Optional custom thermalization function.

        Subclasses can override this method if
        ``use_custom_thermalization=True``. The sampler calls it as
        ``_custom_therm_fun(states, logAccProb, key, numProposed, numAccepted, params,
        sweepSteps, thermSweeps, sweepFunction, arg)``, with the proposer arguments ``arg`` as last
        argument, and expects ``states, logAccProb, key, numProposed, numAccepted, arg`` back.
        """
        if self._use_custom_thermalization:
            raise NotImplementedError(
                "Custom thermalization was enabled, "
                "but '_custom_therm_fun' is not implemented."
            )
        else:
            raise RuntimeError(
                "Custom thermalization not enabled for this proposer."
            )

class AbstractProposeCont(AbstractProposer):
    """
    Interface of the proposers of MCMC updates for continuous particle configurations.

    Args:
        * ``geometry``: Geometry of the system, an instance of \
          ``jVMC_exp.geometry.AbstractGeometry``.
        * ``use_custom_thermalization``: See :class:`~jVMC_exp.sampler.propose.AbstractProposer`.
    """
    def __init__(self, geometry: AbstractGeometry, use_custom_thermalization=False):
        if not isinstance(geometry, AbstractGeometry):
            raise ValueError("The argument 'geometry' has to be an instance of the class 'jVMC.geometry.AbstractGeometry'.")
        self._geometry = geometry
        super().__init__(use_custom_thermalization)

    @property
    def geometry(self):
        """Geometry of the system."""
        return self._geometry

class SpinFlip(AbstractProposer):
    """Flip the local state, 0 or 1, of a randomly chosen site."""
    def __init__(self):
        super().__init__(False)

    def __call__(self, key, s, arg):
        idx = random.randint(key, (1,), 0, s.size)[0]
        idx = jnp.unravel_index(idx, s.shape)
        update = (s[idx] + 1) % 2

        return s.at[idx].set(update), 0
    
class POVMOutcome(AbstractProposer):
    """
    Change the POVM outcome of a randomly chosen site.
    """
    def __init__(self):
        super().__init__(False)

    def __call__(self, key, s, arg):
        key, subkey = random.split(key)
        idx = random.randint(subkey, (1,), 0, s.size)[0]
        idx = jnp.unravel_index(idx, s.shape)
        update = (s[idx] + random.randint(key, (1,), 0, 3) % 4)

        return s.at[idx].set(update), 0

class SpinFlipZ2(AbstractProposer):
    """
    Flip the local state, 0 or 1, of a randomly chosen site, and all the sites with probability 1/5.

    The global flip helps the chains to move between the sectors related by :math:`Z_2` symmetry.
    """
    def __init__(self):
        super().__init__(False)

    def __call__(self, key, s, arg):
        idxKey, flipKey = jax.random.split(key)
        idx = random.randint(idxKey, (1,), 0, s.size)[0]
        idx = jnp.unravel_index(idx, s.shape)
        update = (s[idx] + 1) % 2
        s = s.at[idx].set(update)
        # On average, do a global spin flip every 5 updates to
        # reflect Z_2 symmetry
        doFlip = random.randint(flipKey, (1,), 0, 5)[0]

        return jax.lax.cond(doFlip == 0, lambda x: 1 - x, lambda x: x, s), 0
    
class SpinFlipZeroMag(AbstractProposer):
    """
    Exchange a randomly chosen 1 and 0, and flip all the sites with probability 1/5.

    The moves stay in the zero magnetization sector. They are defined for configurations with two \
    axes that are already in this sector, so the chains need a suitable ``initState``.
    """
    def __init__(self):
        super().__init__(False)

    def __call__(self, key, s, arg):
        # propose spin flips that stay in the zero magnetization sector
        idxKeyUp, idxKeyDown, flipKey = jax.random.split(key, num=3)

        # can't use jnp.where because then it is not jit-compilable
        # find indices based on cumsum
        bound_up = jax.random.randint(idxKeyUp, (1,), 1, s.size / 2 + 1)
        bound_down = jax.random.randint(idxKeyDown, (1,), 1, s.size / 2 + 1)

        id_up = jnp.searchsorted(jnp.cumsum(s), bound_up)
        id_down = jnp.searchsorted(jnp.cumsum(1 - s), bound_down)

        idx_up = jnp.unravel_index(id_up, s.shape)
        idx_down = jnp.unravel_index(id_down, s.shape)

        s = s.at[idx_up].set(0)
        s = s.at[idx_down].set(1)

        # On average, do a global spin flip every 5 updates to
        # reflect Z_2 symmetry
        doFlip = random.randint(flipKey, (1,), 0, 5)[0]

        return jax.lax.cond(doFlip == 0, lambda x: 1 - x, lambda x: x, s), 0

class RWM(AbstractProposeCont):
    """
    Random walk Metropolis: move all particles by a Gaussian step of width ``sigma``.

    With ``use_thermalization=True``, each chain adapts its own ``sigma`` after every thermalization \
    sweep as :math:`\\sigma\\to\\sigma\\exp(\\text{adapt\\_rate}\\,(a-\\text{target\\_rate}))`, with \
    :math:`a` the acceptance rate of the sweep and capped at half of the extent of the geometry. \
    The adapted values carry over to later ``sample()`` calls until the sampler initializes its \
    chains again.

    Args:
        * ``geometry``: Geometry of the system.
        * ``sigma``: Initial step width, either a scalar or one value per spatial dimension. If \
          ``None``, 10% of the extent of the geometry is used.
        * ``use_thermalization``: Whether to adapt ``sigma`` during thermalization.
        * ``adapt_rate``: Rate of the adaptation.
        * ``target_rate``: Acceptance rate targeted by the adaptation.
    """
    def __init__(self, geometry, sigma=None, use_thermalization=True, adapt_rate=0.1, target_rate=0.5):
        super().__init__(geometry, use_thermalization)
        self._adapt_rate = adapt_rate
        self._target_rate = target_rate
        self._sigma = sigma

    @property
    def adapt_rate(self):
        """Rate of the adaptation of ``sigma``."""
        return self._adapt_rate
    
    @property
    def target_rate(self):
        """Acceptance rate targeted by the adaptation of ``sigma``."""
        return self._target_rate
            
    @property
    def arg_in_axes(self):
        return {"sigma": 0}
    
    def init_arg(self, vqs, numChains):
        if self._sigma is None:
            sigma = jnp.array(self.geometry.extent) * 0.1
        else: 
            sigma = jnp.asarray(self._sigma) if not isinstance(self._sigma, jax.Array) else self._sigma
            if sigma.size == 1:
                sigma = jnp.full(self.geometry.n_dim, sigma.item())
            elif sigma.size != self.geometry.n_dim:
                raise ValueError(
                    f"'sigma' must have the same dimension as 'geometry.n_dim' "
                    f"(expected {self.geometry.n_dim}, got {sigma.size})."
                )
        sigma = shard_array_across_chains(jnp.asarray(sigma), numChains, dtype=global_defs.DT_SAMPLES_CONT)

        self._arg = {"sigma": sigma}
    
    def __call__(self, key, s, arg):
        """
        Proposal move for random walk metropolis (RWM).
        
        Args:
            * ``key``: An instance of ``jax.random.PRNGKey``.
            * ``s``: The configuration of a single chain.
            * ``arg``: Dictionary with the step width ``sigma`` of this chain.

        Returns:
            ``s_prime, log_prob_correction``: The new proposed configuration, with the boundary \
            conditions of the geometry applied, and 0, as the move is symmetric.
        """
        sigma = jnp.tile(arg["sigma"], self.geometry.n_particles)
        dx = (sigma * jax.random.normal(format_key(key), s.shape, dtype=s.dtype))   
        return self.geometry.apply_PBC(s + dx), 0
    
    def _get_new_sigma(self, sigma, acceptance_rate):
        new_sigma = sigma * jnp.exp(self.adapt_rate * (acceptance_rate - self.target_rate))
        return jnp.clip(new_sigma, max=jnp.array(self.geometry.extent) / 2)
    
    def _custom_therm_fun(self, states, logAccProb, key, numProposed, numAccepted,
                          params, sweepSteps, thermSweeps, sweepFunction, arg):
        def therm_sweep_fun(carry, i):
            states, logAccProb, key, numProposed, numAccepted, arg = carry
            tmp_numAccepted = numAccepted

            states, logAccProb, key, numProposed, numAccepted = sweepFunction(
                states, logAccProb, key, numProposed, numAccepted, params, sweepSteps, self, arg
            )
            acceptance_rate = (numAccepted - tmp_numAccepted) / jnp.maximum(sweepSteps, 1)
            # Vectorize across chains
            new_sigma = jax.vmap(self._get_new_sigma)(arg["sigma"], acceptance_rate)
            arg = {**arg, "sigma": new_sigma}

            new_carry = (states, logAccProb, key, numProposed, numAccepted, arg)
            return new_carry, None 

        carry = (states, logAccProb, key, numProposed, numAccepted, arg)
        final_carry, _ = jax.lax.scan(therm_sweep_fun, carry, jnp.arange(thermSweeps))
        return final_carry
    
class MALA(AbstractProposeCont):
    """
    Metropolis-adjusted Langevin algorithm: move the particles along the gradient of the log-probability.

    The proposal is :math:`s'=s+\\tau\\nabla\\ln p(s)+\\sqrt{2\\tau}\\,\\xi` with \
    :math:`\\ln p(s)=\\mu\\,\\text{Re}\\ln\\psi(s)` and Gaussian noise :math:`\\xi`.

    With ``use_thermalization=True``, each chain adapts its own ``tau`` after every thermalization \
    sweep as :math:`\\tau\\to\\tau\\exp(\\text{adapt\\_rate}\\,(a-\\text{target\\_rate}))`, with \
    :math:`a` the acceptance rate of the sweep and capped at half of the first extent of the \
    geometry. The adapted values carry over to later ``sample()`` calls until the sampler \
    initializes its chains again.

    With periodic boundaries, the chains sample the right distribution also if the network is not \
    periodic. In that case the wave function has a kink or a jump at the boundary, which estimators \
    based on derivatives, such as the kinetic energy, do not account for.

    Args:
        * ``geometry``: Geometry of the system.
        * ``appy_fun``: Function evaluating :math:`\\ln\\psi` as ``appy_fun(parameters, s)`` for a \
          single configuration, e.g., ``psi.apply_fun``. This argument is spelled without the second \
          ``p``, unlike the ``apply_fun`` property.
        * ``tau``: Initial step size, a positive scalar. If ``None``, 1% of the first extent of the \
          geometry is used.
        * ``use_thermalization``: Whether to adapt ``tau`` during thermalization.
        * ``adapt_rate``: Rate of the adaptation.
        * ``target_rate``: Acceptance rate targeted by the adaptation.
        * ``mu``: Exponent of the distribution used for the drift. It should match the ``mu`` of \
          the sampler to get the intended drift.
    """
    def __init__(self, geometry, apply_fun, tau=None, use_thermalization=True, adapt_rate=0.1, target_rate=0.5, mu=2):
        super().__init__(geometry, use_thermalization)
        self._adapt_rate = adapt_rate
        self._target_rate = target_rate
        self._apply_fun = apply_fun
        self._mu = mu
        self._tau = tau
        
    @property
    def apply_fun(self):
        """Function evaluating :math:`\\ln\\psi` for a single configuration."""
        return self._apply_fun

    @property
    def mu(self):
        """Exponent of the distribution used for the drift."""
        return self._mu

    @property
    def adapt_rate(self):
        """Rate of the adaptation of ``tau``."""
        return self._adapt_rate
    
    @property
    def target_rate(self):
        """Acceptance rate targeted by the adaptation of ``tau``."""
        return self._target_rate

    @property
    def arg_in_axes(self):
        return {"params": None, "tau": 0}
    
    def init_arg(self, vqs, numChains):
        # TODO: implement version for tau in more dims
        if self._tau is None:
            tau = jnp.array(self.geometry.extent)[0] * 0.01
        elif self._tau < 0:
            raise ValueError('"tau" can not be negative.')
        else:
            tau = self._tau
        tau = shard_array_across_chains(jnp.asarray(tau), numChains, dtype=global_defs.DT_SAMPLES_CONT)

        self._arg = {"params": vqs.parameters, "tau": tau}
    
    def update_arg(self, vqs):
        if self._arg is None:
            raise RuntimeError("Must call init_arg before update_arg.")
        self._arg["params"] = vqs.parameters
    
    def __call__(self, key, s, arg):
        """
        Proposal move for the Metropolis-adjusted Langevin algorithm (MALA).

        Args:
            * ``key``: An instance of ``jax.random.PRNGKey``.
            * ``s``: The configuration of a single chain.
            * ``arg``: Dictionary with the network parameters ``params`` and the step size ``tau`` \
              of this chain.

        Returns:
            ``s_prime, log_prob_correction``: The new proposed configuration, with the boundary \
            conditions of the geometry applied, and the correction for the non-symmetric move.
        """
        tau = arg["tau"]
        params = arg["params"]
        log_prob_fun = partial(lambda x, p: self.mu * jnp.real(self.apply_fun(p, x)), p=params)
        log_prob_fun_grad = jax.grad(log_prob_fun)

        xi = jax.random.normal(format_key(key), s.shape, dtype=s.dtype)
        drift = tau * log_prob_fun_grad(s)
        dx = drift + jnp.sqrt(2 * tau).astype(s.dtype) * xi
        sp = s + dx
        sp_wrapped = self.geometry.apply_PBC(sp)

        # The reverse move starts from the wrapped position and undoes the same displacement
        log_q_s_sp = - jnp.sum((s - sp - tau * log_prob_fun_grad(sp_wrapped)) ** 2) / (4 * tau)
        log_q_sp_s = - jnp.sum(xi ** 2) * 0.5
        log_prob_correction = log_q_s_sp - log_q_sp_s

        return sp_wrapped, log_prob_correction.squeeze()
    
    def _get_new_tau(self, tau, acceptance_rate):
        new_tau = tau * jnp.exp(self.adapt_rate * (acceptance_rate - self.target_rate))
        return jnp.clip(new_tau, max=jnp.array(self.geometry.extent)[0] / 2) # TODO: change when tau gets more dims
    
    def _custom_therm_fun(self, states, logAccProb, key, numProposed, numAccepted,
                          params, sweepSteps, thermSweeps, sweepFunction, arg):
        def therm_sweep_fun(carry, i):
            states, logAccProb, key, numProposed, numAccepted, arg = carry
            tmp_numAccepted = numAccepted

            states, logAccProb, key, numProposed, numAccepted = sweepFunction(
                states, logAccProb, key, numProposed, numAccepted, params, sweepSteps, self, arg
            )
            acceptance_rate = (numAccepted - tmp_numAccepted) / jnp.maximum(sweepSteps, 1)
            # Vectorize across chains
            new_tau = jax.vmap(self._get_new_tau)(arg["tau"], acceptance_rate)
            arg = {**arg, "tau": new_tau}

            new_carry = (states, logAccProb, key, numProposed, numAccepted, arg)
            return new_carry, None 

        carry = (states, logAccProb, key, numProposed, numAccepted, arg)
        final_carry, _ = jax.lax.scan(therm_sweep_fun, carry, jnp.arange(thermSweeps))
        return final_carry