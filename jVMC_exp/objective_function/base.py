from abc import ABC, abstractmethod
from chex import dataclass
import jax

from jVMC_exp.stats import SampledObs, LazySampledObs
from jVMC_exp.operator.base import AbstractOperator
from jVMC_exp.sampler.base import AbstractSampler
from jVMC_exp.sharding_config import sharded
from jVMC_exp.util.grads import pick_gradient

@dataclass
class ObjectiveFunctionOutput():
    """
    Output of ``AbstractObjectiveFunction.value_and_grad()``.

    Attributes:
        * ``o_loc``: Local losses :math:`\\ell_\\theta(s_n)` on the samples of the sampler, as a \
          ``SampledObs``. Its ``mean`` estimates the loss :math:`L`.
        * ``grad_log_psi``: Logarithmic derivatives \
          :math:`\\Gamma_k(s_n)=\\partial_{\\theta_k}\\ln\\psi_\\theta(s_n)`, as a ``SampledObs``, or as \
          a ``LazySampledObs`` with ``batched_jacobian=True``.
        * ``grad``: Force :math:`F_k`, a complex array with one entry per real parameter, such \
          that :math:`\\partial_{\\theta_k}L=2\\,\\mathrm{Re}\\,F_k`. ``None`` if the gradient was \
          not computed.
    """
    o_loc: SampledObs | None = None
    grad_log_psi: SampledObs | LazySampledObs | None = None
    grad: jax.Array | None = None

class AbstractObjectiveFunction(ABC):
    """
    Loss of a variational state, estimated from samples.

    The loss is :math:`L(\\theta)=\\langle\\ell_\\theta\\rangle_{p_\\theta}`, the mean of a local \
    loss :math:`\\ell_\\theta(s)` over :math:`p_\\theta(s)\\propto|\\psi_\\theta(s)|^2`, real for all \
    real parameters :math:`\\theta`. ``value_and_grad()`` returns the local losses and the force

    .. math::

        F_k=\\mathrm{Cov}_p(\\Gamma_k^*,\\ell)+\\frac{R_k}{2},\\qquad
        R_k=\\langle\\partial_{\\theta_k}\\mathrm{Re}\\,\\ell\\rangle_p
        -2\\langle\\mathrm{Im}\\,\\Gamma_k\\,\\mathrm{Im}\\,\\ell\\rangle_p,

    with :math:`\\Gamma_k=\\partial_{\\theta_k}\\ln\\psi_\\theta`, so that \
    :math:`\\partial_{\\theta_k}L=2\\,\\mathrm{Re}\\,F_k`. The subclasses differ in the local loss \
    :math:`\\ell`, and therefore in :math:`R_k`. The optimizers of ``jVMC_exp.optimizer`` take an \
    objective function and call ``value_and_grad()`` at every step, with the time ``t`` as a \
    keyword argument.

    The averages are weighted with the weights of the sampler, and use the samples it stores: \
    draw new samples after changing the parameters.

    Args:
        * ``batched_jacobian``: If ``True``, the logarithmic derivatives are computed one batch \
          at a time (``NQS.lazy_gradients()``) and never held in memory at once, see \
          ``jVMC_exp.stats.LazySampledObs``.
    """
    def __init__(self, batched_jacobian: bool = False):
        self._batched_jacobian = batched_jacobian

    @property
    def batched_jacobian(self) -> bool:
        """Whether the logarithmic derivatives are computed one batch at a time."""
        return self._batched_jacobian

    @abstractmethod
    def __call__(self, sampler: AbstractSampler, **kwargs) -> SampledObs:
        """
        Local losses on the samples of ``sampler``.

        Args:
            * ``sampler``: Sampler of the variational state.
            * ``**kwargs``: Keyword arguments of the local loss, e.g. the time ``t``.

        Returns:
            The local losses :math:`\\ell_\\theta(s_n)` as a ``SampledObs``.
        """
        pass

    @abstractmethod
    def value_and_grad(
        self, sampler: AbstractSampler, compute_grad: bool = True, **kwargs
    ) -> ObjectiveFunctionOutput:
        """
        Local losses, logarithmic derivatives and force on the samples of ``sampler``.

        Args:
            * ``sampler``: Sampler of the variational state.
            * ``compute_grad``: Whether to compute the force. The logarithmic derivatives are \
              computed in any case.
            * ``**kwargs``: Keyword arguments of the local loss, e.g. the time ``t``.

        Returns:
            ``ObjectiveFunctionOutput``.
        """
        pass

    def _grad_log_psi(self, sampler: AbstractSampler) -> SampledObs | LazySampledObs:
        """Gamma_k(s_i), returned lazily if ``batched_jacobian`` is set."""
        if self._batched_jacobian:
            return LazySampledObs(
                sampler.psi.lazy_gradients(sampler.samples), sampler.weights
            )
        return SampledObs(sampler.psi.gradients(sampler.samples), sampler.weights)

class Observable(AbstractObjectiveFunction):
    """
    Expectation value :math:`L=\\langle\\hat O\\rangle` of a Hermitian operator that does not \
    depend on the parameters, e.g. the energy.

    The local loss is the local estimator \
    :math:`\\ell(s)=O_\\mathrm{loc}(s)=\\sum_{s'}O_{s,s'}\\psi_\\theta(s')/\\psi_\\theta(s)`. Then \
    :math:`R_k=0` and :math:`F_k=\\mathrm{Cov}_p(\\Gamma_k^*,O_\\mathrm{loc})`.

    Args:
        * ``operator``: Hermitian operator :math:`\\hat O`, see :ref:`Operator`.
        * ``batched_jacobian``: See ``AbstractObjectiveFunction``.
    """
    def __init__(self, operator: AbstractOperator, batched_jacobian: bool = False):
        super().__init__(batched_jacobian)
        self._operator = operator
        # TODO: might add a batch size here so that one can have different batch sizes

    @property
    def operator(self):
        """The operator :math:`\\hat O`."""
        return self._operator

    def __call__(self, sampler: AbstractSampler, **op_kwargs):
        """
        Local estimator :math:`O_\\mathrm{loc}` on the samples of ``sampler``, \
        ``sampler(operator, **op_kwargs)``.

        Args:
            * ``sampler``: Sampler of the variational state.
            * ``**op_kwargs``: Keyword arguments of the operator, e.g. the time ``t`` of a \
              time-dependent prefactor.

        Returns:
            ``SampledObs``.
        """
        return sampler(self.operator, **op_kwargs)

    def value_and_grad(self, sampler: AbstractSampler, compute_grad: bool = True, **op_kwargs):
        """
        Local estimator, logarithmic derivatives and force \
        :math:`F_k=\\mathrm{Cov}_p(\\Gamma_k^*,O_\\mathrm{loc})`.

        Args:
            * ``sampler``: Sampler of the variational state.
            * ``compute_grad``: Whether to compute the force.
            * ``**op_kwargs``: Keyword arguments of the operator, e.g. the time ``t``.

        Returns:
            ``ObjectiveFunctionOutput``.
        """
        o_loc = self(sampler, **op_kwargs)
        grad_log_psi = self._grad_log_psi(sampler)

        if compute_grad:
            grad = grad_log_psi.get_covar(o_loc)
            return ObjectiveFunctionOutput(
                o_loc=o_loc, grad=grad, grad_log_psi=grad_log_psi
            )

        return ObjectiveFunctionOutput(o_loc=o_loc, grad_log_psi=grad_log_psi)

class Estimator(AbstractObjectiveFunction):
    """
    Mean of a real function :math:`f_\\theta(s)` of the configuration and of the parameters.

    The local loss is :math:`\\ell_\\theta(s)=f_\\theta(s)`. Since it is real, \
    :math:`R_k=\\langle\\partial_{\\theta_k}f_\\theta\\rangle_p` and

    .. math::

        F_k=\\mathrm{Cov}_p(\\Gamma_k^*,f_\\theta)+\\frac12\\langle\\partial_{\\theta_k}f_\\theta\\rangle_p,

    with :math:`\\partial_{\\theta_k}f_\\theta` from automatic differentiation.

    Args:
        * ``estimator_fn``: The function ``f(parameters, s)``, with the parameters ``psi.parameters`` \
          of the variational state. It is called on all the samples at once, with ``s`` of shape \
          ``(N, *sampleShape)``, and must return the ``N`` real values; for its derivatives it is \
          called on single configurations, of shape ``sampleShape``, and must return a scalar.
        * ``batched_jacobian``: See ``AbstractObjectiveFunction``. It also applies to the \
          derivatives of ``estimator_fn``.
    """
    def __init__(self, estimator_fn: callable, batched_jacobian: bool = False):
        super().__init__(batched_jacobian)
        self._estimator_fn = estimator_fn
        self._grad_fn = None

    @property
    def estimator_fn(self):
        """The function ``f(parameters, s)``."""
        return self._estimator_fn

    def __call__(self, sampler: AbstractSampler, **kwargs):
        """
        Values :math:`f_\\theta(s_n)` on the samples of ``sampler``.

        Args:
            * ``sampler``: Sampler of the variational state.
            * ``**kwargs``: Not used.

        Returns:
            ``SampledObs``.
        """
        observations = self.estimator_fn(sampler.psi.parameters, sampler.samples)
        return SampledObs(observations, sampler.weights)

    def value_and_grad(self, sampler: AbstractSampler, compute_grad: bool = True, **kwargs):
        """
        Values, logarithmic derivatives and force \
        :math:`F_k=\\mathrm{Cov}_p(\\Gamma_k^*,f_\\theta)+\\frac12\\langle\\partial_{\\theta_k}f_\\theta\\rangle_p`.

        Args:
            * ``sampler``: Sampler of the variational state.
            * ``compute_grad``: Whether to compute the force.
            * ``**kwargs``: Not used.

        Returns:
            ``ObjectiveFunctionOutput``.
        """
        value = self(sampler)
        grad_log_psi = self._grad_log_psi(sampler)

        if compute_grad:
            grad = grad_log_psi.get_covar(value) + 0.5 * self.explicit_grad(sampler)
            return ObjectiveFunctionOutput(
                o_loc=value, grad=grad, grad_log_psi=grad_log_psi
            )

        return ObjectiveFunctionOutput(o_loc=value, grad_log_psi=grad_log_psi)

    def explicit_grad(self, sampler: AbstractSampler) -> jax.Array:
        """
        Mean :math:`\\langle\\partial_{\\theta_k}f_\\theta\\rangle_p` of the derivatives of \
        ``estimator_fn`` with respect to the parameters, on the samples of ``sampler``.

        The derivatives are flattened as the columns of ``NQS.gradients()``. The kind of \
        derivative (real, holomorphic or not) is chosen at the first call, for the parameters of \
        ``sampler.psi``.

        Args:
            * ``sampler``: Sampler of the variational state.

        Returns:
            Array with one entry per real parameter.
        """
        if self._grad_fn is None:
            _, _, self._grad_fn, _ = pick_gradient(
                self.estimator_fn, sampler.psi.parameters, sampler.samples[0]
            )

        grad_kwargs = dict(
            parameters=sampler.psi.parameters, batch_size=sampler.psi.batchSize
        )
        if self._batched_jacobian:
            grad_obs = LazySampledObs(
                self._lazy_grad_fn_sh(sampler.samples, **grad_kwargs), sampler.weights
            )
        else:
            grad_obs = SampledObs(
                self._grad_fn_sh(sampler.samples, **grad_kwargs), sampler.weights
            )

        return grad_obs.mean

    @sharded(automatic_sharding=True) # TODO: Set flag to False once jax problem is solved
    def _grad_fn_sh(self, samples, *, parameters, batch_size):
        return self._grad_fn(self.estimator_fn, parameters, samples)

    @sharded(automatic_sharding=True, yield_iter=True) # TODO: Set flag to False once jax problem is solved
    def _lazy_grad_fn_sh(self, samples, *, parameters, batch_size):
        return self._grad_fn(self.estimator_fn, parameters, samples)

class ParametricObservable(AbstractObjectiveFunction):
    """
    Expectation value :math:`L=\\langle\\hat O(\\theta)\\rangle` of an operator that is Hermitian \
    for all :math:`\\theta` and whose matrix elements depend on the parameters.

    The local loss is :math:`\\ell_\\theta(s)=O_\\mathrm{loc}(s;\\theta)`. The dependence on \
    :math:`\\theta` through :math:`\\psi_\\theta` is treated as in ``Observable``, and the one \
    through the matrix elements gives \
    :math:`R_k=\\langle(\\partial_{\\theta_k}\\hat O)_\\mathrm{loc}\\rangle_p`, so that

    .. math::

        F_k=\\mathrm{Cov}_p(\\Gamma_k^*,O_\\mathrm{loc})
        +\\frac12\\langle(\\partial_{\\theta_k}\\hat O)_\\mathrm{loc}\\rangle_p.

    Args:
        * ``operator``: The operator :math:`\\hat O(\\theta)` at the current parameters, which \
          gives the local losses.
        * ``estimator_fn``: Function ``f(parameters, s)`` whose derivatives are the local \
          estimators of the derivatives of the operator, \
          :math:`\\partial_{\\theta_k}f=(\\partial_{\\theta_k}\\hat O)_\\mathrm{loc}(s)=\\sum_{s'}\\partial_{\\theta_k}O_{s,s'}(\\theta)\\,\\psi(s')/\\psi(s)`: \
          the wave function ratios have to be constants, e.g. through \
          ``jax.lax.stop_gradient``. It is called as in ``Estimator``, and only its derivatives \
          are used.
        * ``batched_jacobian``: See ``AbstractObjectiveFunction``.
    """
    def __init__(
            self,
            operator: AbstractOperator,
            estimator_fn: callable,
            batched_jacobian: bool = False
        ):
        super().__init__(batched_jacobian)
        self._observable = Observable(operator, batched_jacobian)
        self._explicit = Estimator(estimator_fn, batched_jacobian)

    def __call__(self, sampler: AbstractSampler, **op_kwargs):
        """
        Local estimator :math:`O_\\mathrm{loc}(s;\\theta)` on the samples of ``sampler``, as in \
        ``Observable``.

        Args:
            * ``sampler``: Sampler of the variational state.
            * ``**op_kwargs``: Keyword arguments of the operator, e.g. the time ``t``.

        Returns:
            ``SampledObs``.
        """
        return self._observable(sampler, **op_kwargs)

    def value_and_grad(self, sampler: AbstractSampler, compute_grad: bool = True, **op_kwargs):
        """
        Local estimator, logarithmic derivatives and force \
        :math:`F_k=\\mathrm{Cov}_p(\\Gamma_k^*,O_\\mathrm{loc})+\\frac12\\langle(\\partial_{\\theta_k}\\hat O)_\\mathrm{loc}\\rangle_p`.

        Args:
            * ``sampler``: Sampler of the variational state.
            * ``compute_grad``: Whether to compute the force.
            * ``**op_kwargs``: Keyword arguments of the operator, e.g. the time ``t``.

        Returns:
            ``ObjectiveFunctionOutput``.
        """
        obs_out = self._observable.value_and_grad(sampler, compute_grad, **op_kwargs)

        if compute_grad:
            grad = obs_out.grad + 0.5 * self._explicit.explicit_grad(sampler)
            return ObjectiveFunctionOutput(
                o_loc=obs_out.o_loc, grad=grad, grad_log_psi=obs_out.grad_log_psi
            )

        return obs_out