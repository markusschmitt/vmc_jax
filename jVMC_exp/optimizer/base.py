import jax
import jax.numpy as jnp
from abc import ABC, abstractmethod
import tqdm
from typing import Dict, List
import warnings
from typing import Callable
from functools import partial

from jVMC_exp import global_defs
from jVMC_exp.stats import LazySampledObs, SampledObs
from jVMC_exp.vqs import NQS
from jVMC_exp.sampler.base import AbstractSampler
from jVMC_exp.sampler import ExactSampler
from jVMC_exp.util import make_cmplx_array, make_real_array, remove_double
from jVMC_exp.util.output_manager import OutputManager
from jVMC_exp.stepper import AbstractStepper, Euler
from jVMC_exp.util import ObservableEntry, measure
from jVMC_exp.solver.base import AbstractSolver
from jVMC_exp.solver.pinv import PinvSNR
from jVMC_exp.objective_function.base import AbstractObjectiveFunction, ObjectiveFunctionOutput
from jVMC_exp.sharding_config import sharded, MESH, DEVICE_SHARDING

def _init_qgt(grad_log_psi):
    n_params = grad_log_psi.shape[1]

    return jnp.zeros((n_params, n_params), grad_log_psi.dtype, device=DEVICE_SHARDING)

class AbstractOptimizer(ABC):
    """
    Base class of the optimizers, which change the parameters of a variational state with the \
    force of an objective function.

    An optimizer is the right hand side of the equation that a stepper of ``jVMC_exp.stepper`` \
    integrates: ``__call__(parameters, t, ...)`` samples the state at ``parameters``, evaluates \
    the objective function and returns the update :math:`\\dot\\theta` of the parameters. \
    ``ground_state_search()`` and ``time_evolution()`` repeat such steps and record at every \
    step the mean, the variance and the statistical error of the local losses as ``"energy"``, \
    together with the requested observables, in the ``output_manager``.

    Args:
        * ``sampler``: Sampler of ``psi``.
        * ``psi``: Variational state, ``jVMC_exp.vqs.NQS``.
        * ``resample_stepper``: If ``True``, the sampler draws new samples at every substep of \
          the stepper. Otherwise the samples of the first substep are reused, with their \
          weights multiplied by :math:`|\\psi(s)/\\psi_0(s)|^{1/\\kappa}`, where :math:`\\psi_0` \
          is the state at the first substep and :math:`\\kappa` the ``logProbFactor`` of the \
          sampler.
        * ``use_cross_valiadation``: Whether ``cross_validation()`` runs at the first substep of \
          every step.
        * ``output_manager`` [optional]: ``jVMC_exp.util.output_manager.OutputManager`` that \
          records the data. A new one by default.
    """
    def __init__(
            self, sampler: AbstractSampler, psi: NQS, resample_stepper: bool=True,
            use_cross_valiadation: bool=False, output_manager: OutputManager | None = None
        ):
        self._sampler = sampler
        self._psi = psi
        self._resample = resample_stepper
        self.use_cross_valiadation = use_cross_valiadation
        self.meta_data = {}
        self._output_manager = output_manager if output_manager is not None else OutputManager()
        self.o_loc = None
        self._elapsed = 0
        self._sampler_out = (None,) * 3

    @property
    def output_manager(self):
        """``OutputManager`` that records the observables, the metadata and the timings."""
        return self._output_manager

    @property
    def sampler(self):
        """Sampler of the variational state."""
        return self._sampler
    
    @property
    def psi(self):
        """Variational state."""
        return self._psi
    
    @property
    @abstractmethod
    def _needs_grad(self) -> bool:
        pass

    def __call__(
            self, parameters, t, *,
            numSamples=None, intStep=None,
            objective_function: AbstractObjectiveFunction, **objective_function_kwargs
    ):
        """
        Update of the parameters at ``parameters``: the right hand side for the stepper.

        Samples the state at ``parameters`` (see ``resample_stepper``), evaluates \
        ``objective_function.value_and_grad()`` and returns ``get_update()`` of its output. \
        ``psi.parameters`` is restored afterwards. At the first substep, ``intStep=0``, the \
        samples, the local losses and the information of the solver are kept for the \
        measurements, and ``cross_validation()`` runs if it is enabled. The time spent on the \
        sampling, on the objective function and on the solve is recorded in \
        ``output_manager``.

        Args:
            * ``parameters``: Flat parameters, as ``psi.parameters_flat``.
            * ``t``: Time, passed to the objective function as the keyword argument ``t``.
            * ``numSamples`` [optional]: Number of samples, passed to ``sampler.sample()``.
            * ``intStep``: Index of the substep of the stepper.
            * ``objective_function``: ``AbstractObjectiveFunction``.
            * ``**objective_function_kwargs``: Further keyword arguments of \
              ``objective_function.value_and_grad()``.

        Returns:
            The update, a real array with the shape of ``psi.parameters_flat``.

        :meta public:
        """
        tmp_parameters = self.psi.parameters
        self.psi.parameters = parameters
        if intStep is None or intStep == 0:
            self._elapsed = 0

        def stop_timing(name, wait_for=None):
            if wait_for is not None:
                jax.block_until_ready(wait_for)
            return self.output_manager.stop_timing(name)

        # Get sample
        if self._resample or intStep == 0:
            self.output_manager.start_timing("sampling")
            sampler_out = self.sampler.sample(numSamples=numSamples)
            self._elapsed += stop_timing("sampling", wait_for=sampler_out[0])
        # Importance sampling
        elif not self._resample and intStep != 0:
            _, logPsi0, w0 = self._sampler_out
            self.sampler._logPsi = self.psi(self.sampler.samples)
            # The samples are distributed as |psi|^(1 / logProbFactor) at the first substep
            self.sampler._weights = w0 * jnp.exp(
                jnp.real(self.sampler.logPsi - logPsi0) / self.sampler.logProbFactor
            )

        # Evaluate local observables and their gradient
        self.output_manager.start_timing("compute objective function and gradient")
        objective_fn_out = objective_function.value_and_grad(
            self.sampler,
            compute_grad=self._needs_grad, 
            t=t,
            **objective_function_kwargs
        )
        self._elapsed += stop_timing(
            "compute objective function and gradient", 
            wait_for=(objective_fn_out.o_loc, objective_fn_out.grad)
        )

        # Obtain the update from the gradients
        self.output_manager.start_timing("solve")
        update = self.get_update(objective_fn_out)
        self._elapsed += stop_timing("solve")

        self.psi.parameters = tmp_parameters

        if intStep is not None:
            if intStep == 0:
                self._sampler_out = sampler_out
                self.o_loc = objective_fn_out.o_loc
                self._update_meta_data()

                if self.use_cross_valiadation:
                    self.output_manager.start_timing("cross_validation")
                    self.cross_validation(objective_function, t=t, **objective_function_kwargs)
                    self._elapsed += stop_timing("cross_validation")

        return update
    
    def step(self, t, stepper: AbstractStepper, objective_function: AbstractObjectiveFunction, **kwargs):
        """
        One step of ``stepper`` from ``psi.parameters_flat``, with this optimizer as the right \
        hand side. ``psi.parameters`` does not change.

        Args:
            * ``t``: Time.
            * ``stepper``: Stepper of ``jVMC_exp.stepper``.
            * ``objective_function``: ``AbstractObjectiveFunction``.
            * ``**kwargs``: Keyword arguments of the stepper and of ``__call__()``.

        Returns:
            ``new_parameters, dt``: The flat parameters after the step and the time step.
        """
        return stepper.step(
            t,
            self,
            self.psi.parameters_flat,
            objective_function=objective_function,
            **kwargs
        )
    
    def ground_state_search(
            self,
            steps,
            objective_function: AbstractObjectiveFunction,
            stepper: AbstractStepper = Euler(),
            observables: Dict[str, ObservableEntry] | None = None,
            callback: List[Callable] | None = None,
            save_meta_data: bool = False,
            **kwargs # KWARGS ARE FOR BOTH THE STEPPER AND THE OBJECTIVE FUNCTION, TODO: ADD DOCUMENTATION ON THIS
        ):
        """
        Minimize ``objective_function`` with ``steps`` steps of ``stepper``.

        Before each step ``n``, ``stepper.update_dt(n)`` and ``update_hyperparams(n)`` are \
        called. The observables of a step are measured with the samples of the step, before \
        the parameters are updated. The progress bar shows the mean of the local losses.

        Args:
            * ``steps``: Number of steps.
            * ``objective_function``: ``AbstractObjectiveFunction`` to minimize.
            * ``stepper``: Stepper with an ``update_dt(step)`` method, e.g. \
              ``jVMC_exp.stepper.Euler(dt)``, whose time step plays the role of a learning rate.
            * ``observables`` [optional]: Observables to measure at every step, as in \
              ``jVMC_exp.util.measure()``: a dictionary from names to operators, callables \
              returning a ``SampledObs``, or ``(operator, kwargs)`` tuples.
            * ``callback`` [optional]: Callable, or list of callables, called without arguments \
              after the measurements of every step.
            * ``save_meta_data``: Whether to record ``meta_data`` at every step.
            * ``**kwargs``: Keyword arguments of the stepper (e.g. ``normFunction``) and of the \
              objective function.

        Returns:
            ``output_manager.data["observables"]``: a dictionary with the step numbers in \
            ``"times"``, the lists ``"mean"``, ``"variance"`` and ``"MC_error"`` of the local \
            losses in ``"energy"``, and the same for each observable. With ``save_meta_data``, \
            also ``output_manager.data["metadata"]``. Raises ``ValueError`` if the stepper has no \
            ``update_dt()`` method.
        """
        if not hasattr(stepper, "update_dt"):
            raise ValueError(
                "For ground state search the stepper must "
                "implement a method called 'update_dt'"
            )

        pbar = tqdm.tqdm(range(steps), disable=jax.process_index() != 0)
        for n in pbar:
            stepper.update_dt(n)
            self.update_hyperparams(n)

            # Updating the parameters after the measurements, 
            # allows to reuse the samples for the measurement of the observables
            # saving one sample call per step
            new_parameters, _ = self.step(0, stepper, objective_function, **kwargs)
            self.sampler._samples, self.sampler._logPsi, self.sampler._weights = self._sampler_out     
            self._measure_and_store(n, observables, callback, save_meta_data)
            self.psi.parameters = new_parameters

            pbar.set_postfix(E=f"{self.o_loc}")
  
        self.output_manager.print_timings()
        
        if save_meta_data:
            return self.output_manager.data['observables'], self.output_manager.data['metadata']
        
        return self.output_manager.data["observables"]

    def time_evolution(
            self,
            t_max,
            objective_function: AbstractObjectiveFunction,
            stepper: AbstractStepper,
            observables: Dict[str, ObservableEntry] | None = None,
            callback: List[Callable] | None = None,
            save_meta_data: bool = False,
            **kwargs
        ):
        """
        Integrate the equation of motion of the parameters with ``stepper`` up to the time \
        ``t_max``.

        The observables are measured at the time of every step, with its samples, and at the \
        final time, with new samples. ``"energy"`` holds the local losses of the steps, so that \
        its entry at the final time repeats the one of the last step. The last step is the one \
        that reaches or passes ``t_max``.

        Args:
            * ``t_max``: Final time.
            * ``objective_function``: ``AbstractObjectiveFunction``, e.g. \
              ``Observable(hamiltonian)`` for the evolution generated by the Hamiltonian.
            * ``stepper``: Stepper of ``jVMC_exp.stepper``, with a fixed or an adaptive time \
              step, e.g. ``jVMC_exp.stepper.Euler(dt)``.
            * ``observables``, ``callback``, ``save_meta_data``, ``**kwargs``: As in \
              ``ground_state_search()``.

        Returns:
            As ``ground_state_search()``, with the times in ``"times"``.
        """

        pbar = tqdm.tqdm(total=t_max, disable=jax.process_index() != 0)
        t = 0
        while t < t_max:
            new_parameters, dt = self.step(t, stepper, objective_function, **kwargs) 
            self.sampler._samples, self.sampler._logPsi, self.sampler._weights = self._sampler_out
            dt = float(dt)
            self.meta_data['dt'] = dt 
            self._measure_and_store(t, observables, callback, save_meta_data)
            self.psi.parameters = new_parameters

            if t + dt >= t_max:
                t += dt
                break
            t += dt            

            pbar.update(1)
            pbar.set_postfix({
                "t": f"{t:.4f}/{t_max}",
                "dt": f"{dt:.2e}",
                "ETA": f"{(t_max - t) / dt * self._elapsed:.1f}s",
                'Progress': f'{int(t / t_max * 100)}%',
                'E': f"{self.o_loc}",
            })
        
        self.meta_data['dt'] = dt
        self.sampler.sample()
        self._measure_and_store(t, observables, callback, save_meta_data)

        pbar.close()
        self.output_manager.print_timings()

        if save_meta_data:
            return self.output_manager.data['observables'], self.output_manager.data['metadata']
        
        return self.output_manager.data["observables"]
    
    def update_hyperparams(self, step):
        """
        Update the hyperparameters for the step ``step`` of ``ground_state_search()``. Does \
        nothing by default.
        """
        pass

    @abstractmethod
    def get_update(self, objective_function_output: ObjectiveFunctionOutput):
        """
        Update of the parameters from the output of the objective function.

        Args:
            * ``objective_function_output``: ``ObjectiveFunctionOutput``.

        Returns:
            The update, with the shape of ``psi.parameters_flat``.
        """
        self.update = None
        self._additional_info = None

    @abstractmethod
    def cross_validation(self, objective_function: AbstractObjectiveFunction, **objective_function_kwargs):
        """Cross validation of the update at the first substep, see the subclasses."""
        pass

    @abstractmethod
    def _update_meta_data(self):
        '''
        Update the dictionary self.meta_data
        '''
        self.meta_data = {}

    def _measure_and_store(self, t, observables, callback, save_meta_data):
        measures = {}
        energy = dict(
            mean = jnp.real(self.o_loc.mean).squeeze(),
            variance = jnp.real(self.o_loc.var).squeeze(),
            MC_error = jnp.real(self.o_loc.error_of_mean).squeeze()
        )

        if observables is not None:
            measures = measure(observables, self.sampler)
        self.output_manager.write_observables(t, energy=energy, **measures)

        if save_meta_data:
            self.output_manager.write_metadata(t, **self.meta_data)

        if callback is not None:
            if callable(callback):
                callback = (callback,)
            elif not hasattr(callback, '__len__'):
                raise ValueError(
                    'Callback has to be a callable or a list of callables, '
                    f'got {callback}.'
                )
            for cb in callback:
                cb()

class Evolution(AbstractOptimizer):
    """
    Base class of the optimizers that solve the equation of the time-dependent variational \
    principle, ``SR`` and ``TDVP``.

    The update :math:`\\dot\\theta` solves

    .. math::

        S\\dot\\theta=c\\,F,\\qquad S_{kk'}=\\mathrm{Cov}_p(\\Gamma_k^*,\\Gamma_{k'}),

    with the force :math:`F` of the objective function, and :math:`c=-1` in imaginary time or \
    :math:`c=-i` in real time. For a holomorphic network the equation is solved for the complex \
    parameters. Otherwise the parameters are real, and the equation is made real: with \
    ``make_real=True`` it is :math:`\\mathrm{Re}(S)\\dot\\theta=\\mathrm{Re}(cF)`, with \
    ``make_real=False`` :math:`\\mathrm{Im}(S)\\dot\\theta=\\mathrm{Im}(cF)`.

    Before the solve, the diagonal of the (transformed) matrix is multiplied by \
    ``1 + diagonalScale``, and ``diagonalShift`` is added to it. A solver that needs a dense \
    matrix (``solver._needs_dense_matrix``) gets :math:`S` as an array sharded across the \
    devices along its rows, with zero rows and columns appended so that the number of rows is \
    a multiple of the square of the number of devices. A matrix-free solver gets the function \
    :math:`v\\mapsto Sv`.

    Args:
        * ``sampler``, ``psi``, ``resample_stepper``, ``use_cross_valiadation``, \
          ``output_manager``: See ``AbstractOptimizer``.
        * ``imag_time``: Imaginary time, :math:`c=-1`, or real time, :math:`c=-i`.
        * ``make_real``: How the equation is made real for real parameters, see above.
        * ``diagonalShift``: Constant added to the diagonal, or a callable ``step -> value``.
        * ``diagonalScale``: Relative increase of the diagonal, or a callable ``step -> value``.
        * ``solver``: Solver of ``jVMC_exp.solver``, called as \
          ``solver(A, b, grad_log_psi=..., o_loc=..., **solver_state)``.
        * ``store_qgt``: Whether to keep :math:`S`, see ``qgt``.
        * ``compute_tdvp_error``: Whether to compute the TDVP error of every update, which \
          requires ``store_qgt``, see ``get_update()``.
    """
    def __init__(
            self, sampler: AbstractSampler, psi: NQS, resample_stepper: bool,
            imag_time: bool, make_real: bool, use_cross_valiadation: bool=False, 
            diagonalShift: float | Callable=1e-3, diagonalScale: float | Callable=0., 
            solver: AbstractSolver=PinvSNR(), output_manager: OutputManager | None = None,
            store_qgt: bool = False, compute_tdvp_error: bool = False
        ):
        self.rhsPrefactor = -1 if imag_time else -1j 
        if psi.holomorphic:
            self._lhs_trans_fn = lambda x: x
        elif make_real:
            self._lhs_trans_fn = lambda x: jnp.real(x)
        else:
            self._lhs_trans_fn = lambda x: 1j * jnp.imag(x)
        self._rhs_trans_fn = lambda x: self._lhs_trans_fn(self.rhsPrefactor * x)

        self.diag_scale = diagonalScale
        self.diag_shift = diagonalShift

        self._make_cmplx_fn = jax.jit(partial(make_cmplx_array, params_shape=psi.paramShapes))
        self._make_real_fn = jax.jit(partial(make_real_array, params_shape=psi.paramShapes))
        remove_double_fn = partial(remove_double, params_shape=psi.paramShapes)
        self._remove_double_trans = jax.vmap(remove_double_fn)
        self._remove_double_fn = jax.jit(remove_double_fn)
        
        if jax.process_index() == 0:
            warnings.warn(
                "Naming convention changed: "
                "'diag_shift' now adds a constant term to the diagonal, "
                "while 'diag_scale' multiplies the diagonal entries. "
                "This is the opposite of the previous convention.",
                UserWarning,
            )

        self._solver = solver
        self._get_lhs = self._get_lhs_dense if solver._needs_dense_matrix else self._get_lhs_lazy
       
        super().__init__(
            sampler, psi, resample_stepper, use_cross_valiadation, output_manager=output_manager
        )

        double_params = (not psi.realParams) and (not psi.holomorphic)
        num_params = psi.numParameters * (2 if double_params  else 1)
        # S is built one block of rows per device at a time (see _add_qgt),
        # which needs the number of parameters to be divisible by the number of devices squared
        self._params_pad_size = (- num_params) % MESH.shape["devices"] ** 2
        self._pad_obs = jax.jit(
            lambda x: jnp.pad(x, ((0, 0), (0, self._params_pad_size)), mode="constant")
        )
        self._unpad_obs = jax.jit(
            lambda x: x[:, :-self._params_pad_size] if self._params_pad_size else x
        )

        self._solver_state = dict(
            exact_sampler=isinstance(self.sampler, ExactSampler),
            pad_size=self._params_pad_size,
            transformation=self._rhs_trans_fn
        )

        self._qgt = None

        if compute_tdvp_error and not store_qgt:
            raise ValueError(
                "TDVP error computation requires storing the QGT. "
                "Set store_qgt=True when constructing the optimizer."
            )
        self.store_qgt = store_qgt
        self.compute_tdvp_error = compute_tdvp_error

    @property
    def qgt(self):
        """
        Quantum geometric tensor :math:`S` of the last solve, before the transformation and the \
        regularization, if ``store_qgt``: the padded sharded matrix, or the function \
        :math:`v\\mapsto Sv` for a matrix-free solver. ``None`` otherwise.
        """
        return self._qgt

    @property
    def solver(self):
        """The solver."""
        return self._solver
    
    @property
    def solver_state(self):
        """
        Keyword arguments passed to the solver: ``exact_sampler``, ``pad_size`` and \
        ``transformation``, the function that makes the right hand side real.
        """
        return self._solver_state
    
    @property
    def diag_scale(self):
        """
        Relative increase of the diagonal at the current step. It can be set to a number or to \
        a callable ``step -> value``, evaluated by ``update_hyperparams()``.
        """
        return self._diag_scale
    
    @diag_scale.setter
    def diag_scale(self, value):
        self._diag_scale_fn = value if isinstance(value, Callable) else lambda step: value
        self._diag_scale = self._diag_scale_fn(0)

    @property
    def diag_shift(self):
        """
        Constant added to the diagonal at the current step. It can be set to a number or to a \
        callable ``step -> value``, evaluated by ``update_hyperparams()``.
        """
        return self._diag_shift
    
    @diag_shift.setter
    def diag_shift(self, value):
        self._diag_shift_fn = value if isinstance(value, Callable) else lambda step: value
        self._diag_shift = self._diag_shift_fn(0)
    
    @property
    def _needs_grad(self):
        return True

    def update_hyperparams(self, step):
        """Evaluate ``diag_shift`` and ``diag_scale`` for the step ``step``."""
        self._diag_shift = self._diag_shift_fn(step)
        self._diag_scale = self._diag_scale_fn(step)
    
    def get_update(self, objective_function_output: ObjectiveFunctionOutput):
        """
        Solve the equation for the output of the objective function.

        The information returned by the solver becomes ``meta_data`` at the first substep. \
        With ``compute_tdvp_error``, it includes the TDVP error

        .. math::

            r^2=\\left|1+\\frac{\\mathrm{Re}(\\dot\\theta^\\dagger S\\dot\\theta)
            -2\\,\\mathrm{Re}(\\dot\\theta^\\dagger cF)}{\\mathrm{Var}(\\ell)}\\right|

        as ``"tdvp_error"``, the squared distance between :math:`\\partial_t\\psi` from the update \
        and the exact one, relative to the variance of the local losses.

        Args:
            * ``objective_function_output``: ``ObjectiveFunctionOutput``.

        Returns:
            The update, a real array in the layout of ``psi.parameters_flat``.
        """
        if self.psi.holomorphic:
            objective_function_output.grad_log_psi.transform(self._remove_double_trans)
            objective_function_output.grad = self._remove_double_fn(
                objective_function_output.grad
            )

        b = self._get_rhs(objective_function_output.grad)
        A = self._get_lhs(objective_function_output.grad_log_psi)
        
        update, self._additional_info = self.solver(
            A, b, 
            grad_log_psi=objective_function_output.grad_log_psi, 
            o_loc=objective_function_output.o_loc,
            **self.solver_state
        )
        self.update = self._make_real_fn(update) if self.psi.holomorphic else jnp.real(update)

        if self.compute_tdvp_error:
            self._additional_info["tdvp_error"] = self._get_tdvp_error(
                self.update, objective_function_output.grad, objective_function_output.o_loc.var
            ).item() 

        return self.update

    def _get_tdvp_error(self, update, F, var):
        update = self._make_cmplx_fn(update) if self.psi.holomorphic else update

        if callable(self.qgt):
            Sv = self.qgt(update)
        else:
            Sv = self.qgt.dot(jnp.pad(update, (0, self._params_pad_size)))
            Sv = Sv[:-self._params_pad_size] if self._params_pad_size else Sv

        return jnp.abs(
            1. 
            + (jnp.real(jnp.vdot(update, Sv))
            - 2 * jnp.real(jnp.vdot(update, self.rhsPrefactor  * F)))
            / (var + 1e-14)
        )
    
    def _get_lhs_dense(self, grad_log_psi: SampledObs | LazySampledObs):
        '''
        Returns left hand side of the TDVP equation with shape (n_parameters, n_parameters)
        and sharded across devices on the first dimension.
        If n_parameters is not divisible by the number of devices, the output is padded.
        '''
        if self._params_pad_size != 0:
            grad_log_psi.transform(self._pad_obs)
        
        if isinstance(grad_log_psi, SampledObs):
            S = self._add_qgt(
                _init_qgt(grad_log_psi._normalized_obs), 
                grad_log_psi._normalized_obs,
                batch_size=None
            )
        else:
            mean = 0
            S = None
            batches = iter(grad_log_psi._observations)
            for weights in grad_log_psi._weights:
                batch, batch_mean = self._weight_and_mean(
                    next(batches), weights, batch_size=None
                )
                mean += batch_mean

                if S is None:
                    S = _init_qgt(batch)
                S = self._add_qgt(S, batch, batch_size=None)

                del batch

            S = self._subtract_mean_outer(S, mean, batch_size=None)

        if self._params_pad_size != 0:
            grad_log_psi.transform(self._unpad_obs)

        if self.store_qgt:
            self._qgt = S
        S = self._lhs_trans_fn(S)

        if self.diag_scale > 1e-15:
            idx = jnp.arange(S.shape[0] - self._params_pad_size)
            S = S.at[idx, idx].multiply(1 + self.diag_scale)
        if self.diag_shift > 1e-15:
            idx = jnp.arange(S.shape[0] - self._params_pad_size)
            S = S.at[idx, idx].add(self.diag_shift)

        return S

    @sharded(use_vmap=False, donate_argnums=1)
    def _add_qgt(self, S, grad_log_psi, *, batch_size):
        """
        Returns ``S + conj(grad_log_psi).T @ grad_log_psi``, with ``S`` sharded across devices
        on the first axis. ``S`` is donated, so it is updated in place.

        Each device sums over its own samples, and the partial sums are reduced across devices
        one block of rows at a time: the partial sum of a block has as many rows as the share
        of ``S`` of one device, so no device ever holds a matrix of the size of the full ``S``.
        The number of parameters has to be divisible by the number of devices squared.
        """
        n_devices = MESH.size
        n_samples, n_params = grad_log_psi.shape
        block = n_params // n_devices**2
        grad_blocks = grad_log_psi.reshape(n_samples, n_devices, n_devices, block)
        # The products are summed over chunks of at most global_defs.MATMUL_CHUNK_BYTES of samples: XLA
        # autotunes a matrix product on its own copies of the operands, which would otherwise be as
        # large as grad_log_psi
        row_bytes = n_params * grad_log_psi.dtype.itemsize
        chunk = max(1, min(n_samples, (global_defs.MATMUL_CHUNK_BYTES or n_samples * row_bytes) // row_bytes))

        def add_block(j, S):
            # Block j of the rows of every device, in device order
            rows = jax.lax.dynamic_index_in_dim(
                grad_blocks, j, axis=2, keepdims=False
            ).reshape(n_samples, n_devices * block)

            def add_chunk(k, local, size=chunk):
                r, g = (
                    jax.lax.dynamic_slice_in_dim(x, k * chunk, size) 
                    for x in (rows, grad_log_psi)
                )
                # Materialized: XLA would otherwise fuse the slices 
                # into the product and autotune it on whole operands
                r, g = jax.lax.optimization_barrier((r, g))
                return local + jnp.tensordot(jnp.conj(r), g, axes=(0, 0))

            local = jax.lax.fori_loop(1, n_samples // chunk, add_chunk, add_chunk(0, 0))
            if n_samples % chunk:
                local = add_chunk(n_samples // chunk, local, n_samples % chunk)
            local = jax.lax.psum_scatter(local, "devices", scatter_dimension=0, tiled=True)
            S_block = jax.lax.dynamic_slice_in_dim(S, j * block, block, axis=0)

            return jax.lax.dynamic_update_slice_in_dim(S, S_block + local, j * block, axis=0)

        return jax.lax.fori_loop(0, n_devices, add_block, S)

    @sharded(use_vmap=False, donate_argnums=1)
    def _subtract_mean_outer(self, S, mean, *, batch_size):
        """
        Returns ``S - outer(conj(mean), mean)``, with ``S`` and ``mean`` sharded across devices
        on the first axis. ``S`` is donated, so it is updated in place.
        """
        full_mean = jax.lax.all_gather(mean, "devices", tiled=True)

        return S - jnp.outer(jnp.conj(mean), full_mean)

    @sharded(use_vmap=False)
    def _weight_and_mean(self, batch, weights, *, batch_size):
        weighted_batch = jnp.einsum("i, i... -> i...", jnp.sqrt(weights), batch)
        mean = jax.lax.psum_scatter(
            jnp.tensordot(weights, batch, axes=(0, 0)), 
            "devices",
            tiled=True
        )

        return weighted_batch, mean
    
    def _get_lhs_lazy(self, grad_log_psi: SampledObs | LazySampledObs):
        '''
        Returns a function that computes the matrix vector product with the left hand side of the TDVP equation
        '''
        if isinstance(grad_log_psi, LazySampledObs):
            raise ValueError(
                f"Solver '{type(self.solver).__name__}' is matrix-free (_needs_dense_matrix=False) "
                "and builds its matvec directly from the fully materialized Jacobian, but the "
                "Jacobian was computed in batches (LazySampledObs), which has no such materialized "
                "array. Either use a solver with _needs_dense_matrix=True (e.g. PinvSNR), or "
                "compute the Jacobian without batching."
            )

        grad = grad_log_psi._normalized_obs
        
        def raw_matvec(v):
            return (grad.conj().T @ (grad @ v))

        if self.store_qgt:
            self._qgt = raw_matvec

        def matvec(v):
            Sv = self._lhs_trans_fn(raw_matvec(v))
            if self.diag_scale > 1e-15:
                diag = jnp.sum(jnp.abs(grad) ** 2, axis=0)
                Sv = Sv + self.diag_scale * diag * v
            if self.diag_shift > 1e-15:
                Sv = Sv + self.diag_shift * v

            return Sv

        return matvec
    
    def _get_rhs(self, grad):
        b = self._rhs_trans_fn(grad)
        b.block_until_ready()

        return b

    def _update_meta_data(self):
        self.meta_data = dict(**self._additional_info)