from optax import adam
import tqdm
import jax.numpy as jnp
from typing import Dict, List, Callable

from jVMC_exp.optimizer.base import AbstractOptimizer
from jVMC_exp.objective_function.base import ObjectiveFunctionOutput
from jVMC_exp.objective_function.base import AbstractObjectiveFunction
from jVMC_exp.util import ObservableEntry
from jVMC_exp.util.output_manager import OutputManager
from jVMC_exp.stepper import Euler

class Adam(AbstractOptimizer):
    """
    Ground state search with the Adam optimizer of ``optax``.

    Adam gets the gradient :math:`\\partial_{\\theta_k}L=2\\,\\mathrm{Re}\\,F_k` of the objective \
    function, and every step of ``ground_state_search()`` adds its update to the parameters. \
    ``time_evolution()`` raises ``NotImplementedError``.

    Args:
        * ``sampler``: Sampler of ``psi``.
        * ``psi``: Variational state.
        * ``learning_rate``: Learning rate.
        * ``b1``, ``b2``: Decay rates of the first and second moments of the gradient.
        * ``eps``: Constant added to the root of the second moment.
        * ``output_manager`` [optional]: See ``AbstractOptimizer``.
    """
    def __init__(
            self, sampler, psi,
            learning_rate: float=1e-3, b1: float=0.9, b2: float=0.999, eps: float=1e-8,
            output_manager: OutputManager | None = None
    ):
        super().__init__(sampler, psi, True, False, output_manager=output_manager)

        self._adam = adam(learning_rate, b1, b2, eps)
        self._opt_state = self._adam.init(self.psi.parameters_flat)
        self._stepper = Euler(1)

    @property
    def _needs_grad(self):
        return True

    def get_update(self, objective_function_out: ObjectiveFunctionOutput):
        """
        Adam update for the gradient :math:`2\\,\\mathrm{Re}\\,F`: the change of the flat \
        parameters, learning rate included. It advances the state of Adam.

        Args:
            * ``objective_function_out``: ``ObjectiveFunctionOutput`` with the force.

        Returns:
            The update, with the shape of ``psi.parameters_flat``.
        """
        grad = 2 * jnp.real(objective_function_out.grad)
        update, self._opt_state = self._adam.update(grad, self._opt_state, self.psi.parameters_flat)

        return update
    
    def _update_meta_data(self):
        pass

    def cross_validation(self):
        """Does nothing: Adam has no cross validation."""
        pass

    def ground_state_search(
            self,
            steps,
            objective_function: AbstractObjectiveFunction,
            observables: Dict[str, ObservableEntry] | None = None,
            callback: List[Callable] | None = None,
            save_meta_data: bool = False,
            **objective_function_kwargs
        ):
        """
        Minimize ``objective_function`` with ``steps`` steps of Adam, as \
        ``AbstractOptimizer.ground_state_search()`` without a stepper.

        Args:
            * ``steps``: Number of steps.
            * ``objective_function``: ``AbstractObjectiveFunction`` to minimize.
            * ``observables``, ``callback``, ``save_meta_data``: As in \
              ``AbstractOptimizer.ground_state_search()``.
            * ``**objective_function_kwargs``: Keyword arguments of the objective function.

        Returns:
            As ``AbstractOptimizer.ground_state_search()``.
        """
        pbar = tqdm.tqdm(range(steps))
        for n in pbar: 
            self._stepper.update_dt(n)
            self.update_hyperparams(n)

            # Updating the parameters after the measurements, 
            # allows to reuse the samples for the measurement of the observables
            # saving one sample call per step
            new_parameters, _ = self.step(0, self._stepper, objective_function, **objective_function_kwargs)
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
            observables: Dict[str, ObservableEntry] | None = None,
            callback: List[Callable] | None = None,
            save_meta_data: bool = False,
            **kwargs
        ):
        """Not available for Adam: raises ``NotImplementedError``."""
        raise NotImplementedError(
            "The Adam optimizer is intended for ground state search only and "
            "cannot be used for time evolution. Use jVMC_exp.optimizer.TDVP instead."
        )