from typing import Dict, List, Callable

from .base import Evolution
from jVMC_exp.solver import PinvSNR
from jVMC_exp.objective_function import AbstractObjectiveFunction
from jVMC_exp.stepper import AbstractStepper, Euler
from jVMC_exp.util import ObservableEntry
from jVMC_exp.util.output_manager import OutputManager

class SR(Evolution):
    """
    Stochastic reconfiguration, for ground state searches.

    Imaginary time evolution, :math:`S\\dot\\theta=-F` for a holomorphic network and \
    :math:`\\mathrm{Re}(S)\\dot\\theta=-\\mathrm{Re}(F)` otherwise, see ``Evolution``. With \
    the Euler stepper, a step changes the parameters by :math:`\\Delta t\\,\\dot\\theta`. \
    ``time_evolution()`` raises ``NotImplementedError``: real-time evolution is done by ``TDVP``.

    Args:
        * ``sampler``: Sampler of ``psi``.
        * ``psi``: Variational state.
        * ``diagonalShift``: Constant added to the diagonal of :math:`S`, or a callable \
          ``step -> value``.
        * ``diagonalScale``: Relative increase of the diagonal of :math:`S`, or a callable \
          ``step -> value``.
        * ``solver``: Solver of ``jVMC_exp.solver``.
        * ``resample_stepper``: See ``AbstractOptimizer``.
        * ``output_manager`` [optional]: See ``AbstractOptimizer``.
        * ``store_qgt``: Whether to keep :math:`S`, see ``Evolution.qgt``.
    """
    def __init__(
            self, sampler, psi, 
            diagonalShift=0.001, diagonalScale=0., solver=PinvSNR(),
            resample_stepper=True, output_manager: OutputManager | None = None,
            store_qgt: bool = False
        ):
        super().__init__(
            sampler, psi, resample_stepper, True, True, 
            False, diagonalShift, diagonalScale, solver,
            output_manager=output_manager, store_qgt=store_qgt,
            compute_tdvp_error=False
        )

    def time_evolution(
            self,
            t_max,
            objective_function: AbstractObjectiveFunction,
            stepper: AbstractStepper = Euler,
            observables: Dict[str, ObservableEntry] | None = None,
            callback: List[Callable] | None = None,
            save_meta_data: bool = False,
            **kwargs
        ):
        """Not available for SR: raises ``NotImplementedError``, use ``TDVP``."""
        raise NotImplementedError(
            "The SR optimizer is intended for ground state search only and "
            "cannot be used for time evolution. Use jVMC_exp.optimizer.TDVP instead."
        )

    def cross_validation(self):
            """Does nothing: SR has no cross validation."""
            pass