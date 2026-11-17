from abc import ABC, abstractmethod

class AbstractSolver(ABC):
    """
    Solver of the linear equation :math:`Ax=b` of an optimizer.

    The optimizers of ``jVMC_exp.optimizer`` call a solver as ``x, info = solver(A, b, **kwargs)``, \
    with the (regularized) quantum geometric tensor or tangent kernel :math:`A` and the right hand \
    side :math:`b`, and keep the dictionary ``info`` as their ``meta_data``. Depending on \
    ``_needs_dense_matrix``, :math:`A` is

    * a Hermitian matrix, sharded across the devices along its rows, whose last ``pad_size`` rows \
      and columns are zero padding (``True``), or
    * a function :math:`v\\mapsto Av` (``False``).

    The keyword arguments are those of the ``solver_state`` of the optimizer: ``pad_size``, \
    ``exact_sampler`` and, for ``SR`` and ``TDVP``, ``transformation``, together with the \
    ``grad_log_psi`` and ``o_loc`` of the objective function. A solver takes the ones it needs \
    and ignores the others.
    """
    @property
    @abstractmethod
    def _needs_dense_matrix(self) -> bool:
        """Whether :math:`A` is passed as a matrix (``True``) or as a function (``False``)."""
        pass

    @abstractmethod
    def __call__(self, A, b, **kwargs):
        """
        Solve :math:`Ax=b`.

        Args:
            * ``A``: Matrix or function :math:`v\\mapsto Av`, see ``_needs_dense_matrix``.
            * ``b``: Right hand side, a vector without padding.
            * ``**kwargs``: Keyword arguments of the optimizer, see the class description.

        Returns:
            ``x, info``: The solution, a vector without padding, and a dictionary of diagnostics.
        """
        pass