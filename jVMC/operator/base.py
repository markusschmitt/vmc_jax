from abc import ABC, abstractmethod

from jVMC.vqs import NQS


class AbstractOperator(ABC):
    """
    Interface of the operators, whose expectation values are estimated from samples.
    """

    @abstractmethod
    def get_O_loc(self, s, psi: NQS, *, log_psi_s=None, **kwargs):
        """
        Compute the local estimator of the operator.

        For an operator :math:`\\hat O`, the local estimator is \
        :math:`O_{loc}(s)=\\sum_{s'}O_{s,s'}\\psi(s')/\\psi(s)`, whose average over \
        :math:`|\\psi(s)|^2` is the expectation value :math:`\\langle\\hat O\\rangle`.

        Args:
            * ``s``: Configurations, of shape ``(N, *sample_shape)``.
            * ``psi``: Variational state.
            * ``log_psi_s``: :math:`\\ln\\psi(s)`, if already known. Otherwise, it is computed \
              when needed.
            * ``**kwargs``: Keyword arguments of the callable prefactors of the operator, e.g., \
              the time.

        Returns:
            :math:`O_{loc}(s)`, with one entry per configuration along the first axis.
        """
