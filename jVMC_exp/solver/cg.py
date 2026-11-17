from .base import AbstractSolver
from jax.scipy.sparse.linalg import cg

class CG(AbstractSolver):
    """
    Conjugate gradient solver, ``jax.scipy.sparse.linalg.cg``.

    It is matrix-free: :math:`A` is the function :math:`v\\mapsto Av`, which the optimizer \
    computes from the logarithmic derivatives without assembling the matrix. The method assumes \
    a Hermitian positive definite :math:`A`, and starts from :math:`x=0`.

    Args:
        * ``tol``: Relative tolerance: the iteration stops when \
          :math:`\\|Ax-b\\|\\leq\\mathrm{tol}\\,\\|b\\|`.
        * ``maxiter`` [optional]: Maximal number of iterations. If ``None``, ten times the length \
          of :math:`b`.
    """
    def __init__(self, tol: float = 1e-5, maxiter: int | None = None):
        self._tol = tol
        self._maxiter = maxiter

    @property
    def tol(self):
        """Relative tolerance."""
        return self._tol

    @property
    def maxiter(self):
        """Maximal number of iterations, ``None`` for ten times the length of :math:`b`."""
        return self._maxiter

    def __call__(self, A, b, **kwargs):
        """
        Solve :math:`Ax=b`.

        Args:
            * ``A``: Function :math:`v\\mapsto Av`.
            * ``b``: Right hand side.
            * ``**kwargs``: Ignored.

        Returns:
            ``x, info``: The solution and an empty dictionary.
        """
        # The current jax implementaiton of cg does not return any info
        x, info = cg(A, b, tol=self.tol, maxiter=self.maxiter)

        return x, {}

    @property
    def _needs_dense_matrix(self):
        return False