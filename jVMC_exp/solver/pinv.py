import jax.numpy as jnp
import jax
from typing import Literal

from jVMC_exp.solver.base import AbstractSolver
from jVMC_exp.solver.util import diagonalize, smooth_cutoff_fn
from jVMC_exp.stats import LazySampledObs, SampledObs

@jax.jit
def _get_snr(Vtb, Vtb_var, num_samples):
    return jnp.sqrt(
        jnp.abs(num_samples * (jnp.conj(Vtb) * Vtb) / (Vtb_var + 1e-14))
    ).ravel()

@jax.jit
def _get_snr_gaussian(Vtb, ev, num_samples, var):
    return jnp.sqrt(
        jnp.abs(num_samples * (jnp.conj(Vtb) * Vtb) / (jnp.abs(var * ev) + 1e-14))
    ).ravel()

@jax.jit(static_argnums=(4, 5))
def _get_vtb_var(centered_data_1, centered_data_2, weights, eigenvectors, transformation, pad_size):
    """
    Weighted variance over the samples of ``V^H transformation(conj(O_n) E_n)``, for centered
    observations ``O_n`` and ``E_n``, without the ``pad_size`` modes of the padding.
    """
    @jax.vmap
    def _get_covar_per_sample(A, B):
        return jnp.outer(jnp.conj(A), B)

    x = _get_covar_per_sample(centered_data_1, centered_data_2).squeeze(-1)
    # The eigenvectors have pad_size trailing rows of padding
    x = jnp.pad(transformation(x), ((0, 0), (0, pad_size)))
    rho = jnp.einsum("ij, ki -> kj", jnp.conjugate(eigenvectors), x)
    rho = rho - jnp.tensordot(weights, rho, axes=(0, 0))

    var = jnp.tensordot(weights, jnp.abs(rho)**2, axes=(0, 0))

    return var[:var.shape[0] - pad_size]

def _unpad(x, pad_size):
    return x[:-pad_size] if pad_size else x

def _snr_step_gaussian(
        Vtb, ev, o_loc: SampledObs      
    ):
    if o_loc is None: 
        raise ValueError(
            "PinvSNR has snr_tol != 0, but o_loc for the "
            "Pass grad_log_psi, o_loc, and transformation to PinvSNR.__call__, "
            "or set snr_tol=0 to disable SNR regularization."
        )

    return _get_snr_gaussian(Vtb, ev, o_loc.effective_num_samples, o_loc.var)

def _snr_step(
        grad_log_psi: SampledObs, o_loc: SampledObs, 
        V, Vtb, transformation, pad_size
    ):
    if isinstance(grad_log_psi, LazySampledObs):
        raise NotImplementedError(
            "PinvSNR's signal-to-noise regularization (snr_tol != 0) requires a densely "
            "materialized Jacobian, but grad_log_psi is a LazySampledObs (a batched/lazy "
            "Jacobian). Construct the Observable/objective function with "
            "batched_jacobian=False to enable SNR regularization, or set "
            "snr_tol=0 to disable it."
        )
    if grad_log_psi is None or o_loc is None or transformation is None:
        missing = [
            name for name, value in (
                ("grad_log_psi", grad_log_psi), ("o_loc", o_loc), ("transformation", transformation)
            ) if value is None
        ]
        raise ValueError(
            f"PinvSNR has snr_tol != 0, but the following required argument(s) for the "
            f"exact SNR computation were not given: {', '.join(missing)}. Pass "
            "grad_log_psi, o_loc, and transformation to PinvSNR.__call__, "
            "or set snr_tol=0 to disable SNR regularization."
        )

    Vtb_var = _get_vtb_var(
        grad_log_psi._centered_obs,
        o_loc._centered_obs,
        o_loc.weights,
        V,
        transformation,
        pad_size
    )

    return _get_snr(Vtb, Vtb_var, o_loc.effective_num_samples)

class Pinv(AbstractSolver):
    """
    Pseudo-inverse solver, from the eigendecomposition :math:`A=V\\,\\mathrm{diag}(\\lambda)\\,V^\\dagger` \
    of a dense Hermitian matrix.

    The solution is :math:`x=V\\,(\\lambda^+\\circ V^\\dagger b)`, without assembling the \
    pseudo-inverse, with :math:`\\lambda^+_i=1/\\lambda_i` if \
    :math:`|\\lambda_i/\\lambda_\\mathrm{max}|>` ``pinv_cutoff`` and :math:`\\lambda^+_i=0` \
    otherwise, where :math:`\\lambda_\\mathrm{max}` is the largest eigenvalue. The eigenvectors \
    stay in the layout of the diagonalization, sharded and padded, see \
    ``jVMC_exp.solver.util.diagonalize()``.

    The dictionary ``info`` of the solution holds

    * ``"residual"``: The fraction of :math:`b` that the solution does not reproduce, \
      :math:`\\|(\\lambda^+\\circ\\lambda-1)\\circ V^\\dagger b\\|/\\|b\\|`,
    * ``"condition_number"``: :math:`\\lambda_\\mathrm{max}/\\min_i|\\lambda_i|`, over all the \
      eigenvalues.

    Args:
        * ``pinv_cutoff``: Relative cutoff of the eigenvalues.
        * ``diagonalization_mode``: Backend of ``jVMC_exp.solver.util.diagonalize()``, \
          ``"device"``, ``"host"`` or ``"distributed"``.
        * ``T_A`` [optional]: Tile size of ``diagonalize()`` for \
          ``diagonalization_mode="distributed"``.
    """
    def __init__(
            self, pinv_cutoff=1e-14,
            diagonalization_mode: Literal["device", "distributed", "host"] = "device",
            T_A: int | None = None
        ):
        self._pinv_cutoff = pinv_cutoff
        self._diagonalization_mode = diagonalization_mode
        self._T_A = T_A

    @property
    def pinv_cutoff(self):
        """Relative cutoff of the eigenvalues; for ``PinvSNR``, the smallest adaptive cutoff."""
        return self._pinv_cutoff
    
    @property
    def _needs_dense_matrix(self) -> bool:
        return True

    def _diagonalize(self, A, b, pad_size):
        ev, V = diagonalize(
            A, pad_size, mode=self._diagonalization_mode, T_A=self._T_A
        )
        ev = _unpad(ev, pad_size)
        if jnp.max(jnp.abs(ev)) < 1e-14:
            raise RuntimeError(
                f"Largest eigenvalue of the matrix A to invert is {jnp.max(jnp.abs(ev))}. "
                "A is most likely highly ill-conditioned/zero "
            )
        
        Vtb = _unpad(
            jnp.dot(jnp.transpose(jnp.conj(V)), jnp.pad(b, (0, pad_size))),
            pad_size
        )

        return ev, V, Vtb

    def __call__(self, A, b, *, pad_size=0, **kwargs):
        """
        Solve :math:`Ax=b` with the pseudo-inverse.

        Args:
            * ``A``: Hermitian matrix, with ``pad_size`` rows and columns of zero padding at \
              the end.
            * ``b``: Right hand side, without padding.
            * ``pad_size``: Number of rows and columns of padding of :math:`A`.
            * ``**kwargs``: Ignored.

        Returns:
            ``x, info``: The solution, without padding, and the dictionary of the class \
            description.

        Raises ``RuntimeError`` if all the eigenvalues are smaller than :math:`10^{-14}` in \
        absolute value.
        """
        ev, V, Vtb = self._diagonalize(A, b, pad_size)
        
        inv_ev = jnp.where(jnp.abs(ev / ev[-1]) > self.pinv_cutoff, 1. / ev, 0.)
        b_norm = jnp.linalg.norm(b) 
        residual = jnp.linalg.norm((inv_ev * ev - 1) * Vtb) / b_norm

        x = _unpad(
            jnp.dot(V, jnp.pad((inv_ev * Vtb), (0, pad_size))),
            pad_size
        )

        info = dict(
            condition_number=(ev[-1] / jnp.min(jnp.abs(ev))).item(),
            residual=residual.item(),
        )

        return x, info

class PinvSNR(Pinv):
    """
    Pseudo-inverse solver with a smooth, adaptive regularization.

    As in ``Pinv``, the solution is computed in the eigenbasis of :math:`A`, \
    :math:`x=V\\,(\\lambda^+\\circ r\\circ V^\\dagger b)`, where :math:`\\lambda^+_i=1/\\lambda_i` if \
    :math:`|\\lambda_i/\\lambda_\\mathrm{max}|>10^{-14}` and 0 otherwise, and the regularization \
    factors are

    .. math::

        r_i=r\\left(\\left|\\lambda_i/\\lambda_\\mathrm{max}\\right|;\\epsilon\\right)\\,
        r\\left(\\mathrm{SNR}_i;\\mathrm{snr\\_tol}\\right),\\qquad r(x;c)=\\frac{1}{1+(c/x)^6},

    see ``jVMC_exp.solver.util.smooth_cutoff_fn()``: a mode with :math:`x=c` gets half its weight.

    The relative cutoff :math:`\\epsilon` is adaptive. It starts at the larger of \
    :math:`8\\cdot10^{-3}` and ``pinv_cutoff``, and is multiplied by 0.8, but never lowered \
    below ``pinv_cutoff``, until the residual \
    :math:`\\|(\\lambda^+\\circ\\lambda\\circ r-1)\\circ V^\\dagger b\\|/\\|b\\|` is at most \
    ``pinv_tol``. With the default ``pinv_tol`` it ends at ``pinv_cutoff``.

    The second factor suppresses the modes whose component of :math:`b` is not resolved by the \
    samples. It is skipped if ``snr_tol`` is zero (default) or the samples are exact \
    (``exact_sampler=True``). The signal-to-noise ratio of the component \
    :math:`(V^\\dagger b)_i` is

    .. math::

        \\mathrm{SNR}_i=\\sqrt{\\frac{N_\\mathrm{eff}\\,|(V^\\dagger b)_i|^2}{\\sigma_i^2}},

    with the effective number of samples :math:`N_\\mathrm{eff}=1/\\sum_n w_n^2` of ``o_loc``. \
    The variance :math:`\\sigma_i^2` depends on ``snr_mode``:

    * ``"empiric"``: The variance over the samples of :math:`\\rho_{n,i}=(V^\\dagger b_n)_i`, \
      where :math:`b_n=T\\big((\\Gamma(s_n)-\\langle\\Gamma\\rangle)^*(\\ell(s_n)-\\langle\\ell\\rangle)\\big)` \
      are the contributions of the samples to \
      :math:`b=T\\big(\\mathrm{Cov}(\\Gamma^*,\\ell)\\big)=\\sum_n w_n b_n`, with the logarithmic \
      derivatives :math:`\\Gamma` (``grad_log_psi``), the local losses :math:`\\ell` (``o_loc``) \
      and the function :math:`T` that gives the right hand side of the optimizer \
      (``transformation``). It needs the logarithmic derivatives as a ``SampledObs``.
    * ``"gaussian"``: :math:`\\sigma_i^2=\\mathrm{Var}(\\ell)\\,|\\lambda_i|`, which only needs \
      ``o_loc``.

    The dictionary ``info`` of the solution holds ``"residual"``, the final cutoff \
    ``"pinv_cutoff"``, ``"snr"`` (the array of :math:`\\mathrm{SNR}_i`, or ``None`` if skipped), \
    ``"condition_number"`` (as in ``Pinv``), the eigenvalues ``"spectrum"``, and \
    ``"effective_rank"``, the mean of the regularization factors :math:`r_i`.

    Args:
        * ``snr_tol``: Signal-to-noise ratio at which a mode gets half its weight. Zero turns the \
          signal-to-noise regularization off.
        * ``pinv_tol``: Target residual of the adaptive cutoff.
        * ``pinv_cutoff``: Smallest value of the adaptive cutoff :math:`\\epsilon`.
        * ``snr_mode``: ``"empiric"`` or ``"gaussian"``, not case sensitive.
        * ``diagonalization_mode``, ``T_A``: Diagonalization, see ``Pinv``.
    """
    def __init__(
            self, snr_tol=0, pinv_tol=1e-14, pinv_cutoff=1e-8,
            snr_mode: Literal["empiric", "gaussian"] = "empiric", 
            diagonalization_mode: Literal["device", "distributed", "host"] = "device",
            T_A: int | None = None
        ):
        super().__init__(pinv_cutoff, diagonalization_mode, T_A)

        self._snr_tol = snr_tol
        self._pinv_tol = pinv_tol
        self.snr_mode = snr_mode

    @property
    def snr_tol(self):
        """Signal-to-noise ratio at which a mode gets half its weight, zero if off."""
        return self._snr_tol
    
    @property
    def pinv_tol(self):
        """Target residual of the adaptive cutoff."""
        return self._pinv_tol

    @property
    def snr_mode(self):
        """
        Estimate of the noise, ``"empiric"`` or ``"gaussian"``. It can be set, not case \
        sensitive: other values raise ``ValueError``.
        """
        return self._snr_mode

    @snr_mode.setter
    def snr_mode(self, value: str):
        if not isinstance(value, str):
            raise ValueError(
                f"The snr_mode has to be given as a string. Got {value}"
            )
        if value.lower() not in ["empiric", "gaussian"]:
            raise ValueError(
                f"The snr_mode can only be 'empiric' or 'gaussian'. Got {value}"
            )

        self._snr_mode = value.lower()

    def __call__(
            self, A, b,
            *,
            grad_log_psi: None | SampledObs = None, o_loc: None | SampledObs = None,
            transformation=None, pad_size=0, exact_sampler=False,
            **kwargs
        ):
        """
        Solve :math:`Ax=b` with the regularized pseudo-inverse.

        Args:
            * ``A``: Hermitian matrix, with ``pad_size`` rows and columns of zero padding at \
              the end.
            * ``b``: Right hand side, without padding.
            * ``grad_log_psi``: Logarithmic derivatives :math:`\\Gamma`, a ``SampledObs``, for \
              ``snr_mode="empiric"``.
            * ``o_loc``: Local losses :math:`\\ell`, a ``SampledObs``, for the \
              signal-to-noise ratio.
            * ``transformation``: The function :math:`T` with :math:`b=T(F)` for the force \
              :math:`F`, for ``snr_mode="empiric"``.
            * ``pad_size``: Number of rows and columns of padding of :math:`A`.
            * ``exact_sampler``: Whether the samples are exact, which skips the \
              signal-to-noise regularization.
            * ``**kwargs``: Ignored.

        Returns:
            ``x, info``: The solution, without padding, and the dictionary of the class \
            description.

        With the signal-to-noise regularization, raises ``ValueError`` if an argument it needs \
        is ``None``, and ``NotImplementedError`` if ``snr_mode="empiric"`` and ``grad_log_psi`` \
        is a ``LazySampledObs``.
        """
        # Keep V padded so that it doesn't get replicated on devices
        ev, V, Vtb = self._diagonalize(A, b, pad_size)

        snr = None
        if not exact_sampler and self.snr_tol:
            if self.snr_mode == "empiric":
                snr = _snr_step(
                    grad_log_psi, o_loc, V, Vtb, transformation, pad_size
                )
            elif self.snr_mode == "gaussian":
                snr = _snr_step_gaussian(
                    Vtb, ev, o_loc
                )

        # Discard eigenvalues below numerical precision
        invEv = jnp.where(jnp.abs(ev / ev[-1]) > 1e-14, 1. / ev, 0.)

        b_norm = jnp.linalg.norm(b) 
        residual, cutoff, pinvEv, effective_rank = self._regularize(
            snr, ev, invEv, Vtb, b_norm, exact_sampler
        )

        x = _unpad(
            jnp.dot(V, jnp.pad((pinvEv * Vtb), (0, pad_size))),
            pad_size
        )

        info = dict(
            residual=residual.item(),
            pinv_cutoff=cutoff.item(),
            snr=snr,
            condition_number=(ev[-1] / jnp.min(jnp.abs(ev))).item(),
            spectrum=ev,
            effective_rank=effective_rank.item()
        )

        return x, info

    @jax.jit(static_argnums=(0, 6))
    def _regularize(self, snr, eigenvalues, invEv, Vtb, b_norm, exact_sampler):
        def _step(cutoff):
            cutoff = jnp.maximum(0.8 * cutoff, self.pinv_cutoff)
            regularizer = smooth_cutoff_fn(jnp.abs(eigenvalues / eigenvalues[-1]), cutoff)

            if not exact_sampler and snr is not None:
                regularizer = regularizer * smooth_cutoff_fn(snr, self.snr_tol)
            
            pinvEv = invEv * regularizer
            residual = jnp.linalg.norm((pinvEv * eigenvalues - 1) * Vtb) / b_norm

            return residual, cutoff, pinvEv, jnp.mean(regularizer)

        def _cond(state):
            residual, cutoff, _, _ = state
            return (residual > self.pinv_tol) & (cutoff > self.pinv_cutoff)

        return jax.lax.while_loop(_cond, lambda s: _step(s[1]), _step(1e-2))