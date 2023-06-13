import jax
import jax.numpy as jnp
import numpy as np

import jVMC
import jVMC.mpi_wrapper as mpi
import jVMC.global_defs as global_defs
from jVMC.stats import SampledObs

import warnings


def realFun(x):
    return jnp.real(x)


def imagFun(x):
    return 0.5 * (x - jnp.conj(x))


class TDVP:
    """ This class provides functionality to solve a time-dependent variational principle (TDVP).

    With the force vector

        :math:`F_k=\langle \mathcal O_{\\theta_k}^* E_{loc}^{\\theta}\\rangle_c`

    and the quantum Fisher matrix

        :math:`S_{k,k'} = \langle (\mathcal O_{\\theta_k})^* \mathcal O_{\\theta_{k'}}\\rangle_c`

    and for real parameters :math:`\\theta\in\mathbb R`, the TDVP equation reads

        :math:`q\\big[S_{k,k'}\\big]\\theta_{k'} = -q\\big[xF_k\\big]`

    Here, either :math:`q=\\text{Re}` or :math:`q=\\text{Im}` and :math:`x=1` for ground state
    search or :math:`x=i` (the imaginary unit) for real time dynamics.

    For ground state search a regularization controlled by a parameter :math:`\\rho` can be included
    by increasing the diagonal entries and solving

        :math:`q\\big[(1+\\rho\delta_{k,k'})S_{k,k'}\\big]\\theta_{k'} = -q\\big[F_k\\big]`

    The `TDVP` class solves the TDVP equation by computing a pseudo-inverse of :math:`S` via
    eigendecomposition yielding

        :math:`S = V\Sigma V^\dagger`

    with a diagonal matrix :math:`\Sigma_{kk}=\sigma_k`
    Assuming that :math:`\sigma_1` is the smallest eigenvalue, the pseudo-inverse is constructed 
    from the regularized inverted eigenvalues

        :math:`\\tilde\sigma_k^{-1}=\\frac{1}{\\Big(1+\\big(\\frac{\epsilon_{SVD}}{\sigma_j/\sigma_1}\\big)^6\\Big)\\Big(1+\\big(\\frac{\epsilon_{SNR}}{\\text{SNR}(\\rho_k)}\\big)^6\\Big)}`

    with :math:`\\text{SNR}(\\rho_k)` the signal-to-noise ratio of :math:`\\rho_k=V_{k,k'}^{\dagger}F_{k'}` (see `[arXiv:1912.08828] <https://arxiv.org/pdf/1912.08828.pdf>`_ for details).

    Initializer arguments:
        * ``sampler``: A sampler object.
        * ``snrTol``: Regularization parameter :math:`\epsilon_{SNR}`, see above.
        * ``pinvTol``: Regularization parameter :math:`\epsilon_{SVD}`, see above.
        * ``makeReal``: Specifies the function :math:`q`, either `'real'` for :math:`q=\\text{Re}` or `'imag'` for :math:`q=\\text{Im}`.
        * ``rhsPrefactor``: Prefactor :math:`x` of the right hand side, see above.
        * ``diagonalShift``: Regularization parameter :math:`\\rho` for ground state search, see above.
        * ``crossValidation``: Perform cross-validation check as introduced in `[arXiv:2105.01054] <https://arxiv.org/pdf/2105.01054.pdf>`_.
        * ``diagonalizeOnDevice``: Choose whether to diagonalize :math:`S` on GPU or CPU.
    """

    def __init__(self, sampler, snrTol=2, pinvTol=1e-14, svdTol=None, makeReal='imag', rhsPrefactor=1.j, diagonalShift=0., crossValidation=False, diagonalizeOnDevice=True):
        
        if svdTol is not None:
            warnings.warn("Parameter `svdTol` is deprecated. Use `pinvTol` instead.", DeprecationWarning)
            pinvTol = svdTol
            
        self.sampler = sampler
        self.snrTol = snrTol
        self.pinvTol = pinvTol
        self.diagonalShift = diagonalShift
        self.rhsPrefactor = rhsPrefactor
        self.crossValidation = crossValidation

        self.diagonalizeOnDevice = diagonalizeOnDevice

        self.metaData = None

        self.makeReal = realFun
        if makeReal == 'imag':
            self.makeReal = imagFun

        # pmap'd member functions
        self.makeReal_pmapd = global_defs.pmap_for_my_devices(jax.vmap(lambda x: self.makeReal(x)))

    def set_diagonal_shift(self, delta):
        self.diagonalShift = delta

    def set_cross_validation(self, crossValidation=True):
        self.crossValidation = crossValidation

    def _get_tdvp_error(self, update):

        return jnp.abs(1. + jnp.real(update.dot(self.S0.dot(update)) - 2. * jnp.real(update.dot(self.F0))) / self.ElocVar0)

    def get_residuals(self):

        return self.metaData["tdvp_error"], self.metaData["tdvp_residual"]

    def get_snr(self):

        return self.metaData["SNR"]

    def get_spectrum(self):

        return self.metaData["spectrum"]

    def get_metadata(self):

        return self.metaData

    def get_energy_variance(self):

        return self.ElocVar0

    def get_energy_mean(self):

        return jnp.real(self.ElocMean0)

    def get_S(self):

        return self.S

    def get_tdvp_equation(self, Eloc, gradients):

        self.ElocMean = Eloc.mean()[0]
        self.ElocVar = Eloc.var()[0]

        self.F0 = (-self.rhsPrefactor) * gradients.covar(Eloc).ravel()
        F = self.makeReal(self.F0)

        self.S0 = gradients.covar()
        S = self.makeReal(self.S0)

        if self.diagonalShift > 1e-10:
            S = S + jnp.diag(self.diagonalShift * jnp.diag(S))

        return S, F

    def get_sr_equation(self, Eloc, gradients):
        return self.get_tdvp_equation(Eloc, gradients, rhsPrefactor=1.)

    def _transform_to_eigenbasis(self, S, F):
        
        if self.diagonalizeOnDevice:
            try:
                self.ev, self.V = jnp.linalg.eigh(S)
            except ValueError:
                warnings.warn("jax.numpy.linalg.eigh raised an exception. Falling back to numpy.linalg.eigh for "
                              "diagonalization.", RuntimeWarning)
                tmpS = np.array(S)
                tmpEv, tmpV = np.linalg.eigh(tmpS)
                self.ev = jnp.array(tmpEv)
                self.V = jnp.array(tmpV)
        else:
            tmpS = np.array(S)
            tmpEv, tmpV = np.linalg.eigh(tmpS)
            self.ev = jnp.array(tmpEv)
            self.V = jnp.array(tmpV)

        self.VtF = jnp.dot(jnp.transpose(jnp.conj(self.V)), F)

    def _get_snr(self, Eloc, gradients):

        EO = gradients.covar_data(Eloc).transform(
                        fun=lambda x: jnp.matmul(
                                        jnp.transpose(jnp.conj(self.V)), 
                                        self.makeReal((-self.rhsPrefactor) * x)
                                        )
                    )
        self.rhoVar = EO.var().ravel()

        self.snr = jnp.sqrt(jnp.abs(mpi.globNumSamples * (jnp.conj(self.VtF) * self.VtF) / self.rhoVar)).ravel()

    def solve(self, Eloc, gradients):
        # Get TDVP equation from MC data
        self.S, F = self.get_tdvp_equation(Eloc, gradients)
        F.block_until_ready()

        # Transform TDVP equation to eigenbasis and compute SNR
        self._transform_to_eigenbasis(self.S, F) #, Fdata)
        self._get_snr(Eloc, gradients)

        # Discard eigenvalues below numerical precision
        self.invEv = jnp.where(jnp.abs(self.ev / self.ev[-1]) > 1e-14, 1. / self.ev, 0.)

        # Set regularizer for singular value cutoff
        regularizer = 1. / (1. + (self.pinvTol / jnp.abs(self.ev / self.ev[-1]))**6)

        if not isinstance(self.sampler, jVMC.sampler.ExactSampler):
            # Construct a soft cutoff based on the SNR
            regularizer *= 1. / (1. + (self.snrTol / self.snr)**6)

        update = jnp.real(jnp.dot(self.V, (self.invEv * regularizer * self.VtF)))

        return update, jnp.linalg.norm(self.S.dot(update) - F) / jnp.linalg.norm(F)

    def solve_minSR(self, eloc, gradients, p=None, r_pinv=1e-8):
        """
        Uses the techique proposed in arXiv:2302.01941 to compute the updates.
        Efficient only if number of samples << number of parameters.
        """

        # Collect all gradients & local energies
        def soft_cutoff(eigvals):
            return 1 / (eigvals * (1 + (r_pinv * jnp.max(eigvals) / eigvals)**6))

        def hard_cutoff(eigvals):
            return jnp.where(eigvals / jnp.max(eigvals) > r_pinv, 1 / eigvals, 0)

        eloc = mpi.gather(eloc).reshape((-1,))
        gradients = mpi.gather(gradients).reshape((-1, gradients.shape[-1]))
        n_samples = eloc.shape[0]

        eloc_bar = (eloc - jnp.mean(eloc)) / jnp.sqrt(n_samples)
        gradients_bar = (gradients - jnp.mean(gradients, axis=0)) / jnp.sqrt(n_samples)

        T = gradients_bar @ gradients_bar.conj().T
        eigvals, eigvecs = jnp.linalg.eigh(T)
        inv_eigvals = hard_cutoff(eigvals)
        T_inv = eigvecs @ jnp.diag(inv_eigvals) @ eigvecs.conj().T
        self.update = - self.rhsPrefactor * gradients_bar.conj().T @ T_inv @ eloc_bar

        return self.update, None

    def S_dot(self, v):

        return jnp.dot(self.S0, v)

    def __call__(self, netParameters, t, *, psi, hamiltonian, **rhsArgs):
        """ For given network parameters this function solves the TDVP equation.

        This function returns :math:`\\dot\\theta=S^{-1}F`. Thereby an instance of the ``TDVP`` class is a suited
        callable for the right hand side of an ODE to be used in combination with the integration schemes 
        implemented in ``jVMC.stepper``. Alternatively, the interface matches the scipy ODE solvers as well.

        Arguments:
            * ``netParameters``: Parameters of the NQS.
            * ``t``: Current time.
            * ``psi``: NQS ansatz. Instance of ``jVMC.vqs.NQS``.
            * ``hamiltonian``: Hamiltonian operator, i.e., an instance of a derived class of ``jVMC.operator.Operator``. \
                                *Notice:* Current time ``t`` is by default passed as argument when computing matrix elements. 

        Further optional keyword arguments:
            * ``numSamples``: Number of samples to be used by MC sampler.
            * ``outp``: An instance of ``jVMC.OutputManager``. If ``outp`` is given, timings of the individual steps \
                are recorded using the ``OutputManger``.
            * ``intStep``: Integration step number of multi step method like Runge-Kutta. This information is used to store \
                quantities like energy or residuals at the initial integration step.

        Returns:
            The solution of the TDVP equation, :math:`\\dot\\theta=S^{-1}F`.
        """

        tmpParameters = psi.get_parameters()
        psi.set_parameters(netParameters)

        outp = None
        if "outp" in rhsArgs:
            outp = rhsArgs["outp"]
        self.outp = outp

        numSamples = None
        if "numSamples" in rhsArgs:
            numSamples = rhsArgs["numSamples"]

        def start_timing(outp, name):
            if outp is not None:
                outp.start_timing(name)

        def stop_timing(outp, name, waitFor=None):
            if waitFor is not None:
                waitFor.block_until_ready()
            if outp is not None:
                outp.stop_timing(name)

        # Get sample
        start_timing(outp, "sampling")
        sampleConfigs, sampleLogPsi, p = self.sampler.sample(numSamples=numSamples)
        stop_timing(outp, "sampling", waitFor=sampleConfigs)

        # Evaluate local energy
        start_timing(outp, "compute Eloc")
        Eloc = hamiltonian.get_O_loc(sampleConfigs, psi, sampleLogPsi, t)
        stop_timing(outp, "compute Eloc", waitFor=Eloc)
        Eloc = SampledObs( Eloc, p)

        # Evaluate gradients
        start_timing(outp, "compute gradients")
        sampleGradients = psi.gradients(sampleConfigs)
        stop_timing(outp, "compute gradients", waitFor=sampleGradients)
        sampleGradients = SampledObs( sampleGradients, p)

        start_timing(outp, "solve TDVP eqn.")
        update, solverResidual = self.solve(Eloc, sampleGradients)
        stop_timing(outp, "solve TDVP eqn.")

        if outp is not None:
            outp.add_timing("MPI communication", mpi.get_communication_time())

        psi.set_parameters(tmpParameters)

        if "intStep" in rhsArgs:
            if rhsArgs["intStep"] == 0:

                self.ElocMean0 = self.ElocMean
                self.ElocVar0 = self.ElocVar

                self.metaData = {
                    "tdvp_error": self._get_tdvp_error(update),
                    "tdvp_residual": solverResidual,
                    "SNR": self.snr, 
                    "spectrum": self.ev,
                }

                if self.crossValidation:

                    if isinstance(self.sampler, jVMC.sampler.ExactSampler):
                        update_1, _ = self.solve(Eloc[:, 0::2], sampleGradients[:, 0::2], p[:, 0::2])
                        S2, F2, _ = self.get_tdvp_equation(Eloc[:, 1::2], sampleGradients[:, 1::2], p[:, 1::2])
                    else:
                        update_1, _ = self.solve(Eloc[:, 0::2], sampleGradients[:, 0::2])
                        S2, F2, _ = self.get_tdvp_equation(Eloc[:, 1::2], sampleGradients[:, 1::2])

                    validation_tdvpErr = self._get_tdvp_error(update_1)
                    update, solverResidual = self.solve(Eloc, sampleGradients, p)
                    validation_residual = (jnp.linalg.norm(S2.dot(update_1) - F2) / jnp.linalg.norm(F2)) / solverResidual

                    self.crossValidationFactor_residual = validation_residual
                    self.crossValidationFactor_tdvpErr = validation_tdvpErr / self.metaData["tdvp_error"]

                    self.S, _, _ = self.get_tdvp_equation(Eloc, sampleGradients, p)

        return update
