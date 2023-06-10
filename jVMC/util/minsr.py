import jax
import jax.numpy as jnp

import jVMC.mpi_wrapper as mpi
import jVMC.global_defs as global_defs
from jVMC.stats import SampledObs

def realFun(x):
    return jnp.real(x)


def imagFun(x):
    return 0.5 * (x - jnp.conj(x))


class MinSR:
    """ This class provides functionality for energy minimization via MinSR.

    See `[arXiv:2302.01941] <https://arxiv.org/abs/2302.01941>`_ for details.

    Initializer arguments:
        * ``sampler``: A sampler object.
        * ``svdTol``: Regularization parameter :math:`\epsilon_{SVD}`, see above.
        * ``makeReal``: Specifies the function :math:`q`, either `'real'` for :math:`q=\\text{Re}` or `'imag'` for :math:`q=\\text{Im}`.
        * ``diagonalizeOnDevice``: Choose whether to diagonalize :math:`S` on GPU or CPU.
    """

    def __init__(self, sampler, pinvTol=1e-14, makeReal='imag', diagonalizeOnDevice=True):
        self.sampler = sampler
        self.pinvTol = pinvTol

        self.diagonalizeOnDevice = diagonalizeOnDevice

        self.metaData = None

        self.makeReal = realFun
        if makeReal == 'imag':
            self.makeReal = imagFun

        # pmap'd member functions
        self.makeReal_pmapd = global_defs.pmap_for_my_devices(jax.vmap(lambda x: self.makeReal(x)))

    def set_pinv_tol(self, tol):

        self.pinvTol = tol

    def get_metadata(self):

        return self.metaData

    def get_energy_variance(self):

        return self.ElocVar0

    def get_energy_mean(self):

        return jnp.real(self.ElocMean0)

    def solve(self, eloc, gradients):
        """
        Uses the techique proposed in arXiv:2302.01941 to compute the updates.
        Efficient only if number of samples << number of parameters.
        """

        # Collect all gradients & local energies
        # eloc = mpi.gather(eloc).reshape((-1,))
        # gradients = mpi.gather(gradients).reshape((-1, gradients.shape[-1]))
        # n_samples = eloc.shape[0]

        # eloc_bar = (eloc - jnp.mean(eloc)) / jnp.sqrt(n_samples)
        # gradients_bar = (gradients - jnp.mean(gradients, axis=0)) / jnp.sqrt(n_samples)

        # def hard_cutoff(eigvals):
        #     return jnp.where(eigvals / jnp.max(eigvals) > self.pinvTol, 1 / eigvals, 0)

        # T = gradients_bar @ gradients_bar.conj().T
        # eigvals, eigvecs = jnp.linalg.eigh(T)
        # inv_eigvals = hard_cutoff(eigvals)
        # T_inv = eigvecs @ jnp.diag(inv_eigvals) @ eigvecs.conj().T
        # self.update = - self.rhsPrefactor * gradients_bar.conj().T @ T_inv @ eloc_bar
        T = gradients.tangent_kernel()
        T_inv = jnp.linalg.pinv(T, rcond=self.pinvTol, hermitian=True)

        eloc_all = mpi.gather(eloc._data).reshape((-1,))
        gradients_all = mpi.gather(gradients._data)
        update = - gradients_all.conj().T @ T_inv @ eloc_all

        return update

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
            The solution of the MinSR equation, :math:`\\dot\\theta=S^{-1}F`.
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

        start_timing(outp, "solve MinSR eqn.")
        update = self.solve(Eloc, sampleGradients)
        stop_timing(outp, "solve MinSR eqn.")

        if outp is not None:
            outp.add_timing("MPI communication", mpi.get_communication_time())

        psi.set_parameters(tmpParameters)

        if "intStep" in rhsArgs:
            if rhsArgs["intStep"] == 0:

                self.ElocMean0 = Eloc.mean()
                self.ElocVar0 = Eloc.var()

                self.metaData = {}

        return update
