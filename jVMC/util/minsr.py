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
        * ``svdTol``: Regularization parameter :math:`\\epsilon_{SVD}`, see above.
        * ``makeReal``: Specifies the function :math:`q`, either `'real'` for :math:`q=\\text{Re}` \
            or `'imag'` for :math:`q=\\text{Im}`.
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
        Efficient only if number of samples :math:`\\ll` number of parameters.
        """

        T = gradients.tangent_kernel()
        T_inv = jnp.linalg.pinv(T, rtol=self.pinvTol, hermitian=True)

        eloc_all = mpi.gather(eloc._data).reshape((-1,))
        gradients_all = mpi.gather(gradients._data)
        update = - gradients_all.conj().T @ T_inv @ eloc_all

        return update

    def __call__(self, netParameters, t, *, psi, hamiltonian, **rhsArgs):
        """ For given network parameters computes an update step using the MinSR method.

        This function returns :math:`\\dot\\theta=\\bar O^\\dagger (\\bar O\\bar O^\\dagger)^{-1}\\bar E_{loc}`
        (see `[arXiv:2302.01941] <https://arxiv.org/abs/2302.01941>`_ for details). 
        Thereby an instance of the ``MinSR`` class is a suited callable for the right hand side of an ODE to be 
        used in combination with the integration schemes implemented in ``jVMC.stepper``. 
        Alternatively, the interface matches the scipy ODE solvers as well.

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
            The solution of the MinSR equation, :math:`\\dot\\theta=\\bar O^\\dagger (\\bar O\\bar O^\\dagger)^{-1}\\bar E_{loc}`.
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

                self.ElocMean0 = Eloc.mean()[0]
                self.ElocVar0 = Eloc.var()[0]

                self.metaData = {}

        return update
