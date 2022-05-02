import jax
from jax.config import config
config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as random

import time
import numpy as np

import jVMC
import jVMC.util.stepper as jVMCstepper
import jVMC.mpi_wrapper as mpi
import jVMC.nets.activation_functions as act_funs
import jVMC.util.symmetries as sym

import collections


def get_iterable(x):
    if isinstance(x, collections.abc.Iterable):
        return x
    else:
        return (x,)


def init_net(descr, dims, seed=0):

    def get_activation_functions(actFuns):

        if type(actFuns) is list:
            return tuple([act_funs.activationFunctions[fn] for fn in actFuns])

        return act_funs.activationFunctions[actFuns]

    netTypes = {
        "RBM": jVMC.nets.RBM,
        "FFN": jVMC.nets.FFN,
        "CNN": jVMC.nets.CNN,
        "RNN": jVMC.nets.RNN1DGeneral,
        "RNN2D": jVMC.nets.RNN2DGeneral,
        "RNNsym": jVMC.nets.RNN1DGeneralSym,
        "RNN2Dsym": jVMC.nets.RNN2DGeneralSym,
        "CpxRBM": jVMC.nets.CpxRBM,
        "CpxCNN": jVMC.nets.CpxCNN
    }

    def get_net(descr, dims):

        return netTypes[descr["type"]](**descr["parameters"])

    if "actFun" in descr["net1"]["parameters"]:
        descr["net1"]["parameters"]["actFun"] = get_activation_functions(descr["net1"]["parameters"]["actFun"])

    if descr["net1"]["type"][-3:] == "sym":
        L = dims[0]

        # set symmetries ON - turn each one off manually
        kwargs_sym = {"translation": True, "reflection": True, "rotation": True}
        for key in kwargs_sym.keys():
            if key in descr["net1"]:
                kwargs_sym[key] = descr["net1"][key]

        if descr["net1"]["type"][-5:-3] == "2D":
            descr["net1"]["parameters"]["orbit"] = sym.get_orbit_2d_square(L, **kwargs_sym)
        else:
            descr["net1"]["parameters"]["orbit"] = sym.get_orbit_1d(L, **kwargs_sym)

    if "net2" in descr:
        if descr["net2"]["type"][-3:] == "sym":

            # set symmetries ON - turn each one off manually
            kwargs_sym = {"translation": True, "reflection": True, "rotation": True}
            for key in kwargs_sym.keys():
                if key in descr["net2"]:
                    kwargs_sym[key] = descr["net2"][key]

            L = dims[0]
            if descr["net2"]["type"][-5:-3] == "2D":
                descr["net2"]["parameters"]["orbit"] = sym.get_orbit_2d_square(L, **kwargs_sym)
            else:
                descr["net2"]["parameters"]["orbit"] = sym.get_orbit_1d(L, **kwargs_sym)

    if not "net2" in descr:

        model = get_net(descr["net1"], dims)

        psi = jVMC.vqs.NQS(model, batchSize=descr["gradient_batch_size"], seed=seed)

    else:

        if "actFun" in descr["net2"]["parameters"]:

            descr["net2"]["parameters"]["actFun"] = get_activation_functions(descr["net2"]["parameters"]["actFun"])

        model1 = get_net(descr["net1"], dims)
        model2 = get_net(descr["net2"], dims)

        psi = jVMC.vqs.NQS((model1, model2), batchSize=descr["gradient_batch_size"], seed=seed)

    psi(jnp.zeros((jVMC.global_defs.device_count(), 1) + dims, dtype=np.int32))

    return psi


def measure(observables, psi, sampler, numSamples=None):
    ''' This function measures expectation values of a given set of operators given a pure state.

    Arguments:
        * ``observables``: Dictionary of the form with operator names as keys and (lists of) operators as values, e.g.:

            .. code-block:: python

                { "op_name_1": [operator1, operator2], "op_name_2": operator3 }

            If an operator takes additional positional arguments, a tuple of the operator together with the arguments
            has to be passed instead. For example, assuming that ``operator3`` from above requires additional arguments ``*args``:

            .. code-block:: python

                { "op_name_1": [operator1, operator2], "op_name_2": [(operator3, *args)] }

        * ``psi``: Variational wave function (instance of ``jVMC.vqs.NQS``)
        * ``sampler``: Instance of ``jVMC.sampler`` used for sampling.
        * ``numSamples``: Number of samples (optional)

    Returns:
        A dictionary holding expectation values, variances, and MC error estimates for each operator. E.g. for the
        exemplary operator input given in `Arguments`:

        .. code-block:: python

            { 
                "op_name_1": { "mean": [mean1, mean2],
                               "variance": [var1, var2],
                               "MC_error": [err1, err2] },
                "op_name_2": { "mean": [mean3],
                               "variance": [var3],
                               "MC_error": [err3] }
            }

    '''
    # Get sample
    sampleConfigs, sampleLogPsi, p = sampler.sample(numSamples=numSamples)

    result = {}

    for name, ops in observables.items():

        tmpMeans = []
        tmpVariances = []
        tmpErrors = []

        for op in get_iterable(ops):

            args=()
            if isinstance(op, collections.abc.Iterable):
                args = tuple(op[1:])
                op = op[0]
            
            sampleOffdConfigs, matEls = op.get_s_primes(sampleConfigs, *args)
            sampleLogPsiOffd = psi(sampleOffdConfigs)
            Oloc = op.get_O_loc(sampleLogPsi, sampleLogPsiOffd)

            if p is not None:
                tmpMeans.append(mpi.global_mean(Oloc, p))
                tmpVariances.append(mpi.global_variance(Oloc, p))
                tmpErrors.append(0.)
            else:
                tmpMeans.append(mpi.global_mean(Oloc))
                tmpVariances.append(mpi.global_variance(Oloc))
                tmpErrors.append(jnp.sqrt(tmpVariances[-1]) / jnp.sqrt(sampler.get_last_number_of_samples()))

        result[name] = {}
        result[name]["mean"] = jnp.real(jnp.array(tmpMeans))
        result[name]["variance"] = jnp.real(jnp.array(tmpVariances))
        result[name]["MC_error"] = jnp.real(jnp.array(tmpErrors))

    return result


def ground_state_search(psi, ham, tdvpEquation, sampler, numSteps=200, varianceTol=1e-10, stepSize=1e-2, observables=None, outp=None):
    ''' This function performs a ground state search by Stochastic Reconfiguration.

    Arguments:
        * ``psi``: Variational wave function (``jVMC.vqs.NQS``)
        * ``ham``: Hamiltonian operator
        * ``tdvpEquation``: An instance of ``jVMC.util.TDVP``
        * ``numSteps``: Maximal number of steps
        * ``varianceTol``: Stopping criterion
        * ``stepSize``: Update step size (learning rate)
        * ``observables``: Observables to be measured during ground state search
        * ``outp``: ``None`` or instance of ``jVMC.util.OutputManager``.

    '''

    delta = tdvpEquation.diagonalShift

    stepper = jVMCstepper.Euler(timeStep=stepSize)

    n = 0
    if outp is not None:
        if observables is not None:
            obs = measure(observables, psi, sampler)
            outp.write_observables(n, **obs)

    varE = 1.0

    while n < numSteps and varE > varianceTol:

        tic = time.perf_counter()

        dp, _ = stepper.step(0, tdvpEquation, psi.get_parameters(), hamiltonian=ham, psi=psi, numSamples=None, outp=outp)
        psi.set_parameters(dp)
        n += 1

        varE = tdvpEquation.get_energy_variance()

        if outp is not None:
            if observables is not None:
                obs = measure(observables, psi, sampler)
                outp.write_observables(n, **obs)

        delta = 0.95 * delta
        tdvpEquation.set_diagonal_shift(delta)

        if outp is not None:
            outp.print(" STEP %d" % (n))
            outp.print("   Energy mean: %f" % (tdvpEquation.get_energy_mean()))
            outp.print("   Energy variance: %f" % (varE))
            outp.print_timings(indent="   ")
            outp.print("   == Time for step: %fs" % (time.perf_counter() - tic))
