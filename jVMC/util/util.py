import sys
# Find jVMC package
sys.path.append(sys.path[0] + "/..")

import jax
from jax.config import config
config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as random
import flax
import flax.nn as nn

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
            return [act_funs.activationFunctions[fn] for fn in actFuns]

        return act_funs.activationFunctions[actFuns]

    netTypes = {
        "RBM": jVMC.nets.RBM,
        "FFN": jVMC.nets.FFN,
        "CNN": jVMC.nets.CNN,
        "LSTM": jVMC.nets.LSTM,
        "LSTMsym": jVMC.nets.LSTMsym,
        "PhaseRNN": jVMC.nets.RNN,
        "PhaseRNNsym": jVMC.nets.RNNsym,
        "CpxRNN": jVMC.nets.CpxRNN,
        "RNN": jVMC.nets.RNN,
        "RNN2D": jVMC.nets.RNN2D,
        "RNNsym": jVMC.nets.RNNsym,
        "RNN2Dsym": jVMC.nets.RNN2Dsym,
        "CpxRBM": jVMC.nets.CpxRBM,
        "CpxCNN": jVMC.nets.CpxCNN
    }

    def get_net(descr, dims, seed):

        net = netTypes[descr["type"]](**descr["parameters"])
        params = net.init(random.PRNGKey(seed), jnp.zeros(dims, dtype=np.int32))
        return net, params

    if "actFun" in descr["net1"]["parameters"]:
        descr["net1"]["parameters"]["actFun"] = get_activation_functions(descr["net1"]["parameters"]["actFun"])

    if descr["net1"]["type"][-3:] == "sym":
        L = descr["net1"]["parameters"]["L"]
        if descr["net1"]["type"][-5:-3] == "2D":
            descr["net1"]["parameters"]["orbit"] = sym.get_orbit_2d_square(L)
        else:
            # Generate orbit of 1D translations for RNNsym net
            descr["net1"]["parameters"]["orbit"] = jnp.array([jnp.roll(jnp.identity(L, dtype=np.int32), l, axis=1) for l in range(L)])

    if "net2" in descr:
        if descr["net2"]["type"][-3:] == "sym":
            L = descr["net2"]["parameters"]["L"]
            if descr["net2"]["type"][-5:-3] == "2D":
                # Generate orbit of 2D translations for RNNsym net
                descr["net2"]["parameters"]["orbit"] = sym.get_orbit_2d_square(L)
            else:
                # Generate orbit of 1D translations for RNNsym net
                descr["net2"]["parameters"]["orbit"] = jnp.array([jnp.roll(jnp.identity(L, dtype=np.int32), l, axis=1) for l in range(L)])

    if not "net2" in descr:

        model, params = get_net(descr["net1"], dims, seed)

        return jVMC.vqs.NQS(model, params, batchSize=descr["gradient_batch_size"])

    else:

        if "actFun" in descr["net2"]["parameters"]:

            descr["net2"]["parameters"]["actFun"] = get_activation_functions(descr["net2"]["parameters"]["actFun"])

        model1, params1 = get_net(descr["net1"], dims, seed)
        model2, params2 = get_net(descr["net2"], dims, seed)

        return jVMC.vqs.NQS((model1, model2), (params1, params2), batchSize=descr["gradient_batch_size"])


def measure(observables, psi, sampler, numSamples=None):
    ''' This function measures expectation values of a given set of operators given a pure state.

    Arguments:
        * ``observables``: Dictionary of the form with operator names as keys and (lists of) operators as values, e.g.:

            .. code-block:: python
                
                { "op_name_1": [operator1, operator2], "op_name_2": operator3 }
            
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
    sampleConfigs, sampleLogPsi, p = sampler.sample(numSamples)

    result = {}

    for name, ops in observables.items():

        tmpMeans = []
        tmpVariances = []
        tmpErrors = []

        for op in get_iterable(ops):
            sampleOffdConfigs, matEls = op.get_s_primes(sampleConfigs)
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

