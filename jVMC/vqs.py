import jax
from jax.config import config
config.update("jax_enable_x64", True)
from jax import jit, grad, vmap
from jax import numpy as jnp
from jax import random
from jax.tree_util import tree_flatten, tree_unflatten, tree_map
from jax.flatten_util import ravel_pytree
import flax
from flax import nn
from flax.core.frozen_dict import freeze
import numpy as np

import jVMC
import jVMC.global_defs as global_defs
from jVMC.nets import CpxRBM
from jVMC.nets import RBM
import jVMC.mpi_wrapper as mpi

from functools import partial
import collections
import time
from math import isclose

def flat_gradient(fun, params, arg):
    gr = grad(lambda p, y: jnp.real(fun.apply(p, y)))(params, arg)
    gr = tree_flatten(tree_map(lambda x: x.ravel(), gr))[0]
    gi = grad(lambda p, y: jnp.imag(fun.apply(p, y)))(params, arg)
    gi = tree_flatten(tree_map(lambda x: x.ravel(), gi))[0]
    return jnp.concatenate(gr) + 1.j * jnp.concatenate(gi)


def flat_gradient_real(fun, params, arg):
    g = grad(lambda p, y: jnp.real(fun.apply(p, y)))(params, arg)
    g = tree_flatten(tree_map(lambda x: x.ravel(), g))[0]
    return jnp.concatenate(g)

def flat_gradient_holo(fun, params, arg):
    g = grad(lambda p, y: jnp.real(fun.apply(p, y)))(params, arg)
    g = tree_flatten(tree_map(lambda x: [x.ravel(), 1.j*x.ravel()], g))[0]
    return jnp.concatenate(g)

def dict_gradient(fun, params, arg):
    gr = grad(lambda p, y: jnp.real(fun.apply(p, y)))(params, arg)
    gr = tree_map(lambda x: x.ravel(), gr)
    gi = grad(lambda p, y: jnp.imag(fun.apply(p, y)))(params, arg)
    gi = tree_map(lambda x: x.ravel(), gi)
    return tree_map(lambda x,y: x + 1.j*y, gr, gi)


def dict_gradient_real(fun, params, arg):
    g = grad(lambda p, y: jnp.real(fun.apply(p, y)))(params, arg)
    g = tree_map(lambda x: x.ravel(), g)
    return g


class NQS:
    """Wrapper class providing basic functionality of variational states.

    This class can operate in two modi:

        #. Single-network ansatz

            Quantum state of the form :math:`\psi_\\theta(s)\equiv\exp(r_\\theta(s))`, \
            where the network :math:`r_\\theta` is

            a) holomorphic, i.e., parametrized by complex valued parameters :math:`\\theta`.

            b) real, i.e., parametrized by real valued parameters :math:`\\theta`.

        #. Two-network ansatz

            Quantum state of the form 
            :math:`\psi_\\theta(s)\equiv\exp(r_{\\theta_r}(s)+i\\varphi_{\\theta_\\phi}(s))` \
            with an amplitude network :math:`r_{\\theta_{r}}` and a phase network \
            :math:`\\varphi_{\\theta_\phi}` \
            parametrized by real valued parameters :math:`\\theta_r,\\theta_\\phi`.

    Initializer arguments:

        * ``nets``: Variational network or tuple of networks.
        * ``batchSize``: Batch size for batched network evaluation. Choice \
            of this parameter impacts performance: with too small values performance \
            is limited by memory access overheads, too large values can lead \
            to "out of memory" issues.
        * ``seed``: Seed for the PRNG to initialize the network parameters.
    """

    def __init__(self, nets, batchSize=1000, seed=1234):
        """Initializes NQS class.

        This class can operate in two modi:

            #. Single-network ansatz

                Quantum state of the form :math:`\psi_\\theta(s)\equiv\exp(r_\\theta(s))`, \
                where the network :math:`r_\\theta` is

                a) holomorphic, i.e., parametrized by complex valued parameters :math:`\\theta`.

                b) real, i.e., parametrized by real valued parameters :math:`\\theta`.

            #. Two-network ansatz

                Quantum state of the form 
                :math:`\psi_\\theta(s)\equiv\exp(r_{\\theta_r}(s)+i\\varphi_{\\theta_\\phi}(s))` \
                with an amplitude network :math:`r_{\\theta_{r}}` and a phase network \
                :math:`\\varphi_{\\theta_\phi}` \
                parametrized by real valued parameters :math:`\\theta_r,\\theta_\\phi`.

        Args:

            * ``nets``: Variational network or tuple of networks.\
                A network has to be registered as pytree node and provide \
                a ``__call__`` function for evaluation.
            * ``batchSize``: Batch size for batched network evaluation. Choice \
                of this parameter impacts performance: with too small values performance \
                is limited by memory access overheads, too large values can lead \
                to "out of memory" issues.
            * ``seed``: Seed for the PRNG to initialize the network parameters.
        """

        # The net arguments have to be instances of flax.nn.Model
        self.realNets = False
        self.holomorphic = False
        self.flat_gradient_function = flat_gradient_real
        self.dict_gradient_function = dict_gradient_real

        self.initialized = False
        self.seed = seed
        self.parameters = None

        if not isinstance(nets, collections.abc.Iterable):
            self.net = nets
        else:
            self.net = list(nets)
            self.realNets = True


        # Check whether wave function can generate samples
        self._isGenerator = False
        sampleNet = None
        if self.realNets:
            sampleNet = self.net[0]
        else:
            sampleNet = self.net

        if "sample" in dir(sampleNet):
            if callable(sampleNet.sample):
                self._isGenerator = True

        self.batchSize = batchSize

        # Need to keep handles of jit'd functions to avoid recompilation
        self.evalJitdNet1 = global_defs.pmap_for_my_devices(self._eval, in_axes=(None, None, 0, None), static_broadcasted_argnums=(0, 3))
        self.evalJitdNet2 = global_defs.pmap_for_my_devices(self._eval, in_axes=(None, None, 0, None), static_broadcasted_argnums=(0, 3))
        self.evalJitdReal = global_defs.pmap_for_my_devices(self._eval_real, in_axes=(None, None, 0, None), static_broadcasted_argnums=(0, 3))
        self._get_gradients_net1_pmapd = global_defs.pmap_for_my_devices(self._get_gradients, in_axes=(None, None, 0, None, None), static_broadcasted_argnums=(0, 3, 4))
        self._get_gradients_net2_pmapd = global_defs.pmap_for_my_devices(self._get_gradients, in_axes=(None, None, 0, None, None), static_broadcasted_argnums=(0, 3, 4))
        self._append_gradients = global_defs.pmap_for_my_devices(lambda x, y: jnp.concatenate((x[:, :], 1.j * y[:, :]), axis=1), in_axes=(0, 0))
        self._get_gradients_dict_net1_pmapd = global_defs.pmap_for_my_devices(self._get_gradients, in_axes=(None, None, 0, None, None), static_broadcasted_argnums=(0, 3, 4))
        self._get_gradients_dict_net2_pmapd = global_defs.pmap_for_my_devices(self._get_gradients, in_axes=(None, None, 0, None, None), static_broadcasted_argnums=(0, 3, 4))
        self._append_gradients_dict = global_defs.pmap_for_my_devices(lambda x, y: tree_map(lambda a,b: jnp.concatenate((a[:, :], 1.j * b[:, :]), axis=1), x, y), in_axes=(0, 0))
        self._sample_jitd = {}

    # **  end def __init__


    def init_net(self, s):

        if not self.initialized:
    
            if not isinstance(self.net, collections.abc.Iterable):
                
                self.parameters = self.net.init(jax.random.PRNGKey(self.seed), s[0,0,...])
                
                # check Cauchy-Riemann condition to test for holomorphicity
                def make_flat(t):
                    return jnp.concatenate([p.ravel() for p in tree_flatten(t)[0]])
                grads_r = make_flat( jax.grad(lambda a,b: jnp.real(self.net.apply(a,b)))(self.parameters, s[0,0,...]) )
                grads_i = make_flat( jax.grad(lambda a,b: jnp.imag(self.net.apply(a,b)))(self.parameters, s[0,0,...]) )
                if isclose(jnp.linalg.norm(grads_r - 1.j * grads_i), 0.0):
                #if np.concatenate([p.ravel() for p in tree_flatten(self.parameters)[0]]).dtype == np.complex128:
                    self.holomorphic = True
                    self.flat_gradient_function = flat_gradient_holo
                else:
                    self.flat_gradient_function = flat_gradient
                    self.dict_gradient_function = dict_gradient

                self.paramShapes = [(p.size, p.shape) for p in tree_flatten(self.parameters)[0]]
                self.netTreeDef = jax.tree_util.tree_structure(self.parameters)
                self.numParameters = jnp.sum(jnp.array([p.size for p in tree_flatten(self.parameters)[0]]))

            else:
                
                keys = jax.random.split(jax.random.PRNGKey(self.seed), 2)
                self.parameters = [n.init(k, s[0,0,...]) for k,n in zip(keys, self.net)]

                self.paramShapes = [[(p.size, p.shape) for p in tree_flatten(params)[0]] for params in self.parameters]
                self.netTreeDef = [jax.tree_util.tree_structure(params) for params in self.parameters]
                self.numParameters1 = jnp.sum(jnp.array([p.size for p in tree_flatten(self.parameters[0])[0]]))
                self.numParameters2 = jnp.sum(jnp.array([p.size for p in tree_flatten(self.parameters[1])[0]]))
                self.numParameters = self.numParameters1 + self.numParameters2
                

            self.initialized = True

    # ** end init_net


    def __call__(self, s):
        """Evaluate variational wave function.

        Compute the logarithmic wave function coefficients :math:`\ln\psi(s)` for \
        computational configurations :math:`s`.

        Args:

            * ``s``: Array of computational basis states.

        Returns:
            Logarithmic wave function coefficients :math:`\ln\psi(s)`.

        :meta public:
        """

        self.init_net(s)

        if self.realNets:
            logMod = self.evalJitdNet1(self.net[0], self.parameters[0], s, self.batchSize)
            phase = self.evalJitdNet2(self.net[1], self.parameters[1], s, self.batchSize)
            return logMod + 1.j * phase
        else:
            return self.evalJitdNet1(self.net, self.parameters, s, self.batchSize)

    # **  end def __call__

    def real_coefficients(self, s):
        """Evaluate real part of variational wave function.

        Compute the real part of the logarithmic wave function coefficients, \
        :math:`\\text{Re}(\ln\psi(s))`, for computational configurations :math:`s`.

        Args:

            * ``s``: Array of computational basis states.

        Returns:
            Real part of logarithmic wave function coefficients \
            :math:`\\text{Re}(\ln\psi(s))`.
        """

        if self.realNets:
            return self.evalJitdNet1(self.net[0], self.parameters[0], s, self.batchSize)
        else:
            return self.evalJitdReal(self.net, self.parameters, s, self.batchSize)

    # **  end def real_coefficients

    def get_sampler_net(self):
        """Returns net used for sampling

        Returns:
            Variational network that yields :math:`\\text{Re}(\log\psi(s))`.
        """

        if self.realNets:
            return self.net[0], self.parameters[0]
        else:
            return self.net, self.parameters

    # **  end def get_sampler_net

    def _get_gradients(self, net, params, s, batchSize, flat_grad):

        def create_batches(s, b):

            append = b * ((s.shape[0] + b - 1) // b) - s.shape[0]
            pads = [(0, append), ] + [(0, 0)] * (len(s.shape) - 1)

            return jnp.pad(s, pads).reshape((-1, b) + s.shape[1:])

        sb = create_batches(s, batchSize)

        def scan_fun(c, x):
            return c, jax.vmap(lambda y: flat_grad(net, params, y), in_axes=(0,))(x)

        g = jax.lax.scan(scan_fun, None, sb)[1]

        #g = g.reshape((-1,) + g.shape[2:])
        g = tree_map(lambda x: x.reshape((-1,) + x.shape[2:]), g)

        #return g[:s.shape[0]]
        return tree_map(lambda x: x[:s.shape[0]], g)


    def gradients(self, s):
        """Compute gradients of logarithmic wave function.

        Compute gradient of the logarithmic wave function coefficients, \
        :math:`\\nabla\ln\psi(s)`, for computational configurations :math:`s`.

        Args:

            * ``s``: Array of computational basis states.

        Returns:
            A vector containing derivatives :math:`\partial_{\\theta_k}\ln\psi(s)` \
            with respect to each variational parameter :math:`\\theta_k` for each \
            input configuration :math:`s`.
        """
        
        self.init_net(s)

        if self.realNets:  # FOR REAL NETS

            gradOut1 = self._get_gradients_net1_pmapd(self.net[0], self.parameters[0], s, self.batchSize, self.flat_gradient_function)
            gradOut2 = self._get_gradients_net2_pmapd(self.net[1], self.parameters[1], s, self.batchSize, self.flat_gradient_function)

            return self._append_gradients(gradOut1, gradOut2)

        else:             # FOR COMPLEX NET

            gradOut = self._get_gradients_net1_pmapd(self.net, self.parameters, s, self.batchSize, self.flat_gradient_function)

            #if self.holomorphic:
            #    return self._append_gradients(gradOut, gradOut)

            return gradOut

    # **  end def gradients


    def gradients_dict(self, s):
        """Compute gradients of logarithmic wave function and return them as dictionary.

        Compute gradient of the logarithmic wave function coefficients, \
        :math:`\\nabla\ln\psi(s)`, for computational configurations :math:`s`.

        Args:

            * ``s``: Array of computational basis states.

        Returns:
            A dictionary containing derivatives :math:`\partial_{\\theta_k}\ln\psi(s)` \
            with respect to each variational parameter :math:`\\theta_k` for each \
            input configuration :math:`s`.
        """
        
        self.init_net(s)

        if self.realNets:  # FOR REAL NETS

            gradOut1 = self._get_gradients_dict_net1_pmapd(self.net[0], self.parameters[0], s, self.batchSize, self.dict_gradient_function)
            gradOut2 = self._get_gradients_dict_net2_pmapd(self.net[1], self.parameters[1], s, self.batchSize, self.dict_gradient_function)

            return freeze({"net1" : gradOut1, "net2" : gradOut2})

        else:             # FOR COMPLEX NET

            gradOut = self._get_gradients_dict_net1_pmapd(self.net, self.parameters, s, self.batchSize, self.dict_gradient_function)

            if self.holomorphic:
                return self._append_gradients_dict(gradOut, gradOut)

            return gradOut

    # **  end gradients_dict


    def update_parameters(self, deltaP):
        """Update variational parameters.

        Sets new values of all variational parameters by adding given values.

        If parameters are not initialized, parameters are set to ``deltaP``.

        Args:

            * ``deltaP``: Values to be added to variational parameters.
        """

        if not self.initialized:
            self.set_parameters(deltaP)

        if self.realNets:  # FOR REAL NETS

            # Reshape parameter update according to net tree structure
            newParams = self._param_unflatten_real(deltaP)

            # Update model parameters
            for netId in [0, 1]:
                self.parameters[netId] = jax.tree_util.tree_multimap(
                    jax.lax.add, self.parameters[netId],
                    newParams[netId]
                )

        else:             # FOR COMPLEX NET

            # Compute new parameters
            newParams = jax.tree_util.tree_multimap(
                jax.lax.add, self.parameters,
                self._param_unflatten_cpx(deltaP)
            )

            # Update model parameters
            self.parameters = newParams

    # **  end def update_parameters


    def set_parameters(self, P):
        """Set variational parameters.

        Sets new values of all variational parameters.

        Args:

            * ``P``: New values of variational parameters.
        """

        if not self.initialized:
            raise RuntimeError("Error in NQS.set_parameters(): Network not initialized. Evaluate net on example input for initialization.")

        if self.realNets:  # FOR REAL NETS

            newP = self._param_unflatten_real(P)

            # Update model parameters
            for netId in [0, 1]:
                self.parameters[netId] = newP[netId]

        else:             # FOR COMPLEX NET

            # Update model parameters
            self.parameters = self._param_unflatten_cpx(P)

    # **  end def set_parameters

    def _param_unflatten_real(self, P):

        # Reshape parameter update according to net tree structure
        PTreeShape = [[], []]
        start = 0
        for netId in [0, 1]:
            for s in self.paramShapes[netId]:
                PTreeShape[netId].append(P[start:start + s[0]].reshape(s[1]))
                start += s[0]

        # Return unflattened parameters
        return (tree_unflatten(self.netTreeDef[0], PTreeShape[0]), tree_unflatten(self.netTreeDef[1], PTreeShape[1]))

    # **  end def _param_unflatten_real


    def _param_unflatten_cpx(self, P):

        #if self.holomorphic:
        #    # Get complex-valued parameter update vector
        #    P = P[:self.numParameters] + 1.j * P[self.numParameters:]

        # Reshape parameter update according to net tree structure
        PTreeShape = []
        start = 0
        for s in self.paramShapes:
            if self.holomorphic:
                PTreeShape.append( ( P[start:start + s[0]] + 1.j * P[start + s[0]:start + 2*s[0]]).reshape(s[1]) )
                start += 2*s[0]
            else:
                PTreeShape.append(P[start:start + s[0]].reshape(s[1]))
                start += s[0]
        
        # Return unflattened parameters
        return tree_unflatten(self.netTreeDef, PTreeShape)

    # **  end def _param_unflatten_cpx


    def get_parameters(self):
        """Get variational parameters.

        Returns:
            Array holding current values of all variational parameters.
        """

        if not self.initialized:

            return None


        if self.realNets:  # FOR REAL NETS

            paramOut = jnp.empty(self.numParameters, dtype=global_defs.tReal)

            start = 0
            for netId in [0, 1]:
                parameters, _ = tree_flatten(self.parameters[netId])
                # Flatten parameters to give a single vector
                for p in parameters:
                    numParams = p.size
                    paramOut = paramOut.at[start:start + numParams].set(p.reshape(-1))
                    start += numParams

            return paramOut

        else:             # FOR COMPLEX NET

            #paramOut = jnp.concatenate([p.ravel() for p in tree_flatten(self.parameters)[0]])

            if self.holomorphic:
                paramOut = jnp.concatenate([jnp.concatenate([p.ravel().real, p.ravel().imag]) for p in tree_flatten(self.parameters)[0]])
            else:
                paramOut = jnp.concatenate([p.ravel() for p in tree_flatten(self.parameters)[0]])

            return paramOut

    # **  end def set_parameters

    def sample(self, numSamples, key, parameters=None):

        if self._isGenerator:
            net, params = self.get_sampler_net()

            if parameters is not None:
                params = parameters

            numSamplesStr = str(numSamples)

            # check whether _get_samples is already compiled for given number of samples
            if not numSamplesStr in self._sample_jitd:
                self._sample_jitd[numSamplesStr] = global_defs.pmap_for_my_devices(lambda p, n, x: net.apply(p, n, x, method=net.sample),
                                                                                   static_broadcasted_argnums=(1,), in_axes=(None, None, 0))

            samples = self._sample_jitd[numSamplesStr](params, numSamples, key)

            return samples

        return None

    # **  end def sample

    def _sample(self, net, params, numSamples, key):

        return net.apply(params, numSamples, key, method=net.sample)

    @property
    def is_generator(self):
        return self._isGenerator

    def _eval_real(self, net, params, s, batchSize):
        def create_batches(configs, b):

            append = b * ((configs.shape[0] + b - 1) // b) - configs.shape[0]
            pads = [(0, append), ] + [(0, 0)] * (len(configs.shape) - 1)

            return jnp.pad(configs, pads).reshape((-1, b) + configs.shape[1:])

        sb = create_batches(s, batchSize)

        def scan_fun(c, x):
            return c, jax.vmap(lambda y: jnp.real(net.apply(params, y)), in_axes=(0,))(x)

        res = jax.lax.scan(scan_fun, None, jnp.array(sb))[1].reshape((-1,))

        return res[:s.shape[0]]

    def _eval(self, net, params, s, batchSize):

        def create_batches(configs, b):

            append = b * ((configs.shape[0] + b - 1) // b) - configs.shape[0]
            pads = [(0, append), ] + [(0, 0)] * (len(configs.shape) - 1)

            return jnp.pad(configs, pads).reshape((-1, b) + configs.shape[1:])

        sb = create_batches(s, batchSize)

        def scan_fun(c, x):
            return c, jax.vmap(lambda y: net.apply(params, y), in_axes=(0,))(x)

        res = jax.lax.scan(scan_fun, None, jnp.array(sb))[1].reshape((-1,))

        return res[:s.shape[0]]


# **  end class NQS


# Register NQS class as new pytree node

# def flatten_nqs(nqs):
#    auxReal = nqs.realNets
#    if auxReal:
#        flatNet1, auxNet1 = jax.tree_util.tree_flatten(nqs.net[0])
#        flatNet2, auxNet2 = jax.tree_util.tree_flatten(nqs.net[1])
#        return (flatNet1, flatNet2), (auxReal, auxNet1, auxNet2)
#    else:
#        flatNet, auxNet = jax.tree_util.tree_flatten(nqs.net)
#        return (flatNet,), (auxReal, auxNet)
#
# def unflatten_nqs(aux,treeData):
#    if aux[0]:
#        net1 = jax.tree_util.tree_unflatten(aux[1], treeData[0])
#        net2 = jax.tree_util.tree_unflatten(aux[2], treeData[1])
#        return NQS(net1, net2)
#    else:
#        net = jax.tree_util.tree_unflatten(aux[1], treeData[0])
#        return NQS(net)
#
#jax.tree_util.register_pytree_node(NQS, flatten_nqs, unflatten_nqs)


# Register NQS class for flax serialization

def nqs_to_state_dict(nqs):

    stateDict = {}
    if nqs.realNets:
        stateDict['net1'] = flax.serialization.to_state_dict(nqs.net[0])
        stateDict['net2'] = flax.serialization.to_state_dict(nqs.net[1])
    else:
        stateDict['net'] = flax.serialization.to_state_dict(nqs.net)

    return stateDict


def nqs_from_state_dict(nqs, stateDict):

    if nqs.realNets:
        return NQS(
            flax.serialization.from_state_dict(nqs.net[0], stateDict['net1']),
            flax.serialization.from_state_dict(nqs.net[1], stateDict['net2'])
        )
    else:
        return NQS(
            flax.serialization.from_state_dict(nqs.net, stateDict['net'])
        )


flax.serialization.register_serialization_state(NQS, nqs_to_state_dict, nqs_from_state_dict)

