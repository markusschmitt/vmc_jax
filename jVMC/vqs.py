import jax
from jax.config import config
config.update("jax_enable_x64", True)
from jax import jit, grad, vmap
from jax import numpy as jnp
from jax import random
from jax.tree_util import tree_flatten, tree_unflatten, tree_map
from jax.flatten_util import ravel_pytree
import flax
import flax.linen as nn
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

from typing import Sequence

class TwoNets(nn.Module):
    net: Sequence[callable]

    def __call__(self, s):
        return self.net[0](s) + 1j * self.net[1](s)

    def sample(self, *args):
        # Will produce exact samples if net[0] contains a sample function.
        # Won't be called if net[0] does not have a sample method.
        return self.net[0].sample(*args)

    def eval_real(self, s):
        return self.net[0](s)

def create_batches(configs, b):

    append = b * ((configs.shape[0] + b - 1) // b) - configs.shape[0]
    pads = [(0, append), ] + [(0, 0)] * (len(configs.shape) - 1)

    return jnp.pad(configs, pads).reshape((-1, b) + configs.shape[1:])

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
            a) holomorphic, i.e., parametrized by complex valued parameters :math:`\\vartheta`.
            b) non-holomorphic, i.e., parametrized by real valued parameters :math:`\\theta`.
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
                a) holomorphic, i.e., parametrized by complex valued parameters :math:`\\vartheta`.
                b) non-holomorphic, i.e., parametrized by real valued parameters :math:`\\theta`.
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

        self._isGenerator = False
        if isinstance(nets, collections.abc.Iterable):
            if "sample" in dir(nets[0]):
                if callable(nets[0].sample):
                    self._isGenerator = True
            nets = TwoNets(net=nets)
        else:
            if "sample" in dir(nets):
                if callable(nets.sample):
                    self._isGenerator = True
        self.net = nets

        self.batchSize = batchSize

        # Need to keep handles of jit'd functions to avoid recompilation
        self._eval_net_pmapd = global_defs.pmap_for_my_devices(self._eval, in_axes=(None, None, 0, None), static_broadcasted_argnums=(0, 3))
        #self.evalJitdReal = global_defs.pmap_for_my_devices(self._eval_real, in_axes=(None, None, 0, None), static_broadcasted_argnums=(0, 3))
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
    
            self.parameters = self.net.init(jax.random.PRNGKey(self.seed), s[0,0,...])
            
            # check Cauchy-Riemann condition to test for holomorphicity
            def make_flat(t):
                return jnp.concatenate([p.ravel() for p in tree_flatten(t)[0]])
            grads_r = make_flat( jax.grad(lambda a,b: jnp.real(self.net.apply(a,b)))(self.parameters, s[0,0,...]) )
            grads_i = make_flat( jax.grad(lambda a,b: jnp.imag(self.net.apply(a,b)))(self.parameters, s[0,0,...]) )
            if isclose(jnp.linalg.norm(grads_r - 1.j * grads_i), 0.0):
                self.holomorphic = True
                self.flat_gradient_function = flat_gradient_holo
            else:
                self.flat_gradient_function = flat_gradient
                self.dict_gradient_function = dict_gradient

            self.paramShapes = [(p.size, p.shape) for p in tree_flatten(self.parameters)[0]]
            self.netTreeDef = jax.tree_util.tree_structure(self.parameters)
            self.numParameters = jnp.sum(jnp.array([p.size for p in tree_flatten(self.parameters)[0]]))

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

        return self._eval_net_pmapd(self.net, self.parameters, s, self.batchSize)


    def _eval(self, net, params, s, batchSize):

        sb = create_batches(s, batchSize)

        def scan_fun(c, x):
            return c, jax.vmap(lambda y: net.apply(params, y), in_axes=(0,))(x)

        res = jax.lax.scan(scan_fun, None, jnp.array(sb))[1].reshape((-1,))

        return res[:s.shape[0]]

    # **  end def __call__


    def _get_gradients(self, net, params, s, batchSize, flat_grad):

        sb = create_batches(s, batchSize)

        def scan_fun(c, x):
            return c, jax.vmap(lambda y: flat_grad(net, params, y), in_axes=(0,))(x)

        g = jax.lax.scan(scan_fun, None, sb)[1]

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

        return self._get_gradients_net1_pmapd(self.net, self.parameters, s, self.batchSize, self.flat_gradient_function)

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

        gradOut = self._get_gradients_dict_net1_pmapd(self.net, self.parameters, s, self.batchSize, self.dict_gradient_function)

        if self.holomorphic:
            return self._append_gradients_dict(gradOut, gradOut)

        return gradOut

    # **  end gradients_dict


    def grad_dict_to_vec_map(self):

        PTreeShape = []
        start = 0
        P = jnp.arange(2*self.numParameters)
        for s in self.paramShapes:
            if self.holomorphic:
                PTreeShape.append( ( P[start:start + 2*s[0]]) )
                start += 2*s[0]
            else:
                PTreeShape.append(P[start:start + s[0]])
                start += s[0]
        
        # Return unflattened parameters
        return tree_unflatten(self.netTreeDef, PTreeShape)


    def get_sampler_net(self):
        """Get real part of NQS and current parameters

        This function returns a function that evaluates the real part of the NQS,
        :math:`\\text{Re}(\log\psi(s))`, and the current parameters.

        Returns:
            Real part of the NQS and current parameters
        """

        evalReal = lambda p,x: jnp.real( self.net.apply(p,x) )
        if "eval_real" in dir(self.net):
            if callable(self.net.eval_real):
                evalReal = lambda p,x: jnp.real( self.net.apply(p,x,method=self.net.eval_real) )

        return evalReal, self.parameters

    # **  end def get_sampler_net

    def sample(self, numSamples, key, parameters=None):

        if self._isGenerator:
            net, params = self.net, self.parameters

            if parameters is not None:
                params = parameters

            numSamplesStr = str(numSamples)

            # check whether _get_samples is already compiled for given number of samples
            if not numSamplesStr in self._sample_jitd:
                self._sample_jitd[numSamplesStr] = global_defs.pmap_for_my_devices(lambda p, n, x: net.apply(p, n, x, method=net.sample),
                                                                                   static_broadcasted_argnums=(1,), in_axes=(None, None, 0))

            samples = self._sample_jitd[numSamplesStr](params, int(numSamples), key)

            return samples

        return None

    # **  end def sample

    def _sample(self, net, params, numSamples, key):

        return net.apply(params, numSamples, key, method=net.sample)


    def update_parameters(self, deltaP):
        """Update variational parameters.
        Sets new values of all variational parameters by adding given values.
        If parameters are not initialized, parameters are set to ``deltaP``.
        Args:
            * ``deltaP``: Values to be added to variational parameters.
        """

        if not self.initialized:
            self.set_parameters(deltaP)

        # Compute new parameters
        newParams = jax.tree_util.tree_multimap(
            jax.lax.add, self.parameters,
            self._param_unflatten(deltaP)
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

        # Update model parameters
        self.parameters = self._param_unflatten(P)

    # **  end def set_parameters


    def _param_unflatten(self, P):

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

    # **  end def _param_unflatten


    def get_parameters(self):
        """Get variational parameters.
        Returns:
            Array holding current values of all variational parameters.
        """

        if not self.initialized:

            return None


        if self.holomorphic:
            paramOut = jnp.concatenate([jnp.concatenate([p.ravel().real, p.ravel().imag]) for p in tree_flatten(self.parameters)[0]])
        else:
            paramOut = jnp.concatenate([p.ravel() for p in tree_flatten(self.parameters)[0]])

        return paramOut

    # **  end def set_parameters

    @property
    def is_generator(self):
        return self._isGenerator
