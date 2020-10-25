import sys
# Find jVMC package
sys.path.append(sys.path[0]+"/..")

import jax
from jax.config import config
config.update("jax_enable_x64", True)
from jax import jit,grad,vmap
from jax import numpy as jnp
from jax import random
from jax.tree_util import tree_flatten, tree_unflatten
from jax.flatten_util import ravel_pytree
import flax
from flax import nn
import numpy as np

import jVMC
import jVMC.global_defs as global_defs
from jVMC.nets import CpxRBM
from jVMC.nets import RBM
import jVMC.mpi_wrapper as mpi

from functools import partial
import collections
import time

def flat_gradient(fun, arg):
    gr = grad(lambda x, y: jnp.real(x(y)))(fun,arg)
    gr = tree_flatten(jax.tree_util.tree_map(lambda x: x.ravel(), gr))[0]
    gi = grad(lambda x, y: jnp.imag(x(y)))(fun,arg)
    gi = tree_flatten(jax.tree_util.tree_map(lambda x: x.ravel(), gi))[0]
    return jnp.concatenate(gr) + 1.j * jnp.concatenate(gi)

def flat_gradient_real(fun, arg):
    g = grad(lambda x, y: jnp.real(x(y)))(fun,arg)
    g = tree_flatten(jax.tree_util.tree_map(lambda x: x.ravel(), g))[0]
    return jnp.concatenate(g)


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

        * ``nets``: Variational network or tuple of networks.\
            A network has to be registered as pytree node and provide \
            a ``__call__`` function for evaluation.
        * ``batchSize``: Batch size for batched network evaluation. Choice \
            of this parameter impacts performance: with too small values performance \
            is limited by memory access overheads, too large values can lead \
            to "out of memory" issues.
    """

    def __init__(self, nets, batchSize=1000):
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
        """

        # The net arguments have to be instances of flax.nn.Model
        self.realNets = False
        self.holomorphic = False
        self.flat_gradient_function = flat_gradient_real

        if not isinstance(nets, collections.abc.Iterable):
   
            self.net = nets
 
            if np.concatenate([p.ravel() for p in tree_flatten(self.net.params)[0]]).dtype == np.complex128:
                self.holomorphic = True
            else:
                self.flat_gradient_function = flat_gradient

            self.paramShapes = [(p.size,p.shape) for p in tree_flatten(self.net.params)[0]]
            self.netTreeDef = jax.tree_util.tree_structure(self.net.params)
            self.numParameters = jnp.sum(jnp.array([p.size for p in tree_flatten(self.net.params)[0]]))

        else:

            self.net = list(nets)

            self.realNets = True

            self.paramShapes = [ [(p.size,p.shape) for p in tree_flatten(net.params)[0]] for net in self.net ]
            self.netTreeDef = [ jax.tree_util.tree_structure(net.params) for net in self.net ]
            self.numParameters1 = jnp.sum(jnp.array([p.size for p in tree_flatten(self.net[0].params)[0]]))
            self.numParameters2 = jnp.sum(jnp.array([p.size for p in tree_flatten(self.net[1].params)[0]]))
            self.numParameters = self.numParameters1 + self.numParameters2

        # Check whether wave function can generate samples
        self._isGenerator = False
        sampleNet = None
        if self.realNets:
            sampleNet = self.net[0]
        else:
            sampleNet = self.net
        if callable(getattr(sampleNet, 'sample', None)):
            self._isGenerator = True

        self.batchSize = batchSize

        # Need to keep handles of jit'd functions to avoid recompilation
        if global_defs.usePmap:
            self.evalJitdNet1 = global_defs.pmap_for_my_devices(self._eval, in_axes=(None, 0, None), static_broadcasted_argnums=(2,))
            self.evalJitdNet2 = global_defs.pmap_for_my_devices(self._eval, in_axes=(None, 0, None), static_broadcasted_argnums=(2,))
            self.evalJitdReal = global_defs.pmap_for_my_devices(self._eval_real, in_axes=(None, 0, None), static_broadcasted_argnums=(2,))
            self._get_gradients_net1_pmapd = global_defs.pmap_for_my_devices(self._get_gradients, in_axes=(None,0,None,None), static_broadcasted_argnums=(2,3))
            self._get_gradients_net2_pmapd = global_defs.pmap_for_my_devices(self._get_gradients, in_axes=(None,0,None,None), static_broadcasted_argnums=(2,3))
            self._append_gradients = global_defs.pmap_for_my_devices(lambda x,y: jnp.concatenate((x[:,:], 1.j*y[:,:]), axis=1), in_axes=(0,0))
            self._sample_jitd = global_defs.pmap_for_my_devices(self._sample, static_broadcasted_argnums=(1,), in_axes=(None, None, 0))
        else:
            self.evalJitdNet1 = global_defs.jit_for_my_device(self._eval, static_argnums=(2,))
            self.evalJitdNet2 = global_defs.jit_for_my_device(self._eval, static_argnums=(2,))
            self.evalJitdReal = global_defs.jit_for_my_device(self._eval_real, static_argnums=(2,))
            self._get_gradients_net1_pmapd = global_defs.jit_for_my_device(self._get_gradients, static_argnums=(2,3))
            self._get_gradients_net2_pmapd = global_defs.jit_for_my_device(self._get_gradients, static_argnums=(2,3))
            self._append_gradients = global_defs.jit_for_my_device(lambda x,y: jnp.concatenate((x[:,:], 1.j*y[:,:]), axis=1))
            self._sample_jitd = global_defs.jit_for_my_device(self._sample, static_argnums=(1,))

    # **  end def __init__


    def __call__(self, s):
        """Evaluate variational wave function.

        Compute the logarithmic wave function coefficients :math:`\ln\psi(s)` for \
        computational configurations :math:`s`.

        Args:

            * ``s``: Array of computational basis states.

        Returns:
            Logarithmic wave function coefficients :math:`\ln\psi(s)`.
        """

        if self.realNets:
            logMod = self.evalJitdNet1(self.net[0],s,self.batchSize)
            phase = self.evalJitdNet2(self.net[1],s,self.batchSize)
            return logMod + 1.j * phase
        else:
            return self.evalJitdNet1(self.net,s,self.batchSize)

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
            return self.evalJitdNet1(self.net[0],s,self.batchSize)
        else:
            return self.evalJitdReal(self.net,s,self.batchSize)

    # **  end def real_coefficients


    def get_sampler_net(self):
        """Returns net used for sampling

        Returns:
            Variational network that yields :math:`\\text{Re}(\log\psi(s))`.
        """
    
        if self.realNets:
            return self.net[0]
        else:
            return self.net

    # **  end def get_sampler_net


    def _get_gradients(self, net, s, batchSize, flat_grad):
        
        def create_batches(s, b):

            append=b*((s.shape[0]+b-1)//b)-s.shape[0]
            pads=[(0,append),] + [(0,0)]*(len(s.shape)-1)
        
            return jnp.pad(s, pads).reshape((-1,b)+s.shape[1:])

        sb = create_batches(s, batchSize)
  
        def scan_fun(c,x):
            return c, jax.vmap(flat_grad, in_axes=(None,0))(net,x)

        g = jax.lax.scan(scan_fun, None, sb)[1]

        g = g.reshape((-1,) + g.shape[2:])

        return g[:s.shape[0]]


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

        if self.realNets: # FOR REAL NETS
            gradOut1 = self._get_gradients_net1_pmapd(self.net[0], s, self.batchSize, self.flat_gradient_function)
            gradOut2 = self._get_gradients_net2_pmapd(self.net[1], s, self.batchSize, self.flat_gradient_function)
            return self._append_gradients(gradOut1, gradOut2)

        else:             # FOR COMPLEX NET

            gradOut = self._get_gradients_net1_pmapd(self.net, s, self.batchSize, self.flat_gradient_function)

            if self.holomorphic:
                return self._append_gradients(gradOut, gradOut)
            
            return gradOut

    # **  end def gradients


    def update_parameters(self, deltaP):
        """Update variational parameters.
        
        Sets new values of all variational parameters by adding given values.

        Args:

            * ``deltaP``: Values to be added to variational parameters.
        """

        if self.realNets: # FOR REAL NETS
            
            # Reshape parameter update according to net tree structure
            newParams = self._param_unflatten_real(deltaP)
            # Update model parameters
            for netId in [0,1]:
                self.net[netId] = self.net[netId].replace(params=
                                        jax.tree_util.tree_multimap( 
                                            jax.lax.add, self.net[netId].params, 
                                            newParams[netId] 
                                        )
                                    )

        else:             # FOR COMPLEX NET
            
            # Compute new parameters
            newParams = jax.tree_util.tree_multimap( 
                            jax.lax.add, self.net.params, 
                            self._param_unflatten_cpx(deltaP)
                        )

            # Update model parameters
            self.net = self.net.replace(params=newParams)
                
    # **  end def update_parameters

    
    def set_parameters(self, P):
        """Set variational parameters.
        
        Sets new values of all variational parameters.

        Args:

            * ``P``: New values of variational parameters.
        """

        if self.realNets: # FOR REAL NETS
            
            newP = self._param_unflatten_real(P)

            # Update model parameters
            for netId in [0,1]:
                self.net[netId] = self.net[netId].replace( params = newP[netId] )

        else:             # FOR COMPLEX NET

            # Update model parameters
            self.net = self.net.replace(
                            params = self._param_unflatten_cpx(P)
                          )

    # **  end def set_parameters


    def _param_unflatten_real(self, P):
        
        # Reshape parameter update according to net tree structure
        PTreeShape = [[],[]]
        start = 0
        for netId in [0,1]:
            for s in self.paramShapes[netId]:
                PTreeShape[netId].append(P[start:start+s[0]].reshape(s[1]))
                start += s[0]
        
        # Return unflattened parameters
        return ( tree_unflatten( self.netTreeDef[0], PTreeShape[0] ), tree_unflatten( self.netTreeDef[1], PTreeShape[1] ) )

    # **  end def _param_unflatten_real


    def _param_unflatten_cpx(self, P):
            
        if self.holomorphic:
            # Get complex-valued parameter update vector
            P = P[:self.numParameters] + 1.j * P[self.numParameters:]
        
        # Reshape parameter update according to net tree structure
        PTreeShape = []
        start = 0
        for s in self.paramShapes:
            PTreeShape.append(P[start:start+s[0]].reshape(s[1]))
            start += s[0]
        
        # Return unflattened parameters
        return tree_unflatten( self.netTreeDef, PTreeShape ) 

    # **  end def _param_unflatten_cpx
    

    def get_parameters(self):
        """Get variational parameters.
        
        Returns:
            Array holding current values of all variational parameters.
        """

        if self.realNets: # FOR REAL NETS

            paramOut = jnp.empty(self.numParameters, dtype=global_defs.tReal)

            start = 0
            for netId in [0,1]:
                parameters, _ = tree_flatten( self.net[netId].params )
                # Flatten parameters to give a single vector
                for p in parameters:
                    numParams = p.size
                    paramOut = jax.ops.index_update( paramOut, jax.ops.index[start:start+numParams], p.reshape(-1) )
                    start += numParams

            return paramOut

        else:             # FOR COMPLEX NET

            paramOut = jnp.concatenate([p.ravel() for p in tree_flatten(self.net.params)[0]])

            if self.holomorphic:
                paramOut = jnp.concatenate([paramOut.real, paramOut.imag])            

            return paramOut

    # **  end def set_parameters


    def sample(self, numSamples, key):

        if self._isGenerator:
            samples, logP = self._sample_jitd(self.net[0], numSamples, jax.random.split(key,jax.device_count()))
            return samples, self(samples)

        return None, None
    
    # **  end def sample


    def _sample(self, net, numSamples, key):

        return net.sample(numSamples, key)


    @property
    def is_generator(self):
        return self._isGenerator

    def _eval_real(self, net, s, batchSize):
        def create_batches(configs, b):

            append=b*((configs.shape[0]+b-1)//b)-configs.shape[0]
            pads=[(0,append),] + [(0,0)]*(len(configs.shape)-1)
        
            return jnp.pad(configs, pads).reshape((-1,b)+configs.shape[1:])

        sb = create_batches(s, batchSize)
        
        def scan_fun(c,x):
            return c, jax.vmap(lambda m,n: jnp.real(m(n)), in_axes=(None, 0))(net,x)

        res = jax.lax.scan(scan_fun, None, jnp.array(sb))[1].reshape((-1,))

        return res[:s.shape[0]]


    def _eval(self, net, s, batchSize):
        
        def create_batches(configs, b):

            append=b*((configs.shape[0]+b-1)//b)-configs.shape[0]
            pads=[(0,append),] + [(0,0)]*(len(configs.shape)-1)
        
            return jnp.pad(configs, pads).reshape((-1,b)+configs.shape[1:])

        sb = create_batches(s, batchSize)

        def scan_fun(c,x):
            return c, jax.vmap(lambda m,n: m(n), in_axes=(None, 0))(net,x)

        res = jax.lax.scan(scan_fun, None, jnp.array(sb))[1].reshape((-1,))

        return res[:s.shape[0]]


# **  end class NQS


# Register NQS class as new pytree node

def flatten_nqs(nqs):
    auxReal = nqs.realNets
    if auxReal:
        flatNet1, auxNet1 = jax.tree_util.tree_flatten(nqs.net[0])
        flatNet2, auxNet2 = jax.tree_util.tree_flatten(nqs.net[1])
        return (flatNet1, flatNet2), (auxReal, auxNet1, auxNet2)
    else:
        flatNet, auxNet = jax.tree_util.tree_flatten(nqs.net)
        return (flatNet,), (auxReal, auxNet)

def unflatten_nqs(aux,treeData):
    if aux[0]:
        net1 = jax.tree_util.tree_unflatten(aux[1], treeData[0])
        net2 = jax.tree_util.tree_unflatten(aux[2], treeData[1])
        return NQS(net1, net2)
    else:
        net = jax.tree_util.tree_unflatten(aux[1], treeData[0])
        return NQS(net)

jax.tree_util.register_pytree_node(NQS, flatten_nqs, unflatten_nqs)


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


if __name__ == '__main__':
    global_defs.set_pmap_devices(jax.devices()[:2])

    rbm = CpxRBM.partial(numHidden=2,bias=True)
    _,params = rbm.init_by_shape(random.PRNGKey(0),[(1,3)])
    rbmModel = nn.Model(rbm,params)

    print("** Complex net **")
    psiC = NQS(rbmModel)

    shape = (2,3)
    if global_defs.usePmap:
        #shape = (jax.device_count(),) + shape
        shape = (jVMC.global_defs.myDeviceCount,) + shape

    s = jnp.zeros(shape, dtype=np.int32)

    res = psiC(s)
    print(res.shape)
    print(res[1].device_buffer.device())

    s = jnp.zeros(shape, dtype=np.int32)
    G = psiC.gradients(s)

    print(G.shape)
    exit()
    psiC.update_parameters(jnp.real(G[0][0]))
    
    a,b=tree_flatten(psiC)

    print(a)
    print(b)

    psiC = tree_unflatten(b,a)
    exit()
    
    print("** Real nets **")
    rbmR = RBM.partial(numHidden=2,bias=True)
    rbmI = RBM.partial(numHidden=3,bias=True)
    _,paramsR = rbmR.init_by_shape(random.PRNGKey(0),[(1,3)])
    _,paramsI = rbmI.init_by_shape(random.PRNGKey(0),[(1,3)])
    rbmRModel = nn.Model(rbmR,paramsR)
    rbmIModel = nn.Model(rbmI,paramsI)
 
    psiR = NQS(rbmRModel,rbmIModel)

    a,b=tree_flatten(psiR)

    print(a)
    print(b)

    psiR = tree_unflatten(b,a)

    G = psiR.gradients(s)
    print(G)
    psiR.update_parameters(np.abs(G[0]))
