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
#from jax.flatten_util import ravel_pytree
import flax
from flax import nn
import numpy as np

import jVMC.global_defs as global_defs
from jVMC.nets import CpxRBM
from jVMC.nets import RBM

from functools import partial

class NQS:
    def __init__(self, logModNet, phaseNet=None, batchSize=1000):
        # The net arguments have to be instances of flax.nn.Model
        self.realNets = False
        if phaseNet is None:
            self.net = logModNet

            self.paramShapes = [(p.size,p.shape) for p in tree_flatten(self.net.params)[0]]
            self.netTreeDef = jax.tree_util.tree_structure(self.net.params)
            self.numParameters = jnp.sum(jnp.array([p.size for p in tree_flatten(self.net.params)[0]]))
        else:
            self.realNets = True
            self.net = [logModNet, phaseNet] # for [ log|psi(s)|, arg(psi(2)) ]

            self.paramShapes = [ [(p.size,p.shape) for p in tree_flatten(net.params)[0]] for net in self.net ]
            self.netTreeDef = [ jax.tree_util.tree_structure(net.params) for net in self.net ]
            self.numParameters1 = jnp.sum(jnp.array([p.size for p in tree_flatten(self.net[0].params)[0]]))
            self.numParameters2 = jnp.sum(jnp.array([p.size for p in tree_flatten(self.net[1].params)[0]]))
            self.numParameters = self.numParameters1 + self.numParameters2

        # Check whether wave function can generate samples
        self._isGenerator = False
        if callable(getattr(logModNet, 'sample', None)):
            self._isGenerator = True

        self.batchSize = batchSize

        # Need to keep handles of jit'd functions to avoid recompilation
        #self.evalJitdNet1 = jit(vmap(self._eval, in_axes=(None, 0)))
        self.evalJitdNet1 = jax.pmap(vmap(self._eval, in_axes=(None, 0)), in_axes=(None, 0))
        self.evalJitdNet2 = jax.pmap(vmap(self._eval, in_axes=(None, 0)), in_axes=(None, 0))
        self.evalJitdReal = jax.pmap(vmap(self._eval_real, in_axes=(None, 0)), in_axes=(None, 0))
        self._get_gradients_net1_pmapd = jax.pmap(self._get_gradients, in_axes=(None,0))
        self._get_gradients_net2_pmapd = jax.pmap(self._get_gradients, in_axes=(None,0))
        self._append_gradients_holo = jax.pmap(lambda x,n: jnp.concatenate((x[:n], 1.j*x[:n]), axis=1), in_axes=(0,None), static_broadcasted_argnums=1)
        self._append_gradients = jax.pmap(lambda x,y,n: jnp.concatenate((x[:n,:], 1.j*y[:n,:]), axis=1), in_axes=(0,0,None), static_broadcasted_argnums=2)
        self._create_batches_pmapd = jax.pmap(self._create_batches, in_axes=(0,None), static_broadcasted_argnums=1)

    # **  end def __init__


    def __call__(self, s):

        if self.realNets:
            logMod = self.evalJitdNet1(self.net[0],s)
            phase = self.evalJitdNet2(self.net[1],s)
            return logMod + 1.j * phase
        else:
            return self.evalJitdNet1(self.net,s)

    # **  end def __call__
    

    def real_coefficients(self, s):

        if self.realNets:
            return self.evalJitdNet1(self.net[0],s)
        else:
            return self.evalJitdReal(self.net,s)

    # **  end def real_coefficients


    def get_sampler_net(self):
    
        if self.realNets:
            return self.net[0]
        else:
            return self.net

    # **  end def get_sampler_net


    def _get_gradients(self, net, s):
        
        def scan_fun(carry,configs):
            g, _ = tree_flatten(
                        jax.tree_util.tree_map( lambda x: jnp.reshape(x, (configs.shape[0],-1)), 
                            vmap(grad(lambda x, y: jnp.real(x(y))), in_axes=(None, 0))(net, configs)
                        )
                    )

            return None, jnp.concatenate(g, axis=1)

        _, gradients = jax.lax.scan(scan_fun, None, s)

        return gradients.reshape((gradients.shape[0]*gradients.shape[1],-1))


    def _create_batches(self, s, batchSize):

        originalSize = s.shape[0]
        append=batchSize*((s.shape[0]+batchSize-1)//batchSize)-s.shape[0]
        pads=[(0,append),] + [(0,0)]*(len(s.shape)-1)
        
        return jnp.pad(s, pads).reshape((-1,batchSize)+s.shape[1:]), originalSize


    def gradients(self, s):

        # Form batches
        s, originalSize = self._create_batches_pmapd(s,self.batchSize)

        if self.realNets: # FOR REAL NETS
            gradOut1 = self._get_gradients_net1_pmapd(self.net[0], s)
            gradOut2 = self._get_gradients_net2_pmapd(self.net[1], s)
            return self._append_gradients(gradOut1, gradOut2, originalSize[0])

        else:             # FOR COMPLEX NET

            gradOut = self._get_gradients_net1_pmapd(self.net, s)

            return self._append_gradients_holo(gradOut, originalSize[0])

    # **  end def gradients


    def update_parameters(self, deltaP):

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

    # **  end def _param_unflatten_cpx


    def _param_unflatten_cpx(self, P):
            
        # Get complex-valued parameter update vector
        PCpx = P[:self.numParameters] + 1.j * P[self.numParameters:]
        
        # Reshape parameter update according to net tree structure
        PTreeShape = []
        start = 0
        for s in self.paramShapes:
            PTreeShape.append(PCpx[start:start+s[0]].reshape(s[1]))
            start += s[0]
        
        # Return unflattened parameters
        return tree_unflatten( self.netTreeDef, PTreeShape ) 

    # **  end def _param_unflatten_cpx
    

    def get_parameters(self):

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

            paramOut = jnp.empty(2*self.numParameters, dtype=global_defs.tReal)

            parameters, _ = tree_flatten( self.net.params )
            
            # Flatten parameters to give a single vector
            start = 0
            for p in parameters:
                numParams = p.size
                paramOut = jax.ops.index_update(paramOut, jax.ops.index[start:start+numParams], jnp.real(p.reshape(-1)))
                paramOut = jax.ops.index_update(paramOut, jax.ops.index[self.numParameters+start:self.numParameters+start+numParams], jnp.imag(p.reshape(-1)))
                start += numParams

            return paramOut

    # **  end def set_parameters


    def sample(self, numSamples, key):

        if self._isGenerator:
            #samples, logP = self.net[0].sample(numSamples, key)
            samples, logP = jax.pmap(self._sample, static_broadcasted_argnums=(1,), in_axes=(None, None, 0))(self.net[0], numSamples, jax.random.split(key,jax.device_count()))
            return samples, logP

        return 0
    
    # **  end def sample


    def _sample(self, net, numSamples, key):

        return net.sample(numSamples, key)


    @property
    def is_generator(self):
        return self._isGenerator

    def _eval_real(self, net, s):
        return jnp.real(net(s))

    def _eval(self, net, s):
        return net(s)

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
    rbm = CpxRBM.partial(numHidden=2,bias=True)
    _,params = rbm.init_by_shape(random.PRNGKey(0),[(1,3)])
    rbmModel = nn.Model(rbm,params)

    print("** Complex net **")
    psiC = NQS(rbmModel)

    s = jnp.zeros((jax.device_count(), 2,3), dtype=np.int32)

    #print(psiC(s))

    s = jnp.zeros((jax.device_count(), 2,3), dtype=np.int32)
    G = psiC.gradients(s)

    print(G)
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
