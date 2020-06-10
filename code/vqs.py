import flax
from flax import nn
import jax
from jax import jit,grad,vmap
from jax import numpy as jnp
from jax import random
from jax.tree_util import tree_flatten, tree_unflatten
import numpy as np

def cplx_init(rng, shape):
    rng1,rng2 = jax.random.split(rng)
    unif=jax.nn.initializers.uniform()
    return unif(rng1,shape)+1.j*unif(rng2,shape)

class CpxRBM(nn.Module):
    def apply(self, s, L=4, numHidden=2, bias=False):
        layer = nn.Dense.shared(features=numHidden, name='rbm_layer', bias=bias, dtype=np.complex64,
                                kernel_init=cplx_init, bias_init=cplx_init)

        return jnp.sum(jnp.log(jnp.cosh(layer(s))), axis=0)

class RBM(nn.Module):
    def apply(self, s, L=4, numHidden=2, bias=False):
        layer = nn.Dense.shared(features=numHidden, name='rbm_layer', bias=bias, dtype=np.float32)

        return jnp.sum(jnp.log(jnp.cosh(layer(s))))

class NQS:
    def __init__(self, logModNet, phaseNet=None):
        # The net arguments have to be instances of flax.nn.Model
        self.realNets = False
        if phaseNet is None:
            self.cpxNet = logModNet

            self.paramShapes1 = [(p.size,p.shape) for p in tree_flatten(self.cpxNet.params)[0]]
            self.netTreeDef1 = jax.tree_util.tree_structure(self.cpxNet.params)
            self.numParameters = jnp.sum(jnp.array([p.size for p in tree_flatten(self.cpxNet.params)[0]]))
        else:
            self.realNets = True
            self.realNet1 = logModNet # for log|psi(s)|
            self.realNet2 = phaseNet # for arg(psi(2))

            self.paramShapes1 = [(p.size,p.shape) for p in tree_flatten(self.realNet1.params)[0]]
            self.paramShapes2 = [(p.size,p.shape) for p in tree_flatten(self.realNet2.params)[0]]
            self.netTreeDef1 = jax.tree_util.tree_structure(self.realNet1.params)
            self.netTreeDef2 = jax.tree_util.tree_structure(self.realNet2.params)
            self.numParameters1 = jnp.sum(jnp.array([p.size for p in tree_flatten(self.realNet1.params)[0]]))
            self.numParameters2 = jnp.sum(jnp.array([p.size for p in tree_flatten(self.realNet2.params)[0]]))
            self.numParameters = self.numParameters1 + self.numParameters2

    # **  end def __init__


    def __call__(self, s):

        if self.realNets:
            logMod = jit(self.realNet1)(s)
            phase = jit(self.realNet2)(s)
            return logMod + 1.j * phase
        else:
            return jit(self.cpxNet)(s)

    # **  end def __call__


    def gradients(self, s):

        if self.realNets: # FOR REAL NETS
            
            gradOut = jnp.empty((s.shape[0],self.numParameters), dtype=np.complex64)

            # First net
            gradients, self.gradientTreeDef1 = \
                    tree_flatten( 
                        jit(vmap(grad(self._eval_real),in_axes=(None,0)))(self.realNet1,s)
                    )
            
            # Flatten gradients to give a single vector per sample
            start = 0
            for g in gradients:
                numGradients = g[0].size
                gradOut = jax.ops.index_update(gradOut, jax.ops.index[:,start:start+numGradients], g.reshape((s.shape[0],-1)))
                start += numGradients
            
            # Second net
            gradients, self.gradientTreeDef1 = \
                    tree_flatten( 
                        jit(vmap(grad(self._eval_real),in_axes=(None,0)))(self.realNet2,s)
                    )
            
            # Flatten gradients to give a single vector per sample
            for g in gradients:
                numGradients = g[0].size
                gradOut = jax.ops.index_update(gradOut, jax.ops.index[:,start:start+numGradients], 1.j * g.reshape((s.shape[0],-1)))
                start += numGradients

            return gradOut

        else:             # FOR COMPLEX NET

            gradOut = jnp.empty((s.shape[0],2*self.numParameters), dtype=np.complex64)

            gradients, self.gradientTreeDef1 = \
                    tree_flatten( 
                        jit(vmap(grad(self._eval_real),in_axes=(None,0)))(self.cpxNet,s)
                    )
            
            # Flatten gradients to give a single vector per sample
            start = 0
            for g in gradients:
                numGradients = g[0].size
                gradOut = jax.ops.index_update(gradOut, jax.ops.index[:,start:start+numGradients], g.reshape((s.shape[0],-1)))
                start += numGradients

            # Add gradients w.r.t. imaginary parts: g_i = I * g_r
            gradOut = jax.ops.index_update(gradOut, jax.ops.index[:,self.numParameters:], 1.j * gradOut[:,:self.numParameters])

            return gradOut

    # **  end def gradients

    def update_parameters(self, deltaP):

        if self.realNets: # FOR REAL NETS
            
            # Reshape parameter update according to net tree structure
            deltaPTreeShape1 = []
            deltaPTreeShape2 = []
            start = 0
            for s in self.paramShapes1:
                deltaPTreeShape1.append(deltaP[start:start+s[0]].reshape(s[1]))
                start += s[0]
            for s in self.paramShapes2:
                deltaPTreeShape2.append(deltaP[start:start+s[0]].reshape(s[1]))
                start += s[0]

            print(deltaPTreeShape1)
            print(deltaPTreeShape2)
            
            # Compute new parameters
            newParams1 = jax.tree_util.tree_multimap( 
                            jax.lax.add, self.realNet1.params, 
                            tree_unflatten( self.netTreeDef1, deltaPTreeShape1 ) 
                         )

            newParams2 = jax.tree_util.tree_multimap( 
                            jax.lax.add, self.realNet2.params, 
                            tree_unflatten( self.netTreeDef2, deltaPTreeShape2 ) 
                         )

            # Update model parameters
            self.realNet1 = self.realNet1.replace(params=newParams1)
            self.realNet2 = self.realNet2.replace(params=newParams2)

        else:             # FOR COMPLEX NET

            # Get complex-valued parameter update vector
            deltaPCpx = deltaP[:self.numParameters] + 1.j * deltaP[self.numParameters:]
            
            # Reshape parameter update according to net tree structure
            deltaPTreeShape = []
            start = 0
            for s in self.paramShapes1:
                deltaPTreeShape.append(deltaPCpx[start:start+s[0]].reshape(s[1]))
                start += s[0]
            
            # Compute new parameters
            newParams = jax.tree_util.tree_multimap( 
                            jax.lax.add, self.cpxNet.params, 
                            tree_unflatten( self.netTreeDef1, deltaPTreeShape ) 
                        )

            # Update model parameters
            self.cpxNet = self.cpxNet.replace(params=newParams)
                

    # **  end def update_parameters


    def _eval_real(self, net, s):
        return jnp.real(net(s))

def eval_net(model,s):
    return jnp.real(model(s))

if __name__ == '__main__':
    rbm = CpxRBM.partial(L=3,numHidden=2,bias=True)
    _,params = rbm.init_by_shape(random.PRNGKey(0),[(1,3)])
    rbmModel = nn.Model(rbm,params)
    s=2*jnp.zeros((2,3),dtype=np.int32)-1
    s=jax.ops.index_update(s,jax.ops.index[0,1],1)
    print(jit(vmap(rbmModel))(s))
    gradients=jit(vmap(grad(eval_net),in_axes=(None,0)))(rbmModel,s)

    print(gradients.params)
    print(jax.tree_util.tree_flatten(gradients.params))

    print("** Complex net **")
    psiC = NQS(rbmModel)
    G = psiC.gradients(s)
    print(G)
    psiC.update_parameters(jnp.real(G[0]))
    
    print("** Real nets **")
    rbmR = RBM.partial(L=3,numHidden=2,bias=True)
    rbmI = RBM.partial(L=3,numHidden=3,bias=True)
    _,paramsR = rbmR.init_by_shape(random.PRNGKey(0),[(1,3)])
    _,paramsI = rbmI.init_by_shape(random.PRNGKey(0),[(1,3)])
    rbmRModel = nn.Model(rbmR,paramsR)
    rbmIModel = nn.Model(rbmI,paramsI)
 
    psiR = NQS(rbmRModel,rbmIModel)

    G = psiR.gradients(s)
    print(G)
    psiR.update_parameters(np.abs(G[0]))
