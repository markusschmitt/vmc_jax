import jax
import jax.numpy as jnp
import jax.random as jrnd
from functools import partial
#import jVMC.global_defs as global_defs
from flax import linen as nn
from typing import Any, List, Optional, Tuple
from jax import Array, vmap, jit


@jit
def sorting_gumble(sample,logits,gumbel,states):

    numSamples = sample.shape[0]
    LocalHilDim = sample.shape[1]
    L = sample.shape[2]

    indexes = jnp.argsort((-gumbel),axis=None) # only take the most likely probabilities --> cutoff after  numSamples
    indexes_states = (indexes // LocalHilDim)[:numSamples]
    sample = sample.reshape(-1,L)[indexes]
    sample = sample.reshape(LocalHilDim,numSamples,L)
    sample = jnp.swapaxes(sample,0,1) # some shenanigans to select the samples correctly 
    
    logits = logits.ravel()[indexes]
    logits = logits.reshape(LocalHilDim,numSamples).T
    
    gumbel = gumbel.ravel()[indexes]
    gumbel = gumbel.reshape(LocalHilDim,numSamples).T
    vals, treedef  = jax.tree_util.tree_flatten(states)

    vals_ordered = [v[indexes_states] for v in vals] # more shenanigans to select the right states for the network...
    states = jax.tree_util.tree_unflatten(treedef,vals_ordered)
    
    return sample,logits,gumbel,states

class gumbel_wrapper(nn.Module):
    """
    Wrapper module for symmetrization.
    This is a wrapper module for the incorporation of gumbel MC sampling (sampling without repetition). 
    
    see https://arxiv.org/pdf/2408.07625v1     
        
    """
    
    net: callable
    is_gumbel = True
    #avgFun: callable = avgFun_Coefficients_Exp
    def setup(self):
        self.L = self.net.L
        self.LocalHilDim = 2 # default spin with local dim equal 2
        if hasattr(self.net, "LocalHilDim"):
            self.LocalHilDim = self.net.LocalHilDim ## == lDim ** patch_size

        if not "sample" in dir(self.net):
            raise NotImplemented('Gumbel requires autoregressive network')


        
    def __post_init__(self):

        super().__post_init__()
    
    def __call__(self,*args,**kwargs):
        
        return self.net(*args,**kwargs)
        
    def _apply_fun(self, *args,**kwargs):
        return self.net.apply(*args,**kwargs)
    
    def _gumbel_step(self,sample,logits,gumbel,key,states,position):   
        #new samples with (0,..,LocalHilDim-1) at position
        #sample = jnp.array([sample[0].at[position].set(l) for l in jnp.arange(self.LocalHilDim)])
        #right shifted input
        logitnew = jnp.zeros_like(logits)
        sample = jnp.array([sample[0].at[position].set(l) for l in jnp.arange(self.LocalHilDim)])
        #right shifted input
        inputt = jnp.array([jnp.pad(sample[0,:-1],(1,0))])

        #get next conditional probabilities
        logitnew, next_states = self(inputt[:,position],block_states = states, output_state=True)
        #calculate new updated "total" cond. probabilities
        logitnew = logits[0] + logitnew 
        # gumbel "noise"
        gumbelnew = logitnew + jrnd.gumbel(key[0],shape=(self.LocalHilDim,)) 
        Z = jnp.nanmax(gumbelnew)
        gumbelnew = jnp.nan_to_num(-jnp.log(
            jnp.exp(-gumbel[0])-jnp.exp(-Z)+jnp.exp(-gumbelnew) 
            ),nan=-jnp.inf)

        return sample, logitnew, gumbelnew, next_states
    
    def sample(self, numSamples: int, key) -> Array:
        """Autoregressively sample.

        Args:
            * ``numSamples``: The number of configurations to generate.
            * ``key``: JAX random key.

        Returns:
            A batch of system configurations.
        """
        
        if numSamples >= (self.LocalHilDim**self.L):
            raise RuntimeError("number of samples higher than the Hilbert space")
            # Unique samples not possible is the Hilbert space is smaller than the number of samples 
            # This if-condition catches the edge case in order to avoid undefined behavior  
        
        # split the samples
        keys = jrnd.split(key, (self.net.L))
        ## init step
        shape_samples = (numSamples,self.LocalHilDim,self.net.L)
        shape_logits = (numSamples,self.LocalHilDim)
        shape_gumbel = (numSamples,self.LocalHilDim)
        #print(shape_samples,shape_logits)
        working_space_samples = jnp.full(shape_samples,-2,dtype=jnp.int64)
        working_space_logits = jnp.full(shape_logits,-jnp.inf,dtype=jnp.float64)
        working_space_gumbel = jnp.full(shape_gumbel,-jnp.inf,dtype=jnp.float64)
        
        #working_space_samples = working_space_samples.at[0,0,0].set(0)
        working_space_logits = working_space_logits.at[0,0].set(0.)
        working_space_gumbel = working_space_gumbel.at[0,0].set(0.)
        
        states = None
        init_work = partial(self._gumbel_step, position=0,states=states)
        key0 = jrnd.split(keys[0],numSamples)
        key0 = jnp.expand_dims(key0,-2)
        samples,logits,gumbel,states  = jax.vmap(init_work)(working_space_samples,working_space_logits,working_space_gumbel,key0)
        
        init_carry = sorting_gumble(samples,logits,gumbel,states)
        
        res,_ = self._scanning_fn(init_carry,(keys[1:],jnp.arange(1,self.net.L)))
        samples, logits,gumbels,_ = res
        
        kappa = gumbels[0,1] # first non-chosen configuration used to estimate reweighting

        re_weights = jnp.nan_to_num(jnp.exp(logits[:,0]) /(-jnp.expm1(-jnp.exp(logits[:,0]-kappa))),0)

        return samples[:,0,:],logits[:,0]*self.net.logProbFactor,re_weights/jnp.sum(re_weights),kappa
        #return samples[:,0,:],logits[:,0],re_weights/jnp.sum(re_weights)

    @partial(nn.scan,
             variable_broadcast='params',
             split_rngs={'params': False})
    def _scanning_fn(self, carry, key):
        sample = carry[0]
        logits = carry[1]
        gumbel = carry[2]
        states = carry[3]

        position = key[1] # (lattice 0 ... system size)
        keys = jrnd.split(key[0],carry[0].shape[0])
        keys = jnp.expand_dims(keys,-2)

        p_workN = partial(self._gumbel_step,position=position)
        sample,logits,gumbel,states = jax.vmap(p_workN)(sample,logits,gumbel,keys,states)
        #jax.debug.print("gumbelnew new {x}", x=gumbel)

        #### sorting gumble value
        return sorting_gumble(sample,logits,gumbel,states),None
        