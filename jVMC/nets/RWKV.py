from functools import partial
from itertools import repeat
from typing import Any, List, Optional, Tuple

import flax
from flax.training import checkpoints
import jax
from flax.linen import (
    Dense,
    Embed,
    LayerNorm,
    Module,
    Sequential,
    compact,
    gelu,
    log_softmax,
    make_causal_mask,
    scan,
)

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from flax import linen as nn
from jax import random, numpy as jnp

from jax.numpy import arange, expand_dims, full, int64, take_along_axis, zeros, roll, log, ones, pi, sin, log
from jax import Array, vmap, debug, jit
from jax.nn import log_softmax
from jax.random import categorical, split #,KeyArray

# from jVMC.global_defs import tReal
tReal = jnp.float64

def initialize_to_value(x, dtype):
    """
    makes an initializer function that ignores the given PRNGKey
    and always returns the given value. Note that the value
    can be an array or scalar
    """
    return lambda _: x.astype(dtype)
    
class MultiHeadTimeMixing(nn.Module):
    layer_depth: int = 1
    num_layers: int = 1
    embedding_size: int = 2
    num_heads: int = 1
    dtype: type = tReal
    bias: type = False

    def setup(self):
        #####################################################
        # initialize the \mu for time mixing
        #####################################################
        # goes from 0 to 1 along layer depth
        ratio_0_to_1 = self.layer_depth / (self.num_layers - 1)
        # goes from 1 to (almost) 0 along layer depth
        ratio_1_to_almost_0 = 1.0 - (self.layer_depth / self.num_layers)
        zigzag = .5 * (jnp.arange(1, self.embedding_size+1) % 3 - 1)
        time_first = jnp.full(self.embedding_size, jnp.log(.3)) + zigzag
        time_first = jnp.repeat(jnp.expand_dims(time_first,axis=0),self.num_heads,axis=0)
        self.time_first = self.param(
            'time_first', initialize_to_value(time_first, self.dtype))

        h = jnp.arange(0, self.embedding_size)
        h = jnp.repeat(jnp.expand_dims(h,axis=0),self.num_heads,axis=0)
        # the numbers used here were found to work well from experiments
        time_decay = -5 + 8 * (h / (self.embedding_size - 1)
                               ) ** (.7 + 1.3 * ratio_0_to_1)
        time_decay -= jnp.arange(self.num_heads).reshape(self.num_heads,1) * .5 
        self.time_decay = self.param(
            'time_decay', initialize_to_value(time_decay, self.dtype))

        x = (jnp.arange(self.embedding_size) /
             self.embedding_size)
        time_mix_k = jnp.power(x, ratio_1_to_almost_0)
        time_mix_v = time_mix_k + .3 * ratio_0_to_1
        time_mix_r = jnp.power(x, .5 * ratio_1_to_almost_0)
        self.time_mix_k = self.param(
            'time_mix_k', initialize_to_value(time_mix_k, self.dtype))
        self.time_mix_v = self.param(
            'time_mix_v', initialize_to_value(time_mix_v, self.dtype))
        self.time_mix_r = self.param(
            'time_mix_r', initialize_to_value(time_mix_r, self.dtype))

        #####################################################
        # set up the linear RKV layers and the output layer
        #####################################################
        self.layernorm = nn.LayerNorm(epsilon=1e-5, param_dtype=self.dtype)
        self.key = nn.Dense(self.num_heads * self.embedding_size, use_bias=False, param_dtype=self.dtype)
        self.value = nn.Dense(self.num_heads * self.embedding_size, use_bias=False, param_dtype=self.dtype)
        self.receptance = nn.Dense(self.num_heads * self.embedding_size, use_bias=False, param_dtype=self.dtype)
        self.output = nn.Dense(self.embedding_size, use_bias=False, param_dtype=self.dtype)
        self.head_collapse = self.param(
            'head_collapse', initialize_to_value(jnp.ones(self.num_heads), self.dtype))


    def __call__(self, x, time_mix_state):
        # get the state from the previous time step
        sx, aa, bb, pp = time_mix_state # sx is the previous time_mix_state
        xx = x # self.layernorm(x)
        sx = jnp.concatenate((jnp.expand_dims(sx, 0), xx[:-1, :]))
        kx = xx * self.time_mix_k + sx * (1 - self.time_mix_k)
        vx = xx * self.time_mix_v + sx * (1 - self.time_mix_v)
        rx = xx * self.time_mix_r + sx * (1 - self.time_mix_r)
        # apply the gating mechanism
        r = nn.sigmoid(self.receptance(rx)).reshape(rx.shape[0],self.num_heads, self.embedding_size) # does not go into the scan
        # compute the key and value
        k = self.key(kx).reshape(rx.shape[0],self.num_heads, self.embedding_size)
        v = self.value(vx).reshape(rx.shape[0],self.num_heads, self.embedding_size)
        # perform the WKV activation
        def step(state, kv):
            ############
            # KV: key and value pair
            ###########
            (aa, bb, pp), (kk, vv) = state, kv
            ww = self.time_first + kk # time_first -> u (original paper), u + k_t
            q = jnp.maximum(pp, ww) # max(p_{t-1},u + k_t)
            e1 = jnp.exp(pp - q) # exp(p_{t-1} - q)
            e2 = jnp.exp(ww - q) # exp(u + k_t - q)
            out = ((e1 * aa + e2 * vv) / (e1 * bb + e2)).astype(dtype=r.dtype)

            # regularize the exponential moving attention
            ww = pp - jnp.exp(self.time_decay)# weights = nn.elu(self.time_decay) + 1
            qprime = jnp.maximum(ww, kk)
            e1 = jnp.exp(ww - qprime)
            e2 = jnp.exp(kk - qprime)
            aa = e1 * aa + e2 * vv
            bb = e1 * bb + e2
            pp = qprime
            return (aa, bb, pp), out
        (aa, bb, pp), sx = jax.lax.scan(step, (aa, bb, pp), (k, v))
        # collapse the heads
        head_collapse = nn.softmax(self.head_collapse).reshape(self.num_heads,1)  
        sx = jax.vmap(lambda f1: f1*head_collapse)(sx)
        r =  jax.vmap(lambda f1: f1*head_collapse)(r)
        # compute the output and recombine with the input
        out = x + self.output((r * sx).sum(axis=-2))
        # (xx[-1,:], aa, bb, pp) is the next time_mix_state
        return out, (xx[-1, :], aa, bb, pp)

class ChannelMixing(nn.Module):
    layer_depth: int = 1
    num_layers: int = 1
    embedding_size: int = 2
    hidden_size: int = 8
    dtype: type = tReal
    bias: bool = False

    def setup(self):
        #####################################################
        # set up the linear RKV layers and the output layer
        #####################################################
        self.layernorm = nn.LayerNorm(epsilon=1e-5,  param_dtype=self.dtype,name="LayerNormP")
        self.key = nn.Dense(self.hidden_size, use_bias=False, param_dtype=self.dtype,name="KeyP")
        self.receptance = nn.Dense(self.embedding_size, use_bias=False, param_dtype=self.dtype,name="ReceptanceP")
        self.value = nn.Dense(self.embedding_size, use_bias=self.bias, param_dtype=self.dtype,name="ValueP")

        #####################################################
        # initialize the \mu for time mixing
        #####################################################
        x = (jnp.arange(self.embedding_size) /
             self.embedding_size)

        ratio_1_to_almost_0 = 1.0 - (self.layer_depth / self.num_layers)
        time_mix_k = jnp.power(x, ratio_1_to_almost_0)
        time_mix_r = jnp.power(x, .5 * ratio_1_to_almost_0)
        self.time_mix_k = self.param(
            'time_mix_k', initialize_to_value(time_mix_k, self.dtype))
        self.time_mix_r = self.param(
            'time_mix_r', initialize_to_value(time_mix_r, self.dtype))

    def __call__(self, x, channel_mix_state):
        xx = x # self.layernorm(x)
        sx = jnp.concatenate(
            (jnp.expand_dims(channel_mix_state, 0), xx[:-1, :]))
        xk = xx * self.time_mix_k + sx * (1 - self.time_mix_k)
        xr = xx * self.time_mix_r + sx * (1 - self.time_mix_r)
        # sigmoid activation
        r = nn.sigmoid(self.receptance(xr))
        # square ReLU activation
        k = jnp.square(nn.elu(self.key(xk))+1.) # <- this dangerously explodes the gradients
        # k = jnp.square(nn.gelu(self.key(xk))) # <- this dangerously explodes the gradients
        # k = jnp.square(nn.relu(self.key(xk))) # <- this is the original
        # compute the output and recombine with the input
        out = x + r * self.value(k)

        return out, xx[-1, :]
    
def empty_state(num_heads,embedding_size):
    "returns an empty block_state for a given embedding size"
    zeros_multi_head = jnp.zeros((num_heads,embedding_size))
    zeros = jnp.zeros(embedding_size)
    min_values = jnp.full(embedding_size, -jnp.inf)
    time_mix_state = (zeros, zeros_multi_head, zeros_multi_head, (jnp.repeat(jnp.expand_dims(min_values,axis=-2),num_heads,axis=-2)))
    channel_mix_state = zeros
    return time_mix_state, channel_mix_state


class RWKVBlock(nn.Module):
    # Block
    layer_num: int = 1
    dtype: type = tReal
    # time/channel mixing
    num_layers: int = 1
    embedding_size: int =2
    hidden_size: int = 8
    num_heads: int = 1
    bias: bool = False

    def setup(self):
        self.time_mix = MultiHeadTimeMixing(layer_depth=self.layer_num,
                                   embedding_size=self.embedding_size,
                                   num_layers=self.num_layers,
                                   num_heads=self.num_heads,
                                   bias=self.bias,
                                   dtype=self.dtype)
        self.channel_mix = ChannelMixing(layer_depth=self.layer_num,
                                   embedding_size=self.embedding_size,
                                   num_layers=self.num_layers,
                                   hidden_size=self.hidden_size,
                                   dtype=self.dtype)
        self.waist = nn.Dense(self.hidden_size, use_bias=False,name="waist", param_dtype=self.dtype)
        self.torso = nn.Dense(self.embedding_size, use_bias=False,name="torso", param_dtype=self.dtype)

    def __call__(self, x, block_state):
        """
        Takes the embedding from the previous layer, and the `block_state`
        from the previous time_step.

        `block_state` is a tuple of `time_mix_state` and `channel_mix_state`,
        which are used as inputs to the block's `time_mix` and `channel_mix`
        respectively.
        """
        if block_state is None:
            block_state = empty_state(self.num_heads,self.embedding_size)

        time_mix_state, channel_mix_state = block_state
        x, time_mix_state = self.time_mix(x, time_mix_state)
        x, channel_mix_state = self.channel_mix(x, channel_mix_state)
        y = self.waist(x)
        y = nn.gelu(self.torso(y))
        return x + y, (time_mix_state, channel_mix_state)

class CpxRWKV(nn.Module):
    # Main
    L: int = 1 # system size
    LocalHilDim: int = 2
    hidden_size: int = 8
    num_heads: int = 1
    dtype: type = tReal
    # time/channel mixing
    num_layers: int = 1
    embedding_size: int = 2
    # prob correction
    logProbFactor: float = 0.5
    # one hot embedding
    one_hot: bool = False
    # bias
    bias: bool = False
    __name__: str = "RWKV"

    def setup(self):
        if self.one_hot:
            self.embed = nn.Dense(self.embedding_size, use_bias=False,name="Embedding", param_dtype=self.dtype)
        else:
            self.embed = nn.Embed(self.LocalHilDim,
                               self.embedding_size,
                               param_dtype=self.dtype)
        self.input_layernorm = nn.LayerNorm(epsilon=1e-5,name="InputLayerNorm",param_dtype=self.dtype)
        self.blocks = [
                        RWKVBlock(layer_num=i,
                              embedding_size=self.embedding_size,
                              hidden_size=self.hidden_size,
                              num_layers=self.num_layers,
                              num_heads=self.num_heads,
                              bias=self.bias,
                              dtype=self.dtype)
                        for i in range(self.num_layers)
                       ]
        self.output_layernorm = nn.LayerNorm(epsilon=1e-5,name="OutputLayerNorm", param_dtype=self.dtype)
        self.neck = nn.Dense(self.hidden_size, use_bias=self.bias,name="Neck", param_dtype=self.dtype)
        self.head = nn.Dense(self.LocalHilDim, use_bias=False,name="Head", param_dtype=self.dtype)
        self.PhaseNeck = nn.Dense(self.hidden_size, use_bias=self.bias,name="PhaseNeck", param_dtype=self.dtype)
        self.PhaseHead = nn.Dense(1, use_bias=False,name="PhaseHead", param_dtype=self.dtype)
        self.ldim =self.LocalHilDim
        self.lDim = self.LocalHilDim
        
    def __call__(self, s: Array, block_states: Array = None, output_state: bool = False,**kwargs) -> Array:
        # the helper method allows to use nn.jit with static_argnames
        return self.forward_with_state(s, block_states, output_state)

    #@partial(nn.jit, static_argnums=3)
    def forward_with_state(self, s: Array, block_states: Array = None,  output_state: bool = False) -> Array:
        
        if self.one_hot:
            if output_state:
                y = jax.nn.one_hot(s,self.LocalHilDim,axis=-1)
                y = self.embed(y)
            else:
                y = jnp.pad(s,(1,0),mode='constant',constant_values=0)
                y = jax.nn.one_hot(y,self.LocalHilDim,axis=-1) 
                y = self.embed(y)
        else:
            if output_state:
                y = self.embed(s)
            else:
                y = jnp.pad(s,(1,0),mode='constant',constant_values=0)
                y = self.embed(y)

        next_states = []
        if block_states is None:
            block_states = repeat(None)
        for block, state in zip(self.blocks, block_states):
            y, new_state = block(y, state)
            if output_state:
                next_states.append(new_state)
        # the neck precedes the head
        x = nn.elu(self.neck(y))
        # the head tops the neck
        x = self.head(x)
        # necessery: positive activation for the probabilities with some regularization
        x = nn.elu(x) + 1. + 1e-8
        # return here for RNN mode
        if output_state:
            # normalize the product probability distribution
            x = jnp.log(x/jnp.expand_dims(x.sum(axis=-1),axis=-1))
            return x, next_states
        else:
            # normalize the product probability distribution
            x = jnp.log(x[:-1]/jnp.expand_dims(x[:-1].sum(axis=-1),axis=-1)) * self.logProbFactor
        # compute the phase in the auotregressive style
        phase = nn.gelu(self.PhaseNeck(y[-1]))
        # the log-probs according the state
        return (take_along_axis(x, expand_dims(s, -1), axis=-1)
                                .sum(axis=-2)
                                .squeeze(-1) 
                + 1.j * ( self.PhaseHead(phase)).squeeze(-1))

    def sample(self, numSamples: int, key) -> Array:
        """Autoregressively sample a spin configuration.

        Args:
            * ``numSamples``: The number of configurations to generate.
            * ``key``: JAX random key.

        Returns:
            A batch of spin configurations.
        """
        def generate_sample(key):
            key = split(key, self.L)
            logits, carry = self(jnp.zeros(1,dtype=int),block_states = None, output_state=True)
            choice = categorical(key[0], logits.ravel().real)
            x, s = self._scanning_fn((jnp.expand_dims(choice,0),carry),key[1:])
            return jnp.concatenate([jnp.expand_dims(choice,0),s])

        # get the samples
        keys = split(key, numSamples)
        return vmap(generate_sample)(keys)

    @partial(scan,
             variable_broadcast='params',
             split_rngs={'params': False})
    def _scanning_fn(self, carry: Tuple[Array, Array, float], key) -> Tuple[Array,Array, float]:
        logits, next_states = self(carry[0],block_states = carry[1], output_state=True)
        choice = categorical(key, logits.ravel().real)
        return (jnp.expand_dims(choice,0), next_states), choice


class RpxRWKV(nn.Module):
    # Main
    L: int = 1 # system size
    LocalHilDim: int = 2
    hidden_size: int = 8
    num_heads: int = 1
    dtype: type = tReal
    # time/channel mixing
    num_layers: int = 1
    embedding_size: int = 2
    # prob correction
    logProbFactor: float = 0.5
    # one hot embedding
    one_hot: bool = False
    # bias
    bias: bool = False
    __name__: str = "RWKV"


    def setup(self):
        if self.one_hot:
            self.embed = nn.Dense(self.embedding_size, use_bias=False,name="Embedding", param_dtype=self.dtype)
        else:
            self.embed = nn.Embed(self.LocalHilDim,
                               self.embedding_size,
                               param_dtype=self.dtype)
        self.input_layernorm = nn.LayerNorm(epsilon=1e-5,name="InputLayerNorm",param_dtype=self.dtype)
        self.blocks = [
                        RWKVBlock(layer_num=i,
                              embedding_size=self.embedding_size,
                              hidden_size=self.hidden_size,
                              num_layers=self.num_layers,
                              num_heads=self.num_heads,
                              bias=self.bias,
                              dtype=self.dtype)
                        for i in range(self.num_layers)
                       ]
        self.output_layernorm = nn.LayerNorm(epsilon=1e-5,name="OutputLayerNorm", param_dtype=self.dtype)
        self.neck = nn.Dense(self.hidden_size, use_bias=self.bias,name="Neck", param_dtype=self.dtype)
        self.head = nn.Dense(self.LocalHilDim, use_bias=False,name="Head", param_dtype=self.dtype)
        self.ldim =self.LocalHilDim
        self.lDim = self.LocalHilDim
        
    def __call__(self, s: Array, block_states: Array = None, output_state: bool = False) -> Array:
        # the helper method allows to use nn.jit with static_argnames
        #jax.debug.print("s: {s}",s=s)
        return self.forward_with_state(s, block_states, output_state)

    @partial(nn.jit, static_argnums=3)
    def forward_with_state(self, s: Array, block_states: Array = None,  output_state: bool = False) -> Array:
        
        if self.one_hot:
            if output_state:
                y = jax.nn.one_hot(s,self.LocalHilDim,axis=-1)
                y = self.embed(y)
            else:
                y = jnp.pad(s,(1,0),mode='constant',constant_values=0)
                y = jax.nn.one_hot(y,self.LocalHilDim,axis=-1) 
                y = self.embed(y)
        else:
            if output_state:
                y = self.embed(s)
            else:
                y = jnp.pad(s,(1,0),mode='constant',constant_values=0)
                y = self.embed(y)

        next_states = []
        if block_states is None:
            block_states = repeat(None)
        for block, state in zip(self.blocks, block_states):
            y, new_state = block(y, state)
            if output_state:
                next_states.append(new_state)
        # the neck precedes the head
        x = nn.elu(self.neck(y))
        # the head tops the neck
        x = self.head(x)
        # necessery: positive activation for the probabilities with some regularization
        x = nn.elu(x) + 1. + 1e-8
        # return here for RNN mode
        if output_state:
            # normalize the product probability distribution
            x = jnp.log(x/jnp.expand_dims(x.sum(axis=-1),axis=-1))
            return x, next_states
        else:
            # normalize the product probability distribution
            x = jnp.log(x[:-1]/jnp.expand_dims(x[:-1].sum(axis=-1),axis=-1)) * self.logProbFactor
        # compute the phase in the auotregressive style
        # the log-probs according the state
        return (take_along_axis(x, expand_dims(s, -1), axis=-1)
                                .sum(axis=-2)
                                .squeeze(-1))

    def sample(self, numSamples: int, key) -> Array:
        """Autoregressively sample a spin configuration.

        Args:
            * ``numSamples``: The number of configurations to generate.
            * ``key``: JAX random key.

        Returns:
            A batch of spin configurations.
        """
        def generate_sample(key):
            key = split(key, self.L)
            logits, carry = self(jnp.zeros(1,dtype=int),block_states = None, output_state=True)
            choice = categorical(key[0], logits.ravel().real)
            x, s = self._scanning_fn((jnp.expand_dims(choice,0),carry),key[1:])
            return jnp.concatenate([jnp.expand_dims(choice,0),s])

        # get the samples
        keys = split(key, numSamples)
        return vmap(generate_sample)(keys)

    @partial(scan,
             variable_broadcast='params',
             split_rngs={'params': False})
    def _scanning_fn(self, carry: Tuple[Array, Array, float], key) -> Tuple[Array,Array, float]:
        logits, next_states = self(carry[0],block_states = carry[1], output_state=True)
        choice = categorical(key, logits.ravel().real)
        return (jnp.expand_dims(choice,0), next_states), choice
