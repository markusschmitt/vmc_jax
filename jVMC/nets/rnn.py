import jax
from jax.config import config
config.update("jax_enable_x64", True)
import flax
import flax.linen as nn
import numpy as np
import jax.numpy as jnp

import jVMC.global_defs as global_defs

from functools import partial


class RNNCell(nn.Module):
    """
    Implementation of a 'vanilla' RNN-cell, that is part of an RNNCellStack which is scanned over an input sequence.
    The RNNCell therefore receives two inputs, the hidden state (if it is in a deep part of the CellStack) or the 
    input (if it is the first cell of the CellStack) aswell as the hidden state of the previous RNN-cell.
    Both inputs are mapped to obtain a new hidden state, which is what the RNNCell implements.

    Arguments: 
        * ``hiddenSize``: size of the hidden state vector
        * ``actFun``: non-linear activation function
        * ``initScale``: factor by which the initial parameters are scaled

    Returns:
        new hidden state
    """

    hiddenSize: int = 10
    actFun: callable = nn.elu
    initScale: float = 1.0

    def setup(self):

        initFunctionCell = partial(jax.nn.initializers.variance_scaling(scale=1.0, mode="fan_avg", distribution="uniform"),
                                   dtype=global_defs.tReal)
        initFunctionOut = partial(jax.nn.initializers.variance_scaling(scale=self.initScale, mode="fan_in", distribution="uniform"),
                                  dtype=global_defs.tReal)

        self.cellIn = nn.Dense(features=self.hiddenSize,
                               use_bias=False, dtype=global_defs.tReal,
                               kernel_init=initFunctionCell)
        self.cellCarry = nn.Dense(features=self.hiddenSize,
                                  use_bias=True,
                                  kernel_init=initFunctionCell, dtype=global_defs.tReal,
                                  bias_init=partial(jax.nn.initializers.zeros, dtype=global_defs.tReal))

    def __call__(self, carry, x):

        newCarry = self.actFun(self.cellCarry(carry) + self.cellIn(x))
        return newCarry

# ** end class RNNCell


class RNNCellStack(nn.Module):
    """
    Implementation of a stack of RNN-cells which is scanned over an input sequence.
    This is achieved by stacking multiple 'vanilla' RNN-cells to obtain a deep RNN.

    Arguments: 
        * ``hiddenSize``: size of the hidden state vector
        * ``actFun``: non-linear activation function
        * ``initScale``: factor by which the initial parameters are scaled

    Returns:
        New set of hidden states (one for each layer), as well as the last hidden state, that serves as input to the output layer
    """

    hiddenSize: int = 10
    actFun: callable = nn.elu
    initScale: float = 1.0

    @nn.compact
    def __call__(self, carry, x):
        newCarry = jnp.zeros(shape=(carry.shape[0], self.hiddenSize), dtype=global_defs.tReal)

        newR = x
        # Can't use scan for this, because then flax doesn't realize that each cell has different parameters
        for j, c in enumerate(carry):
            newCarry = newCarry.at[j].set( RNNCell(hiddenSize=self.hiddenSize,
                                                                 actFun=self.actFun,
                                                                 initScale=self.initScale)(c, newR) )

            newR = newCarry[j]

        return jnp.array(newCarry), newR

# ** end class RNNCellStack


class RNN(nn.Module):
    """
    Implementation of an RNN which consists of an RNNCellStack with an additional output layer.
    This class defines how sequential input data is treated.

    Arguments: 
        * ``L``: length of the spin chain
        * ``hiddenSize``: size of the hidden state vector
        * ``depth``: number of RNN-cells in the RNNCellStack
        * ``inputDim``: dimension of the input
        * ``actFun``: non-linear activation function
        * ``initScale``: factor by which the initial parameters are scaled
        * ``logProbFactor``: factor defining how output and associated sample probability are related. 0.5 for pure states and 1 for POVMs.

    Returns:
        logarithmic wave-function coefficient or POVM-probability
    """

    L: int = 10
    hiddenSize: int = 10
    depth: int = 1
    inputDim: int = 2
    actFun: callable = nn.elu
    initScale: float = 1.0
    logProbFactor: float = 0.5

    def setup(self):

        self.rnnCell = RNNCellStack(hiddenSize=self.hiddenSize,
                                    actFun=self.actFun,
                                    initScale=self.initScale)
        initFunctionCell = partial(jax.nn.initializers.variance_scaling(scale=1.0, mode="fan_avg", distribution="uniform"),
                                   dtype=global_defs.tReal)
        self.outputDense = nn.Dense(features=self.inputDim,
                                    use_bias=True,
                                    kernel_init=initFunctionCell, dtype=global_defs.tReal,
                                    bias_init=partial(jax.nn.initializers.zeros, dtype=global_defs.tReal))

    def __call__(self, x):

        state = jnp.zeros((self.depth, self.hiddenSize))

        _, probs = self.rnn_cell((state, jnp.zeros(self.inputDim)), jax.nn.one_hot(x, self.inputDim))

        return self.logProbFactor * jnp.sum(probs, axis=0)

    @partial(nn.transforms.scan,
             variable_broadcast='params',
             split_rngs={'params': False})
    def rnn_cell(self, carry, x):
        newCarry, out = self.rnnCell(carry[0], carry[1])
        logProb = nn.log_softmax(self.outputDense(out))
        logProb = jnp.sum(logProb * x, axis=-1)
        return (newCarry, x), jnp.nan_to_num(logProb, nan=-35)

    def sample(self, batchSize, key):
        state = jnp.zeros((batchSize, self.depth, self.hiddenSize))

        keys = jax.random.split(key, self.L)
        _, res = self.rnn_cell_sample((state, jnp.zeros((batchSize, self.inputDim))), keys)

        return jnp.transpose(res[1])

    @partial(nn.transforms.scan,
             variable_broadcast='params',
             split_rngs={'params': False})
    def rnn_cell_sample(self, carry, x):
        newCarry, logits = jax.vmap(self.rnnCell)(carry[0], carry[1])

        logProb = nn.log_softmax(self.outputDense(logits))

        sampleOut = jax.random.categorical(x, logProb)
        sample = jax.nn.one_hot(sampleOut, self.inputDim)
        logProb = jnp.sum(logProb * sample, axis=1)
        return (newCarry, sample), (jnp.nan_to_num(logProb, nan=-35), sampleOut)

# ** end class RNN


class RNNsym(nn.Module):
    """
    Implementation of an RNN which consists of an RNNCellStack with an additional output layer.
    It uses the RNN class to compute probabilities and averages the outputs over all symmetry-invariant configurations.

    Arguments: 
        * ``L``: length of the spin chain
        * ``hiddenSize``: size of the hidden state vector
        * ``depth``: number of RNN-cells in the RNNCellStack
        * ``inputDim``: dimension of the input
        * ``actFun``: non-linear activation function
        * ``initScale``: factor by which the initial parameters are scaled
        * ``logProbFactor``: factor defining how output and associated sample probability are related. 0.5 for pure states and 1 for POVMs.
        * ``orbit``: collection of maps that define symmetries
        * ``z2sym``: for pure states; implement Z2 symmetry

    Returns:
        Symmetry-averaged logarithmic wave-function coefficient or POVM-probability
    """
    L: int = 10
    hiddenSize: int = 10
    depth: int = 1
    inputDim: int = 2
    actFun: callable = nn.elu
    initScale: float = 1.0
    logProbFactor: float = 0.5
    orbit: any = None
    z2sym: bool = False

    def setup(self):

        self.rnn = RNN(L=self.L, hiddenSize=self.hiddenSize, depth=self.depth,
                       inputDim=self.inputDim,  # passDim=self.passDim,
                       actFun=self.actFun, initScale=self.initScale,
                       logProbFactor=self.logProbFactor)

    def __call__(self, x):

        x = jax.vmap(lambda o, s: jnp.dot(o, s), in_axes=(0, None))(self.orbit, x)

        def evaluate(x):
            return self.rnn(x)

        res = jnp.mean(jnp.exp((1. / self.logProbFactor) * jax.vmap(evaluate)(x)), axis=0)

        if self.z2sym:
            res = 0.5 * (res + jnp.mean(jnp.exp((1. / logProbFactor) * jax.vmap(evaluate)(1 - x)), axis=0))

        logProbs = self.logProbFactor * jnp.log(res)

        return logProbs

    def sample(self, batchSize, key):

        key1, key2 = jax.random.split(key)

        configs = self.rnn.sample(batchSize, key1)

        orbitIdx = jax.random.choice(key2, self.orbit.shape[0], shape=(batchSize,))

        configs = jax.vmap(lambda k, o, s: jnp.dot(o[k], s), in_axes=(0, None, 0))(orbitIdx, self.orbit, configs)

        if self.z2sym:
            key3, _ = jax.random.split(key2)
            flipChoice = jax.random.choice(key3, 2, shape=(batchSize,))
            configs = jax.vmap(lambda b, c: jax.lax.cond(b == 1, lambda x: 1 - x, lambda x: x, c), in_axes=(0, 0))(flipChoice, configs)

        return configs

# ** end class RNNsym


class PhaseRNN(nn.Module):
    """
    Implementation of an RNN to encode the phase which consists of an RNNCellStack with an additional output layer.

    Arguments: 
        * ``L``: length of the spin chain
        * ``hiddenSize``: size of the hidden state vector
        * ``depth``: number of RNN-cells in the RNNCellStack
        * ``inputDim``: dimension of the input
        * ``actFun``: non-linear activation function
        * ``initScale``: factor by which the initial parameters are scaled

    Returns:
        phase of the coefficient
    """
    L: int = 10
    hiddenSize: int = 10
    depth: int = 1
    inputDim: int = 2
    actFun: callable = nn.elu
    initScale: float = 1.0

    def setup(self):

        self.rnnCell = RNNCellStack(hiddenSize=self.hiddenSize, actFun=self.actFun, initScale=self.initScale)

        self.dense = nn.Dense(features=8, dtype=global_defs.tReal,
                              kernel_init=jax.nn.initializers.lecun_normal(dtype=global_defs.tReal),
                              bias_init=partial(jax.nn.initializers.zeros, dtype=global_defs.tReal))

    def __call__(self, x):

        state = jnp.zeros((self.depth, self.hiddenSize))

        _, res = self.rnn_cell((state, jnp.zeros(self.inputDim)), jax.nn.one_hot(x, self.inputDim))

        res = self.dense(res.ravel())

        return jnp.mean(self.actFun(res))

    @partial(nn.transforms.scan,
             variable_broadcast='params',
             split_rngs={'params': False})
    def rnn_cell(self, carry, x):
        newCarry, out = self.rnnCell(carry[0], carry[1])
        return (newCarry, x), out

# ** end class PhaseRNN


class PhaseRNNsym(nn.Module):
    """
    Implementation of an RNN to encode the phase which consists of an RNNCellStack with an additional output layer.

    Arguments: 
        * ``L``: length of the spin chain
        * ``hiddenSize``: size of the hidden state vector
        * ``depth``: number of RNN-cells in the RNNCellStack
        * ``inputDim``: dimension of the input
        * ``actFun``: non-linear activation function
        * ``initScale``: factor by which the initial parameters are scaled

    Returns:
        symmetry averaged phase of the coefficient
    """
    L: int = 10
    hiddenSize: int = 10
    depth: int = 1
    inputDim: int = 2
    actFun: callable = nn.elu
    initScale: float = 1.0
    orbit: any = None
    z2sym: bool = False

    @nn.compact
    def __call__(self, x):

        self.rnn = PhaseRNN(L=self.L, hiddenSize=self.hiddenSize, depth=self.depth,
                            inputDim=self.inputDim,  # passDim=self.passDim,
                            actFun=self.actFun,
                            initScale=self.initScale)

        x = jax.vmap(lambda o, s: jnp.dot(o, s), in_axes=(0, None))(self.orbit, x)

        def evaluate(x):
            return self.rnn(x)

        res = jnp.mean(jax.vmap(evaluate)(x), axis=0)

        if self.z2sym:
            res = 0.5 * (res + jnp.mean(jax.vmap(evaluate)(1 - x), axis=0))

        return res

# ** end class PhaseRNNsym


class CpxRNN(nn.Module):
    """
    Implementation of an RNN to encode the phase which consists of an RNNCellStack with an additional output layer.

    Arguments: 
        * ``L``: length of the spin chain
        * ``hiddenSize``: size of the hidden state vector
        * ``inputDim``: dimension of the input
        * ``actFun``: non-linear activation function
        * ``initScale``: factor by which the initial parameters are scaled

    Returns:
        complex coefficient
    """
    L: int = 10
    hiddenSize: int = 10
    inputDim: int = 2
    actFun: callable = nn.elu
    initScale: float = 1.0

    def setup(self):

        self.rnnCell = RNNCell(hiddenSize=self.hiddenSize, actFun=self.actFun, initScale=self.initScale)

        self.probDense = nn.Dense(features=self.inputDim, dtype=global_defs.tReal,
                                  kernel_init=jax.nn.initializers.lecun_normal(dtype=global_defs.tReal),
                                  bias_init=partial(jax.nn.initializers.zeros, dtype=global_defs.tReal))

        self.phaseDense1 = nn.Dense(features=4, dtype=global_defs.tReal,
                                    kernel_init=jax.nn.initializers.lecun_normal(dtype=global_defs.tReal),
                                    bias_init=partial(jax.nn.initializers.zeros, dtype=global_defs.tReal))

        self.phaseDense2 = nn.Dense(features=4, dtype=global_defs.tReal,
                                    kernel_init=jax.nn.initializers.lecun_normal(dtype=global_defs.tReal),
                                    bias_init=partial(jax.nn.initializers.zeros, dtype=global_defs.tReal))

    def __call__(self, x):

        state = jnp.zeros((self.hiddenSize,))

        _, (probs, phaseOut) = self.rnn_cell((state, jnp.zeros(self.inputDim)), jax.nn.one_hot(x, self.inputDim))

        phase = self.actFun(self.phaseDense2(phaseOut))

        return 0.5 * jnp.sum(probs, axis=0) + 1.j * jnp.mean(phase)

    @partial(nn.transforms.scan,
             variable_broadcast='params',
             split_rngs={'params': False})
    def rnn_cell(self, carry, x):
        newCarry = self.rnnCell(carry[0], carry[1])
        logProb = nn.log_softmax(self.actFun(self.probDense(newCarry)))
        logProb = jnp.sum(logProb * x, axis=-1)
        phaseOut = self.actFun(self.phaseDense1(newCarry))
        return (newCarry, x), (jnp.nan_to_num(logProb, nan=-35), phaseOut)

    def sample(self, batchSize, key):
        """sampler
        """
        state = jnp.zeros((batchSize, self.hiddenSize))

        keys = jax.random.split(key, self.L)

        _, res = self.rnn_cell_sampler((state, jnp.zeros((batchSize, self.inputDim))), keys)

        return jnp.transpose(res[1])  # , 0.5 * jnp.sum(res[0], axis=0)

    @partial(nn.transforms.scan,
             variable_broadcast='params',
             split_rngs={'params': False})
    def rnn_cell_sampler(self, carry, x):

        def eval_cell(x, y):
            newCarry = self.rnnCell(x, y)
            return newCarry, self.actFun(self.probDense(newCarry))

        newCarry, logits = jax.vmap(eval_cell)(carry[0], carry[1])
        sampleOut = jax.random.categorical(x, logits)
        sample = jax.nn.one_hot(sampleOut, self.inputDim)
        logProb = jnp.sum(nn.log_softmax(logits) * sample, axis=1)
        return (newCarry, sample), (jnp.nan_to_num(logProb, nan=-35), sampleOut)

# ** end class CpxRNN
