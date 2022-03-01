import jax
from jax.config import config
config.update("jax_enable_x64", True)
import flax
import flax.linen as nn
import numpy as np
import jax.numpy as jnp

import jVMC.global_defs as global_defs
from jVMC.util.symmetries import LatticeSymmetry

from functools import partial


class RNNCell2D(nn.Module):
    """
    Implementation of a 'vanilla' RNN-cell in two dimensions, that is part of an RNNCellStack which is scanned over a (two dimensional) input sequence.
    The RNNCell2D therefore receives three inputs, the hidden state (if it is in a deep part of the CellStack) or the 
    input (if it is the first cell of the CellStack) aswell as the hidden states of the two neighboring spin sites that were already computed.
    All inputs are mapped to obtain a new hidden state, which is what the RNNCell2D implements.

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

    @nn.compact
    def __call__(self, carryH, carryV, newR):

        initFunctionCell = jax.nn.initializers.variance_scaling(scale=1.0, mode="fan_avg", distribution="uniform")
        initFunctionOut = jax.nn.initializers.variance_scaling(scale=self.initScale, mode="fan_in", distribution="uniform")

        cellIn = nn.Dense(features=self.hiddenSize,
                          use_bias=False,
                          kernel_init=initFunctionCell,
                          dtype=global_defs.tReal,
                          param_dtype=global_defs.tReal)
        cellCarry = nn.Dense(features=self.hiddenSize,
                             use_bias=True,
                             kernel_init=initFunctionCell,
                             bias_init=jax.nn.initializers.zeros,
                             dtype=global_defs.tReal,
                             param_dtype=global_defs.tReal)
        newCarry = self.actFun(cellCarry(carryH + carryV) + cellIn(newR))
        return newCarry

# ** end class RNNCell2D


class RNNCellStack2D(nn.Module):
    """
    Implementation of a stack of RNN-cells which is scanned over a two dimensional input sequence.
    This is achieved by stacking multiple 'vanilla' 2D RNN-cells to obtain a deep RNN.

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
    def __call__(self, carryH, carryV, stateH, stateV):
        newCarry = jnp.zeros(shape=(carryH.shape[0], self.hiddenSize), dtype=global_defs.tReal)

        newR = stateH + stateV
        # Can't use scan for this, because then flax doesn't realize that each cell has different parameters
        for j in range(carryH.shape[0]):
            newCarry = newCarry.at[j].set(RNNCell2D(hiddenSize=self.hiddenSize, actFun=self.actFun, initScale=self.initScale)(carryH[j], carryV[j], newR))
            newR = newCarry[j]

        return jnp.array(newCarry), newR

# ** end class RNNCellStack2D


class RNN2D(nn.Module):
    """
    Implementation of an RNN in two dimensions which consists of an RNNCellStack with an additional output layer.
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

        self.rnnCell = RNNCellStack2D(hiddenSize=self.hiddenSize,
                                      actFun=self.actFun,
                                      initScale=self.initScale)
        initFunctionCell = jax.nn.initializers.variance_scaling(scale=1.0, mode="fan_avg", distribution="uniform")
        self.outputDense = nn.Dense(features=self.inputDim,
                                    use_bias=True,
                                    kernel_init=initFunctionCell,
                                    bias_init=jax.nn.initializers.zeros,
                                    dtype=global_defs.tReal,
                                    param_dtype=global_defs.tReal)

    def reverse_line(self, line, b):
        return jax.lax.cond(b == 1, lambda z: z, lambda z: jnp.flip(z, 0), line)

    def __call__(self, x):
        # Scan directions for zigzag path
        direction = np.ones(self.L, dtype=np.int32)
        direction[1::2] = -1
        direction = jnp.asarray(direction)
        _, probs = self.rnn_cell_V(
            (jnp.zeros((self.L, self.depth, self.hiddenSize)), jnp.zeros((self.L, self.inputDim))),
            (jax.nn.one_hot(x, self.inputDim), direction)
        )
        return self.logProbFactor * jnp.sum(probs)

    @partial(nn.transforms.scan,
             variable_broadcast='params',
             split_rngs={'params': False})
    def rnn_cell_V(self, carry, x):
        _, out = self.rnn_cell_H(jnp.zeros((self.depth, self.hiddenSize)),
                                 (self.reverse_line(carry[0], x[1]),
                                  jnp.concatenate((jnp.zeros((1, self.inputDim)), self.reverse_line(x[0], x[1])), axis=0)[:-1],
                                  self.reverse_line(carry[1], x[1]))
                                 )

        logProb = jnp.sum(self.reverse_line(out[1], x[1]) * x[0], axis=-1)
        return (self.reverse_line(out[0], x[1]), x[0]), jnp.nan_to_num(logProb, nan=-35)

    @partial(nn.transforms.scan,
             variable_broadcast='params',
             split_rngs={'params': False})
    def rnn_cell_H(self, carry, x):
        newCarry, out = self.rnnCell(carry, x[0], x[1], x[2])
        out = self.outputDense(out)
        return newCarry, (newCarry, nn.log_softmax(out))

    def sample(self, batchSize, key):
        """sampler
        """

        # Scan directions for zigzag path
        direction = np.ones(self.L, dtype=np.int32)
        direction[1::2] = -1
        direction = jnp.asarray(direction)

        def generate_sample(key):
            myKeys = jax.random.split(key, self.L)
            _, sample = self.rnn_cell_V_sample(
                (jnp.zeros((self.L, self.depth, self.hiddenSize)), jnp.zeros((self.L, self.inputDim))),
                (myKeys, direction)
            )
            return sample

        keys = jax.random.split(key, batchSize)

        return jax.vmap(generate_sample)(keys)

    @partial(nn.transforms.scan,
             variable_broadcast='params',
             split_rngs={'params': False})
    def rnn_cell_V_sample(self, carry, x):
        keys = jax.random.split(x[0], self.L)
        _, out = self.rnn_cell_H_sample((jnp.zeros((self.depth, self.hiddenSize)), jnp.zeros(self.inputDim)),
                                        (self.reverse_line(carry[0], x[1]),
                                         self.reverse_line(carry[1], x[1]),
                                         keys)
                                        )

        configLine = self.reverse_line(out[1], x[1])

        return (self.reverse_line(out[0], x[1]), jax.nn.one_hot(configLine, self.inputDim)), configLine

    @partial(nn.transforms.scan,
             variable_broadcast='params',
             split_rngs={'params': False})
    def rnn_cell_H_sample(self, carry, x):
        newCarry, out = self.rnnCell(carry[0], x[0], carry[1], x[1])
        out = self.outputDense(out)
        sampleOut = jax.random.categorical(x[2], out)
        return (newCarry, jax.nn.one_hot(sampleOut, self.inputDim)), (newCarry, sampleOut)

# ** end class RNN2D


class RNN2Dsym(nn.Module):
    """
    Implementation of an RNN in two dimensions which consists of an RNNCellStack with an additional output layer.
    It uses the RNN class to compute probabilities and averages the outputs over all symmetry-invariant configurations.

    Arguments: 
        * ``orbit``: collection of maps that define symmetries (instance of ``util.symmetries.LatticeSymmetry``)
        * ``L``: length of the spin chain
        * ``hiddenSize``: size of the hidden state vector
        * ``depth``: number of RNN-cells in the RNNCellStack
        * ``inputDim``: dimension of the input
        * ``actFun``: non-linear activation function
        * ``initScale``: factor by which the initial parameters are scaled
        * ``logProbFactor``: factor defining how output and associated sample probability are related. 0.5 for pure states and 1 for POVMs.
        * ``z2sym``: for pure states; implement Z2 symmetry

    Returns:
        Symmetry-averaged logarithmic wave-function coefficient or POVM-probability
    """
    orbit: LatticeSymmetry
    L: int = 10
    hiddenSize: int = 10
    depth: int = 1
    inputDim: int = 2
    actFun: callable = nn.elu
    initScale: float = 1.0
    logProbFactor: float = 0.5
    z2sym: bool = False

    def setup(self):

        self.rnn = RNN2D(L=self.L, hiddenSize=self.hiddenSize, depth=self.depth,
                         inputDim=self.inputDim,
                         actFun=self.actFun, initScale=self.initScale,
                         logProbFactor=self.logProbFactor)

    def __call__(self, x):

        x = jax.vmap(lambda o, s: jnp.dot(o, s.ravel()).reshape((self.L, self.L)), in_axes=(0, None))(self.orbit.orbit, x)

        def evaluate(x):
            return self.rnn(x)

        res = jnp.mean(jnp.exp((1. / self.logProbFactor) * jax.vmap(evaluate)(x)), axis=0)

        if self.z2sym:
            res = 0.5 * (res + jnp.mean(jnp.exp((1. / self.logProbFactor) * jax.vmap(evaluate)(1 - x)), axis=0))

        logProbs = self.logProbFactor * jnp.log(res)

        return logProbs

    def sample(self, batchSize, key):

        key1, key2 = jax.random.split(key)

        configs = self.rnn.sample(batchSize, key1)

        orbitIdx = jax.random.choice(key2, self.orbit.orbit.shape[0], shape=(batchSize,))

        configs = jax.vmap(lambda k, o, s: jnp.dot(o[k], s.ravel()).reshape((self.L, self.L)), in_axes=(0, None, 0))(orbitIdx, self.orbit.orbit, configs)

        if self.z2sym:
            key3, _ = jax.random.split(key2)
            flipChoice = jax.random.choice(key3, 2, shape=(batchSize,))
            configs = jax.vmap(lambda b, c: jax.lax.cond(b == 1, lambda x: 1 - x, lambda x: x, c), in_axes=(0, 0))(flipChoice, configs)

        return configs

# ** end class RNN2Dsym
