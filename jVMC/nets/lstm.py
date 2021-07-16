import jax
from jax.config import config
config.update("jax_enable_x64", True)
import flax
import flax.linen as nn
import numpy as np
import jax.numpy as jnp

import jVMC.global_defs as global_defs

from functools import partial


class LSTMCell(nn.Module):
    """
    Implementation of a LSTM-cell, that is scanned over an input sequence.
    The LSTMCell therefore receives two inputs, the hidden state (if it is in a deep part of the CellStack) or the 
    input (if it is the first cell of the CellStack) aswell as the hidden state of the previous RNN-cell.
    Both inputs are mapped to obtain a new hidden state, which is what the RNNCell implements.

    Arguments: 
        * ``inputDim``: size of the input Dimension
        * ``actFun``: non-linear activation function

    Returns:
        new hidden state
    """

    inputDim: int = 2
    actFun: callable = nn.elu

    @nn.compact
    def __call__(self, carry, x):

        newCarry, out = nn.LSTMCell(kernel_init=partial(flax.nn.linear.default_kernel_init, dtype=global_defs.tReal),
                                    recurrent_kernel_init=partial(flax.nn.initializers.orthogonal(), dtype=global_defs.tReal),
                                    bias_init=partial(flax.nn.initializers.zeros, dtype=global_defs.tReal))(carry, x)

        out = self.actFun(nn.Dense(features=self.inputDim)(out))

        return newCarry, out.reshape((-1))

# ** end class LSTMCell


class LSTM(nn.Module):
    """
    Implementation of an LSTM which consists of an LSTMCell with an additional output layer.
    This class defines how sequential input data is treated.

    Arguments: 
        * ``L``: length of the spin chain
        * ``hiddenSize``: size of the hidden state vector
        * ``inputDim``: dimension of the input
        * ``actFun``: non-linear activation function
        * ``logProbFactor``: factor defining how output and associated sample probability are related. 0.5 for pure states and 1 for POVMs.

    Returns:
        logarithmic wave-function coefficient or POVM-probability
    """
    L: int = 10
    hiddenSize: int = 10
    inputDim: int = 2
    actFun: callable = nn.elu
    logProbFactor: float = 0.5

    def setup(self):

        self.lstmCell = LSTMCell(inputDim=self.inputDim, actFun=self.actFun)

    def __call__(self, x):

        state = nn.LSTMCell.initialize_carry(jax.random.PRNGKey(0), (1,), self.hiddenSize)

        _, probs = self.lstm_cell((state, jnp.zeros(self.inputDim)), jax.nn.one_hot(x, self.inputDim))

        return self.logProbFactor * jnp.sum(probs, axis=0)

    @partial(nn.transforms.scan,
             variable_broadcast='params',
             split_rngs={'params': False})
    def lstm_cell(self, carry, x):

        newCarry, out = self.lstmCell(carry[0], carry[1])
        prob = nn.softmax(out)
        prob = jnp.log(jnp.sum(prob * x, axis=-1))

        return (newCarry, x), prob

    def sample(self, batchSize, key):
        """sampler
        """

        outputs = jnp.asarray(np.zeros((batchSize, self.L, self.L)))

        state = nn.LSTMCell.initialize_carry(jax.random.PRNGKey(0), (batchSize,), self.hiddenSize)

        keys = jax.random.split(key, self.L)
        _, res = self.lstm_cell_sample((state, jnp.zeros((batchSize, self.inputDim))), keys)

        return jnp.transpose(res[1])

    @partial(nn.transforms.scan,
             variable_broadcast='params',
             split_rngs={'params': False})
    def lstm_cell_sample(self, carry, x):
        newCarry, logits = jax.vmap(self.lstmCell)(carry[0], carry[1])
        sampleOut = jax.random.categorical(x, logits)
        sample = jax.nn.one_hot(sampleOut, self.inputDim)
        logProb = jnp.log(jnp.sum(nn.softmax(logits) * sample, axis=1))
        return (newCarry, sample), (logProb, sampleOut)

# ** end class LSTM


class LSTMsym(nn.Module):
    """
    Implementation of an LSTM which consists of an LSTMCellStack with an additional output layer.
    It uses the LSTM class to compute probabilities and averages the outputs over all symmetry-invariant configurations.

    Arguments: 
        * ``L``: length of the spin chain
        * ``hiddenSize``: size of the hidden state vector
        * ``inputDim``: dimension of the input
        * ``actFun``: non-linear activation function
        * ``orbit``: collection of maps that define symmetries

    Returns:
        Symmetry-averaged logarithmic wave-function coefficient or POVM-probability
    """
    L: int = 10
    hiddenSize: int = 10
    inputDim: int = 2
    actFun: callable = nn.elu
    logProbFactor: float = 0.5
    orbit: any = None

    def setup(self):

        self.lstm = LSTM.shared(L=L, hiddenSize=hiddenSize, inputDim=inputDim, actFun=actFun)

    def __call__(self, x, L=10, hiddenSize=10, inputDim=2, actFun=nn.elu, logProbFactor=0.5, orbit=None):

        x = jax.vmap(lambda o, s: jnp.dot(o, s), in_axes=(0, None))(self.orbit, x)

        def evaluate(x):
            return self.lstm(x)

        logProbs = logProbFactor * jnp.log(jnp.mean(jnp.exp((1. / logProbFactor) * jax.vmap(evaluate)(x)), axis=0))

        return logProbs

    def sample(self, batchSize, key):

        key1, key2 = jax.random.split(key)

        configs = self.lstm.sample(batchSize, key1)

        orbitIdx = jax.random.choice(key2, orbit.shape[0], shape=(batchSize,))

        configs = jax.vmap(lambda k, o, s: jnp.dot(o[k], s), in_axes=(0, None, 0))(orbitIdx, self.orbit, configs)

        return configs

# ** end class LSTMsym
