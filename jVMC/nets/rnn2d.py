import jax
from jax.config import config
config.update("jax_enable_x64", True)
import flax
import flax.linen as nn
import numpy as np
import jax.numpy as jnp

import jVMC.global_defs as global_defs

from functools import partial


class RNNCell2D(nn.Module):
    hiddenSize: int = 10
    outDim: int = 2
    actFun: callable = nn.elu
    initScale: float = 1.0

    @nn.compact
    def __call__(self, carryH, carryV, newR):

        initFunctionCell = partial(jax.nn.initializers.variance_scaling(scale=1.0, mode="fan_avg", distribution="uniform"),
                                   dtype=global_defs.tReal)
        initFunctionOut = partial(jax.nn.initializers.variance_scaling(scale=self.initScale, mode="fan_in", distribution="uniform"),
                                  dtype=global_defs.tReal)

        cellIn = nn.Dense(features=self.hiddenSize,
                          use_bias=False, dtype=global_defs.tReal,
                          kernel_init=initFunctionCell)
        # cellCarryH = nn.Dense(features=self.hiddenSize,
        #                             use_bias=False,
        #                             kernel_init=initFunctionCell, dtype=global_defs.tReal,
        #                             bias_init=partial(jax.nn.initializers.zeros,dtype=global_defs.tReal))
        # cellCarryV = nn.Dense(features=self.hiddenSize,
        #                             use_bias=True,
        #                             kernel_init=initFunctionCell, dtype=global_defs.tReal,
        #                             bias_init=partial(jax.nn.initializers.zeros,dtype=global_defs.tReal))
        cellCarry = nn.Dense(features=self.hiddenSize,
                             use_bias=True,
                             kernel_init=initFunctionCell, dtype=global_defs.tReal,
                             bias_init=partial(jax.nn.initializers.zeros, dtype=global_defs.tReal))

        # outputDense = nn.Dense(features=self.outDim,
        #                        kernel_init=initFunctionOut, dtype=global_defs.tReal,
        #                        bias_init=jax.nn.initializers.normal(stddev=0.1, dtype=global_defs.tReal))

        # newCarry = self.actFun(cellCarryH(carryH) + cellCarryV(carryV) + cellIn(state))
        newCarry = self.actFun(cellCarry(carryH + carryV) + cellIn(newR))
        # out = outputDense(newCarry)

        # return newCarry, out
        return newCarry

# ** end class RNNCell


class RNNCellStack2D(nn.Module):
    hiddenSize: int = 10
    outDim: int = 2
    passDim: int = outDim
    actFun: callable = nn.elu
    initScale: float = 1.0

    @nn.compact
    def __call__(self, carryH, carryV, stateH, stateV):

        # outDims = [self.passDim] * carryH.shape[0]
        # outDims[-1] = self.outDim

        # newCarry = [None] * carryH.shape[0]
        newCarry = jnp.zeros(shape=(carryH.shape[0], self.hiddenSize), dtype=global_defs.tReal)

        # newR = jnp.concatenate((stateH, stateV), axis=0)
        newR = stateH + stateV
        # Can't use scan for this, because then flax doesn't realize that each cell has different parameters
        for j in range(carryH.shape[0]):
            # newCarry[j], newR = RNNCell2D(hiddenSize=self.hiddenSize, outDim=outDims[j], actFun=self.actFun, initScale=self.initScale)(carryH[j], carryV[j], newR)
            newCarry = jax.ops.index_update(newCarry, j, RNNCell2D(hiddenSize=self.hiddenSize, actFun=self.actFun, initScale=self.initScale)(carryH[j], carryV[j], newR))
            newR = newCarry[j]

        return jnp.array(newCarry), newR

# ** end class RNNCellStack


class RNN2D(nn.Module):
    """Recurrent neural network for two-dimensional input.
    """
    L: int = 10
    hiddenSize: int = 10
    depth: int = 1
    inputDim: int = 2
    passDim: int = inputDim
    actFun: callable = nn.elu
    initScale: float = 1.0
    logProbFactor: float = 0.5

    def setup(self):

        self.rnnCell = RNNCellStack2D(hiddenSize=self.hiddenSize,
                                      outDim=self.inputDim,
                                      passDim=self.passDim,
                                      actFun=self.actFun,
                                      initScale=self.initScale, name="myCell")
        initFunctionCell = partial(jax.nn.initializers.variance_scaling(scale=1.0, mode="fan_avg", distribution="uniform"),
                                   dtype=global_defs.tReal)
        self.outputDense = nn.Dense(features=self.inputDim,
                                    use_bias=True,
                                    kernel_init=initFunctionCell, dtype=global_defs.tReal,
                                    bias_init=partial(jax.nn.initializers.zeros, dtype=global_defs.tReal),
                                    name="myOutput")

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
    """Recurrent neural network for two-dimensional input with symmetries.
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

        self.rnn = RNN2D(L=self.L, hiddenSize=self.hiddenSize, depth=self.depth,
                         inputDim=self.inputDim,
                         actFun=self.actFun, initScale=self.initScale,
                         logProbFactor=self.logProbFactor, name='myRNN')

    def __call__(self, x):

        x = jax.vmap(lambda o, s: jnp.dot(o, s.ravel()).reshape((self.L,self.L)), in_axes=(0, None))(self.orbit, x)

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

        orbitIdx = jax.random.choice(key2, self.orbit.shape[0], shape=(batchSize,))

        configs = jax.vmap(lambda k, o, s: jnp.dot(o[k], s.ravel()).reshape((self.L,self.L)), in_axes=(0, None, 0))(orbitIdx, self.orbit, configs)

        if self.z2sym:
            key3, _ = jax.random.split(key2)
            flipChoice = jax.random.choice(key3, 2, shape=(batchSize,))
            configs = jax.vmap(lambda b, c: jax.lax.cond(b == 1, lambda x: 1 - x, lambda x: x, c), in_axes=(0, 0))(flipChoice, configs)

        return configs

# ** end class RNN2Dsym
