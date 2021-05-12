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
    hiddenSize: int = 10
    outDim: int = 2
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

        # self.outputDense = nn.Dense(features=self.outDim,
        #                               kernel_init=initFunctionOut, dtype=global_defs.tReal,
        #                               bias_init=jax.nn.initializers.normal(stddev=0.1,dtype=global_defs.tReal))

    def __call__(self, carry, x):

        newCarry = self.actFun(self.cellCarry(carry) + self.cellIn(x))
        # out = self.outputDense(newCarry)

        # return newCarry, out
        return newCarry

# ** end class RNNCell


class RNNCellStack(nn.Module):
    hiddenSize: int = 10
    # outDim: int = 2
    actFun: callable = nn.elu
    initScale: float = 1.0
    # passDim: int = outDim

    @nn.compact
    def __call__(self, carry, x):

        # outDims = [self.passDim] * carry.shape[0]
        # outDims[-1] = self.outDim

        # newCarry = [None] * carry.shape[0]
        newCarry = jnp.zeros(shape=(carry.shape[0], self.hiddenSize), dtype=global_defs.tReal)

        newR = x
        # Can't use scan for this, because then flax doesn't realize that each cell has different parameters
        for j, c in enumerate(carry):
            # newCarry[j] = RNNCell(hiddenSize=self.hiddenSize, outDim=outDims[j], actFun=self.actFun, initScale=self.initScale)(c, newR)
            newCarry = jax.ops.index_update(newCarry, j, RNNCell(hiddenSize=self.hiddenSize,
                                                                 # outDim=outDims[j],
                                                                 actFun=self.actFun,
                                                                 initScale=self.initScale)(c, newR))
            newR = newCarry[j]

        return jnp.array(newCarry), newR

# ** end class RNNCellStack


class RNN(nn.Module):
    """Recurrent neural network.
    """
    L: int = 10
    hiddenSize: int = 10
    depth: int = 1
    inputDim: int = 2
    # passDim: int = inputDim
    actFun: callable = nn.elu
    initScale: float = 1.0
    logProbFactor: float = 0.5

    def setup(self):

        self.rnnCell = RNNCellStack(hiddenSize=self.hiddenSize,
                                    # outDim=self.inputDim,
                                    # passDim=self.passDim,
                                    actFun=self.actFun,
                                    initScale=self.initScale,
                                    name="myCell")
        initFunctionCell = partial(jax.nn.initializers.variance_scaling(scale=1.0, mode="fan_avg", distribution="uniform"),
                                   dtype=global_defs.tReal)
        self.outputDense = nn.Dense(features=self.inputDim,
                                    use_bias=True,
                                    kernel_init=initFunctionCell, dtype=global_defs.tReal,
                                    bias_init=partial(jax.nn.initializers.zeros, dtype=global_defs.tReal),
                                    name="myOutput")

    def __call__(self, x):

        state = jnp.zeros((self.depth, self.hiddenSize))

        _, probs = self.rnn_cell((state, jnp.zeros(self.inputDim)), jax.nn.one_hot(x, self.inputDim))

        return self.logProbFactor * jnp.sum(probs, axis=0)

    @partial(nn.transforms.scan,
             variable_broadcast='params',
             split_rngs={'params': False})
    def rnn_cell(self, carry, x):
        newCarry, out = self.rnnCell(carry[0], carry[1])

        # logProb = nn.log_softmax(out)
        logProb = nn.log_softmax(self.outputDense(out))
        logProb = jnp.sum(logProb * x, axis=-1)
        return (newCarry, x), jnp.nan_to_num(logProb, nan=-35)

    def sample(self, batchSize, key):
        """sampler
        """

        outputs = jnp.asarray(np.zeros((batchSize, self.L, self.L)))

        state = jnp.zeros((batchSize, self.depth, self.hiddenSize))

        keys = jax.random.split(key, self.L)
        _, res = self.rnn_cell_sample((state, jnp.zeros((batchSize, self.inputDim))), keys)

        return jnp.transpose(res[1])  # , 0.5 * jnp.sum(res[0], axis=0)

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
    """Recurrent neural network with symmetries.
    """
    L: int = 10
    hiddenSize: int = 10
    depth: int = 1
    inputDim: int = 2
    #passDim: int = inputDim
    actFun: callable = nn.elu
    initScale: float = 1.0
    logProbFactor: float = 0.5
    orbit: any = None
    z2sym: bool = False

    def setup(self):

        self.rnn = RNN(L=self.L, hiddenSize=self.hiddenSize, depth=self.depth,
                       inputDim=self.inputDim,# passDim=self.passDim,
                       actFun=self.actFun, initScale=self.initScale,
                       logProbFactor=self.logProbFactor, name='myRNN')

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
    """Recurrent neural network.
    """
    L: int = 10
    hiddenSize: int = 10
    depth: int = 1
    inputDim: int = 2
    actFun: callable = nn.elu
    initScale: float = 1.0
    #passDim: int = inputDim

    def setup(self):

        self.rnnCell = RNNCellStack(hiddenSize=self.hiddenSize, outDim=self.inputDim,
                                    #passDim=self.passDim, 
                                    actFun=self.actFun, initScale=self.initScale)

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
    """Recurrent neural network with symmetries.
    """
    L: int = 10
    hiddenSize: int = 10
    depth: int = 1
    inputDim: int = 2
    actFun: callable = nn.elu
    initScale: float = 1.0
    orbit: any = None
    z2sym: bool = False
    #passDim: int = inputDim

    @nn.compact
    def __call__(self, x):

        self.rnn = PhaseRNN(L=self.L, hiddenSize=self.hiddenSize, depth=self.depth,
                            inputDim=self.inputDim, #passDim=self.passDim,
                            actFun=aself.ctFun,
                            initScale=self.initScale, name='myRNN')

        x = jax.vmap(lambda o, s: jnp.dot(o, s), in_axes=(0, None))(self.orbit, x)

        def evaluate(x):
            return self.rnn(x)

        res = jnp.mean(jax.vmap(evaluate)(x), axis=0)

        if self.z2sym:
            res = 0.5 * (res + jnp.mean(jax.vmap(evaluate)(1 - x), axis=0))

        return res

# ** end class PhaseRNNsym


class CpxRNN(nn.Module):
    """Recurrent neural network.
    """
    L: int = 10
    hiddenSize: int = 10
    inputDim: int = 2
    actFun: callable = nn.elu
    initScale: float = 1.0
    logProbFactor: float = 0.5

    def setup(self):

        self.rnnCell = RNNCell(hiddenSize=self.hiddenSize, outDim=self.hiddenSize,
                               actFun=self.actFun, initScale=self.initScale, name="myCell")

        self.probDense = nn.Dense(features=self.inputDim, name="probDense", dtype=global_defs.tReal,
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

        return self.logProbFactor * jnp.sum(probs, axis=0) + 1.j * jnp.mean(phase)

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

        outputs = jnp.asarray(np.zeros((batchSize, self.L, self.L)))

        state = jnp.zeros((batchSize, self.hiddenSize))

        keys = jax.random.split(key, self.L)

        _, res = self.rnn_cell_sampler((state, jnp.zeros((batchSize, self.inputDim))), keys)

        return jnp.transpose(res[1])  # , 0.5 * jnp.sum(res[0], axis=0)

    @partial(nn.transforms.scan,
             variable_broadcast='params',
             split_rngs={'params': False})
    def rnn_cell_sampler(carry, x):

        def eval_cell(x, y):
            newCarry = self.rnnCell(x, y)
            return newCarry, actFun(self.probDense(newCarry))

        newCarry, logits = jax.vmap(eval_cell)(carry[0], carry[1])
        sampleOut = jax.random.categorical(x, logits)
        sample = jax.nn.one_hot(sampleOut, self.inputDim)
        logProb = jnp.sum(nn.log_softmax(logits) * sample, axis=1)
        return (newCarry, sample), (jnp.nan_to_num(logProb, nan=-35), sampleOut)
