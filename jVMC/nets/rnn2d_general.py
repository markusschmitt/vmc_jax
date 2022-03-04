import jax
from jax.config import config
config.update("jax_enable_x64", True)
# config.update('jax_disable_jit', True)
import flax
import flax.linen as nn
import numpy as np
import jax.numpy as jnp

from typing import Union

import sys
# Find jVMC package
sys.path.append(sys.path[0] + "/../..")

import jVMC
import jVMC.global_defs as global_defs
from jVMC.nets.initializers import init_fn_args
from jVMC.util.symmetries import LatticeSymmetry

from functools import partial


class RNNCellStack(nn.Module):
    """
    Implementation of a stack of RNN-cells which is scanned over an input sequence.
    This is achieved by stacking multiple 'vanilla' RNN-cells to obtain a deep RNN.

    Initialization arguments:
        * ``hiddenSize``: size of the hidden state vector
        * ``actFun``: non-linear activation function
        * ``initScale``: factor by which the initial parameters are scaled

    Returns:
        New set of hidden states (one for each layer), as well as the last hidden state, that serves as input to the output layer
    """

    cells: list
    dtype: type = global_defs.tReal
    initFun: callable = partial(jax.nn.initializers.variance_scaling(scale=0.1, mode="fan_avg", distribution="uniform"),
                                dtype=global_defs.tReal)

    @nn.compact
    def __call__(self, carryH, carryV, xH, xV):
        newCarry = jnp.zeros_like(carryH, dtype=self.dtype)
        newR = jnp.concatenate([xH, xV])
        
        for j, (cH, cV, cell) in enumerate(zip(carryH, carryV, self.cells)):
            #carry = jnp.concatenate([cH, cV], axis=-1)
            #current_carry, newR = cell(carry, newR)
            current_carry, newR = cell(cH, cV, newR)
            #newCarry = jax.ops.index_update(newCarry, j, current_carry)
            newCarry = newCarry.at[j].set(current_carry)
        return newCarry, newR

# ** end class RNNCellStack


class RNN2DGeneral(nn.Module):
    """
    Implementation of a multi-layer RNN for one-dimensional data with arbitrary cell.
    This implementation follows approximately the original proposal for RNN wave functions in
    `Hibat-Allah et al., Phys. Rev. Research 2, 023358 (2020) <https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.2.023358>`_.

    The ``cell`` parameter can be a string ("RNN", "LSTM", or "GRU") indicating a pre-implemented
    cell. Alternatively, a custom cell can be passed in the form of a tuple containing a flax
    module that implements the hidden state update rule and the initial value of the hidden state 
    (i.e., the initial ``carry``).
    The signature of the ``__call__`` function of the cell flax module has to be 
    ``(carry, state) -> (new_carry, output)``.

    This model can produce real positive or complex valued output. In either case the output is
    normalized such that

        :math:`\sum_s |RNN(s)|^{1/\kappa}=1`.

    Here, :math:`\kappa` corresponds to the initialization parameter ``logProbFactor``. Thereby, the RNN
    can represent both probability distributions and wave functions. Real or complex valued output is 
    chosen through the parameter ``realValuedOutput``.

    The RNN allows for autoregressive sampling through the ``sample`` member function.

    Initialization arguments:
        * ``L``: length of the spin chain
        * ``hiddenSize``: size of the hidden state vector
        * ``depth``: number of RNN-cells in the RNNCellStack
        * ``inputDim``: dimension of the input
        * ``actFun``: non-linear activation function
        * ``initScale``: factor by which the initial parameters are scaled
        * ``logProbFactor``: factor defining how output and associated sample probability are related. 0.5 for pure states and 1 for POVMs.
        * ``realValuedOutput``: Boolean indicating whether the network output is a real or complex number.
        * ``realValuedParams``: Boolean indicating whether the network parameters are real or complex parameters.
        * ``cell``: String ("RNN", "LSTM", or "GRU") or custom definition indicating which type of cell to use for hidden state  transformations.

    """

    L: int = 10
    hiddenSize: int = 10
    depth: int = 1
    inputDim: int = 2
    actFun: callable = nn.elu
    initScale: float = 1.0
    logProbFactor: float = 0.5
    realValuedOutput: bool = False
    realValuedParams: bool = True
    cell: Union[str, list] = "RNN"

    def setup(self):
        if self.realValuedParams:
            self.dtype = global_defs.tReal
            self.initFunction = jax.nn.initializers.variance_scaling(scale=self.initScale, mode="fan_avg", distribution="uniform")
        else:
            self.dtype = global_defs.tCpx
            self.initFunction = partial(jVMC.nets.initializers.cplx_variance_scaling, scale=self.initScale)

        if isinstance(self.cell, str):
            if self.cell in ["LSTM", "GRU"] and not self.realValuedParams:
                ValueError("Complex parameters for LSTM/GRU not yet implemented.")

            self.zero_carry = jnp.zeros((self.L, self.depth, 1, self.hiddenSize), dtype=self.dtype)
            if self.cell == "RNN":
                self.cells = [RNNCell(actFun=self.actFun, initFun=self.initFunction, dtype=self.dtype) for _ in range(self.depth)]
            elif self.cell == "LSTM":
                self.cells = [LSTMCell() for _ in range(self.depth)]
                self.zero_carry = jnp.zeros((self.L, self.depth, 2, self.hiddenSize), dtype=self.dtype)
            elif self.cell == "GRU":
                self.cells = [GRUCell() for _ in range(self.depth)]
            else:
                ValueError("Cell name not recognized.")
        else:
            self.cells = self.cell[0]
            self.zero_carry = self.cell[1]

        self.rnnCell = RNNCellStack(self.cells, dtype=self.dtype, initFun=self.initFunction)
        self.outputDense = nn.Dense(features=(self.inputDim-1) * (2 - self.realValuedOutput),
                                    use_bias=True, 
                                    **init_fn_args(bias_init=jax.nn.initializers.zeros,
                                                    kernel_init=self.initFunction, 
                                                    dtype=self.dtype)
                                    )

    def log_coeffs_to_log_probs(self, logCoeffs):
        phase = jnp.zeros((self.inputDim))
        if not self.realValuedOutput and self.realValuedParams:
            phase = 1.j*jnp.concatenate([jnp.array([0.0]), logCoeffs[self.inputDim-1:]]).transpose()
        amp = jnp.concatenate([jnp.array([0.0]), logCoeffs[:self.inputDim-1]]).transpose()

        return (self.logProbFactor * jax.nn.log_softmax(amp)).transpose() + phase 


    def reverse_line(self, line, b):
        return jax.lax.cond(b == 1, lambda z: z, lambda z: jnp.flip(z, 0), line)

    def __call__(self, x):
        # Scan directions for zigzag path
        direction = np.ones(self.L, dtype=np.int32)
        direction[1::2] = -1
        direction = jnp.asarray(direction)
        _, probs = self.rnn_cell_V(
            (self.zero_carry, jnp.zeros((self.L, self.inputDim), dtype=np.int32)),
            (jax.nn.one_hot(x, self.inputDim, dtype=np.int32), direction)
        )
        return jnp.sum(probs)

    @partial(nn.transforms.scan,
             variable_broadcast='params',
             split_rngs={'params': False})
    def rnn_cell_V(self, carry, x):
        _, out = self.rnn_cell_H(self.zero_carry[0],
                                 (self.reverse_line(carry[0], x[1]),
                                  jnp.concatenate((jnp.zeros((1, self.inputDim), dtype=np.int32), self.reverse_line(x[0], x[1])), axis=0)[:-1],
                                  self.reverse_line(carry[1], x[1]),
                                  self.reverse_line(x[0], x[1]))
                                 )
        logProb = jnp.sum(self.reverse_line(out[1], x[1]) * x[0], axis=-1)
        return (self.reverse_line(out[0], x[1]), x[0]), jnp.nan_to_num(logProb, nan=-35)

    @partial(nn.transforms.scan,
             variable_broadcast='params',
             split_rngs={'params': False})
    def rnn_cell_H(self, carry, x):
        newCarry, out = self.rnnCell(carry, x[0], x[1], x[2])
        out = self.outputDense(out)

        logCoeff = self.log_coeffs_to_log_probs(out)
        return newCarry, (newCarry, logCoeff)

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
                (self.zero_carry, jnp.zeros((self.L, self.inputDim))),
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
        _, out = self.rnn_cell_H_sample((self.zero_carry[0], jnp.zeros(self.inputDim)),
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
        logCoeffs = self.log_coeffs_to_log_probs(out)
        sampleOut = jax.random.categorical(x[2], jnp.real(logCoeffs) / self.logProbFactor)

        return (newCarry, jax.nn.one_hot(sampleOut, self.inputDim)), (newCarry, sampleOut)

# ** end class RNN2D


class GRUCell(nn.Module):
    @nn.compact
    def __call__(self, carryH, carryV, state):
        cellCarryH = nn.Dense(features=carryH.shape[-1],
                              use_bias=True,
                              dtype=global_defs.tReal)
        cellCarryV = nn.Dense(features=carryV.shape[-1],
                              use_bias=False,
                              dtype=global_defs.tReal)
        current_carry, newR = nn.GRUCell()(cellCarryH(carryH) + cellCarryV(carryV), state)

        return current_carry, newR[0]


class LSTMCell(nn.Module):
    @nn.compact
    def __call__(self, carryH, carryV, state):
        cellCarryH = nn.Dense(features=carryH.shape[-1],
                              use_bias=True,
                              dtype=global_defs.tReal)
        cellCarryV = nn.Dense(features=carryV.shape[-1],
                              use_bias=False,
                              dtype=global_defs.tReal)
        current_carry, newR = nn.OptimizedLSTMCell()(cellCarryH(carryH) + cellCarryV(carryV), state)

        return jnp.asarray(current_carry), newR


class RNNCell(nn.Module):
    """
    Implementation of a 'vanilla' RNN-cell, that is part of an RNNCellStack which is scanned over an input sequence.
    The RNNCell therefore receives two inputs, the hidden state (if it is in a deep part of the CellStack) or the
    input (if it is the first cell of the CellStack) aswell as the hidden state of the previous RNN-cell.
    Both inputs are mapped to obtain a new hidden state, which is what the RNNCell implements.

    Initialization arguments:
        * ``hiddenSize``: size of the hidden state vector
        * ``actFun``: non-linear activation function
        * ``initScale``: factor by which the initial parameters are scaled

    Returns:
        New hidden state
    """

    initFun: callable = jax.nn.initializers.variance_scaling(scale=0.1, mode="fan_avg", distribution="uniform")
    dtype: type = global_defs.tReal
    actFun: callable = nn.elu

    @nn.compact
    def __call__(self, carryH, carryV, state):
        kernel_init_fun = partial(jax.nn.initializers.variance_scaling(scale=0.1, mode="fan_avg", distribution="uniform"),
                                  dtype=global_defs.tReal)
        cellCarryH = nn.Dense(features=carryH.shape[-1],
                              use_bias=False,
                              **init_fn_args(bias_init=jax.nn.initializers.zeros,
                                             kernel_init=self.initFun, 
                                             dtype=self.dtype)
                             )
        cellCarryV = nn.Dense(features=carryV.shape[-1],
                              use_bias=False,
                              **init_fn_args(bias_init=jax.nn.initializers.zeros,
                                             kernel_init=self.initFun, 
                                             dtype=self.dtype)
                             )
        cellState = nn.Dense(features=carryH.shape[-1],
                             use_bias=False,
                              **init_fn_args(bias_init=jax.nn.initializers.zeros,
                                             kernel_init=self.initFun, 
                                             dtype=self.dtype)
                             )

        newCarry = self.actFun(cellCarryH(carryH[0]) + cellCarryV(carryV[0]) + cellState(state))[None, :]
        return newCarry, newCarry[0]


class RNN2DGeneralSym(nn.Module):
    """
    Implementation of an RNN which consists of an RNNCellStack with an additional output layer.
    It uses the RNN class to compute probabilities and averages the outputs over all symmetry-invariant configurations.

    Initialization arguments:
        * ``orbit``: collection of maps that define symmetries (instance of ``util.symmetries.LatticeSymmetry``)
        * ``L``: length of the spin chain
        * ``hiddenSize``: size of the hidden state vector
        * ``depth``: number of RNN-cells in the RNNCellStack
        * ``inputDim``: dimension of the input
        * ``actFun``: non-linear activation function
        * ``initScale``: factor by which the initial parameters are scaled
        * ``logProbFactor``: factor defining how output and associated sample probability are related. 0.5 for pure states and 1 for POVMs.
        * ``z2sym``: for pure states; implement Z2 symmetry

    """
    orbit: LatticeSymmetry
    L: int = 10
    hiddenSize: int = 10
    depth: int = 1
    inputDim: int = 2
    actFun: callable = nn.elu
    initScale: float = 1.0
    logProbFactor: float = 0.5
    realValuedOutput: bool = False
    realValuedParams: bool = True
    cell: Union[str, list] = "RNN"
    z2sym: bool = False

    def setup(self):

        self.rnn = RNN2DGeneral(L=self.L, hiddenSize=self.hiddenSize, depth=self.depth,
                                inputDim=self.inputDim,
                                actFun=self.actFun, initScale=self.initScale,
                                logProbFactor=self.logProbFactor,
                                realValuedOutput=self.realValuedOutput,
                                realValuedParams=self.realValuedParams)


    def __call__(self, x):

        x = jax.vmap(lambda o, s: jnp.dot(o, s), in_axes=(0, None))(self.orbit.orbit, x)

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

        orbitIdx = jax.random.choice(key2, self.orbit.orbit.shape[0], shape=(batchSize,))

        configs = jax.vmap(lambda k, o, s: jnp.dot(o[k], s), in_axes=(0, None, 0))(orbitIdx, self.orbit.orbit, configs)

        if self.z2sym:
            key3, _ = jax.random.split(key2)
            flipChoice = jax.random.choice(key3, 2, shape=(batchSize,))
            configs = jax.vmap(lambda b, c: jax.lax.cond(b == 1, lambda x: 1 - x, lambda x: x, c), in_axes=(0, 0))(flipChoice, configs)

        return configs

# ** end class RNN2DGeneralSym
