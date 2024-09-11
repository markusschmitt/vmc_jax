import jax
jax.config.update("jax_enable_x64", True)
import flax
import flax.linen as nn
import numpy as np
import jax.numpy as jnp


# Find jVMC package
import sys
sys.path.append(sys.path[0] + "../..")
import jVMC
import jVMC.global_defs as global_defs
from jVMC.nets.initializers import init_fn_args
from jVMC.util.symmetries import LatticeSymmetry

from typing import Union

from functools import partial


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

    cells: list
    dtype: type = global_defs.tReal
    actFun: callable = nn.elu
    initFun: callable = jax.nn.initializers.variance_scaling(scale=0.1, mode="fan_avg", distribution="uniform")

    @ nn.compact
    def __call__(self, carry, newR):
        newCarry = jnp.zeros_like(carry)
       
        newR = nn.Dense(features=carry.shape[-1], use_bias=False,
                        **init_fn_args(kernel_init=self.initFun, dtype=self.dtype), 
                        name="data_in_dense")(newR)
        newR = self.actFun(newR)

        for j, (c, cell) in enumerate(zip(carry, self.cells)):
            current_carry, newR = cell(c, newR)
            newCarry = newCarry.at[j].set(current_carry)
        return newCarry, newR

# ** end class RNNCellStack


class RNN1DGeneral(nn.Module):
    """
    Implementation of a multi-layer RNN for one-dimensional data with arbitrary cell.

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
        if isinstance(self.cell, str) and self.cell != "RNN":
            ValueError("Complex parameters for LSTM/GRU not yet implemented.")

        if self.realValuedParams:
            self.dtype = global_defs.tReal
            self.initFunction = jax.nn.initializers.variance_scaling(scale=self.initScale, mode="fan_avg", distribution="uniform")
        else:
            self.dtype = global_defs.tCpx
            self.initFunction = jVMC.nets.initializers.cplx_variance_scaling

        if isinstance(self.cell, str):
            self.zero_carry = jnp.zeros((self.depth, 1, self.hiddenSize), dtype=self.dtype)
            if self.cell == "RNN":
                self.cells = [RNNCell(actFun=self.actFun, initFun=self.initFunction, dtype=self.dtype) for _ in range(self.depth)]
            elif self.cell == "LSTM":
                self.cells = [LSTMCell(features=self.hiddenSize) for _ in range(self.depth)]
                self.zero_carry = jnp.zeros((self.depth, 2, self.hiddenSize), dtype=self.dtype)
            elif self.cell == "GRU":
                self.cells = [GRUCell(features=self.hiddenSize) for _ in range(self.depth)]
            else:
                ValueError("Cell name not recognized.")
        else:
            self.cells = self.cell[0]
            self.zero_carry = self.cell[1]

        self.rnnCell = RNNCellStack(self.cells, actFun=self.actFun, dtype=self.dtype)
        init_args = init_fn_args(dtype=self.dtype, bias_init=jax.nn.initializers.zeros, kernel_init=self.initFunction)
        self.outputDense = nn.Dense(features=(self.inputDim-1) * (2 - self.realValuedOutput),
                                    use_bias=True, **init_args)

    def log_coeffs_to_log_probs(self, logCoeffs):
        phase = jnp.zeros((self.inputDim))
        if not self.realValuedOutput and self.realValuedParams:
            phase = 1.j*jnp.concatenate([jnp.array([0.0]), logCoeffs[self.inputDim-1:]]).transpose()
        amp = jnp.concatenate([jnp.array([0.0]), logCoeffs[:self.inputDim-1]]).transpose()

        return (self.logProbFactor * jax.nn.log_softmax(amp)).transpose() + phase 

    def __call__(self, x):
        _, probs = self.rnn_cell((self.zero_carry, jnp.zeros(self.inputDim)), jax.nn.one_hot(x, self.inputDim))

        return jnp.sum(probs)

    @partial(nn.transforms.scan,
             variable_broadcast='params',
             split_rngs={'params': False})
    def rnn_cell(self, carry, x):
        newCarry, out = self.rnnCell(carry[0], carry[1])
        logProb = self.log_coeffs_to_log_probs(self.outputDense(out))
        logProb = jnp.sum(logProb * x, axis=-1)
        return (newCarry, x), jnp.nan_to_num(logProb, nan=-35)

    def sample(self, batchSize, key):
        def generate_sample(key):
            myKeys = jax.random.split(key, self.L)
            _, sample = self.rnn_cell_sample(
                (self.zero_carry, jnp.zeros(self.inputDim)),
                (myKeys)
            )
            return sample[1]

        keys = jax.random.split(key, batchSize)
        return jax.vmap(generate_sample)(keys)

    @partial(nn.transforms.scan,
             variable_broadcast='params',
             split_rngs={'params': False})
    def rnn_cell_sample(self, carry, x):
        newCarry, out = self.rnnCell(carry[0], carry[1])
        logCoeffs = self.log_coeffs_to_log_probs(self.outputDense(out))
        sampleOut = jax.random.categorical(x, jnp.real(logCoeffs) / self.logProbFactor)
        return (newCarry, jax.nn.one_hot(sampleOut, self.inputDim)), (jnp.nan_to_num(logCoeffs, nan=-35), sampleOut)


class GRUCell(nn.Module):
    features: int

    @nn.compact
    def __call__(self, carry, state):
        current_carry, newR = nn.GRUCell(features=self.features, **init_fn_args(recurrent_kernel_init=jax.nn.initializers.orthogonal(dtype=global_defs.tReal)))(carry, state)
        return current_carry, newR[0]


class LSTMCell(nn.Module):
    features: int

    @nn.compact
    def __call__(self, carry, state):
        current_carry, newR = nn.OptimizedLSTMCell(features=self.features, **init_fn_args(recurrent_kernel_init=jax.nn.initializers.orthogonal(dtype=global_defs.tReal)))(carry, state)
        return jnp.asarray(current_carry), newR


class RNNCell(nn.Module):
    """
    Implementation of a 'vanilla' RNN-cell, that is part of an RNNCellStack which is scanned over an input sequence.
    The RNNCell therefore receives two inputs, the hidden state (if it is in a deep part of the CellStack) or the 
    input (if it is the first cell of the CellStack) aswell as the hidden state of the previous RNN-cell.
    Both inputs are mapped to obtain a new hidden state, which is what the RNNCell implements.
    
    Arguments: 
        * ``initFun``: initialization function for parameters
        * ``actFun``: non-linear activation function
        * ``dtype``: data type of parameters

    Returns:
        New hidden state
    """

    initFun: callable = jax.nn.initializers.variance_scaling(scale=1e-1, mode="fan_avg", distribution="uniform")
    actFun: callable = nn.elu
    dtype: type = global_defs.tReal

    @nn.compact
    def __call__(self, carry, state):
        cellCarry = nn.Dense(features=carry.shape[-1],
                             use_bias=False,
                             **init_fn_args(dtype=self.dtype,
                                            bias_init=jax.nn.initializers.zeros,
                                            kernel_init=self.initFun),
                             name="cell_carry_dense")

        newCarry = (self.actFun(cellCarry(carry[0])) + state)[None, :]

        return newCarry, newCarry[0]
