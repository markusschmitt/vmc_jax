import jax
import flax.linen as nn
import jax.numpy as jnp
from typing import Union
from functools import partial

import jVMC_exp
from jVMC_exp import global_defs
from jVMC_exp.nets.initializers import init_fn_args

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
    dtype: type = global_defs.DT_PARAMS_REAL
    actFun: callable = nn.elu
    initFun: callable = jax.nn.initializers.variance_scaling(scale=0.1, mode="fan_avg", distribution="uniform")

    @ nn.compact
    def __call__(self, carry, newR):
        newCarry = jnp.zeros_like(carry)
       
        newR = nn.Dense(
            features=carry.shape[-1], 
            use_bias=False,
            **init_fn_args(kernel_init=self.initFun, param_dtype=self.dtype), 
            name="data_in_dense"
        )(newR)
        newR = self.actFun(newR)

        for j, (c, cell) in enumerate(zip(carry, self.cells)):
            current_carry, newR = cell(c, newR)
            newCarry = newCarry.at[j].set(current_carry)
        return newCarry, newR

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

        :math:`\\sum_s |RNN(s)|^{1/\\kappa}=1`.

    Here, :math:`\\kappa` corresponds to the initialization parameter ``logProbFactor``. Thereby, the RNN
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
        if isinstance(self.cell, str) and self.cell != "RNN" and not self.realValuedParams:
            raise ValueError("Complex parameters for LSTM/GRU not yet implemented.")

        if self.realValuedParams:
            self.dtype = global_defs.DT_PARAMS_REAL
            self.initFunction = jax.nn.initializers.variance_scaling(
                scale=self.initScale, mode="fan_avg", distribution="uniform", dtype=self.dtype
            )
        else:
            self.dtype = global_defs.DT_PARAMS_CPX
            self.initFunction = jVMC_exp.nets.initializers.cplx_variance_scaling 

        if isinstance(self.cell, str):
            self.zero_carry = jnp.zeros((self.depth, 1, self.hiddenSize), dtype=self.dtype)
            if self.cell == "RNN":
                self.cells = [
                    RNNCell(
                        actFun=self.actFun, initFun=self.initFunction, dtype=self.dtype
                    ) for _ in range(self.depth)
                ]
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
        init_args = init_fn_args(
            param_dtype=self.dtype, 
            bias_init=jax.nn.initializers.zeros, 
            kernel_init=self.initFunction
        )
        self.outputDense = nn.Dense(
            features=(self.inputDim - 1) * (2 - self.realValuedOutput), use_bias=True, **init_args
        )

    def log_coeffs_to_log_probs(self, logCoeffs):
        phase = jnp.zeros((self.inputDim))
        if not self.realValuedOutput and self.realValuedParams:
            phase = 1.j*jnp.concatenate([jnp.array([0.0]), logCoeffs[self.inputDim-1:]]).transpose()
        amp = jnp.concatenate([jnp.array([0.0]), logCoeffs[:self.inputDim-1]]).transpose()

        return (self.logProbFactor * jax.nn.log_softmax(amp)).transpose() + phase 

    def _initial_carry(self, axis_marker):
        # Same method as in the RWKV to make the input device-varying, otherwise it doesn't work with shard_map
        axis_marker = jnp.asarray(axis_marker, dtype=jnp.int32) * jnp.asarray(0, dtype=jnp.int32)
        return (self.zero_carry + axis_marker, jnp.zeros(self.inputDim) + axis_marker)

    def __call__(self, x):
        x = jnp.asarray(x, dtype=jnp.int32)
        _, probs = self.rnn_cell(self._initial_carry(x[0]), jax.nn.one_hot(x, self.inputDim))

        return jnp.sum(probs)

    @partial(nn.transforms.scan,
             variable_broadcast='params',
             split_rngs={'params': False})
    def rnn_cell(self, carry, x):
        newCarry, out = self.rnnCell(carry[0], carry[1])
        logProb = self.log_coeffs_to_log_probs(self.outputDense(out))
        logProb = jnp.sum(logProb * x, axis=-1)
        return (newCarry, x), jnp.nan_to_num(logProb, nan=-35)
    
    def sample(self, key):
        myKeys = jax.random.split(key, self.L)
        _, sample = self.rnn_cell_sample(
            self._initial_carry(key[0]),
            (myKeys)
        )
        return sample[1]

    @partial(nn.transforms.scan,
             variable_broadcast='params',
             split_rngs={'params': False})
    def rnn_cell_sample(self, carry, x):
        newCarry, out = self.rnnCell(carry[0], carry[1])
        logCoeffs = self.log_coeffs_to_log_probs(self.outputDense(out))
        sampleOut = jax.random.categorical(x, jnp.real(logCoeffs) / self.logProbFactor).astype(global_defs.DT_SAMPLES)
        return (newCarry, jax.nn.one_hot(sampleOut, self.inputDim)), (jnp.nan_to_num(logCoeffs, nan=-35), sampleOut)

class GRUCell(nn.Module):
    features: int

    @nn.compact
    def __call__(self, carry, state):
        current_carry, newR = nn.GRUCell(
            features=self.features, 
            **init_fn_args(
                param_dtype=global_defs.DT_PARAMS_REAL, 
                recurrent_kernel_init=jax.nn.initializers.orthogonal(dtype=global_defs.DT_PARAMS_REAL)
            )
        )(carry, state)

        return current_carry, newR[0]

class LSTMCell(nn.Module):
    features: int

    @nn.compact
    def __call__(self, carry, state):
        current_carry, newR = nn.OptimizedLSTMCell(
            features=self.features, 
            **init_fn_args(
                param_dtype=global_defs.DT_PARAMS_REAL, 
                recurrent_kernel_init=jax.nn.initializers.orthogonal(dtype=global_defs.DT_PARAMS_REAL)
            )
        )(carry, state)

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
    dtype: type = global_defs.DT_PARAMS_REAL

    @nn.compact
    def __call__(self, carry, state):
        cellCarry = nn.Dense(
            features=carry.shape[-1],
            use_bias=False,
            **init_fn_args(
                param_dtype=self.dtype,
                bias_init=jax.nn.initializers.zeros,
                kernel_init=self.initFun
            ),
            name="cell_carry_dense"
        )

        newCarry = (self.actFun(cellCarry(carry[0])) + state)[None, :]

        return newCarry, newCarry[0]
