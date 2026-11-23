from functools import partial
from typing import Union

import flax.linen as nn
import jax
import jax.numpy as jnp

import jVMC
from jVMC import global_defs
from jVMC.nets.initializers import init_fn_args


class RNNCellStack(nn.Module):
    """
    Implementation of a stack of RNN-cells which is scanned over an input sequence.
    This is achieved by stacking multiple 'vanilla' RNN-cells to obtain a deep RNN.

    Arguments:
    * ``hidden_size``: size of the hidden state vector
    * ``act_fun``: non-linear activation function
    * ``init_scale``: factor by which the initial parameters are scaled

    Returns:
    New set of hidden states (one for each layer), as well as the last hidden state, that serves as input to the output layer
    """

    cells: list
    dtype: type = global_defs.DT_PARAMS_REAL
    act_fun: callable = nn.elu
    init_fun: callable = jax.nn.initializers.variance_scaling(
        scale=0.1, mode="fan_avg", distribution="uniform"
    )

    @nn.compact
    def __call__(self, carry, new_r):
        new_carry = jnp.zeros_like(carry)

        new_r = nn.Dense(
            features=carry.shape[-1],
            use_bias=False,
            **init_fn_args(kernel_init=self.init_fun, param_dtype=self.dtype),
            name="data_in_dense",
        )(new_r)
        new_r = self.act_fun(new_r)

        for j, (c, cell) in enumerate(zip(carry, self.cells)):
            current_carry, new_r = cell(c, new_r)
            new_carry = new_carry.at[j].set(current_carry)
        return new_carry, new_r


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

    Here, :math:`\\kappa` corresponds to the initialization parameter ``log_prob_factor``. Thereby, the RNN
    can represent both probability distributions and wave functions. Real or complex valued output is
    chosen through the parameter ``real_valued_output``.

    The RNN allows for autoregressive sampling through the ``sample`` member function.

    Initialization arguments:
        * ``L``: length of the spin chain
        * ``hidden_size``: size of the hidden state vector
        * ``depth``: number of RNN-cells in the RNNCellStack
        * ``input_dim``: dimension of the input
        * ``act_fun``: non-linear activation function
        * ``init_scale``: factor by which the initial parameters are scaled
        * ``log_prob_factor``: factor defining how output and associated sample probability are related. 0.5 for pure states and 1 for POVMs.
        * ``real_valued_output``: Boolean indicating whether the network output is a real or complex number.
        * ``real_valued_params``: Boolean indicating whether the network parameters are real or complex parameters. Complex parameters are not supported.
        * ``cell``: String ("RNN", "LSTM", or "GRU") or custom definition indicating which type of cell to use for hidden state  transformations.

    """

    L: int = 10
    hidden_size: int = 10
    depth: int = 1
    input_dim: int = 2
    act_fun: callable = nn.elu
    init_scale: float = 1.0
    log_prob_factor: float = 0.5
    real_valued_output: bool = False
    real_valued_params: bool = True
    cell: Union[str, list] = "RNN"

    def setup(self):
        if not self.real_valued_params:
            raise ValueError("Complex parameters are not supported.")

        if self.real_valued_params:
            self.dtype = global_defs.DT_PARAMS_REAL
            self.init_function = jax.nn.initializers.variance_scaling(
                scale=self.init_scale,
                mode="fan_avg",
                distribution="uniform",
                dtype=self.dtype,
            )
        else:
            self.dtype = global_defs.DT_PARAMS_CPX
            self.init_function = jVMC.nets.initializers.cplx_variance_scaling

        if isinstance(self.cell, str):
            self.zero_carry = jnp.zeros(
                (self.depth, 1, self.hidden_size), dtype=self.dtype
            )
            if self.cell == "RNN":
                self.cells = [
                    RNNCell(
                        act_fun=self.act_fun, init_fun=self.init_function, dtype=self.dtype
                    )
                    for _ in range(self.depth)
                ]
            elif self.cell == "LSTM":
                self.cells = [
                    LSTMCell(features=self.hidden_size) for _ in range(self.depth)
                ]
                self.zero_carry = jnp.zeros(
                    (self.depth, 2, self.hidden_size), dtype=self.dtype
                )
            elif self.cell == "GRU":
                self.cells = [
                    GRUCell(features=self.hidden_size) for _ in range(self.depth)
                ]
            else:
                ValueError("Cell name not recognized.")
        else:
            self.cells = self.cell[0]
            self.zero_carry = self.cell[1]

        self.cell_stack = RNNCellStack(self.cells, act_fun=self.act_fun, dtype=self.dtype)
        init_args = init_fn_args(
            param_dtype=self.dtype,
            bias_init=jax.nn.initializers.zeros,
            kernel_init=self.init_function,
        )
        self.output_dense = nn.Dense(
            features=(self.input_dim - 1) * (2 - self.real_valued_output),
            use_bias=True,
            **init_args,
        )

    def log_coeffs_to_log_probs(self, log_coeffs):
        phase = jnp.zeros(self.input_dim)
        if not self.real_valued_output and self.real_valued_params:
            phase = (
                1.0j
                * jnp.concatenate(
                    [jnp.array([0.0]), log_coeffs[self.input_dim - 1 :]]
                ).transpose()
            )
        amp = jnp.concatenate(
            [jnp.array([0.0]), log_coeffs[: self.input_dim - 1]]
        ).transpose()

        return (self.log_prob_factor * jax.nn.log_softmax(amp)).transpose() + phase

    def _initial_carry(self, axis_marker):
        # Same method as in the RWKV to make the input device-varying, otherwise it doesn't work with shard_map
        axis_marker = jnp.asarray(axis_marker, dtype=jnp.int32) * jnp.asarray(
            0, dtype=jnp.int32
        )
        return (self.zero_carry + axis_marker, jnp.zeros(self.input_dim) + axis_marker)

    def __call__(self, x):
        x = jnp.asarray(x, dtype=jnp.int32)
        _, probs = self.rnn_cell(
            self._initial_carry(x[0]), jax.nn.one_hot(x, self.input_dim)
        )

        return jnp.sum(probs)

    @partial(
        nn.transforms.scan, variable_broadcast="params", split_rngs={"params": False}
    )
    def rnn_cell(self, carry, x):
        new_carry, out = self.cell_stack(carry[0], carry[1])
        log_prob = self.log_coeffs_to_log_probs(self.output_dense(out))
        log_prob = jnp.sum(log_prob * x, axis=-1)
        return (new_carry, x), jnp.nan_to_num(log_prob, nan=-35)

    def sample(self, key):
        my_keys = jax.random.split(key, self.L)
        _, sample = self.rnn_cell_sample(self._initial_carry(key[0]), (my_keys))
        return sample[1]

    @partial(
        nn.transforms.scan, variable_broadcast="params", split_rngs={"params": False}
    )
    def rnn_cell_sample(self, carry, x):
        new_carry, out = self.cell_stack(carry[0], carry[1])
        log_coeffs = self.log_coeffs_to_log_probs(self.output_dense(out))
        sample_out = jax.random.categorical(
            x, jnp.real(log_coeffs) / self.log_prob_factor
        ).astype(global_defs.DT_SAMPLES)
        return (new_carry, jax.nn.one_hot(sample_out, self.input_dim)), (
            jnp.nan_to_num(log_coeffs, nan=-35),
            sample_out,
        )


class GRUCell(nn.Module):
    features: int

    @nn.compact
    def __call__(self, carry, state):
        current_carry, new_r = nn.GRUCell(
            features=self.features,
            **init_fn_args(
                param_dtype=global_defs.DT_PARAMS_REAL,
                recurrent_kernel_init=jax.nn.initializers.orthogonal(
                    dtype=global_defs.DT_PARAMS_REAL
                ),
            ),
        )(carry, state)

        return current_carry, new_r[0]


class LSTMCell(nn.Module):
    features: int

    @nn.compact
    def __call__(self, carry, state):
        current_carry, new_r = nn.OptimizedLSTMCell(
            features=self.features,
            **init_fn_args(
                param_dtype=global_defs.DT_PARAMS_REAL,
                recurrent_kernel_init=jax.nn.initializers.orthogonal(
                    dtype=global_defs.DT_PARAMS_REAL
                ),
            ),
        )(carry, state)

        return jnp.asarray(current_carry), new_r


class RNNCell(nn.Module):
    """
    Implementation of a 'vanilla' RNN-cell, that is part of an RNNCellStack which is scanned over an input sequence.
    The RNNCell therefore receives two inputs, the hidden state (if it is in a deep part of the CellStack) or the
    input (if it is the first cell of the CellStack) aswell as the hidden state of the previous RNN-cell.
    Both inputs are mapped to obtain a new hidden state, which is what the RNNCell implements.

    Arguments:
        * ``init_fun``: initialization function for parameters
        * ``act_fun``: non-linear activation function
        * ``dtype``: data type of parameters

    Returns:
        New hidden state
    """

    init_fun: callable = jax.nn.initializers.variance_scaling(
        scale=1e-1, mode="fan_avg", distribution="uniform"
    )
    act_fun: callable = nn.elu
    dtype: type = global_defs.DT_PARAMS_REAL

    @nn.compact
    def __call__(self, carry, state):
        cell_carry = nn.Dense(
            features=carry.shape[-1],
            use_bias=False,
            **init_fn_args(
                param_dtype=self.dtype,
                bias_init=jax.nn.initializers.zeros,
                kernel_init=self.init_fun,
            ),
            name="cell_carry_dense",
        )

        new_carry = (self.act_fun(cell_carry(carry[0])) + state)[None, :]

        return new_carry, new_carry[0]
