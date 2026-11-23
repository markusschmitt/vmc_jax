from collections.abc import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp

import jVMC.nets.activation_functions as act_funs
import jVMC.nets.initializers
from jVMC import global_defs
from jVMC.nets.initializers import init_fn_args


class CNN(nn.Module):
    """Convolutional neural network with real parameters.

    Initialization arguments:
        * ``F``: Filter diameter
        * ``channels``: Number of channels
        * ``strides``: Number of pixels the filter shifts over
        * ``act_fun``: Non-linear activation function
        * ``bias``: Whether to use biases
        * ``first_layer_bias``: Whether to use biases in the first layer
        * ``periodic_boundary``: Whether to use periodic boundary conditions

    """

    F: Sequence[int] = (8,)
    channels: Sequence[int] = (10,)
    strides: Sequence[int] = (1,)
    act_fun: Sequence[callable] = (nn.elu,)
    bias: bool = True
    first_layer_bias: bool = False
    periodic_boundary: bool = True

    @nn.compact
    def __call__(self, x):
        init_function = jax.nn.initializers.variance_scaling(
            scale=1.0, mode="fan_avg", distribution="uniform"
        )

        # Set up padding for periodic boundary conditions
        # Padding size must be 1 - filter diameter
        pads = [(0, 0)]
        for f in self.F:
            if self.periodic_boundary:
                pads.append((0, f - 1))
            else:
                pads.append((f - 1, f - 1))
        pads.append((0, 0))

        bias = [self.bias] * len(self.channels)
        bias[0] = self.first_layer_bias

        activation_functions = [f for f in self.act_fun]
        for _ in range(len(activation_functions), len(self.channels)):
            activation_functions.append(self.act_fun[-1])

        # List of axes that will be summed for symmetrization
        reduce_dims = tuple([-i - 1 for i in range(len(self.strides) + 2)])

        init_args = init_fn_args(
            param_dtype=global_defs.DT_PARAMS_REAL, kernel_init=init_function
        )

        # Add feature dimension
        x = jnp.expand_dims(jnp.expand_dims(2 * x - 1, axis=0), axis=-1)
        for c, fun, b in zip(self.channels, activation_functions, bias):
            if self.periodic_boundary:
                x = jnp.pad(x, pads, "wrap")
            else:
                x = jnp.pad(x, pads, "constant", constant_values=0)

            x = fun(
                nn.Conv(
                    features=c,
                    kernel_size=tuple(self.F),
                    strides=self.strides,
                    padding=[(0, 0)] * len(self.strides),
                    use_bias=b,
                    **init_args,
                )(x)
            )

        nrm = jnp.sqrt(jnp.prod(jnp.array(x.shape[reduce_dims[-1] :])))

        return jnp.sum(x, axis=reduce_dims) / nrm


class CpxCNN(nn.Module):
    """Convolutional neural network with complex parameters.

    Initialization arguments:
        * ``F``: Filter diameter
        * ``channels``: Number of channels
        * ``strides``: Number of pixels the filter shifts over
        * ``act_fun``: Non-linear activation function
        * ``bias``: Whether to use biases
        * ``first_layer_bias``: Whether to use biases in the first layer
        * ``periodic_boundary``: Whether to use periodic boundary conditions

    """

    F: Sequence[int] = (8,)
    channels: Sequence[int] = (10,)
    strides: Sequence[int] = (1,)
    act_fun: Sequence[callable] = (act_funs.poly6,)
    bias: bool = True
    first_layer_bias: bool = False
    periodic_boundary: bool = True

    @nn.compact
    def __call__(self, x):

        init_function = jVMC.nets.initializers.cplx_variance_scaling

        # Set up padding for periodic boundary conditions
        # Padding size must be 1 - filter diameter
        pads = [(0, 0)]
        for f in self.F:
            if self.periodic_boundary:
                pads.append((0, f - 1))
            else:
                pads.append((f - 1, f - 1))
        pads.append((0, 0))

        bias = [self.bias] * len(self.channels)
        bias[0] = self.first_layer_bias

        activation_functions = [f for f in self.act_fun]
        for l in range(len(activation_functions), len(self.channels)):
            activation_functions.append(self.act_fun[-1])

        init_args = init_fn_args(
            param_dtype=global_defs.DT_PARAMS_CPX, kernel_init=init_function
        )

        # List of axes that will be summed for symmetrization
        reduce_dims = tuple([-i - 1 for i in range(len(self.strides) + 2)])

        # Add feature dimension
        x = jnp.expand_dims(jnp.expand_dims(2 * x - 1, axis=0), axis=-1)
        for c, f, b in zip(self.channels, activation_functions, bias):
            if self.periodic_boundary:
                x = jnp.pad(x, pads, "wrap")
            x = f(
                nn.Conv(
                    features=c,
                    kernel_size=tuple(self.F),
                    strides=self.strides,
                    padding=[(0, 0)] * len(self.strides),
                    use_bias=b,
                    **init_args,
                )(x)
            )

        nrm = jnp.sqrt(jnp.prod(jnp.array(x.shape[reduce_dims[-1] :])))

        return jnp.sum(x, axis=reduce_dims) / nrm
