import jax
from jax.config import config
config.update("jax_enable_x64", True)
import flax
import flax.linen as nn
import jax.numpy as jnp

import jVMC.global_defs as global_defs
import jVMC.nets.activation_functions as act_funs
from jVMC.util.symmetries import LatticeSymmetry

from functools import partial
from typing import List, Sequence

import jVMC.nets.initializers


class CNN(nn.Module):
    """ Convolutional neural network with real parameters.

    Arguments:
        * ``F``: Filter diameter
        * ``channels``: Number of channels
        * ``strides``: Number of pixels the filter shifts over
        * ``actFun``: Non-linear activation function
        * ``bias``: Whether to use biases
        * ``firstLayerBias``: Whether to use biases in the first layer
        * ``periodicBoundary``: Whether to use periodic boundary conditions

    Returns:
        Real wave-function amplitude
    """

    F: Sequence[int] = (8,)
    channels: Sequence[int] = (10,)
    strides: Sequence[int] = (1,)
    actFun: Sequence[callable] = (nn.elu,)
    bias: bool = True
    firstLayerBias: bool = False
    periodicBoundary: bool = True

    @nn.compact
    def __call__(self, x):
        initFunction = jax.nn.initializers.variance_scaling(scale=1.0, mode="fan_avg", distribution="uniform")

        # Set up padding for periodic boundary conditions
        # Padding size must be 1 - filter diameter
        pads = [(0, 0)]
        for f in self.F:
            if self.periodicBoundary:
                pads.append((0, f - 1))
            else:
                pads.append((f - 1, f - 1))
        pads.append((0, 0))

        bias = [self.bias] * len(self.channels)
        bias[0] = self.firstLayerBias

        activationFunctions = [f for f in self.actFun]
        for l in range(len(activationFunctions), len(self.channels)):
            activationFunctions.append(self.actFun[-1])

        # List of axes that will be summed for symmetrization
        reduceDims = tuple([-i - 1 for i in range(len(self.strides) + 2)])

        # Add feature dimension
        x = jnp.expand_dims(jnp.expand_dims(2 * x - 1, axis=0), axis=-1)
        for c, fun, b in zip(self.channels, activationFunctions, bias):
            if self.periodicBoundary:
                x = jnp.pad(x, pads, 'wrap')
            else:
                x = jnp.pad(x, pads, 'constant', constant_values=0)

            x = fun(nn.Conv(features=c, kernel_size=tuple(self.F),
                            strides=self.strides, padding=[(0, 0)] * len(self.strides),
                            use_bias=b, dtype=global_defs.tReal, param_dtype=global_defs.tReal,
                            kernel_init=initFunction)(x))

        nrm = jnp.sqrt(jnp.prod(jnp.array(x.shape[reduceDims[-1]:])))

        return jnp.sum(x, axis=reduceDims) / nrm

# ** end class CNN


class CpxCNN(nn.Module):
    """Convolutional neural network with complex parameters.

    Arguments:
        * ``F``: Filter diameter
        * ``channels``: Number of channels
        * ``strides``: Number of pixels the filter shifts over
        * ``actFun``: Non-linear activation function
        * ``bias``: Whether to use biases
        * ``firstLayerBias``: Whether to use biases in the first layer
        * ``periodicBoundary``: Whether to use periodic boundary conditions

    Returns:
        Complex wave-function amplitude
    """
    F: Sequence[int] = (8,)
    channels: Sequence[int] = (10,)
    strides: Sequence[int] = (1,)
    actFun: Sequence[callable] = (act_funs.poly6,)
    bias: bool = True
    firstLayerBias: bool = False
    periodicBoundary: bool = True

    @nn.compact
    def __call__(self, x):

        #initFunction = jax.nn.initializers.variance_scaling(scale=1.0, mode="fan_avg", distribution="uniform", dtype=global_defs.tReal)
        initFunction = jVMC.nets.initializers.cplx_variance_scaling

        # Set up padding for periodic boundary conditions
        # Padding size must be 1 - filter diameter
        pads = [(0, 0)]
        for f in self.F:
            if self.periodicBoundary:
                pads.append((0, f - 1))
            else:
                pads.append((f - 1, f - 1))
        pads.append((0, 0))

        bias = [self.bias] * len(self.channels)
        bias[0] = self.firstLayerBias

        activationFunctions = [f for f in self.actFun]
        for l in range(len(activationFunctions), len(self.channels)):
            activationFunctions.append(self.actFun[-1])

        # List of axes that will be summed for symmetrization
        reduceDims = tuple([-i - 1 for i in range(len(self.strides) + 2)])

        # Add feature dimension
        x = jnp.expand_dims(jnp.expand_dims(2 * x - 1, axis=0), axis=-1)
        for c, f, b in zip(self.channels, activationFunctions, bias):
            if self.periodicBoundary:
                x = jnp.pad(x, pads, 'wrap')
            # else:
            #    x = jnp.pad(x, pads, 'constant', constant_values=0)
            x = f(nn.Conv(features=c, kernel_size=tuple(self.F),
                          strides=self.strides,
                          use_bias=b, dtype=global_defs.tCpx,
                          kernel_init=initFunction)(x))

        # strides=self.strides, padding=[(0, 0)] * len(self.strides),
        nrm = jnp.sqrt(jnp.prod(jnp.array(x.shape[reduceDims[-1]:])))

        return jnp.sum(x, axis=reduceDims) / nrm

# ** end class CpxCNN


class CpxCNNSym(nn.Module):
    """
    Complex symmetric CNN.
    It uses the CpxCNN class to compute probabilities and averages the outputs over all symmetry-invariant configurations.

    Arguments:
        * ``orbit``: orbits which define the symmetry operations (instance of ``util.symmetries.LatticeSymmetry``)
        * ``F``: Filter diameter
        * ``channels``: Number of channels
        * ``strides``: Number of pixels the filter shifts over
        * ``actFun``: Non-linear activation function
        * ``bias``: Whether to use biases
        * ``firstLayerBias``: Whether to use biases in the first layer

    Returns:
        Symmetry-averaged logarithmic wave-function coefficient or POVM-probability
    """
    orbit: LatticeSymmetry
    F: Sequence[int] = (8,)
    channels: Sequence[int] = (10,)
    strides: Sequence[int] = (1,)
    actFun: Sequence[callable] = (act_funs.poly6,)
    bias: bool = True
    firstLayerBias: bool = False

    def setup(self):

        self.cnn = CpxCNN(F=self.F, channels=self.channels,
                          strides=self.strides, actFun=self.actFun,
                          bias=self.bias, firstLayerBias=self.firstLayerBias)

    def __call__(self, x):

        inShape = x.shape
        x = jax.vmap(lambda o, s: jnp.dot(o, s.ravel()).reshape(inShape), in_axes=(0, None))(self.orbit.orbit, x)

        def evaluate(x):
            return self.cnn(x)

        res = jnp.mean(jax.vmap(evaluate)(x), axis=0)

        return res

# ** end class CpxCNNSym
