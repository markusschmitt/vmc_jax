import jax
from flax import nn

def square(x):
    return x**2

def poly6(x):
    x = x**2
    return ((0.022222222 * x -0.083333333) * x + 0.5) * x

def poly5(x):
    return jax.grad(poly6)(x) 

activationFunctions = {
    "square" : square ,
    "poly5" : poly5 ,
    "poly6" : poly6 ,
    "elu" : nn.elu ,
    "relu" : nn.relu
}
