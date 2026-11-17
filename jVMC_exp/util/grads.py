"""
Gradients of a network with respect to its parameters, for ``NQS``.

``pick_gradient()`` chooses the functions for a network: its parameters are real or complex, and a
network with complex parameters is holomorphic or not. The flat gradients are the derivatives with
respect to the real parameters, in the layout of ``NQS.parameters_flat``: for every leaf of the
parameter tree, the derivatives with respect to the real parts of its entries, followed, for
complex parameters, by those with respect to the imaginary parts. They are complex if the output
of the network is complex.

All the functions take ``(fun, params, arg)``: ``fun(params, arg)`` evaluates the network on one
input ``arg``, and ``params`` is the dictionary of variables with the ``"params"`` collection.
"""
import jax
from jax import grad
from jax.tree_util import tree_flatten, tree_map
import jax.numpy as jnp

def flat_gradient_real(fun, params, arg):
    """Flat gradient of a real ``fun`` of real parameters."""
    g = grad(lambda p, y: jnp.real(fun(p, y)))(params, arg)["params"]
    g = tree_flatten(tree_map(lambda x: x.ravel(), g))[0]

    return jnp.concatenate(g)

def flat_gradient(fun, params, arg):
    """
    Flat gradient of a complex ``fun`` of real parameters, \
    :math:`\\partial_k\\mathrm{Re}f+i\\,\\partial_k\\mathrm{Im}f`.
    """
    gr = grad(lambda p, y: jnp.real(fun(p, y)))(params, arg)["params"]
    gr = tree_flatten(tree_map(lambda x: x.ravel(), gr))[0]
    gi = grad(lambda p, y: jnp.imag(fun(p, y)))(params, arg)["params"]
    gi = tree_flatten(tree_map(lambda x: x.ravel(), gi))[0]

    return jnp.concatenate(gr) + 1.j * jnp.concatenate(gi)

def flat_gradient_nonholo(fun, params, arg):
    """
    Flat gradient of a complex ``fun`` of complex parameters that is not holomorphic: \
    ``2 * size`` entries per leaf, the derivatives with respect to the real and the imaginary \
    parts of the parameters.
    """
    gr = grad(lambda p, y: jnp.real(fun(p, y)))(params, arg)["params"]
    gr = tree_flatten(tree_map(lambda x: [jnp.real(x.ravel()), -jnp.imag(x.ravel())], gr))[0]
    gi = grad(lambda p, y: jnp.imag(fun(p, y)))(params, arg)["params"]
    gi = tree_flatten(tree_map(lambda x: [jnp.real(x.ravel()), -jnp.imag(x.ravel())], gi))[0]

    return jnp.concatenate(gr) + 1.j * jnp.concatenate(gi)

def flat_gradient_holo(fun, params, arg):
    """
    Flat gradient of a holomorphic ``fun`` of complex parameters: for every leaf, the complex \
    derivatives :math:`f'` followed by :math:`if'`, which are the derivatives with respect to the \
    real and the imaginary parts of the parameters.
    """
    g = grad(lambda p, y: jnp.real(fun(p, y)))(params, arg)["params"]
    g = tree_flatten(tree_map(lambda x: [x.ravel(), 1.j * x.ravel()], g))[0]

    return jnp.concatenate(g)

def dict_gradient(fun, params, arg):
    """
    Gradient of a complex ``fun`` of real parameters, as a dictionary with the structure of \
    ``params["params"]`` and flattened leaves.
    """
    gr = grad(lambda p, y: jnp.real(fun(p, y)))(params, arg)["params"]
    gr = tree_map(lambda x: x.ravel(), gr)
    gi = grad(lambda p, y: jnp.imag(fun(p, y)))(params, arg)["params"]
    gi = tree_map(lambda x: x.ravel(), gi)

    return tree_map(lambda x,y: x + 1.j * y, gr, gi)

def dict_gradient_real(fun, params, arg):
    """
    Gradient of a real ``fun`` of real parameters, or complex derivative :math:`f'` of a \
    holomorphic ``fun``, as a dictionary with the structure of ``params["params"]`` and \
    flattened leaves.
    """
    g = grad(lambda p, y: jnp.real(fun(p, y)))(params, arg)["params"]
    g = tree_map(lambda x: x.ravel(), g)

    return g

def dict_gradient_nonholo(fun, params, arg):
    """
    Gradient of a non-holomorphic ``fun`` of complex parameters, as a dictionary with the \
    structure of ``params["params"]``: every leaf holds the derivatives with respect to the real \
    parts of its entries, followed by those with respect to the imaginary parts.
    """
    split = lambda x: jnp.concatenate([jnp.real(x.ravel()), -jnp.imag(x.ravel())])
    gr = tree_map(split, grad(lambda p, y: jnp.real(fun(p, y)))(params, arg)["params"])
    gi = tree_map(split, grad(lambda p, y: jnp.imag(fun(p, y)))(params, arg)["params"])

    return tree_map(lambda x, y: x + 1.j * y, gr, gi)

def pick_gradient(fn, params, single_input):
    """
    Choose the gradient functions of a network.

    All the parameters must have the same data type, otherwise ``Exception`` is raised. A \
    network with complex parameters counts as holomorphic if the gradients ``jax.grad`` of \
    :math:`\\mathrm{Re}f` and of :math:`\\mathrm{Im}f` at ``single_input`` satisfy \
    :math:`\\nabla\\mathrm{Re}f=i\\nabla\\mathrm{Im}f`, the Cauchy-Riemann equations.

    Args:
        * ``fn``: The network, called as ``fn(params, single_input)``.
        * ``params``: Its variables, with the ``"params"`` collection.
        * ``single_input``: One input of the network.

    Returns:
        ``real_params, holomorphic, flat_gradient_fn, dict_gradient_fn``: whether the parameters \
        are real, whether the network is holomorphic, and the functions for the flat and the \
        dictionary gradients from this module.
    """
    dtypes = [a.dtype for a in tree_flatten(params)[0]]
    if not all(d == dtypes[0] for d in dtypes):
        raise Exception("Network uses different parameter data types. This is not supported.")

    real_params = jnp.issubdtype(dtypes[0], jnp.floating)
    if real_params:
        holomorphic = False
    else:
        def make_flat(t):
            return jnp.concatenate([p.ravel() for p in tree_flatten(t)[0]])
        
        grads_r = make_flat(
            jax.grad(lambda a, b: jnp.real(fn(a, b)))(params, single_input)["params"]
        )
        grads_i = make_flat(
            jax.grad(lambda a, b: jnp.imag(fn(a, b)))(params, single_input)["params"]
        )
        holomorphic = jnp.isclose(
            jnp.linalg.norm(grads_r - 1.j * grads_i) / grads_r.shape[0], 0.0, atol=1e-14
        )

    if holomorphic:
       return real_params, holomorphic, flat_gradient_holo, dict_gradient_real

    if real_params:
        out = jax.eval_shape(fn, params, single_input)
        if not jnp.issubdtype(out.dtype, jnp.complexfloating):
            return real_params, holomorphic, flat_gradient_real, dict_gradient_real

        return real_params, holomorphic, flat_gradient, dict_gradient
    else:
        return real_params, holomorphic, flat_gradient_nonholo, dict_gradient_nonholo