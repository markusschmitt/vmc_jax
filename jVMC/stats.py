import jax
import jax.numpy as jnp
from jax.tree_util import tree_flatten, tree_unflatten, tree_map

import jVMC
import jVMC.mpi_wrapper as mpi
import jVMC.global_defs as global_defs
from jVMC.global_defs import pmap_for_my_devices

import numpy as np

from functools import partial

_mean_helper = None
_data_prep = None
_covar_helper = None
_covar_var_helper = None
_covar_data_helper = None
_trafo_helper_1 = None
_trafo_helper_2 = None
_select_helper = None
_get_subset_helper = None
_subset_mean_helper = None
_subset_data_prep = None

statsPmapDevices = None

def jit_my_stuff():
    # This is a helper function to make sure that pmap'd functions work with the actual choice of devices
    # at all times.

    global _mean_helper
    global _covar_helper
    global _covar_var_helper
    global _covar_data_helper
    global _trafo_helper_1
    global _trafo_helper_2
    global _select_helper
    global _data_prep
    global _get_subset_helper
    global _subset_mean_helper
    global _subset_data_prep

    global statsPmapDevices

    if jVMC.global_defs.pmap_devices_updated(statsPmapDevices):

        statsPmapDevices = global_defs.myPmapDevices

        _mean_helper = jVMC.global_defs.pmap_for_my_devices(lambda data, w: jnp.tensordot(w, data, axes=(0,0)), in_axes=(0, 0))
        _data_prep = jVMC.global_defs.pmap_for_my_devices(lambda data, w, mean: jax.vmap(lambda d, w, m: jnp.sqrt(w) * (d - m), in_axes=(0,0,None))(data, w, mean), in_axes=(0, 0, None))
        _covar_helper = jVMC.global_defs.pmap_for_my_devices(
                                lambda data1, data2:
                                    jnp.tensordot(
                                        jnp.conj(data1),
                                        data2, axes=(0,0)), 
                                in_axes=(0, 0)
                                )
        _covar_var_helper = jVMC.global_defs.pmap_for_my_devices(
                                    lambda data1, data2, w: 
                                        jnp.sum(
                                            jnp.abs( 
                                                jax.vmap(lambda a,b: jnp.outer(a,b))(jnp.conj(data1), data2),
                                            )**2 / w[...,None,None],
                                            axis=0),
                                    in_axes=(0, 0, 0)
                                    )
        _covar_data_helper = jVMC.global_defs.pmap_for_my_devices(lambda data1, data2, w: jax.vmap(lambda a,b,w: jnp.outer(a,b) / w)(jnp.conj(data1), data2, w), in_axes=(0, 0, 0))
        _trafo_helper_1 = jVMC.global_defs.pmap_for_my_devices(
                                lambda data, w, mean, f: f(
                                    jax.vmap(lambda x,y: x/jnp.sqrt(y), in_axes=(0,0))(data, w) 
                                    + mean
                                    ), 
                                in_axes=(0, 0, None), static_broadcasted_argnums=(3,))
        _trafo_helper_2 = jVMC.global_defs.pmap_for_my_devices(
                                lambda data, w, mean, v, f: 
                                    jnp.matmul(v, 
                                                f(
                                                jax.vmap(lambda x,y: x/jnp.sqrt(y), in_axes=(0,0))(data, w) 
                                                + mean
                                                )
                                    ), 
                                in_axes=(0, 0, None, None), static_broadcasted_argnums=(4,))
        _select_helper = jVMC.global_defs.pmap_for_my_devices( lambda ix,g: jax.vmap(lambda ix,g: g[ix], in_axes=(None, 0))(ix,g), in_axes=(None, 0) )
        _get_subset_helper = jVMC.global_defs.pmap_for_my_devices(lambda x, ixs: x[slice(*ixs)], in_axes=(0,), static_broadcasted_argnums=(1,))
        _subset_mean_helper = jVMC.global_defs.pmap_for_my_devices(lambda d, w, m: jnp.tensordot(jnp.sqrt(w), d, axes=(0,0)) + m, in_axes=(0,0,None))
        _subset_data_prep = jVMC.global_defs.pmap_for_my_devices(jax.vmap(lambda d, w, m1, m2: d+jnp.sqrt(w)*(m1-m2), in_axes=(0,0,None,None)), in_axes=(0,0,None,None))


# def get_op_estimator(psi, operator, *args):

#     op_fun = operator.compile()
#     if type(op_fun) is tuple:
#         op_fun_args = op_fun[1](*args)
#         op_fun = op_fun[0]
#     net_fun = psi.net.apply

#     def op_estimator(params, config):

#         sp, matEls = op_fun(config, *op_fun_args)

#         log_psi_s = net_fun(params, config)
#         log_psi_sp = jax.vmap(lambda s: net_fun(params,s))(sp)

#         return jnp.dot(matEls, jnp.exp(log_psi_sp - log_psi_s))

#     return op_estimator


def flat_grad(fun):

    def grad_fun(*args):
        grad_tree = jax.grad(fun)(*args)

        dtypes = [a.dtype for a in tree_flatten(args[0])[0]]
        if dtypes[0] == np.single or dtypes[0] == np.double:
            grad_vec = tree_flatten(
                        tree_map(
                            lambda x: x.ravel(), 
                            grad_tree
                            )
                        )[0]
        else:
            grad_vec = tree_flatten(
                    tree_map(
                        lambda x: [jnp.real(x.ravel()), -jnp.imag(x.ravel())], 
                        grad_tree
                        )
                    )[0]
            
        return jnp.concatenate(grad_vec)
    
    return grad_fun


class SampledObs():
    """This class implements the computation of statistics from Monte Carlo or exact samples.

    Initializer arguments:
        * ``observations``: Observations :math:`O_n` in the sample. This can be the value of an observable `O(s_n)` or the \
                plain configuration `s_n`. The array must have a leading device dimension plus a batch dimension.
        * ``weights``: Weights :math:`w_n` associated with observation :math:`O_n`.
        * ``estimator``: [optional] Function :math:`O(\\theta, s)` that computes an estimator parametrized by :math:`\\theta`
        * ``params``: [optional] A set of parameters for the estimator function.
    """

    def __init__(
            self,
            observations=None, 
            weights=None,
            estimator=None,
            params=None
            ):
        """Initializes SampledObs class.

        Args:
            * ``observations``: Observations :math:`O_n` in the sample. This can be the value of an observable `O(s_n)` or the \
                plain configuration `s_n`. The array must have a leading device dimension plus a batch dimension.
            * ``weights``: Weights :math:`w_n` associated with observation :math:`O_n`.
            * ``estimator``: [optional] Function :math:`O(\\theta, s)` that computes an estimator parametrized by :math:`\\theta`
            * ``params``: [optional] A set of parameters for the estimator function.
        """

        jit_my_stuff()

        self._weights = weights
        self._data = observations
        self._mean = None
        self._configs = None
        def estimator_not_implemented(p,s):
            raise Exception("No estimator function given.")

        self._estimator = estimator_not_implemented
        self._estimator_grad = estimator_not_implemented

        if estimator is not None:

            self._configs = observations

            self._estimator = jVMC.global_defs.pmap_for_my_devices(
                jax.vmap(lambda p, s: estimator(p,s), in_axes=(None, 0)), 
                in_axes=(None, 0)
                )
            
            self._estimator_grad = jVMC.global_defs.pmap_for_my_devices(
                                    jax.vmap(lambda p, s: flat_grad(lambda a, b: jnp.real(estimator(a,b)))(p,s) + 1.j*flat_grad(lambda a, b: jnp.imag(estimator(a,b)))(p,s), in_axes=(None, 0)), 
                                    #jax.vmap(lambda p, s: flat_grad(lambda a, b: jnp.real(estimator(a,b)))(p,s), in_axes=(None, 0)), 
                                    in_axes=(None, 0)
                                    )
            
            if params is not None:

                observations = self._estimator(params, self._configs)

        self._compute_data_and_mean(observations)
        

    def _compute_data_and_mean(self, observations):

        if (observations is not None) and (self._weights is not None):
            if len(observations.shape) == 2:
                observations = observations[...,None]

            #self._weights = weights
            self._mean = mpi.global_sum( _mean_helper(observations,self._weights)[:, None,...]  )
            self._data = _data_prep(observations, self._weights, self._mean)


    def mean(self, params=None):
        """Returns the mean.
        """

        if params is not None:
            observations = self._estimator(params, self._configs)
            self._compute_data_and_mean(observations)

        return self._mean
    

    def mean_and_grad(self, psi, params):
        """Returns the mean and gradient of the given estimator.
        """

        obs = self._estimator(params, self._configs)
        self._compute_data_and_mean(obs)

        obsGrad = self._estimator_grad(params, self._configs)
        obsGradMean = mpi.global_sum( _mean_helper(obsGrad,self._weights)[None,...] )

        psiGrad = SampledObs( 2.0*jnp.real( psi.gradients(self._configs) ), self._weights )

        return self._mean, psiGrad.covar(self).ravel() + obsGradMean



    def covar(self, other=None):
        """Returns the covariance.

        Args:
            * ``other`` [optional]: Another instance of `SampledObs`.
        """

        if other is None:
            other = self
        
        return mpi.global_sum( _covar_helper(self._data, other._data)[:, None,...]  )
    

    def var(self):
        """Returns the variance.
        """

        return mpi.global_sum( jnp.abs(self._data)**2 )
    

    def covar_data(self, other=None):
        """Returns the covariance.

        Args:
            * ``other`` [optional]: Another instance of `SampledObs`.
        """

        if other is None:
            other = self

        return SampledObs( _covar_data_helper(self._data, other._data, self._weights), self._weights )
    

    def covar_var(self, other=None):
        """Returns the variance of the covariance.

        Args:
            * ``other`` [optional]: Another instance of `SampledObs`.
        """

        if other is None:
            other = self
        
        return mpi.global_sum( _covar_var_helper(self._data, other._data, self._weights)[:, None,...]  ) \
                    - jnp.abs(self.covar(other))**2


    def transform(self, nonLinearFun=lambda x: x, linearFun=None):
        """Returns a `SampledObs` for the transformed data.

        Args:
            * ``fun``: A function.
        """

        if linearFun is None:
            return SampledObs( _trafo_helper_1(self._data, self._weights, self._mean, nonLinearFun), self._weights )
        
        return SampledObs( _trafo_helper_2(self._data, self._weights, self._mean, linearFun, nonLinearFun), self._weights )


    def select(self, ixs):
        """Returns a `SampledObs` for the data selection indicated by the given indices.

        Args:
            * ``ixs``: Indices of selected data.
        """

        newObs = SampledObs()
        newObs._data = _select_helper(ixs, self._data)
        newObs._mean = self._mean[ixs]
        newObs._weights = self._weights

        return newObs
    
    
    def subset(self, start=None, end=None, step=None):
        """Returns a `SampledObs` for a subset of the data.

        Args:
            * ``start``: Start sample index for subset selection
            * ``end``: End sample index for subset selection
            * ``step``: Sample index step for subset selection
        """ 

        newObs = SampledObs()
        newObs._weights = _get_subset_helper(self._weights, (start, end, step))
        normalization = mpi.global_sum(newObs._weights)
        newObs._data = _get_subset_helper(self._data, (start, end, step))
        newObs._weights = newObs._weights / normalization
        newObs._data = newObs._data / jnp.sqrt(normalization)

        newObs._mean = mpi.global_sum( _subset_mean_helper(newObs._data, newObs._weights, 0.0)[:,None,...] )  + self._mean
        newObs._data = _subset_data_prep(newObs._data, newObs._weights, self._mean, newObs._mean)

        return newObs


    def tangent_kernel(self):

        all_data = mpi.gather(self._data)
        
        return jnp.matmul(all_data, jnp.conj(jnp.transpose(all_data)))


