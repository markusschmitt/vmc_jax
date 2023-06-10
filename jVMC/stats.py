import jax
import jax.numpy as jnp

import jVMC
import jVMC.mpi_wrapper as mpi
from jVMC.global_defs import pmap_for_my_devices


class SampledObs():
    """This class implements the computation of statistics from Monte Carlo or exact samples.

    Initializer arguments:
        * ``observations``: Observations :math:`O_n` in the sample. The array must have a leading device \
            dimension plus a batch dimension.
        * ``weights``: Weights :math:`w_n` associated with observation :math:`O_n`.
    """

    def __init__(self, observations, weights):
        """Initializes SampledObs class.

        Args:
            * ``observations``: Observations :math:`O_n` in the sample. The array must have a leading device \
                dimension plus a batch dimension.
            * ``weights``: Weights :math:`w_n` associated with observation :math:`O_n`.
        """

        self.jit_my_stuff()

        if len(observations.shape) == 2:
            observations = observations[...,None]

        self._weights = weights
        self._mean = mpi.global_sum( self._mean_helper(observations,self._weights)[None,...] )
        #self._data = self._data_prep(observations, self._mean)
        self._data = self._data_prep(observations, self._weights, self._mean)


    def mean(self):
        """Returns the mean.
        """

        return self._mean
    

    def covar(self, other=None):
        """Returns the covariance.

        Args:
            * ``other`` [optional]: Another instance of `SampledObs`.
        """

        self.jit_my_stuff()

        if other is None:
            other = self
        
        #return mpi.global_sum( self._covar_helper(self._data, other._data, self._weights)[None,...] )
        return mpi.global_sum( self._covar_helper(self._data, other._data)[None,...] )
    

    def var(self):
        """Returns the variance.
        """

        #return mpi.global_sum( self._mean_helper(jnp.abs(self._data)**2, self._weights)[None,...] )
        return mpi.global_sum( jnp.abs(self._data)**2 )
    

    def covar_data(self, other=None):
        """Returns the covariance.

        Args:
            * ``other`` [optional]: Another instance of `SampledObs`.
        """

        self.jit_my_stuff()

        if other is None:
            other = self

        #return SampledObs( self._covar_data_helper(self._data, other._data), self._weights )
        return SampledObs( self._covar_data_helper(self._data, other._data, self._weights), self._weights )
    

    def covar_var(self, other=None):
        """Returns the variance of the covariance.

        Args:
            * ``other`` [optional]: Another instance of `SampledObs`.
        """

        self.jit_my_stuff()

        if other is None:
            other = self
        
        return mpi.global_sum( self._covar_var_helper(self._data, other._data, self._weights)[None,...] ) \
                    - jnp.abs(self.covar(other))**2


    def transform(self, fun=lambda x: x):
        """Returns a `SampledObs` for the transformed data.

        Args:
            * ``fun``: A function.
        """

        #return SampledObs( self._trafo_helper(self._data, self._mean, fun), self._weights )
        return SampledObs( self._trafo_helper(self._data, self._weights, self._mean, fun), self._weights )
    

    def tangent_kernel(self):

        all_data = mpi.gather(self._data)
        
        return jnp.matmul(all_data, jnp.conj(jnp.transpose(all_data)))


    def jit_my_stuff(self):
        # This is a helper function to make sure that pmap'd functions work with the actual choice of devices
        # at all times.

        if jVMC.global_defs.pmap_devices_updated():
            self._mean_helper = pmap_for_my_devices(lambda data, w: jnp.tensordot(w, data, axes=(0,0)), in_axes=(0, 0))
            #self._data_prep = pmap_for_my_devices(lambda data, mean: data - mean, in_axes=(0, None))
            self._data_prep = pmap_for_my_devices(lambda data, w, mean: jax.vmap(lambda d, w, m: jnp.sqrt(w) * (d - m), in_axes=(0,0,None))(data, w, mean), in_axes=(0, 0, None))
            # self._covar_helper = pmap_for_my_devices(
            #                         lambda data1, data2, w:
            #                             jnp.tensordot(
            #                                 jnp.conj(
            #                                     jax.vmap(lambda a,b: a*b, in_axes=(0,0))(w, data1)
            #                                     ),
            #                                 data2, axes=(0,0)), 
            #                         in_axes=(0, 0, 0)
            #                         )
            self._covar_helper = pmap_for_my_devices(
                                    lambda data1, data2:
                                        jnp.tensordot(
                                            jnp.conj(data1),
                                            data2, axes=(0,0)), 
                                    in_axes=(0, 0)
                                    )
            # self._covar_var_helper = pmap_for_my_devices(
            #                             lambda data1, data2, w: 
            #                                 jnp.sum(
            #                                     w[...,None,None] * 
            #                                     jnp.abs( 
            #                                         jax.vmap(lambda a,b: jnp.outer(a,b))(jnp.conj(data1), data2),
            #                                     )**2,
            #                                     axis=0),
            #                             in_axes=(0, 0, 0)
            #                             )
            self._covar_var_helper = pmap_for_my_devices(
                                        lambda data1, data2, w: 
                                            jnp.sum(
                                                jnp.abs( 
                                                    jax.vmap(lambda a,b: jnp.outer(a,b))(jnp.conj(data1), data2),
                                                )**2 / w[...,None,None],
                                                axis=0),
                                        in_axes=(0, 0, 0)
                                        )
            self._covar_data_helper = pmap_for_my_devices(lambda data1, data2, w: jax.vmap(lambda a,b,w: jnp.outer(a,b) / w)(jnp.conj(data1), data2, w), in_axes=(0, 0, 0))
            #self._trafo_helper = pmap_for_my_devices(lambda data, mean, f: f(data + mean), in_axes=(0, None), static_broadcasted_argnums=(2,))
            self._trafo_helper = pmap_for_my_devices(
                                    lambda data, w, mean, f: f(
                                        jax.vmap(lambda x,y: x/jnp.sqrt(y), in_axes=(0,0))(data, w) 
                                        + mean
                                        ), 
                                    in_axes=(0, 0, None), static_broadcasted_argnums=(3,))


        

        