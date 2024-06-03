import jax
from jax import jit, vmap, grad
import jax.numpy as jnp
import numpy as np

import abc

import sys

import jVMC.global_defs as global_defs

opDtype = global_defs.tCpx

def expand_batch(batch, batchSize):
                outShape = list(batch.shape)
                outShape[0] = batchSize
                outp = jnp.zeros(tuple(outShape), dtype=batch.dtype)
                return outp.at[:batch.shape[0]].set(batch)

class Operator(metaclass=abc.ABCMeta):
    """This class defines an interface and provides functionality to compute operator matrix elements

    This is an abstract class. It provides the general interface to compute operator matrix elements, \
    but it lacks the implementation of operator definitions. Arbitrary operators can be constructed \
    as classes that inherit from ``Operator`` and implement the ``compile()`` method.

    The ``compile()`` method has to return a *jit-able* pure function with the following interface:

        Arguments:
            * ``s``: A single basis configuration.
            * ``*args``: Further positional arguments. E.g. time in the case of time-dependent operators.
        Returns: 
            A tuple ``sp, matEls``, where ``sp`` is the list of connected basis configurations \
            (as ``jax.numpy.array``) and ``matEls`` the corresponding matrix elements.

    Alternatively, ``compile()`` can return a tuple of two functions, the first as described above and
    and the second a preprocessor for the additional positional arguments ``*args``. Assuming that ``compile()``
    returns the tuple ``(f1, f2)``, ``f1`` will be called as ``f1(s, f2(*args))`` .

    *Important:* Any child class inheriting from ``Operator`` has to call ``super().__init__()`` in \
    its constructor.

    **Example:**

        Here we define a :math:`\hat\sigma_1^x` operator acting on lattice site :math:`1` of a spin-1/2 chain.

        .. literalinclude:: ../../examples/ex1_custom_operator.py
                :linenos:
                :language: python
                :lines: 14-

        Then we indeed obtain the correct ``sp`` and ``matEl``:

        >>> print(sp)
        [0 1 0 0]
        >>> print(matEl)
        [1.+0.j]

    """

    def __init__(self, ElocBatchSize=-1):
        """Initialize ``Operator``.
        """

        self.compiled = False
        self.compiled_argnum = -1
        self.ElocBatchSize = ElocBatchSize

        # pmap'd member functions
        self._get_s_primes_pmapd = None
        self._find_nonzero_pmapd = global_defs.pmap_for_my_devices(vmap(self._find_nonzero, in_axes=0))
        self._set_zero_to_zero_pmapd = global_defs.pmap_for_my_devices(jax.vmap(self.set_zero_to_zero, in_axes=(0, 0, 0)), in_axes=(0, 0, 0))
        self._array_idx_pmapd = global_defs.pmap_for_my_devices(jax.vmap(lambda data, idx: data[idx], in_axes=(0, 0)), in_axes=(0, 0))
        self._get_O_loc_pmapd = global_defs.pmap_for_my_devices(self._get_O_loc)
        self._flatten_pmapd = global_defs.pmap_for_my_devices(lambda x: x.reshape(-1, *x.shape[2:]))
        self._alloc_Oloc_cpx_pmapd = global_defs.pmap_for_my_devices(lambda s: jnp.zeros(s.shape[0],
                                                                                         dtype=global_defs.tCpx))
        self._alloc_Oloc_real_pmapd = global_defs.pmap_for_my_devices(lambda s: jnp.zeros(s.shape[0],
                                                                                          dtype=global_defs.tReal))
        self._get_config_batch_pmapd = global_defs.pmap_for_my_devices(lambda d, startIdx, sliceSize: jax.lax.dynamic_slice_in_dim(d, startIdx, sliceSize), in_axes=(0, None, None), static_broadcasted_argnums=(2,))
        self._get_logPsi_batch_pmapd = global_defs.pmap_for_my_devices(lambda d, startIdx, sliceSize: jax.lax.dynamic_slice_in_dim(d, startIdx, sliceSize), in_axes=(0, None, None), static_broadcasted_argnums=(2,))
        self._insert_Oloc_batch_pmapd = global_defs.pmap_for_my_devices(
                                            lambda dst, src, beg: jax.lax.dynamic_update_slice(dst, src, [beg, ]),
                                            in_axes=(0, 0, None)
                                        )
        self._get_Oloc_slice_pmapd = global_defs.pmap_for_my_devices(
                                            lambda d, startIdx, sliceSize: jax.lax.dynamic_slice_in_dim(d, startIdx, sliceSize), 
                                            in_axes=(0, None, None), static_broadcasted_argnums=(2,)
                                        )

    def _find_nonzero(self, m):

        choice = jnp.zeros(m.shape, dtype=np.int64) + m.shape[0] - 1

        def scan_fun(c, x):
            b = jnp.abs(x[0]) > 1e-6
            out = jax.lax.cond(b, lambda z: z[0], lambda z: z[1], (c[1], x[1]))
            newcarry = jax.lax.cond(b, lambda z: (z[0] + 1, z[1] + 1), lambda z: (z[0], z[1] + 1), c)
            return newcarry, out

        carry, choice = jax.lax.scan(scan_fun, (0, 0), (m, choice))

        return jnp.sort(choice), carry[0]

    def set_zero_to_zero(self, m, idx, numNonzero):

        def scan_fun(c, x):
            out = jax.lax.cond(c[1] < c[0], lambda z: z[0], lambda z: z[1], (x, 0. * x))  # use 0.*x to get correct data type
            newCarry = (c[0], c[1] + 1)
            return newCarry, out

        _, m = jax.lax.scan(scan_fun, (numNonzero, 0), m[idx])

        return m

    def get_s_primes(self, s, *args):
        """Compute matrix elements

        For a list of computational basis states :math:`s` this member function computes the corresponding \
        matrix elements :math:`O_{s,s'}=\langle s|\hat O|s'\\rangle` and their respective configurations \
        :math:`s'`.

        Arguments:
            * ``s``: Array of computational basis states.
            * ``*args``: Further positional arguments that are passed on to the specific operator implementation.

        Returns:
            An array holding `all` configurations :math:`s'` and the corresponding matrix elements :math:`O_{s,s'}`.

        """

        def id_fun(*args):
            return args

        if (not self.compiled) or self.compiled_argnum!=len(args):
            fun = self.compile()
            self.compiled = True
            self.compiled_argnum = len(args)
            if type(fun) is tuple:
                self.arg_fun = fun[1]
                args = self.arg_fun(*args)
                fun = fun[0]
            else:
                self.arg_fun = id_fun
            _get_s_primes = jax.vmap(fun, in_axes=(0,)+(None,)*len(args))
            #_get_s_primes = jax.vmap(self.compile(), in_axes=(0,)+(None,)*len(args))
            self._get_s_primes_pmapd = global_defs.pmap_for_my_devices(_get_s_primes, in_axes=(0,)+(None,)*len(args))
        else:
            args = self.arg_fun(*args)

        # Compute matrix elements
        #self.sp, self.matEl = self._get_s_primes_pmapd(s, *args)
        self.sp, self.matEl = self._get_s_primes_pmapd(s, *args)

        # Get only non-zero contributions
        idx, self.numNonzero = self._find_nonzero_pmapd(self.matEl)
        self.matEl = self._set_zero_to_zero_pmapd(self.matEl, idx[..., :jnp.max(self.numNonzero)], self.numNonzero)
        self.sp = self._array_idx_pmapd(self.sp, idx[..., :jnp.max(self.numNonzero)])

        return self._flatten_pmapd(self.sp), self.matEl

    def _get_O_loc(self, matEl, logPsiS, logPsiSP):

        return jax.vmap(lambda x, y, z: jnp.sum(x * jnp.exp(z - y)), in_axes=(0, 0, 0))(matEl, logPsiS, logPsiSP.reshape(matEl.shape))

    def get_O_loc(self, samples, psi, logPsiS=None, *args):
        """Compute :math:`O_{loc}(s)`.

        If the instance parameter ElocBatchSize is larger than 0 :math:`O_{loc}(s)` is computed in a batch-wise manner
        to avoid out-of-memory issues.

        Arguments:
            * ``samples``: Sample of computational basis configurations :math:`s`.
            * ``psi``: Neural quantum state.
            * ``logPsiS``: Logarithmic amplitudes :math:`\\ln(\psi(s))`
            * ``*args``: Further positional arguments for the operator.

        Returns:
            :math:`O_{loc}(s)` for each configuration :math:`s`.
        """

        if logPsiS is None:
            logPsiS = psi(samples)

        if self.ElocBatchSize > 0:
            return self.get_O_loc_batched(samples, psi, logPsiS, self.ElocBatchSize, *args)
        else:
            sampleOffdConfigs, _ = self.get_s_primes(samples, *args)
            logPsiSP = psi(sampleOffdConfigs)
            return self.get_O_loc_unbatched(logPsiS, logPsiSP)

    def get_O_loc_unbatched(self, logPsiS, logPsiSP):
        """Compute :math:`O_{loc}(s)`.

        This member function assumes that ``get_s_primes(s)`` has been called before, as \
        internally stored matrix elements :math:`O_{s,s'}` are used.

        Computes :math:`O_{loc}(s)=\sum_{s'} O_{s,s'}\\frac{\psi(s')}{\psi(s)}`, given the \
        logarithmic wave function amplitudes of the involved configurations :math:`\\ln(\psi(s))` \
        and :math:`\\ln\psi(s')`

        Arguments:
            * ``logPsiS``: Logarithmic amplitudes :math:`\\ln(\psi(s))`
            * ``logPsiSP``: Logarithmic amplitudes :math:`\\ln(\psi(s'))`

        Returns:
            :math:`O_{loc}(s)` for each configuration :math:`s`.
        """

        return self._get_O_loc_pmapd(self.matEl, logPsiS, logPsiSP)

    def get_O_loc_batched(self, samples, psi, logPsiS, batchSize, *args):
        """Compute :math:`O_{loc}(s)` in batches.

        Computes :math:`O_{loc}(s)=\sum_{s'} O_{s,s'}\\frac{\psi(s')}{\psi(s)}` in a batch-wise manner
        to avoid out-of-memory issues.

        Arguments:
            * ``samples``: Sample of computational basis configurations :math:`s`.
            * ``psi``: Neural quantum state.
            * ``logPsiS``: Logarithmic amplitudes :math:`\\ln(\psi(s))`
            * ``batchSize``: Batch size.
            * ``*args``: Further positional arguments for the operator.

        Returns:
            :math:`O_{loc}(s)` for each configuration :math:`s`.
        """

        Oloc = None

        numSamples = samples.shape[1]
        numBatches = numSamples // batchSize
        remainder = numSamples % batchSize

        # Minimize mismatch
        if remainder > 0:
            batchSize = numSamples // (numBatches+1)
            numBatches = numSamples // batchSize
            remainder = numSamples % batchSize

        for b in range(numBatches):

            batch = self._get_config_batch_pmapd(samples, b * batchSize, batchSize)
            logPsiSbatch = self._get_logPsi_batch_pmapd(logPsiS, b * batchSize, batchSize)

            sp, _ = self.get_s_primes(batch, *args)

            OlocBatch = self.get_O_loc_unbatched(logPsiSbatch, psi(sp))

            if Oloc is None:
                if OlocBatch.dtype == global_defs.tCpx:
                    Oloc = self._alloc_Oloc_cpx_pmapd(samples)
                else:
                    Oloc = self._alloc_Oloc_real_pmapd(samples)

            Oloc = self._insert_Oloc_batch_pmapd(Oloc, OlocBatch, b * batchSize)
        
        if remainder > 0:

            batch = self._get_config_batch_pmapd(samples, numBatches * batchSize, remainder)
            batch = global_defs.pmap_for_my_devices(expand_batch, static_broadcasted_argnums=(1,))(batch, batchSize)
            logPsiSbatch = self._get_logPsi_batch_pmapd(logPsiS, numBatches * batchSize, numSamples % batchSize)
            logPsiSbatch = global_defs.pmap_for_my_devices(expand_batch, static_broadcasted_argnums=(1,))(logPsiSbatch, batchSize)

            sp, _ = self.get_s_primes(batch, *args)

            OlocBatch = self.get_O_loc_unbatched(logPsiSbatch, psi(sp))
        
            OlocBatch = self._get_Oloc_slice_pmapd(OlocBatch, 0, remainder)

            Oloc = self._insert_Oloc_batch_pmapd(Oloc, OlocBatch, numBatches * batchSize)

        return Oloc

    @abc.abstractmethod
    def compile():
        """An implementation of this method should return a *jit-able* function that returns for a given \
            basis configuration ``s`` the connected configurations ``s'`` and the \
            corresponding matrix elements.
        """

    def get_estimator_function(self, psi, *args):
        """Get a function that computes :math:`O_{loc}(\\theta, s)`.

        Returns a function that computes :math:`O_{loc}(\\theta, s)=\sum_{s'} O_{s,s'}\\frac{\psi_\\theta(s')}{\psi_\\theta(s)}` 
        for a given configuration :math:`s` and parameters :math:`\\theta` of a given ansatz :math:`\psi_\\theta(s)`.

        Arguments:
            * ``psi``: Neural quantum state.
            * ``*args``: Further positional arguments for the operator.

        Returns:
            A function :math:`O_{loc}(\\theta, s)`.
        """

        op_fun = self.compile()
        if type(op_fun) is tuple:
            op_fun_args = op_fun[1](*args)
            op_fun = op_fun[0]
        net_fun = psi.net.apply

        def op_estimator(params, config):

            sp, matEls = op_fun(config, *op_fun_args)

            log_psi_s = net_fun(params, config)
            log_psi_sp = jax.vmap(lambda s: net_fun(params,s))(sp)

            #return jnp.dot(matEls, jnp.exp(log_psi_sp - log_psi_s))
            return jnp.sum(matEls * jnp.exp(log_psi_sp - log_psi_s))

        return op_estimator
