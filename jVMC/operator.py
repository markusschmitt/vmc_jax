"""Implementation of quantum operators.

The many-body Hilbert space is a product of local Hilbert spaces, :math:`\mathscr H=\\bigotimes_l \mathscr h_l`.
Operators generally take the form :math:`\hat O=\sum_k \hat O_k`, where :math:`\hat O_k=\prod_l \hat o_l^k` are operator strings 
made up of elementary operators :math:`\hat o_l` acting on the factors :math:`\mathscr h_l`.

In variational Monte Carlo a key quantity is

\t:math:`O_{loc}(s) = \sum_{s'}O_{s,s'}\psi(s')/\psi(s)`

where :math:`s,s'` denote computational basis states.

This module provides functionality to assemble general operators from elementary operators, to compute matrix elements :math:`O_{s,s'}`,
and finally to compute :math:`O_{loc}(s)`.

Elementary operators
^^^^^^^^^^^^^^^^^^^^

At the core the operator class works with a general description of elementary operators \
:math:`\hat o_l`. Elementary operators are defined by dictionaries that hold four items:

* ``'idx'``: Index of the corresponding local Hilbert space.
* ``'map'``: An array (jax.numpy.array) defining the mapping between local basis indices. \
The array entry :math:`j` holds the image of the basis index :math:`j` under the map.
* ``'matEls'``: An array (jax.numpy.array) defining the corresponding matrix elements.
* ``'diag'``: Boolean stating whether this operator is a diagonal operator \
(exploiting this information enhances efficiency).

For concreteness, consider the Pauli operator

\t:math:`\hat\sigma^x=\\begin{pmatrix}0&1\\\\1&0\end{pmatrix}`

acting on lattice site :math:`l=1`. The corresponding dictionary is::

    Sx = {
            'idx': 1,
            'map': jax.numpy.array([1,0],dtype=np.int32),
            'matEl': jax.numpy.array([1.,1.],dtype=jVMC.global_defs.tReal),
            'diag': False
         }

A number of frequently used operators is pre-defined in this module, see below.

Operator strings
^^^^^^^^^^^^^^^^

Operator strings are treated as tuples of elementary operators. For example, using the \
pre-defined Pauli operators ``Sx`` and ``Sz``, the operator string :math:`\hat\sigma_1^x\hat\sigma_2^z` \
is obtained as::

    X1Z2 = ( Sx(1), Sz(2) )

Prefactors can be added to operator strings using the ``scal_opstr()`` function. For \
example, to obtain :math:`\\frac{1}{2}\hat\sigma_1^x\hat\sigma_2^z`::

    X1Z2_with_prefactor = scal_opstr(0.5, X1Z2)

Assembling operators
^^^^^^^^^^^^^^^^^^^^

Finally, arbitrary operators can be assembled from operator strings. Consider, e.g., the \
Hamiltonian of the spin-1/2 quantum Ising chain of length L,

\t:math:`\hat H=-\sum_{l=0}^{L-2}\hat\sigma_l^z\hat\sigma_{l+1}^z-g\sum_{l=0}^{L-1}\hat\sigma_l^x`

Again, using the pre-defined Pauli operators ``Sx`` and ``Sz``, an ``Operator`` object \
for this Hamiltonian can be obtained as::

    hamiltonian = Operator()
    for l in range(L-1):
        hamiltonian.add( scal_opstr( -1., ( Sz(l), Sz(l+1) ) ) )
        hamiltonian.add( scal_opstr( -g, ( Sx(l), ) ) )
    hamiltonian.add( scal_opstr( -g, ( Sx(L-1), ) ) )

Detailed documentation
^^^^^^^^^^^^^^^^^^^^^^
"""

import jax
from jax import jit, vmap, grad, partial
import jax.numpy as jnp
import numpy as np

import sys
# Find jVMC package
sys.path.append(sys.path[0]+"/..")
import jVMC.global_defs as global_defs

import jVMC.global_defs as global_defs

# Common operators
def Id(idx=0,lDim=2):
    """Returns an identity operator

    Args:

    * ``idx``: Index of the local Hilbert space.
    * ``lDim``: Dimension of local Hilbert space.

    Returns:
        Dictionary defining an identity operator

    """

    return {'idx': idx, 'map': jnp.array([j for j in range(lDim)],dtype=np.int32),
            'matEls':jnp.array([1. for j in range(lDim)],dtype=global_defs.tReal), 'diag': True}


def Sx(idx):
    """Returns a :math:`\hat\sigma^x` Pauli operator

    Args:

    * ``idx``: Index of the local Hilbert space.

    Returns:
        Dictionary defining :math:`\hat\sigma^x` Pauli operator

    """

    return {'idx': idx, 'map': jnp.array([1,0],dtype=np.int32), 'matEls':jnp.array([1.0,1.0],dtype=global_defs.tReal), 'diag': False}


def Sz(idx):
    """Returns a :math:`\hat\sigma^z` Pauli operator

    Args:

    * ``idx``: Index of the local Hilbert space.

    Returns:
        Dictionary defining :math:`\hat\sigma^z` Pauli operator

    """

    return {'idx': idx, 'map': jnp.array([0,1],dtype=np.int32), 'matEls':jnp.array([-1.0,1.0],dtype=global_defs.tReal), 'diag': True}


def Sp(idx):
    """Returns a :math:`S^+` spin-1/2 ladder operator

    Args:

    * ``idx``: Index of the local Hilbert space.

    Returns:
        Dictionary defining :math:`S^+` ladder operator

    """

    return {'idx': idx, 'map': jnp.array([1,0],dtype=np.int32), 'matEls':jnp.array([1.0,0.0],dtype=global_defs.tReal), 'diag': False}


def Sm(idx):
    """Returns a :math:`S^-` spin-1/2 ladder operator

    Args:

    * ``idx``: Index of the local Hilbert space.

    Returns:
        Dictionary defining :math:`S^-` ladder operator

    """

    return {'idx': idx, 'map': jnp.array([1,0],dtype=np.int32), 'matEls':jnp.array([0.0,1.0],dtype=global_defs.tReal), 'diag': False}


def scal_opstr(a,op):
    """Add prefactor to operator string

    Args:

    * ``a``: Scalar prefactor.
    * ``op``: Operator string.

    Returns:
        Rescaled operator string. Effectively, the matrix elements of the first element of \
        the operator string are multiplied by ``a``.

    """

    op[0]['matEls'] = a * op[0]['matEls']
    return op


def apply_fun(s,matEl,idx,sMap,matEls):
    matEl=matEl*matEls[s[idx]]
    s=jax.ops.index_update(s,jax.ops.index[idx],sMap[s[idx]])
    return s,matEl

@jit
def apply_multi(s,matEl,opIdx,opMap,opMatEls,diag):
    for idx,mp,me in zip(opIdx,opMap,opMatEls):
        s,matEl=vmap(apply_fun, in_axes=(0,0,None,None,None))(s,matEl,idx,mp,me)
    
    return s,matEl


class Operator:
    """This class provides functionality to compute operator matrix elements
    """

    def __init__(self,lDim=2):
        """Initialize ``Operator``.

        Args:

        * ``lDim``: Dimension of local Hilbert space.
        """
        self.ops=[]
        self.lDim=lDim
        self.compiled=False

        # jit'd member functions
        if global_defs.usePmap:
            self._get_s_primes_pmapd = global_defs.pmap_for_my_devices(self._get_s_primes, static_broadcasted_argnums=(1,2,3,4))
            self._find_nonzero_pmapd = global_defs.pmap_for_my_devices(vmap(self._find_nonzero, in_axes=1))
            self._set_zero_to_zero_pmapd = global_defs.pmap_for_my_devices(jax.vmap(self.set_zero_to_zero, in_axes=(1,0,0), out_axes=1), in_axes=(0,0,0))
            self._array_idx_pmapd = global_defs.pmap_for_my_devices(jax.vmap(lambda data, idx: data[idx], in_axes=(1,0), out_axes=1), in_axes=(0,0))
            self._get_O_loc_pmapd = global_defs.pmap_for_my_devices(self._get_O_loc)
            self._flatten_pmapd = global_defs.pmap_for_my_devices(lambda x: x.reshape(-1,*x.shape[2:]))
        else:
            self._get_s_primes_pmapd = global_defs.jit_for_my_device(self._get_s_primes, static_argnums=(1,2,3,4))
            self._find_nonzero_pmapd = global_defs.jit_for_my_device(vmap(self._find_nonzero, in_axes=1))
            self._set_zero_to_zero_pmapd = global_defs.jit_for_my_device(jax.vmap(self.set_zero_to_zero, in_axes=(1,0,0), out_axes=1))
            self._array_idx_pmapd = global_defs.jit_for_my_device(jax.vmap(lambda data, idx: data[idx], in_axes=(1,0), out_axes=1))
            self._get_O_loc_pmapd = global_defs.jit_for_my_device(self._get_O_loc)
            self._flatten_pmapd = global_defs.jit_for_my_device(lambda x: x.reshape(-1,*x.shape[2:]))

    def add(self,opDescr):
        """Add another operator to the operator

        Args:

        * ``opDescr``: Operator string to be added to the operator.

        """

        self.ops.append(opDescr)
        self.compiled=False


    def compile(self):
        self.idx=[]
        self.map=[]
        self.matEls=[]
        self.diag=[]
        self.maxOpStrLength=0
        for op in self.ops:
            if len(op) > self.maxOpStrLength:
                self.maxOpStrLength = len(op)
        IdOp=Id(lDim=self.lDim)
        o=0
        for op in self.ops:
            self.idx.append([])
            self.map.append([])
            self.matEls.append([])
            isDiagonal = True
            for k in range(self.maxOpStrLength):
                if k < len(op):
                    if not op[k]['diag']:
                        isDiagonal=False
                    self.idx[o].append(op[k]['idx'])
                    self.map[o].append(op[k]['map'])
                    self.matEls[o].append(op[k]['matEls'])
                else:
                    self.idx[o].append(IdOp['idx'])
                    self.map[o].append(IdOp['map'])
                    self.matEls[o].append(IdOp['matEls'])

            if isDiagonal:
                self.diag.append(o)
            o=o+1

        #if len(self.diag) == 0:
        #    self.diag.append(0) # append dummy diagonal entry

        self.idxC = jnp.array(self.idx,dtype=np.int32)
        self.mapC = jnp.array(self.map,dtype=np.int32)
        self.matElsC = jnp.array(self.matEls,dtype=global_defs.tReal)
        self.diag = jnp.array(self.diag, dtype=np.int32)

        self.compiled=True


    def _get_s_primes(self, s, idxC, mapC, matElsC, diag):

        numInStates = s.shape[0]
        numOps = idxC.shape[0]
        matEl=jnp.ones((numOps,numInStates),dtype=global_defs.tReal)
        sp=jnp.vstack([[s]]*numOps)

        # vmap over operators
        sp, matEl = vmap(apply_multi,in_axes=(0,0,0,0,0,None))(sp,matEl,idxC,mapC,matElsC,diag)

        if len(diag) > 1:
            matEl = jax.ops.index_update(matEl, jax.ops.index[diag[0],:], jnp.sum(matEl[diag], axis=0))
            matEl = jax.ops.index_update(matEl, jax.ops.index[diag[1:],:],
                                         jnp.zeros((diag.shape[0]-1,matEl.shape[1]), dtype=global_defs.tReal))

        return sp, matEl


    def _find_nonzero(self, m):

        choice = jnp.zeros(m.shape, dtype=np.int64) + m.shape[0] - 1

        def scan_fun(c, x):
            b = jnp.abs(x[0])>1e-6
            out = jax.lax.cond(b, lambda z: z[0], lambda z: z[1], (c[1], x[1]))
            newcarry = jax.lax.cond(b, lambda z:  (z[0]+1,z[1]+1), lambda z: (z[0],z[1]+1), c)
            return newcarry, out

        carry, choice = jax.lax.scan(scan_fun, (0,0), (m,choice))

        return jnp.sort(choice), carry[0]


    def set_zero_to_zero(self, m, idx, numNonzero):

        def scan_fun(c, x):
            out = jax.lax.cond(c[1]<c[0], lambda z: z[0], lambda z: z[1], (x, 0.))
            newCarry = (c[0], c[1]+1)
            return newCarry, out

        _, m = jax.lax.scan(scan_fun, (numNonzero, 0), m[idx])

        return m


    def get_s_primes(self,s):
        """Compute matrix elements

        For a list of computational basis states :math:`s` this member function computes the corresponding \
        matrix elements :math:`O_{s,s'}=\langle s|\hat O|s'\\rangle` and their respective configurations \
        :math:`s'`.

        Args:

        * ``s``: Array of computational basis states.

        Returns:
            An array holding `all` configurations :math:`s'` and the corresponding matrix elements :math:`O_{s,s'}`.

        """

        if not self.compiled:
            self.compile()

        # Compute matrix elements
        self.sp, self.matEl = self._get_s_primes_pmapd(s, self.idxC, self.mapC, self.matElsC, self.diag)

        # Get only non-zero contributions
        idx, self.numNonzero = self._find_nonzero_pmapd(self.matEl)
        #self.matEl = self._array_idx_pmapd(self.matEl, idx[:,:,:jnp.max(self.numNonzero)])
        #self.matEl = self._set_zero_to_zero_pmapd(self.matEl, idx[:,:,:jnp.max(self.numNonzero)], self.numNonzero)
        self.matEl = self._set_zero_to_zero_pmapd(self.matEl, idx[...,:jnp.max(self.numNonzero)], self.numNonzero)
        #self.sp = self._array_idx_pmapd(self.sp, idx[:,:,:jnp.max(self.numNonzero)])
        self.sp = self._array_idx_pmapd(self.sp, idx[...,:jnp.max(self.numNonzero)])
        
        return self._flatten_pmapd(self.sp), self.matEl


    def _get_O_loc(self, matEl, logPsiS, logPsiSP):

        return jax.vmap(lambda x,y,z: jnp.sum(x * jnp.exp(z-y), axis=0), in_axes=(1,0,1), out_axes=1)(matEl, logPsiS, logPsiSP.reshape(matEl.shape))


    def get_O_loc(self,logPsiS,logPsiSP):
        """Compute :math:`O_{loc}(s)`.

        This member function assumes that ``get_s_primes(s)`` has been called before, as \
        internally stored matrix elements :math:`O_{s,s'}` are used.

        Computes :math:`O_{loc}(s)=\sum_{s'} O_{s,s'}\\frac{\psi(s')}{\psi(s)}`, given the \
        logarithmic wave function amplitudes of the involved configurations :math:`\\ln(\psi(s))` \
        and :math:`\\ln\psi(s')`

        Args:

        * ``logPsiS``: Logarithmic amplitudes :math:`\\ln(\psi(s))`
        * ``logPsiSP``: Logarithmic amplitudes :math:`\\ln(\psi(s'))`

        Returns:
            :math:`O_{loc}(s)` for each configuration :math:`s`.
        """
      
        return self._get_O_loc_pmapd(self.matEl, logPsiS, logPsiSP)


if __name__ == '__main__':
    L=4
    lDim=2
    deviceCount = 1
    shape = (2,L)
    if global_defs.usePmap:
        deviceCount = jax.local_device_count()
        shape = (deviceCount,) + shape
    s=jnp.zeros(shape,dtype=np.int32)

    h=Operator()

    h.add(scal_opstr(-1.,(Sx(0),Sx(1))))
    h.add((Sx(2),))
    h.add((Sz(0),))
    h.add((Sz(1),))
    h.add((Sm(0),))

    sp,matEl=h.get_s_primes(s)

    logPsi=jnp.ones(s.shape[:-1])*(0.5j)
    logPsiSP=0.3*jnp.ones(sp.shape[:-1])
    print(logPsi.shape)
    print(logPsiSP.shape)

    print(h.get_O_loc(logPsi,logPsiSP))
