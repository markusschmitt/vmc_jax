import jax
from jax import jit, vmap, grad
import jax.numpy as jnp
import numpy as np

import jVMC.global_defs as global_defs
from mpi4py import MPI
from . import Operator

import functools
import sys

opDtype = global_defs.tCpx

# Common operators


def Id(idx=0, lDim=2):
    """Returns an identity operator

    Args:

    * ``idx``: Index of the local Hilbert space.
    * ``lDim``: Dimension of local Hilbert space.

    Returns:
        Dictionary defining an identity operator

    """

    return {'idx': idx, 'map': jnp.array([j for j in range(lDim)], dtype=np.int32),
            'matEls': jnp.array([1. for j in range(lDim)], dtype=opDtype), 'diag': True}


def Sx(idx):
    """Returns a :math:`\hat\sigma^x` Pauli operator

    Args:

    * ``idx``: Index of the local Hilbert space.

    Returns:
        Dictionary defining :math:`\hat\sigma^x` Pauli operator

    """

    return {'idx': idx, 'map': jnp.array([1, 0], dtype=np.int32), 'matEls': jnp.array([1.0, 1.0], dtype=opDtype), 'diag': False}


def Sy(idx):
    """Returns a :math:`\hat\sigma^x` Pauli operator

    Args:

    * ``idx``: Index of the local Hilbert space.

    Returns:
        Dictionary defining :math:`\hat\sigma^x` Pauli operator

    """

    return {'idx': idx, 'map': jnp.array([1, 0], dtype=np.int32), 'matEls': jnp.array([1.j, -1.j], dtype=opDtype), 'diag': False}


def Sz(idx):
    """Returns a :math:`\hat\sigma^z` Pauli operator

    Args:

    * ``idx``: Index of the local Hilbert space.

    Returns:
        Dictionary defining :math:`\hat\sigma^z` Pauli operator

    """

    return {'idx': idx, 'map': jnp.array([0, 1], dtype=np.int32), 'matEls': jnp.array([-1.0, 1.0], dtype=opDtype), 'diag': True}


def Sp(idx):
    """Returns a :math:`S^+` spin-1/2 ladder operator

    Args:

    * ``idx``: Index of the local Hilbert space.

    Returns:
        Dictionary defining :math:`S^+` ladder operator

    """

    return {'idx': idx, 'map': jnp.array([1, 0], dtype=np.int32), 'matEls': jnp.array([1.0, 0.0], dtype=opDtype), 'diag': False}


def Sm(idx):
    """Returns a :math:`S^-` spin-1/2 ladder operator

    Args:

    * ``idx``: Index of the local Hilbert space.

    Returns:
        Dictionary defining :math:`S^-` ladder operator

    """

    return {'idx': idx, 'map': jnp.array([0, 0], dtype=np.int32), 'matEls': jnp.array([0.0, 1.0], dtype=opDtype), 'diag': False}


######################
# fermionic number operator
def number(idx):
    """Returns a :math:`c^\dagger c` femrioic number operator

    Args:

    * ``idx``: Index of the local Hilbert space.

    Returns:
        Dictionary defining :math:`c^\dagger c` femrioic number operator

    """

    return {
        'idx': idx,
        'map': jax.numpy.array([0,1],dtype=np.int32),
        'matEls': jax.numpy.array([0.,1.],dtype=opDtype),
        'diag': True,
        'fermionic': False
    }

######################
# fermionic creation operator
def creation(idx): 
    """Returns a :math:`c^\dagger` femrioic creation operator

    Args:

    * ``idx``: Index of the local Hilbert space.

    Returns:
        Dictionary defining :math:`c^\dagger` femrioic creation operator

    """
    
    return {
        'idx': idx,
        'map': jax.numpy.array([1,0],dtype=np.int32),
        'matEls': jax.numpy.array([1.,0.],dtype=opDtype),
        'diag': False,
        "fermionic": True
    }

######################
# fermionic annihilation operator
def annihilation(idx): 
    """Returns a :math:`c` femrioic creation operator

    Args:

    * ``idx``: Index of the local Hilbert space.

    Returns:
        Dictionary defining :math:`c` femrioic creation operator

    """ 
    
    return {
        'idx': idx,
        'map': jax.numpy.array([1,0],dtype=np.int32),
        'matEls': jax.numpy.array([0.,1.],dtype=opDtype),
        'diag': False,
        "fermionic": True
     }


import copy

@jax.jit
def _id_prefactor(*args, val=1.0, **kwargs):
    return val

def _prod_fun(f1, f2, *args, **kwargs):
    return f1(*args) * f2(*args)

def scal_opstr(a, op):
    """Add prefactor to operator string

    Arguments:
        * ``a``: Scalar prefactor or function.
        * ``op``: Operator string.

    Returns:
        Operator string rescaled by ``a``.

    """

    newOp = [copy.deepcopy(o) for o in op]
    if not callable(a):
        a = functools.partial(_id_prefactor, val=a)

    if callable(newOp[0]):
        newOp[0] = functools.partial(_prod_fun, f1=a, f2=newOp[0])
    else:
        newOp = [a] + newOp

    return tuple(newOp)

class BranchFreeOperator(Operator):
    """This class provides functionality to compute operator matrix elements

    Initializer arguments:

        * ``lDim``: Dimension of local Hilbert space.
    """

    def __init__(self, lDim=2, **kwargs):
        """Initialize ``Operator``.

        Arguments:
            * ``lDim``: Dimension of local Hilbert space.
        """
        self.ops = []
        self.lDim = lDim

        super().__init__(**kwargs)

    def add(self, opDescr):
        """Add another operator to the operator

        Arguments:
            * ``opDescr``: Operator string to be added to the operator.

        """

        self.ops.append(opDescr)
        self.compiled = False

    def compile(self):
        """Compiles a operator mapping function from the given operator strings.

        """

        self.idx = []
        self.map = []
        self.matEls = []
        self.diag = []
        self.prefactor = []
        ######## fermions ########
        self.fermionic = []
        ##########################
        self.maxOpStrLength = 0
        for op in self.ops:
            tmpLen = len(op)
            if callable(op[0]):
                tmpLen -= 1
            if tmpLen > self.maxOpStrLength:
                self.maxOpStrLength = tmpLen
        IdOp = Id(lDim=self.lDim)
        o = 0
        for op in self.ops:
            self.idx.append([])
            self.map.append([])
            self.matEls.append([])
            self.fermionic.append([])
            # check whether string contains prefactor
            k0=0
            if callable(op[0]):
                self.prefactor.append((o, jax.jit(op[0])))
                k0=1
            #else:
            #    self.prefactor.append(_id_prefactor)
            isDiagonal = True
            for k in range(k0, k0+self.maxOpStrLength):
                if k < len(op):
                    if not op[k]['diag']:
                        isDiagonal = False
                    self.idx[o].append(op[k]['idx'])
                    self.map[o].append(op[k]['map'])
                    self.matEls[o].append(op[k]['matEls'])
                    ######## fermions ########
                    fermi_check = True
                    if "fermionic" in op[k]:
                        if op[k]["fermionic"]:  
                            fermi_check = False
                            self.fermionic[o].append(1.)
                    if fermi_check:
                        self.fermionic[o].append(0.)
                    ##########################
                else:
                    self.idx[o].append(IdOp['idx'])
                    self.map[o].append(IdOp['map'])
                    self.matEls[o].append(IdOp['matEls'])
                    self.fermionic[o].append(0.)

            if isDiagonal:
                self.diag.append(o)
            o = o + 1

        self.idxC = jnp.array(self.idx, dtype=np.int32)
        self.mapC = jnp.array(self.map, dtype=np.int32)
        self.matElsC = jnp.array(self.matEls, dtype=opDtype)
        ######## fermions ########
        self.fermionicC = jnp.array(self.fermionic, dtype=np.int32)
        ##########################
        self.diag = jnp.array(self.diag, dtype=np.int32)

        def arg_fun(*args, prefactor, init):
            N = len(prefactor)
            if N<50:
                res = init
                for i,f in prefactor:
                    res[i] = f(*args)
            else:
                # parallelize this, because jit compilation for each element can be slow
                comm = MPI.COMM_WORLD
                commSize = comm.Get_size()
                rank = comm.Get_rank()
                nEls = (N + commSize - 1) // commSize
                myStart = nEls * rank
                myEnd = min(myStart+nEls, N)
                res = init[myStart:myEnd]
                for i,f in prefactor[myStart:myEnd]:
                    res[i-myStart] = f(*args)

                res = np.concatenate(comm.allgather(res), axis=0)
                
            return (jnp.array(res), )

        return functools.partial(self._get_s_primes, idxC=self.idxC, mapC=self.mapC, matElsC=self.matElsC, diag=self.diag, fermiC=self.fermionicC, prefactor=self.prefactor),\
                functools.partial(arg_fun, prefactor=self.prefactor, init=np.ones(self.idxC.shape[0], dtype=self.matElsC.dtype))

    def _get_s_primes(self, s, *args, idxC, mapC, matElsC, diag, fermiC, prefactor):

        numOps = idxC.shape[0]
        #matEl = jnp.ones(numOps, dtype=matElsC.dtype)
        matEl = args[0]
        
        sp = jnp.array([s] * numOps)

        ######## fermions ########
        mask = jnp.tril(jnp.ones((s.shape[-1],s.shape[-1])),-1).T
        ##########################

        def apply_fun(c, x):
            config, configMatEl = c
            idx, sMap, matEls, fermi = x

            configShape = config.shape
            config = config.ravel()
            ######## fermions ########
            configMatEl = configMatEl * matEls[config[idx]] * (-1.)**(fermi * (sum(mask[idx]*config)))
            ##########################
            config = config.at[idx].set(sMap[config[idx]])

            return (config.reshape(configShape), configMatEl), None

        #def apply_multi(config, configMatEl, opIdx, opMap, opMatEls, prefactor):
        def apply_multi(config, configMatEl, opIdx, opMap, opMatEls, opFermi):

            (config, configMatEl), _ = jax.lax.scan(apply_fun, (config, configMatEl), (opIdx, opMap, opMatEls, opFermi))

            #return config, prefactor*configMatEl
            return config, configMatEl

        # vmap over operators
        #sp, matEl = vmap(apply_multi, in_axes=(0, 0, 0, 0, 0, 0))(sp, matEl, idxC, mapC, matElsC, jnp.array([f(*args) for f in prefactor]))
        sp, matEl = vmap(apply_multi, in_axes=(0, 0, 0, 0, 0, 0))(sp, matEl, idxC, mapC, matElsC, fermiC)
        if len(diag) > 1:
            matEl = matEl.at[diag[0]].set(jnp.sum(matEl[diag], axis=0))
            matEl = matEl.at[diag[1:]].set(jnp.zeros((diag.shape[0] - 1,), dtype=matElsC.dtype))

        return sp, matEl

