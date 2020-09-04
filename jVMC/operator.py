import jax
from jax import jit, vmap, grad, partial
import jax.numpy as jnp
import numpy as np

import sys
# Find jVMC package
sys.path.append(sys.path[0]+"/..")
import jVMC.global_defs as global_defs

# Common operators
def Id(idx=0,lDim=2):
    return {'idx': idx, 'map': jnp.array([j for j in range(lDim)],dtype=np.int32),
            'matEls':jnp.array([1. for j in range(lDim)],dtype=global_defs.tReal), 'diag': True}

def Sx(idx):
    return {'idx': idx, 'map': jnp.array([1,0],dtype=np.int32), 'matEls':jnp.array([1.0,1.0],dtype=global_defs.tReal), 'diag': False}

def Sz(idx):
    return {'idx': idx, 'map': jnp.array([0,1],dtype=np.int32), 'matEls':jnp.array([-1.0,1.0],dtype=global_defs.tReal), 'diag': True}

def Sp(idx):
    return {'idx': idx, 'map': jnp.array([1,0],dtype=np.int32), 'matEls':jnp.array([1.0,0.0],dtype=global_defs.tReal), 'diag': False}

def Sm(idx):
    return {'idx': idx, 'map': jnp.array([1,0],dtype=np.int32), 'matEls':jnp.array([0.0,1.0],dtype=global_defs.tReal), 'diag': False}

def scal_opstr(a,op):
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

    def __init__(self,lDim=2):
        self.ops=[]
        self.lDim=lDim
        self.compiled=False

        # pmap'd member functions
        self._get_s_primes_pmapd = jax.pmap(self._get_s_primes, static_broadcasted_argnums=(1,2,3,4))
        self._find_nonzero_pmapd = jax.pmap(vmap(self._find_nonzero, in_axes=1))
        self._set_zero_to_zero_pmapd = jax.pmap(jax.vmap(self.set_zero_to_zero, in_axes=(1,0,0), out_axes=1), in_axes=(0,0,0))
        self._array_idx_pmapd = jax.pmap(jax.vmap(lambda data, idx: data[idx], in_axes=(1,0), out_axes=1), in_axes=(0,0))
        #self._get_O_loc_pmapd = jax.pmap(jax.vmap(self._get_O_loc, in_axes=(1,0,1), out_axes=1))
        self._get_O_loc_pmapd = jax.pmap(self._get_O_loc)
        self._flatten_pmapd = jax.pmap(lambda x: x.reshape(-1,*x.shape[2:]))

    def add(self,opDescr):
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

        if not self.compiled:
            self.compile()

        # Compute matrix elements
        self.sp, self.matEl = self._get_s_primes_pmapd(s, self.idxC, self.mapC, self.matElsC, self.diag)

        # Get only non-zero contributions
        idx, self.numNonzero = self._find_nonzero_pmapd(self.matEl)
        #self.matEl = self._array_idx_pmapd(self.matEl, idx[:,:,:jnp.max(self.numNonzero)])
        self.matEl = self._set_zero_to_zero_pmapd(self.matEl, idx[:,:,:jnp.max(self.numNonzero)], self.numNonzero)
        self.sp = self._array_idx_pmapd(self.sp, idx[:,:,:jnp.max(self.numNonzero)])
        
        return self._flatten_pmapd(self.sp), self.matEl


    def _get_O_loc(self, matEl, logPsiS, logPsiSP):

        return jax.vmap(lambda x,y,z: jnp.sum(x * jnp.exp(z-y), axis=0), in_axes=(1,0,1), out_axes=1)(matEl, logPsiS, logPsiSP.reshape(matEl.shape))
        #return jnp.sum(matEl * jnp.exp(logPsiSP-logPsiS), axis=0)


    def get_O_loc(self,logPsiS,logPsiSP):
      
        return self._get_O_loc_pmapd(self.matEl, logPsiS, logPsiSP)


if __name__ == '__main__':
    L=4
    lDim=2
    s=jnp.array([jnp.zeros((2,L),dtype=np.int32)] * jax.local_device_count())

    h=Operator()

    h.add(scal_opstr(-1.,(Sx(0),Sx(1))))
    h.add((Sx(2),))
    h.add((Sz(0),))
    h.add((Sz(1),))
    h.add((Sm(0),))

    sp,matEl=h.get_s_primes(s)

    logPsi=jax.pmap(lambda s: jnp.ones(s.shape[0])*(0.5j))(s)
    logPsiSP=jax.pmap(lambda sp: 0.3*jnp.ones((sp.shape[0], sp.shape[1])))(sp)

    print(h.get_O_loc(logPsi,logPsiSP))
