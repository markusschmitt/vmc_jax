import jax
from jax import jit, vmap, grad, partial
import jax.numpy as jnp
import numpy as np

# Common operators
def Id(idx=0,lDim=2):
    return {'idx': idx, 'map': jnp.array([j for j in range(lDim)],dtype=np.int32), 'matEls':jnp.array([1. for j in range(lDim)],dtype=np.float64)}

def Sx(idx):
    return {'idx': idx, 'map': jnp.array([1,0],dtype=np.int32), 'matEls':jnp.array([0.5,0.5],dtype=np.float64)}

def Sz(idx):
    return {'idx': idx, 'map': jnp.array([0,1],dtype=np.int32), 'matEls':jnp.array([0.5,0.5],dtype=np.float64)}

def Sp(idx):
    return {'idx': idx, 'map': jnp.array([1,0],dtype=np.int32), 'matEls':jnp.array([1.0,0.0],dtype=np.float64)}

def Sm(idx):
    return {'idx': idx, 'map': jnp.array([1,0],dtype=np.int32), 'matEls':jnp.array([0.0,1.0],dtype=np.float64)}

def scal_opstr(a,op):
    op[0]['matEls'] = a * op[0]['matEls']
    return op

def op_map(s,idx,sMap):
    s=jax.ops.index_update(s,jax.ops.index[idx],sMap[s[idx]])
    return s

@jit
def apply_fun(s,matEl,idx,sMap,matEls):
    matEl=matEl*matEls[idx]
    s=vmap(op_map,in_axes=(0,None,None))(s,idx,sMap)
    return s,matEl

def apply_multi(s,matEl,opIdx,opMap,opMatEls):
    for idx,mp,me in zip(opIdx,opMap,opMatEls):
        s,matEl=apply_fun(s,matEl,idx,mp,me)
    return s,matEl

@jit
def get_O_loc_fun(matEl, logPsiS, logPsiSP):
    return jnp.sum(matEl * jnp.exp(logPsiSP-logPsiS), axis=0)

class Operator:
    def __init__(self,lDim=2):
        self.ops=[]
        self.lDim=lDim
        self.compiled=False

    def add(self,opDescr):
        self.ops.append(opDescr)
        self.compiled=False

    def compile(self):
        self.idx=[]
        self.map=[]
        self.matEls=[]
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
            for k in range(self.maxOpStrLength):
                if k < len(op):
                    self.idx[o].append(op[k]['idx'])
                    self.map[o].append(op[k]['map'])
                    self.matEls[o].append(op[k]['matEls'])
                else:
                    self.idx[o].append(IdOp['idx'])
                    self.map[o].append(IdOp['map'])
                    self.matEls[o].append(IdOp['matEls'])

            o=o+1

        self.idxC = jnp.array(self.idx,dtype=np.int32)
        self.mapC = jnp.array(self.map,dtype=np.int32)
        self.matElsC = jnp.array(self.matEls,dtype=np.float64)

        self.compiled=True

    def get_s_primes(self,s):
        if not self.compiled:
            self.compile()
        self.numInStates = s.shape[0]
        self.matEl=jnp.ones((len(self.ops),self.numInStates),dtype=np.float64)
        self.sp=jnp.vstack([[s.copy()]]*len(self.ops))

        self.sp,self.matEl = vmap(apply_multi,in_axes=0)(self.sp,self.matEl,self.idxC,self.mapC,self.matElsC)

        self.nonzero=jnp.where(jnp.abs(self.matEl)>1e-6)

        return self.sp[self.nonzero], self.matEl[self.nonzero]

    def get_O_loc(self,logPsiS,logPsiSP):
        self.logPsiSP = jnp.zeros((len(self.ops),self.numInStates),dtype=np.complex64)
        self.logPsiSP = jax.ops.index_update(self.logPsiSP, jax.ops.index[self.nonzero], logPsiSP)

        #return jnp.sum(self.matEl * jnp.exp(self.logPsiSP-logPsiS), axis=0)
        return get_O_loc_fun(self.matEl, logPsiS, self.logPsiSP)

if __name__ == '__main__':
    L=4
    lDim=2
    s=jnp.zeros((2,L),dtype=np.int32)
    matEl=jnp.ones((2),dtype=np.float64)

    idx=2
    sMap=jnp.array([1,0],dtype=np.int32)
    matEls=jnp.array([0.5,0.5],dtype=np.float64)
    print(s)

    h=Operator()

    h.add(scal_opstr(-1.,(Sx(0),Sx(1))))
    h.add((Sx(2),))
    h.add((Sz(0),))
    h.add((Sm(0),))

    sp,matEl=h.get_s_primes(s)
    logPsi=jnp.ones(s.shape[0])*(0.5j)
    logPsiSP=0.3*jnp.ones(sp.shape[0])

    print(h.get_O_loc(logPsi,logPsiSP))
