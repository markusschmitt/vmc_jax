import sys
# Find jVMC package
sys.path.append(sys.path[0]+"/..")

import jax
from jax.config import config
config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

import jax.random as random
import flax

import jVMC
import jVMC.global_defs as global_defs
import jVMC.activation_functions
import jVMC.mpi_wrapper as mpi

from flax import nn

import json

class OrthoCpxCNN(nn.Module):

    def apply(self, x, lbda=0., F=[8,], channels=[10], strides=[1], actFun=[nn.elu], bias=True, orbit=None):

        net1=jVMC.nets.CpxCNNSym.shared(F=F, channels=channels, strides=strides, actFun=actFun, bias=bias, orbit=orbit, name="net")
        net2=jVMC.nets.CpxCNNSym.shared(F=F, channels=channels, strides=strides, actFun=actFun, bias=bias, orbit=orbit, name="orthoNet")

        return jnp.log(jnp.exp(net1(x))-lbda*jnp.exp(net2(x)))


class OrthoNQS(jVMC.vqs.NQS):

    def __init__(self, psi, orthoPsi, overlap, overlapDerivative, inputShape, batchSize=100, **netArgs):

        self.psi = psi
        self.orthoPsi = orthoPsi

        self.inputShape = inputShape
        self.netArgs = netArgs
        self.set_overlap(overlap, overlapDerivative)

        self._isGenerator = False

        self.batchSize = batchSize

        self.evalJitd = global_defs.pmap_for_my_devices(self._eval, in_axes=(None, 0, None), static_broadcasted_argnums=(2,))
        self.evalJitdReal = global_defs.pmap_for_my_devices(self._eval_real, in_axes=(None, 0, None), static_broadcasted_argnums=(2,))


    def set_overlap(self, overlap, overlapDerivative):
        
        self.overlap = overlap
        self.overlapDerivative = overlapDerivative

        net = OrthoCpxCNN.partial(lbda=overlap, **self.netArgs)
        _, params = net.init_by_shape(jax.random.PRNGKey(4321),[self.inputShape])

        params['net'] = self.psi.get_sampler_net().params
        params['orthoNet'] = self.orthoPsi.get_sampler_net().params

        self.orthoNet = flax.nn.Model(net,params)

    def __call__(self, s):
        
        return self.evalJitd(self.orthoNet,s,self.batchSize)


    def real_coefficients(self, s):

        return self.evalJitdReal(self.orthoNet,s,self.batchSize)


    def gradients(self, s):

        netGradients = self.psi.gradients(s)

        psiCoeffs = self.psi(s)
        orthoPsiCoeffs = self.orthoPsi(s)

        return global_defs.pmap_for_my_devices(jax.vmap(lambda x,y,z,lbda,dLbda: x/(1.-lbda*jnp.exp(z-y)) - dLbda / (jnp.exp(y-z)-lbda), in_axes=(0,0,0,None,None)), in_axes=(0,0,0,None,None))(netGradients,psiCoeffs,orthoPsiCoeffs,self.overlap,self.overlapDerivative)

    def get_sampler_net(self):

        return self.orthoNet


    def _eval_real(self, net, s, batchSize):
        def create_batches(configs, b):

            append=b*((configs.shape[0]+b-1)//b)-configs.shape[0]
            pads=[(0,append),] + [(0,0)]*(len(configs.shape)-1)

            return jnp.pad(configs, pads).reshape((-1,b)+configs.shape[1:])

        sb = create_batches(s, batchSize)

        def scan_fun(c,x):
            return c, jax.vmap(lambda m,n: jnp.real(m(n)), in_axes=(None, 0))(net,x)

        res = jax.lax.scan(scan_fun, None, jnp.array(sb))[1].reshape((-1,))

        return res[:s.shape[0]]


    def _eval(self, net, s, batchSize):

        def create_batches(configs, b):

            append=b*((configs.shape[0]+b-1)//b)-configs.shape[0]
            pads=[(0,append),] + [(0,0)]*(len(configs.shape)-1)

            return jnp.pad(configs, pads).reshape((-1,b)+configs.shape[1:])

        sb = create_batches(s, batchSize)

        def scan_fun(c,x):
            return c, jax.vmap(lambda m,n: m(n), in_axes=(None, 0))(net,x)

        res = jax.lax.scan(scan_fun, None, jnp.array(sb))[1].reshape((-1,))

        return res[:s.shape[0]]


    def get_parameters(self):

        return self.psi.get_parameters()


    def set_parameters(self, p):

        self.psi.set_parameters(p)


def get_point_orbit(L):

    trafos = []

    idx = np.arange(L*L).reshape((L,L))

    for _ in range(2):
        for _ in range(4):
            idx = np.array(list(zip(*idx[::-1]))) # rotation
            trafos.append(idx)
        idx = np.transpose(idx) # reflection

    orbit = []
    
    idx = np.arange(L*L)

    for t in trafos:

        o = np.zeros((L*L,L*L), dtype=np.int32)

        o[idx,t.ravel()] = 1

        orbit.append(o)

    orbit = jnp.array(orbit)

    return orbit


def get_lambda(vsLogPsi, gsLogPsi, vsGrads):

    data = global_defs.pmap_for_my_devices(lambda x,y: jnp.exp(x-y))(vsLogPsi,gsLogPsi)
    lbda = mpi.global_mean(data)
    data = global_defs.pmap_for_my_devices(lambda x,y,z: jnp.multiply(jnp.exp(x-y)[:,None],z))(vsLogPsi,gsLogPsi,vsGrads)
    dLbda = mpi.global_mean(data)

    return lbda, dLbda


inp = None
if len(sys.argv) > 1:
    # if an input file is given
    with open(sys.argv[1],'r') as f:
        inp = json.load(f)
else:

   if mpi.rank == 0:
        print("Error: No input file given.")
        exit() 

global_defs.set_pmap_devices(jax.devices()[mpi.rank % jax.device_count()])
print(" -> Rank %d working with device %s" % (mpi.rank, global_defs.devices()), flush=True)

wdir=inp["general"]["working_directory"]

# Initialize output manager
outp = jVMC.util.OutputManager(wdir+inp["general"]["data_output"], append=inp["general"]["append_data"])

L = inp["system"]["L"]
N=L**2
g = -inp["system"]["g"]
inputShape=(L,L)

orbit = get_point_orbit(L)

## 1D orbit
#o1 = np.identity(L,dtype=np.int32)
#o2 = np.zeros(L*L,dtype=np.int32).reshape((L,L))
#idx=np.arange(L)
#o2[idx,idx[::-1]] = 1
#orbit = jnp.array([o1,o2])

numChains=inp["sampler"]["numChains"]
numSamples=inp["sampler"]["numSamples"]

###
#   GROUND STATE
###
outp.set_group("ground_state")

netArgs = {
            'F' : inp["network"]["F"],
            'strides' : [1,1],
            'channels' : inp["network"]["channels"],
            'bias' : False,
            'actFun' : [jVMC.activation_functions.poly6,jVMC.activation_functions.poly5],
            'orbit' : orbit
          }

# Initialize net
net = jVMC.nets.CpxCNNSym.partial(**netArgs)
_, params = net.init_by_shape(jax.random.PRNGKey(9876),[inputShape])
gsModel = flax.nn.Model(net,params)

gsPsi = jVMC.vqs.NQS(gsModel) # Variational wave function

# Set up hamiltonian
hamiltonian = jVMC.operator.Operator()
#for l in range(L):
#    hamiltonian.add( jVMC.operator.scal_opstr( -1., ( jVMC.operator.Sz(l), jVMC.operator.Sz((l+1)%L) ) ) )
#    hamiltonian.add( jVMC.operator.scal_opstr( g, ( jVMC.operator.Sx(l), ) ) )
for i in range(L):
    for j in range(L):
        hamiltonian.add( jVMC.operator.scal_opstr( -1., ( jVMC.operator.Sz(i*L+j), jVMC.operator.Sz(((i+1)%L)*L+j) ) ) )
        hamiltonian.add( jVMC.operator.scal_opstr( -1., ( jVMC.operator.Sz(i*L+j), jVMC.operator.Sz(i*L+(j+1)%L) ) ) )
        hamiltonian.add( jVMC.operator.scal_opstr( g, ( jVMC.operator.Sx(i*L+j), ) ) )

# Set up sampler
sampler = jVMC.sampler.MCMCSampler( random.PRNGKey(inp["sampler"]["seed"]), jVMC.sampler.propose_spin_flip_Z2, inputShape,
                                    numChains=inp["sampler"]["numChains"],
                                    numSamples=inp["sampler"]["numSamples"],
                                    thermalizationSweeps=inp["sampler"]["num_thermalization_sweeps"],
                                    sweepSteps=L*L )

# Set up TDVP
delta=inp["search"]["init_regularizer"]
tdvpEquation = jVMC.tdvp.TDVP(sampler, rhsPrefactor=1.,
                              svdTol=1e-8, diagonalShift=delta, makeReal='real')

stepper = jVMC.stepper.Euler(timeStep=1e-2) # ODE integrator

for n in range(inp["search"]["num_steps"]):
    
    #gsSample, gsLogPsi, _ = sampler.sample(gsPsi)
    dp, _ = stepper.step(0, tdvpEquation, gsPsi.get_parameters(), hamiltonian=hamiltonian, psi=gsPsi, numSamples=None)
    gsPsi.set_parameters(dp)
    
    delta=0.95*delta
    tdvpEquation.set_diagonal_shift(delta)

    energy = jax.numpy.real(tdvpEquation.ElocMean0)/N
    energyVariance = tdvpEquation.ElocVar0/N**2
    outp.print("%d\t%f\t%f" % (n, energy, energyVariance))
    # Write network parameters
    outp.write_network_checkpoint(n, gsPsi.get_parameters())
    # Write observables
    outp.write_observables(n, energy={"mean":energy, "variance":energyVariance})

    # Write simulation meta data
    outp.write_metadata(n,MC_acceptance_ratio=sampler.acceptance_ratio())

    if energyVariance<inp["search"]["convergence_variance"]:
        break

np.savetxt(wdir+"gs_parameters.txt", np.array(gsPsi.get_parameters()))
#gsPsi.set_parameters(jnp.array(np.loadtxt("tmp.txt"), dtype=global_defs.tReal))

###
#   EXCITED STATE
###
outp.set_group("excited_state")

net1 = jVMC.nets.CpxCNNSym.partial(**netArgs)
_, params = net1.init_by_shape(jax.random.PRNGKey(2143),[inputShape])
vsModel = flax.nn.Model(net1,params)
psi = jVMC.vqs.NQS(vsModel) # Variational wave function

gsSample, gsLogPsi, _ = sampler.sample(gsPsi)

# get coefficients for projection
vsLogPsi = psi(gsSample)
vsGrads = psi.gradients(gsSample)
lbda, dLbda = get_lambda(vsLogPsi, gsLogPsi, vsGrads)

psiOrtho = OrthoNQS(psi, gsPsi, lbda, dLbda, inputShape, batchSize=numChains, **netArgs) # Variational wave function

delta=2*inp["search"]["init_regularizer"]
for n in range(inp["search"]["num_steps"]):

    psi.set_parameters(psiOrtho.get_parameters())

    # get coefficients for projection
    vsLogPsi = psi(gsSample)
    vsGrads = psi.gradients(gsSample)
    lbda, dLbda = get_lambda(vsLogPsi, gsLogPsi, vsGrads)
    psiOrtho.set_overlap(lbda,dLbda)

    dp, _ = stepper.step(0, tdvpEquation, psiOrtho.get_parameters(), hamiltonian=hamiltonian, psi=psiOrtho, numSamples=None)
    psiOrtho.set_parameters(dp)
    
    energy = jax.numpy.real(tdvpEquation.ElocMean0)/N
    energyVariance = tdvpEquation.ElocVar0/N**2
    outp.print("%d\t%f\t%f" % (n, energy, energyVariance))
    # Write network parameters
    outp.write_network_checkpoint(n, psiOrtho.get_parameters())
    # Write observables
    outp.write_observables(n, energy={"mean":energy, "variance":energyVariance})
    
    # Write simulation meta data
    outp.write_metadata(n,MC_acceptance_ratio=sampler.acceptance_ratio())

    if energyVariance<inp["search"]["convergence_variance"]:
        break
        
    delta=0.95*delta
    tdvpEquation.set_diagonal_shift(delta)

    if (n+1)%10 == 0:
        # get new sample from GS
        gsSample, gsLogPsi, _ = sampler.sample(gsPsi)

np.savetxt(wdir+"es_parameters.txt", np.array(psiOrtho.get_parameters()))


#import numpy as np
#res = np.array(res)
#import matplotlib.pyplot as plt
#plt.plot(res[:,0], res[:,1]+1.1272225, '-')
#plt.legend()
#plt.xlabel('iteration')
#plt.ylabel(r'$(E-E_0)/L$')
#plt.tight_layout()
#plt.savefig('gs_search.pdf')
