import sys
# Find jVMC package
sys.path.append(sys.path[0]+"/../..")

import jVMC
import jVMC.nets as nets
from jVMC.vqs import NQS
import numpy as np
import jax
import jax.numpy as jnp
import flax
import flax.nn as nn

import time
    
from functools import partial

@jax.jit
def eval_net(model, states):
    return model(states)

def time_net_eval(states, timingReps, get_net):
    L=states.shape[1]
    
    model = get_net()
    psi = NQS(model)

    t0 = time.perf_counter()
    #jax.vmap(eval_net, in_axes=(None,0))(model,states)
    #eval_net(model,states)
    psi(states).block_until_ready()
    t1 = time.perf_counter()
    print("      Time elapsed (incl. jit): %f seconds" % (t1-t0))

    t=0
    for i in range(timingReps):
        t0 = time.perf_counter()
        #jax.vmap(eval_net, in_axes=(None,0))(model,states).block_until_ready()
        #eval_net(model,states).block_until_ready()
        psi(states).block_until_ready()
        t1 = time.perf_counter()
        t += t1-t0
    print("      Avg. time elapsed (jit'd, %d repetitions): %f seconds" % (timingReps, t/timingReps))

stateNums=[100,1000,10000]

timingReps = 10

print("* RBM")
print("  - evaluation")

def get_rbm(L):

    rbm = nets.RBM.partial(numHidden=32)
    _,params = rbm.init_by_shape(jax.random.PRNGKey(0),[(L,)])
    return nn.Model(rbm,params)

Ls=[64,256]
for numStates in stateNums:
    for L in Ls:
        print("    > number of states: %d, L = %d" % (numStates, L))
        states = jnp.array(jax.random.bernoulli(jax.random.PRNGKey(0), shape=(numStates,L)), dtype=np.int32)
        time_net_eval(states,timingReps, partial(get_rbm, L=L))

print("* CNN 1D")
print("  - evaluation")

def get_1d_cnn(L):

    cnn1D = nets.CNN.partial(F=(8,), channels=[6,5,4])
    _,params = cnn1D.init_by_shape(jax.random.PRNGKey(0),[(L,)])
    return nn.Model(cnn1D,params)

Ls=[64,256]
for numStates in stateNums:
    for L in Ls:
        print("    > number of states: %d, L = %d" % (numStates, L))
        states = jnp.array(jax.random.bernoulli(jax.random.PRNGKey(0), shape=(numStates,L)), dtype=np.int32)
        time_net_eval(states,timingReps, partial(get_1d_cnn, L=L))


print("* CNN 2D")
print("  - evaluation")

def get_2d_cnn(L):

    cnn2D = nets.CNN.partial(F=(6,6), channels=[6,5,4], strides=[1,1])
    _,params = cnn2D.init_by_shape(jax.random.PRNGKey(0),[(L,L)])
    return nn.Model(cnn2D,params)

Ls=[10,20]
for numStates in stateNums:
    for L in Ls:
        print("    > number of states: %d, L = %d" % (numStates, L))
        states = jnp.array(jax.random.bernoulli(jax.random.PRNGKey(0), shape=(numStates,L,L)), dtype=np.int32)
        time_net_eval(states,timingReps, partial(get_2d_cnn, L=L))



print("* RNN 1D")
print("  - evaluation")

def get_1d_rnn(L):

    rnn1D = nets.RNN.partial(L=L,units=[20])
    _,params = rnn1D.init_by_shape(jax.random.PRNGKey(0),[(L,)])
    return nn.Model(rnn1D,params)

Ls=[64,256]
for numStates in stateNums:
    for L in Ls:
        print("    > number of states: %d, L = %d" % (numStates, L))
        states = jnp.array(jax.random.bernoulli(jax.random.PRNGKey(0), shape=(numStates,L)), dtype=np.int32)
        time_net_eval(states,timingReps, partial(get_1d_rnn, L=L))
