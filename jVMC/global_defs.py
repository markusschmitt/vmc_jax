import numpy as np

# Complex floating point
tCpx=np.complex128
# Real floating point
tReal=np.float64

from mpi4py import MPI

import jax

from functools import partial

usePmap = True

myDevice = jax.devices()[MPI.COMM_WORLD.Get_rank() % len(jax.devices())]
myPmapDevices = jax.devices()#[myDevice]
myDeviceCount = len(myPmapDevices)
jit_for_my_device = partial(jax.jit, device=myDevice)
pmap_for_my_devices = partial(jax.pmap, devices=myPmapDevices)

import collections
def get_iterable(x):
    if isinstance(x, collections.Iterable):
        return x
    else:
        return (x,)

def set_use_pmap(b=True):
    global usePmap
    usePmap = b

def set_pmap_devices(devices):
    devices = list(get_iterable(devices))
    global myPmapDevices
    global myDeviceCount
    global pmap_for_my_devices
    myPmapDevices = devices
    myDeviceCount = len(myPmapDevices)
    pmap_for_my_devices = partial(jax.pmap, devices=myPmapDevices)
    myDevice = myPmapDevices[0]

def device_count():
    return len(myPmapDevices)

def devices():
    if usePmap:
        return myPmapDevices
    return myDevice
