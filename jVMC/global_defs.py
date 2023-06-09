import numpy as np

# Complex floating point
tCpx = np.complex128
# Real floating point
tReal = np.float64

from mpi4py import MPI

import jax

from functools import partial
import collections

try:
    myDevice = jax.devices()[MPI.COMM_WORLD.Get_rank() % len(jax.devices())]
except:
    myDevice = jax.devices()[0]
    print("WARNING: Could not assign devices based on MPI ranks. Assigning default device ", myDevice)

myPmapDevices = jax.devices()  # [myDevice]
myDeviceCount = len(myPmapDevices)
pmap_for_my_devices = partial(jax.pmap, devices=myPmapDevices)

pmapDevices = None
def pmap_devices_updated():
    if collections.Counter(pmapDevices) == collections.Counter(myPmapDevices):
        return False
    return True


def get_iterable(x):
    if isinstance(x, collections.abc.Iterable):
        return x
    else:
        return (x,)


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
    return myPmapDevices
