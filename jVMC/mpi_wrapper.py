from mpi4py import MPI
import jax
import jax.numpy as jnp
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
commSize = comm.Get_size()

globNumSamples=0
myNumSamples=0

def distribute_sampling(numSamples):

    global globNumSamples
    globNumSamples = numSamples

    mySamples = numSamples // commSize

    if rank < numSamples % commSize:
        mySamples+=1

    return mySamples


def first_sample_id():

    global globNumSamples

    mySamples = globNumSamples // commSize

    firstSampleId = rank * mySamples

    if rank < globNumSamples % commSize:
        firstSampleId += rank
    else:
        firstSampleId += globNumSamples % commSize

    return firstSampleId


def global_mean(data):

    # Compute sum locally
    localSum = np.array( jnp.sum(data, axis=0) )
    
    # Allocate memory for result
    res = np.empty_like(localSum, dtype=localSum.dtype)

    # Global sum
    global globNumSamples
    comm.Allreduce(localSum, res, op=MPI.SUM)

    return jnp.array(res) / globNumSamples


def global_variance(data):

    mean = global_mean(data)

    # Compute sum locally
    localSum = np.array( jnp.linalg.norm(data - mean, axis=0)**2 )
    
    # Allocate memory for result
    res = np.empty_like(localSum, dtype=localSum.dtype)

    # Global sum
    global globNumSamples
    comm.Allreduce(localSum, res, op=MPI.SUM)

    return jnp.array(res) / globNumSamples


def global_sum(data):

    # Compute sum locally
    def sum_up(data):
        s = jnp.sum(data, axis=0)
        return jax.lax.psum(s, 'i')

    localSum = np.array( jax.pmap(sum_up, axis_name='i')(data)[0] )
    
    # Allocate memory for result
    res = np.empty_like(localSum, dtype=localSum.dtype)

    # Global sum
    comm.Allreduce(localSum, res, op=MPI.SUM)

    return jnp.array(res)



if __name__ == "__main__":
    data=jnp.array(np.arange(720*4).reshape((720,4)))
    myNumSamples = distribute_sampling(720)
    
    myData=data[rank*myNumSamples:(rank+1)*myNumSamples]

    print(global_mean(myData)-jnp.mean(data,axis=0))
