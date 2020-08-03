# Compiling JAX with GPU support on a compute cluster

Currently (July 2020), installing JAX via pip does not work on CentOS 7, see this [Jax issue](https://github.com/google/jax/issues/854). Unfortunately, however, many large scale compute clusters run on CentOS 7 and installing JAX might not be the highest priority for the corresponding support teams. Hence, scientific JAX users face the challenge of installing the library themselves on those computers. The following is a summary of the steps that I found to avoid various pitfalls when installing JAX on the [JUWELS](https://www.fz-juelich.de/ias/jsc/EN/Expertise/Supercomputers/JUWELS/JUWELS_news.html) compute cluster.

General instructions for building from scratch are part of the [JAX documentation](https://jax.readthedocs.io/en/latest/developer.html#building-from-source).

Some of the main steps are also explained in this [Jax issue](https://github.com/google/jax/issues/2083).

## Prerequisites

Required software:

* C++ compiler
* CUDA
* cuDNN
* Python
* pip

Paths you need to know:

* Path to the CUDA installation. You can determine it via
`
which nvcc
`
and the part you need is probably something like  
```
/your/path/to/cuda/CUDA/cuda.version.number/
```  
&nbsp;&nbsp;&nbsp;We refer to it as `MY_PATH_TO_CUDA` in the following.

* Path to the cuDNN installation. Given you know the path to CUDA from above, it is probably
```
your/path/to/cuda/cuDNN/version.number-CUDA-cuda.version.number/
```
&nbsp;&nbsp;&nbsp;We refer to it as `MY_PATH_TO_CUDNN` in the following.

* Path to your python binary. Get it via
```
which python
```
&nbsp;&nbsp;&nbsp;We refer to it as `MY_PATH_TO_PYTHON` in the following and it should include the name of the binary.

## Installation steps

We generally follow the [JAX documentation](https://jax.readthedocs.io/en/latest/developer.html#building-from-source), but add some fine tuning.


1. clone the JAX repo from GitHub and go into the new directory: 
```
git clone https://github.com/google/jax.git
cd jax
```

2. Install JAX dependencies
```	
pip install --user --upgrade numpy scipy cython six
```

3. Make sure that `PYTHONPATH` points to these locally installed modules:
```
PYTHONPATH=$HOME/.local/lib/python3.6/site-packages/:$PYTHONPATH
export PYTHONPATH
```

&nbsp;&nbsp;&nbsp;Add the local directory also to `PATH` and `LD_LIBRARY_PATH`:
```
PATH=$HOME/.local/bin:$PATH
export PATH
LD_LIBRARY_PATH=$HOME/.local/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH
```
4. Tell tensorflow where CUDA is, see this [Jax issue](https://github.com/google/jax/issues/2083)
```
export TF_CUDA_PATHS=$MY_PATH_TO_CUDA
```

5. Attempt to build jaxlib:
```
python build/build.py --enable_cuda --cuda_path $MY_PATH_TO_CUDA --cudnn_path $MY_PATH_TO_CUDNN --python_bin_path $MY_PATH_TO_PYTHON
```
&nbsp;&nbsp;&nbsp;This will first download and compile bazel. Bazel creates a local cache directory under $HOME/.cache/bazel. This can, however, easily exhaust the home directory quota. If so, a corresponding error will appear while building jaxlib. To circumvent this, go to the next step, otherwise skip it.

6. To avoid cramming the home directory, we create a wrapper script for bazel that temporarily overwrites the HOME environment variable to fool bazel into using a different directory, see this [Bazel issue](https://github.com/bazelbuild/bazel/issues/4248).

&nbsp;&nbsp;&nbsp;First move the bazel binary into a new subdirectory:
```
mkdir build/bazel
mv build/bazel-2.0.0-linux-x86_64 build/basel/
```
&nbsp;&nbsp;&nbsp;Then create a script named `bazel-2.0.0-linux-x86_64` in the `build/` directory with the content
```
#!/bin/sh

export HOME=/p/project/your_project/

exec /your/path/jax/build/bazel/bazel-2.0.0-linux-x86_64 "$@"
```

&nbsp;&nbsp;&nbsp;Now you should be able to run
```
python build/build.py --enable_cuda --cuda_path $MY_PATH_TO_CUDA --cudnn_path $MY_PATH_TO_CUDNN --python_bin_path $MY_PATH_TO_PYTHON
```
&nbsp;&nbsp;&nbsp;without exceeding disk quota. On JUWELS the building process took about 15-20 minutes. 

7. Now proceed as described in the [JAX documentation](https://jax.readthedocs.io/en/latest/developer.html#building-from-source):
```
pip install --user -e build
pip install --user -e .
```

8. Finally, JAX should be installed. For a first quick test:
```
python -c 'import jax'
```
&nbsp;&nbsp;&nbsp;If your cluster has a slurm queue: To run all JAX tests, submit a script similar to
```
#!/bin/bash -x
#SBATCH --account=your_account
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=12
#SBATCH --output=gpu-out.%j
#SBATCH --error=gpu-err.%j
#SBATCH --time=02:00:00
#SBATCH --partition=gpu_partition
#SBATCH --gres=gpu:1
 
srun pytest -n auto /your/path/jax/tests/
```

9. Add the export statements from 3) and 4) also to your `~/.bash_profile`
