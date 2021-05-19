0. Create a new directory:

	mkdir jvmc; cd jvmc

1. Clone the jVMC repository and check out the development branch:

	git clone https://github.com/markusschmitt/vmc_jax.git .
	git checkout dev_0.1.0

2. Create and activate a conda environment (give your consent if conda asks to install some packages):

	conda create -n jax_env python=3.8
	conda activate test_jax

3. Install JAX and other dependencies:

	pip install --upgrade jax jaxlib
	pip install flax mpi4py h5py

4. Install sphinx to be able to build the documentation:

	conda install sphinx sphinx_rtd_theme mock

5. Compile the documentation:

	cd documentation
	make html
	cd ..

	Now you should be able to open documentation/build/html/index.html in your browser.

6. Run an example to see whether everything works:

	python examples/ex3_custom_net.py 

	If you see the output "coeffs.shape: (1, 13)", everything is good. Don't worry about the "WARNING".

