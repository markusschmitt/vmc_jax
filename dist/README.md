How to obtain a wheel of the package, following https://dzone.com/articles/executable-package-pip-install.

1. place 'jvmc_exec' and 'setup.py' one level higher, i.e. on the level where the jVMC package is located.
2. change 'setup.py' to your liking.
3. follow the steps as they are layed out in the above link, i.e. 
	a) 'chmod +x jvmc_exec'
	b) 'python setup.py bdist_wheel'
	c) 'python -m pip install dist/*.whl', ideally into a clean environment to check that package dependencies are treated correctly.
4. Test that everything worked, e.g. run 'python -c "import jVMC"' from a place different than 'vmc_jax'.

