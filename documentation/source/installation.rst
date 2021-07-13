..

Installation
===============

Currently, there exist two ways to install the python package, which both rely on ``pip``.
An installation into a clean virtual environment is recommended, but not essential. 
The following lines assume that a clean virtual environment was created and activated, like so: 
``conda create --name my_environment python``
followed by
``conda activate my_environment``.
Dependent packages are taken care of automatically. 

Build from local wheel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the folder ``vmc_jax/dist/`` a ``.whl`` file exists, which may be installed using ``pip`` using the following command: ``python -m pip install vmc_jax/dist/jVMC-1.0-py3-none-any.whl``


Build with pip (online)
^^^^^^^^^^^^^^^^^^^^^^^^^^
Coming soon.

To check that everything went as expected, you may run ``python -c "import jVMC"``.