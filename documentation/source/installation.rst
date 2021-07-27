..

Installation
===============


Option 1: ``pip``-install
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
1. We recommend you create a new conda environment to work with jVMC:

.. code-block:: none

    conda create -n jvmc python=3.8
    conda activate jvmc


2. ``pip``-install the package

.. code-block:: none

    pip install jVMC

Option 2: Clone and ``pip``-install
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Clone the jVMC repository and check out the development branch:

.. code-block:: none

    git clone https://github.com/markusschmitt/vmc_jax.git
    cd vmc_jax


2. We recommend you create a new conda environment to work with jVMC:

.. code-block:: none

    conda create -n jvmc python=3.8
    conda activate jvmc


3. Create a wheel and ``pip``-install the package

.. code-block:: none

    python setup.py bdist_wheel
    python -m pip install dist/\*.whl



Test that everything worked, e.g. run 'python -c "import jVMC"' from a place different than ``vmc_jax``.

Option 3: Manually install dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you want to work on the jVMC code you might prefer to `install dependencies and set up jVMC <https://github.com/markusschmitt/vmc_jax/blob/master/documentation/readme/compile_jax_on_cluster.md>`_ without ``pip``-install.