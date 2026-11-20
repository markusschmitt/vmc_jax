Installation
============

jVMC requires Python 3.11 or later. Like JAX, it runs on Linux and macOS; GPUs are supported on
Linux. We recommend installing it in a fresh virtual environment, e.g.

.. code-block:: none

    python -m venv jvmc-env
    source jvmc-env/bin/activate

CPU
^^^

.. code-block:: none

    pip install jVMC

GPU
^^^

Choose the extra that matches your CUDA version:

.. code-block:: none

    pip install "jVMC[cuda12]"     # or "jVMC[cuda13]"

This installs the CUDA build of JAX together with the CUDA libraries (see the
`JAX installation guide <https://docs.jax.dev/en/latest/installation.html>`_), and
`JAXMg <https://github.com/flatironinstitute/jaxmg>`_ for distributed linear algebra across
several GPUs.

On clusters that provide CUDA through environment modules, use the system installation instead,
so that JAX uses the libraries matched to the local drivers:

.. code-block:: none

    module load cuda                    # cluster-specific
    pip install "jVMC[cuda12-local]"    # or "jVMC[cuda13-local]"

Development installation
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: none

    git clone https://github.com/computational-quantum-science/jvmc-2.0.git
    cd jvmc-2.0
    pip install -e ".[dev]"

This also installs the test and documentation tools. Run the tests with ``pytest tests/`` and
build this documentation with

.. code-block:: none

    sphinx-build -b html -W --keep-going documentation/source documentation/_build/html

Check the installation from outside the repository folder with
``python -c "import jVMC; print(jVMC.__version__)"``.
