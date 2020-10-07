..

Operator module
===============

The many-body Hilbert space is a product of local Hilbert spaces, :math:`\mathscr H=\bigotimes_l \mathscr h_l`.
Operators generally take the form :math:`\hat O=\sum_k \hat O_k`, where :math:`\hat O_k=\prod_l \hat o_l^k` are operator strings 
made up of elementary operators :math:`\hat o_l` acting on the factors :math:`\mathscr h_l`.

In variational Monte Carlo a key quantity is

    :math:`O_{loc}(s) = \sum_{s'}O_{s,s'}\psi(s')/\psi(s)`

where :math:`s,s'` denote computational basis states.

This module provides functionality to assemble general operators from elementary operators, to compute matrix elements :math:`O_{s,s'}`,
and finally to compute :math:`O_{loc}(s)`.

Elementary operators
^^^^^^^^^^^^^^^^^^^^

At the core the operator class works with a general description of elementary operators 
:math:`\hat o_l`. Elementary operators are defined by dictionaries that hold four items:

    * ``'idx'``: Index of the corresponding local Hilbert space.
    * ``'map'``: An array (jax.numpy.array) defining the mapping between local basis indices. The array entry :math:`j` holds the image of the basis index :math:`j` under the map.
    * ``'matEls'``: An array (jax.numpy.array) defining the corresponding matrix elements.
    * ``'diag'``: Boolean stating whether this operator is a diagonal operator (exploiting this information enhances efficiency).

For concreteness, consider the Pauli operator

    :math:`\hat\sigma^x=\begin{pmatrix}0&1\\1&0\end{pmatrix}`

acting on lattice site :math:`l=1`. The corresponding dictionary is::

    Sx = {
            'idx': 1,
            'map': jax.numpy.array([1,0],dtype=np.int32),
            'matEl': jax.numpy.array([1.,1.],dtype=jVMC.global_defs.tReal),
            'diag': False
         }

A number of frequently used operators is pre-defined in this module, see below.

Operator strings
^^^^^^^^^^^^^^^^

Operator strings are treated as tuples of elementary operators. For example, using the \
pre-defined Pauli operators ``Sx`` and ``Sz``, the operator string :math:`\hat\sigma_1^x\hat\sigma_2^z` \
is obtained as::

    X1Z2 = ( Sx(1), Sz(2) )

Prefactors can be added to operator strings using the ``scal_opstr()`` function. For \
example, to obtain :math:`\frac{1}{2}\hat\sigma_1^x\hat\sigma_2^z`::

    X1Z2_with_prefactor = scal_opstr(0.5, X1Z2)

Assembling operators
^^^^^^^^^^^^^^^^^^^^

Finally, arbitrary operators can be assembled from operator strings. Consider, e.g., the \
Hamiltonian of the spin-1/2 quantum Ising chain of length L,

    :math:`\hat H=-\sum_{l=0}^{L-2}\hat\sigma_l^z\hat\sigma_{l+1}^z-g\sum_{l=0}^{L-1}\hat\sigma_l^x`

Again, using the pre-defined Pauli operators ``Sx`` and ``Sz``, an ``Operator`` object \
for this Hamiltonian can be obtained as::

    hamiltonian = Operator()
    for l in range(L-1):
        hamiltonian.add( scal_opstr( -1., ( Sz(l), Sz(l+1) ) ) )
        hamiltonian.add( scal_opstr( -g, ( Sx(l), ) ) )
    hamiltonian.add( scal_opstr( -g, ( Sx(L-1), ) ) )

Detailed documentation
^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: jVMC.operator
    :members:
