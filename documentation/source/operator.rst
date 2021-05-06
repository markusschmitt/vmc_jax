..

Operator module
===============

The operator module provides functionality to evaluate the action of operators on
computational basis configurations. 
In terms of an operator acting on the pure state Hilbert space this corresponds to
''on-the-fly'' generation of matrix elements.
Generally, this amounts to a mapping

.. math:: s \rightarrow \{s_j'\}, \{O_{s,s_j'}\} 

Here, :math:`s` denotes the input basis configuration and the :math:`s'` are the connected
basis configurations produced by the operator. The :math:`O_{s,s'}` are the coefficients
associated with the map :math:`s\to s'`.

Abstract operator class
^^^^^^^^^^^^^^^^^^^^^^^

The abstract operator class ``Operator`` defines the general interface of operators
acting on computational basis configurations. Any specific implementation should be
a child of ``Operator`` and implement a ``compile`` function.


.. autoclass:: jVMC.operator.Operator
    :members:


Branch-free operator class
^^^^^^^^^^^^^^^^^^^^^^^^^^
These operators are intended to act on the computational basis of a many-body Hilbert space.
The many-body Hilbert space is a product of local Hilbert spaces, :math:`\mathscr H=\bigotimes_l \mathscr h_l`.
Operators generally take the form :math:`\hat O=\sum_k \hat O_k`, where :math:`\hat O_k=\prod_l \hat o_l^k` are operator strings 
made up of elementary operators :math:`\hat o_l` acting on the factors :math:`\mathscr h_l`.

.. note:: The key assumption of the ``BranchFreeOperator`` is that the elementary operators :math:`\hat o_l` are ''branch-free'', meaning that the correspondig matrices have only one non-zero entry per row.

In variational Monte Carlo a key quantity is

    :math:`O_{loc}(s) = \sum_{s'}O_{s,s'}\psi(s')/\psi(s)`

where :math:`s,s'` denote computational basis states.

This module provides functionality to assemble general operators from elementary operators, to compute matrix elements :math:`O_{s,s'}`,
and finally to compute :math:`O_{loc}(s)`.

Elementary operators
--------------------

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
----------------

Operator strings are treated as tuples of elementary operators. For example, using the \
pre-defined Pauli operators ``Sx`` and ``Sz``, the operator string :math:`\hat\sigma_1^x\hat\sigma_2^z` \
is obtained as::

    X1Z2 = ( Sx(1), Sz(2) )

Prefactors can be added to operator strings using the ``scal_opstr()`` function. For \
example, to obtain :math:`\frac{1}{2}\hat\sigma_1^x\hat\sigma_2^z`::

    X1Z2_with_prefactor = scal_opstr(0.5, X1Z2)

Assembling operators
--------------------

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
----------------------

.. autoclass:: jVMC.operator.BranchFreeOperator
    :members:

.. autofunction:: jVMC.operator.Id
.. autofunction:: jVMC.operator.Sx
.. autofunction:: jVMC.operator.Sy
.. autofunction:: jVMC.operator.Sz
.. autofunction:: jVMC.operator.Sp
.. autofunction:: jVMC.operator.Sm
.. autofunction:: jVMC.operator.scal_opstr


POVM operator class
^^^^^^^^^^^^^^^^^^^

The POVM-formalism builds on the observation that a density matrix :math:`\rho` may equivalently (without information loss) be represented as a probability distribution :math:`P` with :math:`4^N` entries.
Elements of this distribution are obtained as expectation values of the density matrix :math:`P^\textbf{a}=\mathrm{tr}(\rho M^\textbf{a})` for the product POVM-operators :math:`M^\textbf{a}=M^{a_1}\otimes M^{a_2}\otimes..\otimes M^{a_N}`. In order to simulate dissipative dynamics in Open Quantum Systems (OQS) the Lindbladian evolution is expressed as :math:`\dot{P}^\textbf{a}=\mathcal{L}^\textbf{ab}P^\textbf{b}`, where :math:`\mathcal{L}` is the operator that generates the evolution.
The POVM operator class comprises an implementation of :math:`\mathcal{L}^\textbf{ab}`, whose main purpose it is to output all off-diagonal configurations :math:`\textbf{b}=b_1b_2..b_N` associated with an input configuration :math:`\textbf{a}=a_1a_2..a_N`. This is achieved by building on the sparsity that is typically present in many-body systems, as this sparsity crucially is preserved in the POVM-formalism allowing the described operations to scale linearly in the system size :math:`N`.

Elementary operators
--------------------

Elementary operators in the POVM-formalism are build by reexpressing the equation of motion of the density operator :math:`\dot{\rho}=\mathcal{L}[\rho]` in terms of :math:`P`, \
resulting in 

    :math:`\dot{P}^\textbf{a} = \mathrm{tr}(\mathcal{L}[M^\textbf{c}]M^\textbf{a}) T^{-1\textbf{cb}} P^\textbf{b} = \mathcal{L}^\textbf{ab}P^\textbf{b}`.

Typically :math:`\mathcal{L}[\rho]` consists of 2-body (1-body) operators which translate to (real) :math:`16\times 16` (:math:`4\times 4`) matrices in the POVM-picture. Importantly, only objects of this size need to be stored.
Frequently encountered unitary and dissipative operators are pre-defined and can be constructed as explained below.

Assembling operators
--------------------

Consider again the above example of a spin-1/2 quantum Ising chain of length :math:`L` (here with periodic boundary conditions),

    :math:`\hat H=-\sum_{l=1}^{L}\hat\sigma_l^z\hat\sigma_{(l+1) \% L}^z-g\sum_{l=1}^{L}\hat\sigma_l^x`.

However, let us now additionally assume a single-qubit decay on each spin in the chain, \
such that the time-derivative of the density matrix is

    :math:`\dot{\rho} = -i[H, \rho] + \gamma \sum_i \sigma^-_i\rho\sigma^+_i - \frac{1}{2}\lbrace \sigma^+_i\sigma^-_i,\rho\rbrace`

Using the POVM-operator class, the expression for the operator that corresponds to these dynamics reads::

    Lindbladian = jVMC.operator.POVMOperator()
    for l in range(L):
        Lindbladian.add({"name": "ZZ", "strength": -1.0, "sites": (l, (l + 1) % L)})
        Lindbladian.add({"name": "X", "strength": -g, "sites": (l,)})
        Lindbladian.add({"name": "decaydown", "strength": gamma, "sites": (l,)})

Adding terms to the Lindbladian is done using dictionaries, which have three entries: The name of the operator to be added, its prefactor and the site-ids that are involved.
Valid names that are recognized are the unitary 2-body (1-body) operators [``"XX"``, ``"YY"``, ``"ZZ"``] ([``"X"``, ``"Y"``, ``"Z"``]) corresponding to couplings and external magnetic fields aswell as the single-particle dissipation terms [``"dephasing"``, ``"decaydown"``, ``"decayup"``] corresponding to the dissipation operators [:math:`\sigma^z`, :math:`\sigma^-`, :math:`\sigma^+`].

Detailed documentation
----------------------

.. autoclass:: jVMC.operator.POVM
    :members:

.. autoclass:: jVMC.operator.POVMOperator
    :members:

.. autofunction:: jVMC.operator.measure_povm
.. autofunction:: jVMC.operator.get_M
.. autofunction:: jVMC.operator.get_dissipators
.. autofunction:: jVMC.operator.get_unitaries
.. autofunction:: jVMC.operator.get_observables
.. autofunction:: jVMC.operator.get_1_particle_distributions
.. autofunction:: jVMC.operator.get_paulis