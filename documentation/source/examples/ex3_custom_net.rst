.. _ex3_custom_net:

Definition of custom network architectures
==========================================

This example shows exemplarily how to define a custom complex RBM architecture, i.e.,

.. math:: \log\psi(\mathbf s) = \sum_i \log\big[\cosh\big(b_i + \sum_j W_{ij} s_j\big)\big]

with :math:`b_i, W_{ij}\in\mathbb C`.


.. literalinclude:: ../../../examples/ex3_custom_net.py
        :linenos:
        :language: python
        :lines: 10-34
