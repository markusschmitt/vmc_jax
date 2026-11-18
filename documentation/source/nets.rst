.. _Nets:

Networks
========

``jVMC_exp.nets`` provides ready-made variational networks. Any Flax module whose ``__call__``
maps one configuration to :math:`\ln\psi(s)` can be used instead, see :ref:`VQS`.

Restricted Boltzmann machines
-----------------------------

.. autoclass:: jVMC_exp.nets.RBM
.. autoclass:: jVMC_exp.nets.CpxRBM
.. autoclass:: jVMC_exp.nets.CpxRBMNoSpinFlip
.. autoclass:: jVMC_exp.nets.CpxRBMRatio

Convolutional and feed-forward networks
---------------------------------------

.. autoclass:: jVMC_exp.nets.CNN
.. autoclass:: jVMC_exp.nets.CpxCNN
.. autoclass:: jVMC_exp.nets.FFN

Autoregressive networks
-----------------------

.. autoclass:: jVMC_exp.nets.RNN1DGeneral
.. autoclass:: jVMC_exp.nets.RNN2DGeneral
.. autoclass:: jVMC_exp.nets.RWKVCPM
.. autoclass:: jVMC_exp.nets.ParticleConservingAutoregressive
.. autoclass:: jVMC_exp.nets.LogRWKV2DAutoregressiveJVMC

Transformers
------------

.. autoclass:: jVMC_exp.nets.LogViterittiSpatialViTJVMC
.. autoclass:: jVMC_exp.nets.HiddenPfaffianViT

Other networks and wrappers
---------------------------

.. autoclass:: jVMC_exp.nets.FullConfigurationState
.. autoclass:: jVMC_exp.nets.TwoNets

Initializers and activation functions
-------------------------------------

.. automodule:: jVMC_exp.nets.initializers
    :members:

.. automodule:: jVMC_exp.nets.activation_functions
    :members:
