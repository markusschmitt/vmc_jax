.. _Nets:

Networks
========

``jVMC.nets`` provides ready-made variational networks. Any Flax module whose ``__call__``
maps one configuration to :math:`\ln\psi(s)` can be used instead, see :ref:`VQS`.

Restricted Boltzmann machines
-----------------------------

.. autoclass:: jVMC.nets.RBM
.. autoclass:: jVMC.nets.CpxRBM
.. autoclass:: jVMC.nets.CpxRBMNoSpinFlip
.. autoclass:: jVMC.nets.CpxRBMRatio

Convolutional and feed-forward networks
---------------------------------------

.. autoclass:: jVMC.nets.CNN
.. autoclass:: jVMC.nets.CpxCNN
.. autoclass:: jVMC.nets.FFN

Autoregressive networks
-----------------------

.. autoclass:: jVMC.nets.RNN1DGeneral
.. autoclass:: jVMC.nets.RNN2DGeneral
.. autoclass:: jVMC.nets.RWKVCPM
.. autoclass:: jVMC.nets.ParticleConservingAutoregressive
.. autoclass:: jVMC.nets.LogRWKV2DAutoregressiveJVMC

Transformers
------------

.. autoclass:: jVMC.nets.LogViterittiSpatialViTJVMC
.. autoclass:: jVMC.nets.HiddenPfaffianViT

Other networks and wrappers
---------------------------

.. autoclass:: jVMC.nets.FullConfigurationState
.. autoclass:: jVMC.nets.TwoNets

Initializers and activation functions
-------------------------------------

.. automodule:: jVMC.nets.initializers
    :members:

.. automodule:: jVMC.nets.activation_functions
    :members:
