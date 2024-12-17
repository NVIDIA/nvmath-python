
************************************************
cuRAND Device APIs (:mod:`nvmath.device.random`)
************************************************

.. _device-api-curand-overview:

Overview
========

This module provides access to the device APIs of NVIDIA cuRAND library, which allows
random number generation on the GPU.
Detailed documentation of cuRAND device APIs can be found in the
`cuRAND documentation
<https://docs.nvidia.com/cuda/curand/group__DEVICE.html#group__DEVICE>`_.

.. _device-api-curand-reference:

API Reference
=============

.. module:: nvmath.device.random

Utilities
^^^^^^^^^

.. currentmodule:: nvmath.device.random
.. autosummary::
   :toctree: generated/

   Compile

Bit Generator and State APIs (:mod:`nvmath.device.random`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: nvmath.device.random
.. autosummary::
   :toctree: generated/

   init
   rand
   rand4

   StatesMRG32k3a
   StatesPhilox4_32_10
   StatesSobol32
   StatesSobol64
   StatesScrambledSobol32
   StatesScrambledSobol64
   StatesXORWOW


Distribution Sampling APIs (:mod:`nvmath.device.random`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: nvmath.device.random
.. autosummary::
   :toctree: generated/

   normal
   normal_double
   normal2
   normal2_double
   normal4
   log_normal
   log_normal_double
   log_normal2
   log_normal2_double
   log_normal4
   poisson
   poisson4
   uniform
   uniform_double
   uniform2_double
   uniform4

Skip Ahead APIs  (:mod:`nvmath.device.random`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: nvmath.device.random
.. autosummary::
   :toctree: generated/

   skipahead
   skipahead_sequence
   skipahead_subsequence

Helper Host APIs (:mod:`nvmath.device.random_helpers`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. module:: nvmath.device.random_helpers
.. currentmodule:: nvmath.device.random_helpers
.. autosummary::
   :toctree: generated/

    get_direction_vectors32
    get_direction_vectors64
    get_scramble_constants32
    get_scramble_constants64
    DirectionVectorSet
