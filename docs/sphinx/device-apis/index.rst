
*************************
nvmath-python Device APIs
*************************

.. _device-api-overview:

Overview
========

The device module of nvmath-python :mod:`nvmath.device` offers integration with NVIDIA's high-performance computing libraries through device APIs for cuFFTDx, cuBLASDx, and cuRAND.
Detailed documentation for these libraries can be found at `cuFFTDx <https://docs.nvidia.com/cuda/cufftdx/1.2.0>`_, `cuBLASDx <https://docs.nvidia.com/cuda/cublasdx/0.1.1>`_, and
`cuRAND device APIs <https://docs.nvidia.com/cuda/curand/group__DEVICE.html#group__DEVICE>`_ respectively.

Users may take advantage of the device module via the two approaches below:

- Numba Extensions: Users can access these device APIs via Numba by utilizing specific extensions that simplify the process of defining functions,
  querying device traits, and calling device functions.
- Third-party JIT Compilers: The APIs are also available through low-level interfaces in other JIT compilers,
  allowing advanced users to work directly with the raw device code.

.. note::

   The :class:`~nvmath.device.fft` and :class:`~nvmath.device.matmul` device APIs in module :mod:`nvmath.device` currently supports cuFFTDx 1.2.0 and cuBLASDx 0.1.1, also available as part of MathDx 24.04.
   All functionalities from the C++ libraries are supported with the exception of cuFFTDx C++ APIs with a workspace argument, which are currently not available in nvmath-python.


.. _device-api-reference:

API Reference
=============

.. module:: nvmath.device

Utility APIs (:mod:`nvmath.device`)
-----------------------------------

.. autosummary::
   :toctree: generated/

   current_device_lto
   float16x2
   float16x4
   float32x2
   float64x2
   float16x2_type
   float16x4_type
   float32x2_type
   float64x2_type

   :template: namedtuple.rst

   ISAVersion
   Code
   CodeType
   ComputeCapability
   CodeType
   Symbol
   Dim3

cuBLASDx APIs (:mod:`nvmath.device`)
------------------------------------

.. autosummary::
   :toctree: generated/

   matmul
   BlasOptions

   :template: namedtuple.rst

   LeadingDimension
   TransposeMode

cuFFTDx APIs (:mod:`nvmath.device`)
-----------------------------------

.. autosummary::
   :toctree: generated/

   fft
   FFTOptions

cuRAND Device APIs (:mod:`nvmath.device.random`)
------------------------------------------------

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

.. currentmodule:: nvmath.device.random_helpers
.. autosummary::
   :toctree: generated/

    get_direction_vectors32
    get_direction_vectors64
    get_scramble_constants32
    get_scramble_constants64
    DirectionVectorSet
