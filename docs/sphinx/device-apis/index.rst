
*************************
nvmath-python Device APIs
*************************

.. _device-api-overview:

Overview
========

The device module of nvmath-python :mod:`nvmath.device` offers integration with NVIDIA's high-performance computing libraries through device APIs for cuFFTDx and cuBLASDx. 
Detailed documentation for these libraries can be found at `cuFFTDx <https://docs.nvidia.com/cuda/cufftdx/1.2.0>`_ and `cuBLASDx <https://docs.nvidia.com/cuda/cublasdx/0.1.1>`_.

Users may take advantage of the device module via the two approaches below:

- Numba Extensions: Users can access these device APIs via Numba by utilizing specific extensions that simplify the process of defining functions, 
  querying device traits, and calling device functions.
- Third-party JIT Compilers: The APIs are also available through low-level interfaces in other JIT compilers, 
  allowing advanced users to work directly with the raw device code.

.. note::

   The device module :mod:`nvmath.device` currently supports cuFFTDx 1.2.0 and cuBLASDx 0.1.1, also available as part of MathDx 24.04. 
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
