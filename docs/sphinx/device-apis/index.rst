
*************************
nvmath-python Device APIs
*************************

.. module:: nvmath.device

.. _device-api-overview:

The device module of nvmath-python :mod:`nvmath.device` offers integration with NVIDIA's
high-performance computing libraries through device APIs for cuFFTDx, cuBLASDx, and cuRAND.
Detailed documentation for these libraries can be found at `cuFFTDx
<https://docs.nvidia.com/cuda/cufftdx/1.2.0>`_, `cuBLASDx
<https://docs.nvidia.com/cuda/cublasdx/0.1.1>`_, and `cuRAND device APIs
<https://docs.nvidia.com/cuda/curand/group__DEVICE.html#group__DEVICE>`_ respectively.
Device APIs can only be called from CUDA device or kernel code, and execute on the GPU.

Users may take advantage of the device module via the two approaches below:

- Numba Extensions: Users can access these device APIs via Numba by utilizing specific
  extensions that simplify the process of defining functions, querying device traits, and
  calling device functions.
- Third-party JIT Compilers: The APIs are also available through low-level interfaces in
  other JIT compilers, allowing advanced users to work directly with the raw device code.

.. note::

   The :class:`~nvmath.device.fft` and :class:`~nvmath.device.matmul` device APIs in module
   :mod:`nvmath.device` currently supports cuFFTDx 1.2.0 and cuBLASDx 0.1.1, also available
   as part of MathDx 24.04. All functionalities from the C++ libraries are supported with
   the exception of cuFFTDx C++ APIs with a workspace argument, which are currently not
   available in nvmath-python.

.. toctree::
   :caption: Contents
   :maxdepth: 1

   Device API utilities <utils.rst>
   cuBLASDx <cublas.rst>
   cuFFTDx <cufft.rst>
   cuRAND Device APIs <curand.rst>
