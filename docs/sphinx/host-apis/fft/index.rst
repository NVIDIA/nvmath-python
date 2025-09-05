**********************
Fast Fourier Transform
**********************

.. _fft-overview:

Overview
========

The Fast Fourier Transform (FFT) module :mod:`nvmath.fft` in nvmath-python leverages the
NVIDIA cuFFT library and provides a powerful suite of APIs that can be directly called from
the host to efficiently perform discrete Fourier Transformations. Both stateless
function-form APIs and stateful class-form APIs are provided to support a spectrum of
N-dimensional FFT operations. These include forward and inverse complex-to-complex (C2C)
transformations, as well as complex-to-real (C2R) and real-to-complex (R2C) transforms:

- N-dimensional forward C2C FFT transform by :func:`nvmath.fft.fft`.
- N-dimensional inverse C2C FFT transform by :func:`nvmath.fft.ifft`.
- N-dimensional forward R2C FFT transform by :func:`nvmath.fft.rfft`.
- N-dimensional inverse C2R FFT transform by :func:`nvmath.fft.irfft`.
- All types of N-dimensional FFT by stateful :class:`nvmath.fft.FFT`.

Furthermore, the :class:`nvmath.fft.FFT` class includes utility APIs designed to help users
cache FFT plans, facilitating the efficient execution of repeated calculations across
various computational tasks (see :meth:`~nvmath.fft.FFT.create_key`).

The FFT transforms performed on GPU can be fused with other operations using :ref:`FFT
callbacks <fft-callback>`. This enables users to write custom functions in Python for pre or
post-processing, while leveraging Just-In-Time (JIT) and Link-Time Optimization (LTO).

Users can also choose :ref:`CPU execution <fft-gpu-cpu-execution>` to utilize all available
computational resources.

.. note::

    The API :func:`~nvmath.fft.fft` and related function-form APIs perform **N-D FFT**
    operations, similar to :func:`numpy.fft.fftn`. There are no special 1-D
    (:func:`numpy.fft.fft`) or 2-D FFT (:func:`numpy.fft.fft2`) APIs. This not only reduces
    the API surface, but also avoids the potential for incorrect use because the number of
    batch dimensions is :math:`N - 1` for :func:`numpy.fft.fft` and :math:`N - 2` for
    :func:`numpy.fft.fft2`, where :math:`N` is the operand dimension.


.. _fft-callback:

FFT Callbacks
=============

User-defined functions can be `compiled to the LTO-IR format
<https://docs.nvidia.com/cuda/cufft/index.html#lto-load-and-store-callback-routines>`_ and
provided as epilog or prolog to the FFT operation, allowing for Link-Time Optimization and
fusing. This can be used to implement DFT-based convolutions or scale the FFT output, for
example.

The FFT module comes with convenient helper functions :func:`nvmath.fft.compile_prolog` and
:func:`nvmath.fft.compile_epilog` that compile functions written in Python to LTO-IR format.
Under the hood, the helpers rely on Numba as the compiler. The compiled callbacks can be
passed to functional or stateful FFT APIs as :class:`~nvmath.fft.DeviceCallable`.
Alternatively, users can compile the callbacks to LTO-IR format with a compiler of their
choice and pass them as :class:`~nvmath.fft.DeviceCallable` to the FFT call.

Examples illustrating use of prolog and epilog functions can be found in the `FFT examples
directory <https://github.com/NVIDIA/nvmath-python/tree/main/examples/fft>`_.

.. note::

    FFT Callbacks are not currently supported on Windows.

Setting-up
----------

The fastest way to start using cuFFT LTO with nvmath is to install it with device API
dependencies. Pip users should run the following command:

.. code-block:: bash

   pip install nvmath-python[cu12,dx]


Required dependencies
---------------------

For those who need to collect the required dependencies manually:

- LTO callbacks are supported by cuFFT 11.3 which is shipped with `CUDA Toolkit 12.6 Update
  2 and newer <https://developer.nvidia.com/cuda-downloads>`_.
- Using cuFFT LTO callbacks requires nvJitLink from the same CUDA toolkit or newer (within
  the same major CUDA release, for example version 12).
- Compiling the callbacks with the :func:`nvmath.fft.compile_prolog` and
  :func:`nvmath.fft.compile_epilog` helpers requires Numba 0.59+ and nvcc/nvvm from the same
  CUDA toolkit as nvJitLink or older (within the same major CUDA release). The helpers
  require the target device to have compute capability 7.0 or higher.

For further details, refer to the `cuFFT LTO documentation
<https://docs.nvidia.com/cuda/cufft/index.html#lto-load-and-store-callback-routines>`_.


Older CTKs
^^^^^^^^^^

Adventurous users who want to try callback functionality and cannot upgrade the CUDA Toolkit
to 12.6U2, can download and install the older preview release `cuFFT LTO EA version 11.1.3.0
<https://docs.nvidia.com/cuda/archive/12.6.1/cufft/ltoea/release_notes.html
#cufft-lto-ea-preview-11-1-3-0>`_ from `here <https://developer.nvidia.com/cufftea>`_, which
requires at least CUDA Toolkit 12.2. When using LTO EA, setting environmental variables may
be needed for nvmath to pick the desired cuFFT version. Users should adjust the
``LD_PRELOAD`` variable, so that the right cuFFT shared library is used:

.. code-block:: bash

   export LD_PRELOAD="/path_to_cufft_lto_ea/libcufft.so"


.. _fft-gpu-cpu-execution:

Execution space
===============

FFT transforms can be executed either on NVIDIA GPU or CPU. By default, the execution space
is selected based on the memory space of the operand passed to the FFT call, but it can be
explicitly controlled with :class:`~nvmath.fft.ExecutionCUDA` and
:class:`~nvmath.fft.ExecutionCPU` passed as the ``execution`` option to the call (for
example :class:`~nvmath.fft.FFT` or :func:`~nvmath.fft.fft`).

.. note::

    CPU execution is not currently supported on Windows.

Required dependencies
---------------------

With ARM CPUs, such as NVIDIA Grace, nvmath-python can utilize `NVPL (Nvidia Performance
Libraries) <https://developer.nvidia.com/nvpl>`_ FFT to run the transform. On x86_64
architecture, the `MKL library <https://pypi.org/project/mkl/>`_ can be used.

For pip users, the fastest way to get the required dependencies is to use ``'cu12'`` /
``'cu11'`` and ``'cpu'`` extras:

.. code-block:: bash

   # for CPU-only dependencies
   pip install nvmath-python[cpu]

   # for CUDA-only dependencies (assuming CUDA 12)
   pip install nvmath-python[cu12]

   # for CUDA 12 and CPU dependencies
   pip install nvmath-python[cu12,cpu]


Custom CPU library
^^^^^^^^^^^^^^^^^^

Other libraries that conform to FFTW3 API and ship single and double precision symbols in
the single ``so`` file can be used to back the CPU FFT execution. Users who would like to
use different library for CPU FFT, or point to a custom installation of NVPL or MKL library,
can do so by including the library path in ``LD_LIBRARY_PATH`` and specifying the library
name with ``NVMATH_FFT_CPU_LIBRARY``. For example:


.. code-block:: bash

   # nvpl
   export LD_LIBRARY_PATH=/path/to/nvpl/:$LD_LIBRARY_PATH
   export NVMATH_FFT_CPU_LIBRARY=libnvpl_fftw.so.0

   # mkl
   export LD_LIBRARY_PATH=/path/to/mkl/:$LD_LIBRARY_PATH
   export NVMATH_FFT_CPU_LIBRARY=libmkl_rt.so.2


.. _fft-api-reference:

Host API Reference
==================

.. module:: nvmath.fft


FFT support (:mod:`nvmath.fft`)
-------------------------------

.. autosummary::
   :toctree: generated/

   fft
   ifft
   rfft
   irfft
   FFT
   compile_prolog
   compile_epilog
   UnsupportedLayoutError

   :template: dataclass.rst

   FFTOptions
   FFTDirection
   ExecutionCUDA
   ExecutionCPU
   DeviceCallable
