**********************
nvmath-python Bindings
**********************

Overview
========

.. warning::

    All Python bindings documented in this section are *experimental* and subject to future
    changes. Use it at your own risk.

Low-level Python bindings for C APIs from NVIDIA Math Libraries are exposed under the
corresponding modules in :mod:`nvmath.bindings`. To access the Python bindings, use the
modules for the corresponding libraries. Under the hood, nvmath-python handles the run-time
linking to the libraries for you lazily.

The currently supported libraries along with the corresponding module names are listed as
follows:

.. module:: nvmath
   :no-index:

.. list-table::
   :widths: 20 40
   :header-rows: 1

   * - Library name
     - Python access
   * - cuBLAS
     - :mod:`nvmath.bindings.cublas`
   * - cuBLASLt
     - :mod:`nvmath.bindings.cublasLt`
   * - cuFFT
     - :mod:`nvmath.bindings.cufft`
   * - cuRAND
     - :mod:`nvmath.bindings.curand`
   * - cuSOLVER
     - :mod:`nvmath.bindings.cusolver`
   * - cuSOLVERDn
     - :mod:`nvmath.bindings.cusolverDn`
   * - cuSPARSE
     - :mod:`nvmath.bindings.cusparse`

Support for more libraries will be added in the future.


Naming & Calling Convention
===========================

Inside each of the modules, all public APIs of the corresponding NVIDIA Math library are
exposed following the `PEP 8`_ style guide along with the following changes:

* All library name prefixes are stripped
* The function names are broken by words and follow the camel case
* The first letter in each word in the enum names are capitalized
* Each enum's name prefix is stripped from its values' names
* Whenever applicable, the outputs are stripped away from the function arguments and
  returned directly as Python objects
* Pointers are passed as Python :class:`int`
* Exceptions are raised instead of returning the C error code

Below is a non-exhaustive list of examples of such C-to-Python mappings:

.. currentmodule:: nvmath.bindings

- Function: ``cublasDgemm`` -> :func:`cublas.dgemm`.
- Function: ``curandSetGeneratorOrdering`` -> :func:`curand.set_generator_ordering`
- Enum type: ``cublasLtMatmulTile_t`` -> :class:`cublasLt.MatmulTile`
- Enum type: ``cufftXtSubFormat`` -> :class:`cufft.XtSubFormat`
- Enum value name: ``CUSOLVER_EIG_MODE_NOVECTOR`` -> :data:`cusolver.EigMode.NOVECTOR`
- Enum value name: ``CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED`` ->
  :data:`cusparse.Status.MATRIX_TYPE_NOT_SUPPORTED`
- Returns: The outputs of ``cusolverDnXpotrf_bufferSize`` are the workspace sizes on device
  and host, which are wrapped as a 2-tuple in the corresponding
  :func:`cusolverDn.xpotrf_buffer_size` Python API.

There may be exceptions for the above rules, but they would be self-evident and will be
properly documented. In the next section we discuss pointer passing in Python.

.. _PEP 8: https://www.python.org/dev/peps/pep-0008/


Memory management
=================

Pointer and data lifetime
.........................

Unlike in C/C++, Python does not provide low-level primitives to allocate/deallocate host
memory (nor device memory). In order to make the C APIs work with Python, it is important
that memory management is properly done through Python proxy objects. In nvmath-python, we
ask users to address such needs using NumPy (for host memory) and CuPy (for device memory).

.. note::

    It is also possible to use :class:`array.array` (plus :class:`memoryview` as needed) to
    manage host memory. However it is more laborious compared to using
    :class:`numpy.ndarray`, especially when it comes to array manipulation and computation.

.. note::

    It is also possible to use `CUDA Python`_ to manage device memory, but as of CUDA 11
    there is no simple, pythonic way to modify the contents stored on GPU, which requires
    custom kernels. CuPy is a lightweight, NumPy-compatible array library that addresses
    this need.

To pass data from Python to C, using pointer addresses (as Python :class:`int`) of various
objects is required. We illustrate this using NumPy/CuPy arrays as follows:

.. code-block:: python

    # create a host buffer to hold 5 int
    buf = numpy.empty((5,), dtype=numpy.int32)
    # pass buf's pointer to the wrapper
    # buf could get modified in-place if the function writes to it
    my_func(..., buf.ctypes.data, ...)
    # examine/use buf's data
    print(buf)

    # create a device buffer to hold 10 double
    buf = cupy.empty((10,), dtype=cupy.float64)
    # pass buf's pointer to the wrapper
    # buf could get modified in-place if the function writes to it
    my_func(..., buf.data.ptr, ...)
    # examine/use buf's data
    print(buf)

    # create an untyped device buffer of 128 bytes
    buf = cupy.cuda.alloc(128)
    # pass buf's pointer to the wrapper
    # buf could get modified in-place if the function writes to it
    my_func(..., buf.ptr, ...)
    # buf is automatically destroyed when going out of scope

The underlying assumption is that the arrays must be contiguous in
memory (unless the C interface allows for specifying the array strides).

As a consequence, all C structs in NVIDIA Math libraries (including handles and descriptors)
are *not exposed* as Python classes; that is, they do not have their own types and are
simply cast to plain Python :class:`int` for passing around. Any downstream consumer should
create a wrapper class to hold the pointer address if so desired. In other words, users have
full control (and responsibility) for managing the *pointer lifetime*.

However, in certain cases we are able to convert Python objects for users (if *readonly,
host* arrays are needed) so as to alleviate users' burden. For example, in functions that
require a sequence or a nested sequence, the following operations are equivalent:

.. code-block:: python

    # passing a host buffer of int type can be done like this
    buf = numpy.array([0, 1, 3, 5, 6], dtype=numpy.int32)
    my_func(..., buf.ctypes.data, ...)

    # or just this
    buf = [0, 1, 3, 5, 6]
    my_func(..., buf, ...)  # the underlying data type is determined by the C API

which is particularly useful when users need to pass multiple sequences or nested sequences
to C (For example, :func:`nvmath.bindings.cufft.plan_many`).

.. note::

    Some functions require their arguments to be in the device memory. You need to pass
    device memory (for example, :class:`cupy.ndarray`) to such arguments. nvmath-python
    neither validates the memory pointers nor implicitly transfers the data.
    Passing host memory where device memory is expected (and vice versa) results in
    undefined behavior.


.. _CUDA Python: https://nvidia.github.io/cuda-python/index.html


.. _python-bindings-reference-label:

API Reference
=============

This reference describes all nvmath-python's math primitives.

.. module:: nvmath.bindings

.. toctree::
   :maxdepth: 2
   :includehidden:

   cublas
   cublasLt
   cufft
   cusolver
   cusolverDn
   cusparse
   curand
