
************************************
cuBLASDx APIs (:mod:`nvmath.device`)
************************************

.. _device-api-cublas-overview:

Overview
========

These APIs offer integration with the NVIDIA cuBLASDx library.
Detailed documentation of cuBLASDx can be found in the
:cublasdx_doc:`cuBLASDx documentation <index.html>`.

.. note::

   The :class:`~nvmath.device.Matmul` device API in module
   :mod:`nvmath.device` currently supports cuBLASDx |cublasdx_version|, also available
   as part of MathDx |mathdx_version|.

.. _device-api-cublas-traits:

Traits Feature Readiness
========================

This table outlines the readiness of cuBLASDx traits in the Python API
(:mod:`nvmath.device`).

1. Description Traits
---------------------
These traits provide information about the function descriptor constructed using Operators.

.. list-table::
   :widths: 30 35 10 25
   :header-rows: 1

   * - C++ Trait
     - Python ``nvmath.device`` Implementation
     - Status
     - Notes
   * - ``size_of``
     - :attr:`~nvmath.device.Matmul.size`
     - ✅
     - Returns ``(m, n, k)`` tuple.
   * - ``type_of``
     - :attr:`~nvmath.device.Matmul.data_type`
     - ✅
     - Returns ``'real'`` or ``'complex'``.
   * - ``precision_of``
     - :attr:`~nvmath.device.Matmul.precision`
     - ✅
     - Returns :class:`~nvmath.device.Precision` named tuple.
   * - ``function_of``
     - :attr:`~nvmath.device.Matmul.function`
     - ✅
     - Returns the string (e.g., ``'MM'``).
   * - ``arrangement_of``
     - :attr:`~nvmath.device.Matmul.arrangement`
     - ✅
     - Returns :class:`~nvmath.device.Arrangement` named tuple.
   * - ``transpose_mode_of``
     - :attr:`~nvmath.device.Matmul.transpose_mode`
     - ✅
     - Returns :class:`~nvmath.device.TransposeMode` named tuple (marked as deprecated).
   * - ``alignment_of``
     - :attr:`~nvmath.device.Matmul.alignment`
     - ✅
     - Returns :class:`~nvmath.device.Alignment` named tuple.
   * - ``leading_dimension_of``
     - :attr:`~nvmath.device.Matmul.leading_dimension`
     - ✅
     - Returns :class:`~nvmath.device.LeadingDimension` named tuple.
   * - ``sm_of``
     - :attr:`~nvmath.device.Matmul.sm`
     - ✅
     - Returns :class:`~nvmath.device.ComputeCapability`.
   * - ``is_blas``
     - N/A
     - ❌
     - Unnecessary in Python. The :class:`~nvmath.device.Matmul` class acts as
       the guaranteed descriptor.
   * - ``is_blas_execution``
     - N/A
     - ❌
     - The execution state is handled internally/implicitly.
   * - ``is_complete_blas``
     - N/A
     - ❌
     - Construction of :class:`~nvmath.device.Matmul` inherently validates completeness.
   * - ``is_complete_blas_execution``
     - N/A
     - ❌
     - Same as above.

2. Execution Traits (Block Traits)
----------------------------------
These traits describe execution configuration when using ``Block()`` operators.

.. list-table::
   :widths: 30 35 10 25
   :header-rows: 1

   * - C++ Trait
     - Python ``nvmath.device`` Implementation
     - Status
     - Notes
   * - ``<a/b/c>_value_type``
     - :attr:`~nvmath.device.Matmul.a_value_type`, :attr:`~nvmath.device.Matmul.b_value_type`, :attr:`~nvmath.device.Matmul.c_value_type`
     - ✅
     - Returns the NumPy compute data type for A, B, and C.
   * - ``<a/b/c>_dim``
     - :attr:`~nvmath.device.Matmul.a_dim`, :attr:`~nvmath.device.Matmul.b_dim`, :attr:`~nvmath.device.Matmul.c_dim`
     - ✅
     - Returns the dimensions as ``(rows, columns)`` tuples.
   * - ``ld<a/b/c>``
     - :attr:`~nvmath.device.Matmul.leading_dimension`
     - ✅
     - Exposed as part of the :class:`~nvmath.device.LeadingDimension` tuple.
   * - ``<a/b/c>_alignment``
     - :attr:`~nvmath.device.Matmul.alignment`
     - ✅
     - Exposed as part of the :class:`~nvmath.device.Alignment` tuple.
   * - ``<a/b/c>_size``
     - :attr:`~nvmath.device.Matmul.a_size`, :attr:`~nvmath.device.Matmul.b_size`, :attr:`~nvmath.device.Matmul.c_size`
     - ✅
     - Number of elements in matrices, inclusive of padding.
   * - ``block_dim``
     - :attr:`~nvmath.device.Matmul.block_dim`
     - ✅
     - Returns :class:`~nvmath.device.Dim3` representing CUDA block dimensions.
   * - ``suggested_block_dim``
     - N/A
     - ✅
     - Automatically calculated and used if ``block_dim="suggested"`` is passed
       during :class:`~nvmath.device.Matmul` initialization.
   * - ``max_threads_per_block``
     - :attr:`~nvmath.device.Matmul.max_threads_per_block`
     - ✅
     - Calculated as ``x * y * z`` threads.

3. Other Traits
---------------
Helper traits regarding hardware support and performance suggestions.

.. list-table::
   :widths: 30 35 10 25
   :header-rows: 1

   * - C++ Trait
     - Python ``nvmath.device`` Implementation
     - Status
     - Notes
   * - ``is_supported_smem_restrict``
     - N/A
     - ❌
     - Not currently implemented or exposed to the user.
   * - ``is_supported_rmem_restrict``
     - N/A
     - ❌
     - Not currently implemented or exposed to the user.
   * - ``suggested_leading_dimension_of``
     - N/A
     - ✅
     - Automatically calculated and used if ``leading_dimension="suggested"``
       is passed during :class:`~nvmath.device.Matmul` initialization.
   * - ``suggested_alignment_of``
     - N/A
     - ❌
     - Not explicitly implemented (although the backend imports
       ``MAX_ALIGNMENT``, there is no trait method returning the suggested
       tuple for A, B, C).

.. _device-api-cublas-reference:

API Reference
=============

.. currentmodule:: nvmath.device

.. autosummary::
   :toctree: generated/

   Matmul
   matmul
   make_tensor
   axpby
   copy
   copy_fragment
   clear
   copy_wait

   OpaqueTensor
   Layout

   Accumulator

   DevicePipeline
   TilePipeline

   SharedStorageCalc

.. autosummary::
   :toctree: generated/
   :template: namedtuple

   LeadingDimension
   TransposeMode
   Precision
   Arrangement
   Alignment
