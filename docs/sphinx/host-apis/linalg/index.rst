**************
Linear Algebra
**************

.. _linalg-overview:

Overview
========

The Linear Algebra module :mod:`nvmath.linalg` in nvmath-python leverages various NVIDIA
math libraries to support dense [#]_ linear algebra computations. As of version 0.7.0, we
offer both a generic matrix multiplication API based on the cuBLAS and NVPL libraries and a
specialized matrix multiplication API (:mod:`nvmath.linalg.advanced`) based on the cuBLASLt
library. See :ref:`Generic and Specialized APIs <generic specialized>` for motivation.

At a high-level, if your use case is predominantly GEMM and requires particular flexibility
in matrix data layouts, input and/or compute types, and also in choosing the algorithmic
implementation, look at the specialized APIs. Otherwise, look at the generic APIs.

.. _linalg-api-reference:

API Reference
=============

.. module:: nvmath.linalg

Generic Linear Algebra APIs (:mod:`nvmath.linalg`)
--------------------------------------------------

The generic linear algebra module includes matrix multiplication APIs which accept
structured matrices as input, but do not allow for control over computational precision or
algorithm selection and planning.

.. autosummary::
   :toctree: generated/

   matmul
   Matmul
   matrix_qualifiers_dtype
   DiagonalMatrixQualifier
   GeneralMatrixQualifier
   HermitianMatrixQualifier
   InvalidMatmulState
   SymmetricMatrixQualifier
   TriangularMatrixQualifier

   :template: dataclass.rst

   ExecutionCPU
   ExecutionCUDA
   MatmulOptions

.. module:: nvmath.linalg.advanced

Specialized Linear Algebra APIs (:mod:`nvmath.linalg.advanced`)
---------------------------------------------------------------

The specialized linear algebra module includes a matrix multiplication API which only
accepts general matrices, but provides extra functionality such as epilog functions, more
options and controls over computational precision, and control over algorithm selection and
planning.

.. autosummary::
   :toctree: generated/

   matmul
   matrix_qualifiers_dtype
   Algorithm
   Matmul
   MatmulComputeType
   MatmulEpilog
   MatmulInnerShape
   MatmulNumericalImplFlags
   MatmulReductionScheme

   :template: dataclass.rst

   MatmulEpilogPreferences
   MatmulOptions
   MatmulPlanPreferences
   MatmulQuantizationScales

Helpers
^^^^^^^

The Specialized Linear Algebra helpers module :mod:`nvmath.linalg.advanced.helpers`
provides helper functions to facilitate working with some of the complex features of
:mod:`nvmath.linalg.advanced` module.

Matmul helpers (:mod:`nvmath.linalg.advanced.helpers.matmul`)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

.. module:: nvmath.linalg.advanced.helpers.matmul

.. autosummary::
   :toctree: generated/

   create_mxfp8_scale
   invert_mxfp8_scale
   apply_mxfp8_scale
   get_mxfp8_scale_offset

.. rubric:: Footnotes

.. [#] See :ref:`Sparse Linear Algebra <sparse-overview>` for sparse operations.
