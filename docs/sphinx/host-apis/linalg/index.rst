**************
Linear Algebra
**************

.. _linalg-overview:

Overview
========

The Linear Algebra module :mod:`nvmath.linalg` in nvmath-python leverages various NVIDIA
math libraries to support multiple linear algebra computations. As of the initial Beta
release, we offer the specialized matrix multiplication API based on the cuBLASLt library.

.. _linalg-api-reference:

API Reference
=============

.. module:: nvmath.linalg

Generic Linear Algebra APIs (:mod:`nvmath.linalg`)
--------------------------------------------------

Generic APIs will be available in a later release.

.. module:: nvmath.linalg.advanced

Specialized Linear Algebra APIs (:mod:`nvmath.linalg.advanced`)
---------------------------------------------------------------

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
