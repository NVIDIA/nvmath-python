*********************
Sparse Linear Algebra
*********************

.. _sparse-overview:

Overview
========

The sparse linear algebra module :mod:`nvmath.sparse` in nvmath-python leverages various
NVIDIA math libraries to support sparse linear algebra computations. As of the current Beta
release, we offer the specialized sparse direct solver API based on the `cuDSS library
<https://docs.nvidia.com/cuda/cudss/>`_.

.. _sparse-api-reference:

API Reference
=============

.. module:: nvmath.sparse

Generic Linear Algebra APIs (:mod:`nvmath.sparse`)
--------------------------------------------------

Generic APIs will be available in a later release.

.. module:: nvmath.sparse.advanced

Specialized Linear Algebra APIs (:mod:`nvmath.sparse.advanced`)
---------------------------------------------------------------

.. autosummary::
   :toctree: generated/

   direct_solver
   DirectSolver
   DirectSolverFactorizationConfig
   DirectSolverFactorizationInfo
   DirectSolverPlanConfig
   DirectSolverPlanInfo
   DirectSolverSolutionConfig
   memory_estimates_dtype

   :template: dataclass.rst

   DirectSolverAlgType
   DirectSolverMatrixType
   DirectSolverMatrixViewType
   DirectSolverOptions
   ExecutionCUDA
   ExecutionHybrid
   HybridMemoryModeOptions
