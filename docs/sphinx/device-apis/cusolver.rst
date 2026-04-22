
**************************************
cuSOLVERDx APIs (:mod:`nvmath.device`)
**************************************

.. _device-api-cusolver-overview:

Overview
========

These APIs offer integration with the NVIDIA cuSOLVERDx library, providing device-level
dense matrix factorization and linear solver functions that execute directly within CUDA
kernels. Detailed documentation of cuSOLVERDx can be found in the
:cusolverdx_doc:`cuSOLVERDx documentation <index.html>`.

Architecture
------------

The nvmath-python cuSOLVERDx API is organized into two layers:

**Solver Class: 1:1 C++ Binding**

The :class:`~nvmath.device.Solver` class provides a direct 1:1 binding to the cuSOLVERDx
C++ API, offering complete control over all library features and parameters.
This interface allows users to access the flexibility of cuSOLVERDx with Python
syntax, maintaining compatibility with the underlying C++ library semantics.

**Pythonic Adapters**

The API provides several adapter classes to simplify common use cases:

* :class:`~nvmath.device.CholeskySolver` - Specialized for Cholesky factorization and
  solve operations on symmetric positive definite matrices
* :class:`~nvmath.device.LUSolver` - Specialized for LU factorization without pivoting
  and solve operations on general matrices
* :class:`~nvmath.device.LUPivotSolver` - Specialized for LU factorization with partial
  pivoting and solve operations on general matrices
* :class:`~nvmath.device.QRFactorize` - Specialized for QR factorization
  of general matrices
* :class:`~nvmath.device.LQFactorize` - Specialized for LQ factorization
  of general matrices
* :class:`~nvmath.device.QRMultiply` - Specialized for multiplication by the unitary
  matrix Q from a QR factorization
* :class:`~nvmath.device.LQMultiply` - Specialized for multiplication by the unitary
  matrix Q from an LQ factorization
* :class:`~nvmath.device.TriangularSolver` - Specialized for solve
  operations on triangular matrices
* :class:`~nvmath.device.LeastSquaresSolver` - Specialized for solving
  overdetermined or underdetermined least squares problems

.. _device-api-cusolver-supported-functions:

Supported Functions
-------------------

The following table lists all supported cuSOLVERDx
functions with their corresponding Python adapters:

.. list-table::
   :header-rows: 1
   :widths: 25 40 35

   * - Function
     - Description
     - Adapter Class
   * - :cusolverdx_doc:`potrf <get_started/functions/potrf.html>`
     - Cholesky factorization
     - :class:`~nvmath.device.CholeskySolver`
   * - :cusolverdx_doc:`potrs <get_started/functions/potrs.html>`
     - Linear system solve after Cholesky factorization
     - :class:`~nvmath.device.CholeskySolver`
   * - :cusolverdx_doc:`posv <get_started/functions/posv.html>`
     - Fused Cholesky factorization with solve
     - :class:`~nvmath.device.Solver`
   * - :cusolverdx_doc:`getrf_no_pivot <get_started/functions/getrf.html>`
     - LU factorization without pivoting
     - :class:`~nvmath.device.LUSolver`
   * - :cusolverdx_doc:`getrs_no_pivot <get_started/functions/getrs.html>`
     - LU solve without pivoting
     - :class:`~nvmath.device.LUSolver`
   * - :cusolverdx_doc:`gesv_no_pivot <get_started/functions/gesv.html>`
     - Fused LU without pivoting factorization with solve
     - :class:`~nvmath.device.Solver`
   * - :cusolverdx_doc:`getrf_partial_pivot <get_started/functions/getrf.html>`
     - LU factorization with partial pivoting
     - :class:`~nvmath.device.LUPivotSolver`
   * - :cusolverdx_doc:`getrs_partial_pivot <get_started/functions/getrs.html>`
     - LU solve with partial pivoting
     - :class:`~nvmath.device.LUPivotSolver`
   * - :cusolverdx_doc:`gesv_partial_pivot <get_started/functions/gesv.html>`
     - Fused LU with partial pivoting factorization with solve
     - :class:`~nvmath.device.Solver`
   * - :cusolverdx_doc:`geqrf <get_started/functions/geqrf.html>`
     - QR factorization
     - :class:`~nvmath.device.QRFactorize`
   * - :cusolverdx_doc:`unmqr <get_started/functions/unmqr.html>`
     - Multiplication of Q from QR factorization
     - :class:`~nvmath.device.QRMultiply`
   * - :cusolverdx_doc:`ungqr <get_started/functions/ungqr.html>` :sup:`[0.3.2]`
     - Unitary matrix Q generation from QR factorization
     - :class:`~nvmath.device.Solver`
   * - :cusolverdx_doc:`gelqf <get_started/functions/gelqf.html>`
     - LQ factorization
     - :class:`~nvmath.device.LQFactorize`
   * - :cusolverdx_doc:`unmlq <get_started/functions/unmlq.html>`
     - Multiplication of Q from LQ factorization
     - :class:`~nvmath.device.LQMultiply`
   * - :cusolverdx_doc:`unglq <get_started/functions/unglq.html>` :sup:`[0.3.2]`
     - Unitary matrix Q generation from LQ factorization
     - :class:`~nvmath.device.Solver`
   * - :cusolverdx_doc:`trsm <get_started/functions/trsm.html>`
     - Triangular matrix-matrix solve
     - :class:`~nvmath.device.TriangularSolver`
   * - :cusolverdx_doc:`gels <get_started/functions/gels.html>`
     - Overdetermined or underdetermined least squares problems
     - :class:`~nvmath.device.LeastSquaresSolver`
   * - :cusolverdx_doc:`gtsv_no_pivot <get_started/functions/gtsv.html>` :sup:`[0.3.2]`
     - General tridiagonal matrix solve without pivoting
     - :class:`~nvmath.device.Solver`
   * - :cusolverdx_doc:`htev <get_started/functions/htev.html>` :sup:`[0.3.2]`
     - Eigenvalue solver for Hermitian tridiagonal matrices
     - :class:`~nvmath.device.Solver`
   * - :cusolverdx_doc:`heev <get_started/functions/heev.html>` :sup:`[0.3.2]`
     - Eigenvalue solver for Hermitian dense matrices
     - :class:`~nvmath.device.Solver`
   * - :cusolverdx_doc:`gesvdj <get_started/functions/gesvdj.html>` :sup:`[0.3.2]`
     - Singular value decomposition for general dense matrices
     - :class:`~nvmath.device.Solver`

Version Support
---------------

.. note::

    All functionality up to cuSOLVERDx 0.2.1 is fully supported. Functionality from
    cuSOLVERDx |cusolverdx_version| or later are partially supported.

    Functions marked with :sup:`[0.3.2]` requires libmathdx 0.3.2 or later and are
    only accessible through the base :class:`~nvmath.device.Solver` class.

.. _device-api-cusolver-reference:

API Reference
=============

.. currentmodule:: nvmath.device

.. autosummary::
   :toctree: generated/

    Solver
    CholeskySolver
    LUSolver
    LUPivotSolver
    QRFactorize
    LQFactorize
    QRMultiply
    LQMultiply
    TriangularSolver
    LeastSquaresSolver
