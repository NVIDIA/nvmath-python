
************************************
cuBLASDx APIs (:mod:`nvmath.device`)
************************************

.. _device-api-cublas-overview:

Overview
========

These APIs offer integration with the NVIDIA cuBLASDx library.
Detailed documentation of cuBLASDx can be found in the
`cuBLASDx documentation <https://docs.nvidia.com/cuda/cublasdx/0.4.1>`_.

.. note::

   The :class:`~nvmath.device.Matmul` device API in module
   :mod:`nvmath.device` currently supports cuBLASDx 0.4.1, also available
   as part of MathDx 25.06.

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

   Partition
   Partitioner

   SharedStorageCalc

   :template: namedtuple.rst

   LeadingDimension
   TransposeMode
