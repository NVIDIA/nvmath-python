
************************************
cuBLASDx APIs (:mod:`nvmath.device`)
************************************

.. _device-api-cublas-overview:

Overview
========

These APIs offer integration with the NVIDIA cuBLASDx library.
Detailed documentation of cuBLASDx can be found in the
`cuBLASDx documentation <https://docs.nvidia.com/cuda/cublasdx/0.1.1>`_.

.. note::

   The :class:`~nvmath.device.matmul` device API in module
   :mod:`nvmath.device` currently supports cuBLASDx 0.1.1, also available
   as part of MathDx 24.04.

.. _device-api-cublas-reference:

API Reference
=============

.. currentmodule:: nvmath.device

.. autosummary::
   :toctree: generated/

   matmul
   BlasOptions

   :template: namedtuple.rst

   LeadingDimension
   TransposeMode
