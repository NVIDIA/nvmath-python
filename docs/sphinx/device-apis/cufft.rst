
***********************************
cuFFTDx APIs (:mod:`nvmath.device`)
***********************************

.. _device-api-cufft-overview:

Overview
========

These APIs offer integration with the NVIDIA cuFFTDx library.
Detailed documentation of cuBLASDx can be found in the
`cuFFTDx documentation <https://docs.nvidia.com/cuda/cufftdx/1.2.0>`_.

.. note::

   The :class:`~nvmath.device.fft` device APIs in module
   :mod:`nvmath.device` currently support cuFFTDx 1.2.0, also available
   as part of MathDx 24.04. All functionalities from the C++ library are supported with
   the exception of cuFFTDx C++ APIs with a workspace argument, which are currently not
   available in nvmath-python.

.. _device-api-cufft-reference:

API Reference
=============

.. currentmodule:: nvmath.device

.. autosummary::
   :toctree: generated/

   fft
   FFTOptions
