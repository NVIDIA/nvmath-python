**********************
Fast Fourier Transform
**********************

.. _fft-overview:

Overview
========

The Fast Fourier Transform (FFT) module :mod:`nvmath.fft` in nvmath-python leverages the NVIDIA cuFFT library and provides a powerful suite of APIs 
that can be directly called from the host to efficiently perform discrete Fourier Transformations. Both stateless function-form APIs and stateful class-form APIs are 
provided to support a spectrum of N-dimensional FFT operations. These include forward and inverse transformations, as well as complex-to-complex (C2C), complex-to-real (C2R), 
and real-to-complex (R2C) transforms:

- N-dimensional forward C2C FFT transform by :func:`nvmath.fft.fft`.
- N-dimensional inverse C2C FFT transform by :func:`nvmath.fft.ifft`.
- N-dimensional forward R2C FFT transform by :func:`nvmath.fft.rfft`.
- N-dimensional inverse C2R FFT transform by :func:`nvmath.fft.irfft`.
- All types of N-dimensional FFT by stateful :class:`nvmath.fft.FFT`.

Furthermore, the :class:`nvmath.fft.FFT` class includes utility APIs designed to help users cache FFT plans, facilitating the efficient execution of repeated calculations across various computational tasks
(see :meth:`~nvmath.fft.FFT.create_key`).

.. note::
    The API :func:`~nvmath.fft.fft` and related function-form APIs perform **N-D FFT** operations, similar to :func:`numpy.fft.fftn`. There are no special 1-D (:func:`numpy.fft.fft`) or 2-D FFT (:func:`numpy.fft.fft2`) APIs.
    This not only reduces the API surface, but also avoids the potential for incorrect use because the number of batch dimensions is :math:`N - 1` for :func:`numpy.fft.fft` and :math:`N - 2` for :func:`numpy.fft.fft2`, where :math:`N` is the operand dimension.

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
   UnsupportedLayoutError
   FFT

   :template: dataclass.rst   

   FFTOptions
   FFTDirection
   DeviceCallable
