nvmath-python Release Notes
***************************

nvmath-python v0.2.0
====================

Beta2 release.

* CPU execution space support for FFT libraries that conform to FFTW3 API (e.g. MKL, NVPL).
* Support for prolog and epilog callback for FFT, written in Python.
* New device APIs for random number generation.
* Notebooks to illustrate use of advanced matrix multiplication APIs.
* Introduced hypothesis-based tests for host APIs.
* Reduced Python overhead in `execute` methods.

Bugs Fixed
----------

* Matrix multiplication may fail with "illegal memory access" for K=1 with DRELU and DGELU epilogs.

Packaging
---------

* Added support for NumPy 2.
* Removed Python 3.9 support.
* Patching changes and pynvjitlink version.

Known issues
------------

* When ``compute_type`` argument of :class:`nvmath.linalg.advanced.Matmul` is set to
  ``COMPUTE_16F``, an incompatible default for ``scale_type`` is chosen, resulting in
  incorrect results for CTKs older than 12.6 and an error for CTK 12.6 and newer.
  As a workaround we recommend setting both ``compute_type`` and ``scale_type`` in a
  compatible manner according to `supported data types table <https://docs.nvidia.com/cuda/cublas/#cublasltmatmul>`_.

nvmath-python v0.1.0
====================

Initial beta release, with single-GPU support only.

* FFT APIs based on cuFFT.
* Specialized matrix multiplication APIs based on cuBLASLt.
* Device APIs for FFT and matrix multiplication based on the MathDx libraries.

The required and optional dependencies are summarized in the :ref:`cheatsheet <cheatsheet>`.

*Limitations:*

* Many matrix multiplication epilogs require CTK 11.5+, and a few require CTK 11.8+.
  Please refer to `cuBLAS Release Notes <https://docs.nvidia.com/cuda/archive/11.8.0/cuda-toolkit-release-notes/index.html#title-cublas-library>`_ for more details.

Disclaimer
==========

nvmath-python is in a Beta state. Beta products may not be fully functional, may contain errors or design flaws, and may be changed at any time without notice. We appreciate your feedback to improve and iterate on our Beta products.
