nvmath-python Release Notes
***************************

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

