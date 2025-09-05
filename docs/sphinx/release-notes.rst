nvmath-python Release Notes
***************************

nvmath-python v0.6.0
====================

Beta6 release.

* This will be the last release to support CUDA 11.
* Added support for distributed R2C/C2R FFTs, along with support for non-uniform partition
  sizes across PEs.
* The ``distribution`` option for distributed FFTs is now a required keyword-only argument.
* To enable making CuPy an optional dependency, an internal ``NDBuffer`` datastructure
  was introduced that facilitates copying tensors across memory spaces and layouts. Users
  may notice a one-time latency for each unique layout since the copy kernel is JIT compiled
  and cached.
* Replaced internal logic with
  `cuda-pathfinder <https://github.com/NVIDIA/cuda-python/tree/main/cuda_pathfinder>`_ for
  locating libraries and components.

Bugs Fixed
----------

* The :meth:`nvmath.linalg.advanced.Matmul.autotune` method in the advanced Matmul APIs may
  not have selected the best kernel, since the L2-cache wasn't cleared.
* The return status of an internal call to a CUDA API  wasn't checked, resulting
  in a misleading error regarding memory limit.
* Fixed a use-after-free issue with the batched direct sparse solver.
* Fixed a deadlock that may occur in certain circumstances during distributed FFT.
* Added appropriate constraints for cuda-bindings based on the CTK version.
* Fixed missing logging messages when a Python logger was not created with ``force=True``.


Known Issues
------------

* The minimum supported versions for CuPy and PyTorch are out-of-date and will be increased
  in the next release.
* An internal symbol table used when loading symbols from libraries needs to be made
  thread-safe. This will be done in the next release.

nvmath-python v0.5.0
====================

Beta5 release.

* New single-GPU and hybrid CPU-GPU sparse direct solver APIs supporting SciPy, CuPy,
  and PyTorch.

Known Issues
------------

* Python overhead for matmul host-APIs has increased since v0.3.0 by 21 microseconds on
  average. We are investigating.

* CUDA 12.8.0, 12.8.1 and 12.9.0 have been known to miscompile cuBLASDx in some
  rare slow-path cases (see
  `cuBLASDx <https://docs.nvidia.com/cuda/cublasdx/0.4.0/index.html>`_ for more
  details).

nvmath-python v0.4.0
====================

Beta4 release.

* New distributed FFT APIs to run on multi-node/multi-GPU systems.
* New device matrix multiplication tensor API to enable advanced techniques such as
  cooperative copy and floating-point emulation using integer tensor cores.
* Transition from CuPy to `cuda-python (cuda.core)
  <https://nvidia.github.io/cuda-python/cuda-core/latest/>`_ for core CUDA constructs.

Bugs Fixed
----------

* FFT prolog or epilog fails to compile on ``SM >= 100``.

Known Issues
------------

* Python overhead for matmul host-APIs has increased since v0.3.0 by 21 microseconds on
  average. We are investigating.

nvmath-python v0.3.0
====================

Beta3 release.

* FP8 and MXFP8 support for the advanced matrix multiplication API.
* Notebook to illustrate use of FP8 and MXFP8 in the advanced matrix multiplication API.
* Added bindings for new APIs introduced in CTK version 12.8.

Bugs Fixed
----------

* The advanced matrix multiplication API may return an incorrect result when a bias vector
  is used along with 1-D A and C operands.

API Changes
-----------

* The ``last_axis_size`` option in :class:`nvmath.fft.FFTOptions` is removed in favor of
  ``last_axis_parity`` to better reflect its semantics.

nvmath-python v0.2.1
====================

Beta2 update 1 with improved diagnostics, testing enhancements, and bug fixes.

* New tests for batched epilogs and autotuning with epilogs for the advanced matrix
  multiplication APIs.
* Added more hypothesis-based tests for host APIs.
* Improved algorithm for detecting overlapping memory operands for certain sliced tensors,
  thereby supporting such layouts for FFTs.
* Added bindings for new APIs introduced in CTK versions 12.5 and 12.6.
* Further coding style fixes toward meeting PEP8 recommendations.
* Clarified batched semantics for matrix multiplication epilogs in the documentation.
* Code snippets in API docstrings are now tested.

Bugs Fixed
----------

* C2R FFT may fail with "illegal memory access" on sliced tensors.
* Improved diagnostics to detect incompatible combinations of scale and compute types for
  matrix multiplication, that previously may have resulted in incorrect results.
* Matrix multiplication provided incorrect results when operand A is a vector (number of
  dimensions=1).

API Changes
-----------

* The ``last_axis_size`` option in :class:`nvmath.fft.FFTOptions` is now deprecated in favor
  of ``last_axis_parity`` to better reflect its semantics.

.. note::

   Deprecated APIs will be removed in the next release.

nvmath-python v0.2.0
====================

Beta2 release.

* CPU execution space support for FFT libraries that conform to FFTW3 API (for example MKL,
  NVPL).
* Support for prolog and epilog callback for FFT, written in Python.
* New device APIs for random number generation.
* Notebooks to illustrate use of advanced matrix multiplication APIs.
* Introduced hypothesis-based tests for host APIs.
* Reduced Python overhead in ``execute`` methods.

Bugs Fixed
----------

* Matrix multiplication may fail with "illegal memory access" for K=1 with DRELU and DGELU
  epilogs.

Packaging
---------

* Added support for NumPy 2.
* Removed Python 3.9 support.
* Patching changes and pynvjitlink version.

Known issues
------------

* When ``compute_type`` argument of :class:`nvmath.linalg.advanced.Matmul` is set to
  ``COMPUTE_16F``, an incompatible default for ``scale_type`` is chosen, resulting in
  incorrect results for CTKs older than 12.6 and an error for CTK 12.6 and newer. As a
  workaround we recommend setting both ``compute_type`` and ``scale_type`` in a compatible
  manner according to `supported data types table
  <https://docs.nvidia.com/cuda/cublas/#cublasltmatmul>`_.

nvmath-python v0.1.0
====================

Initial beta release, with single-GPU support only.

* FFT APIs based on cuFFT.
* Specialized matrix multiplication APIs based on cuBLASLt.
* Device APIs for FFT and matrix multiplication based on the MathDx libraries.

The required and optional dependencies are summarized in the :ref:`cheatsheet <cheatsheet>`.

*Limitations:*

* Many matrix multiplication epilogs require CTK 11.5+, and a few require CTK 11.8+.
  Refer to `cuBLAS Release Notes
  <https://docs.nvidia.com/cuda/archive/11.8.0/cuda-toolkit-release-notes/index.html
  #title-cublas-library>`_
  for more details.

Disclaimer
==========

nvmath-python is in a Beta state. Beta products may not be fully functional, may contain
errors or design flaws, and may be changed at any time without notice. We appreciate your
feedback to improve and iterate on our Beta products.
