Getting Started
***************

Install nvmath-python
=====================

nvmath-python, like most modern Python packages, provides pre-built binaries (wheels and later conda packages) to the end users.
The full source code is hosted in the `NVIDIA/nvmath-python <https://github.com/NVIDIA/nvmath-python>`_ repository.

In terms of CUDA Toolkit (CTK) choices, nvmath-python is designed and implemented to allow building and running against 1. ``pip``-wheel, 2. ``conda``, or 3. system installation of CTK. Having a full CTK installation at either build- or run- time is not necessary; just a small fraction as explained below is enough.

Host & device APIs (see :ref:`nvmath overview`) have different run-time dependencies and requirements. Even among
host APIs the needed underlying libraries are different (for example, :func:`~nvmath.fft.fft` on GPUs only needs cuFFT and not cuBLAS). Libraries
are loaded when only needed. Therefore, nvmath-python is designed to have most of its dependencies *optional*, but provides
convenient installation commands for users to quickly spin up a working Python environment. 

The :ref:`cheatsheet <cheatsheet>` below captures nvmath-python's required/optional, build-/run- time dependencies.
Using the installation commands from the sections below should support most of your needs.


.. _install from pypi:

Install from PyPI
-----------------

The pre-built wheels can be ``pip``-installed from the public PyPI. There are several optional dependencies expressible in the standard "extras" bracket notation. The following assumes that **CTK components are also installed via pip** (so no extra step from users is needed; the dependencies are pulled via extras).

.. list-table::
   :widths: 25 50
   :header-rows: 1

   * - Command
     - Description
   * - ``pip install nvmath-python[cu11]``
     - Install nvmath-python along with all CUDA 11 optional
       dependencies (wheels for cuBLAS/cuFFT/... and CuPy) to support
       nvmath host APIs.

       **Note**: Currently this does not support linux-aarch64.
   * - ``pip install nvmath-python[cu12]``
     - Install nvmath-python along with all CUDA 12 optional
       dependencies (wheels for cuBLAS/cuFFT/... and CuPy) to support
       nvmath host APIs.
   * - ``pip install nvmath-python[cu12,dx]``
     - Install nvmath-python along with all CUDA 12 optional
       dependencies (wheels for cuBLAS/cuFFT/..., CuPy, Numba,
       pynvjitlink, ...) to support nvmath host & device APIs (which
       only supports CUDA 12) [8]_.

       **Note**: Currently this does not support linux-aarch64.

The options below are for adventurous users who want to manage most of the dependencies themselves. The following assumes that **system CTK is installed**.

.. list-table::
   :widths: 25 50
   :header-rows: 1

   * - Command
     - Description
   * - ``pip install nvmath-python cupy-cuda11x``
     - Install nvmath-python along with CuPy for CUDA 11 to support
       nvmath host APIs.

       **Note**: ``LD_LIBRARY_PATH`` should be set
       properly to include CUDA libraries.
   * - ``pip install nvmath-python cupy-cuda12x``
     - Install nvmath-python along with CuPy for CUDA 12 to support
       nvmath host APIs.

       **Note**: ``LD_LIBRARY_PATH`` should be set
       properly to include CUDA libraries.
   * - ``pip install nvmath-python[dx] cupy-cuda12x``
     - Install nvmath-python along with CuPy for CUDA 12 to support
       nvmath host & device APIs.

       **Note**:

       1. ``LD_LIBRARY_PATH`` should be set properly to include CUDA libraries.
       2. For using :mod:`nvmath.device` APIs, ``CUDA_HOME`` (or ``CUDA_PATH``) should be
          set to point to the local CTK.

For system admins or ninja users, ``pip install nvmath-python`` would be a bare minimal installation (very lightweight). This allows fully explicit control of all dependencies.


Install from conda
------------------

**Coming soon!** conda packages will be released on the `conda-forge <https://conda-forge.org>`_ channel shortly after the beta 1 (v0.1.0) release. 


Build from source
-----------------

Once you clone the repository and go into the root directory, you can build the project from source. There are several ways to build it since we need some CUDA headers at build time.

.. list-table::
   :widths: 25 50
   :header-rows: 1

   * - Command
     - Description
   * - ``pip install -v .``             
     - Set up a build isolation (as per `PEP 517 <https://peps.python.org/pep-0517/>`_),
       install CUDA wheels and other build-time dependencies to the
       build environment, build the project, and install it to the
       current user environment together with the run-time
       dependencies.

       **Note**: in this case we get CUDA headers by installing pip wheels to the isolated
       build environment.
   * - ``CUDA_PATH=/path/to/your/cuda/installation pip install --no-build-isolation -v .``
     - Skip creating a build isolation (it'd use CUDA headers from ``$CUDA_PATH/include``
       instead), build the project, and install it to the current
       user environment together with the run-time dependencies. One can use:

       - conda: After installing CUDA 12 conda packages, set the environment variable ``CUDA_PATH``

         * linux-64: ``CUDA_PATH=$CONDA_PREFIX/targets/x86_64-linux/``
         * linux-aarch64: ``CUDA_PATH=$CONDA_PREFIX/targets/sbsa-linux/``
         * win-64: ``CUDA_PATH=$CONDA_PREFIX\Library``

       - local CTK: Just set ``CUDA_PATH`` to the local CTK location.

**Notes**:

- If you add the "extras" notation after the dot ``.``, e.g., ``.[cu11]``, ``.[cu12,dx]``, ..., it has the same meaning
  as explained in the :ref:`previous section <install from pypi>`.
- If you don't want the run-time dependencies to be automatically handled, add ``--no-deps`` after the ``pip install``
  command above; in this case, however, it's your responsibility to make sure that all the run-time requirements are met.
- By replacing ``install`` by ``wheel``, a wheel can be built targeting the current OS and CPython version.
- If you want inplace/editable install, add the ``-e`` flag to the command above (before the dot ``.``).
  This is suitable for local development. However, our wheels rely on *non-editable builds* so that the RPATH
  hack can kick in. DO NOT pass the ``-e`` flag when building wheels!
- All optional run-time dependencies as listed below need to be manually installed.


.. _cheatsheet:

Cheatsheet
----------

Below we provide a summary of requirements to support all nvmath-python functionalities.
A dependency is *required* unless stated otherwise.

.. list-table::
   :widths: 25 25 25 25
   :header-rows: 1

   * - 
     - When Building
     - When Running - host APIs
     - When Running - device APIs
   * - CPU architecture & OS
     - linux-64, linux-aarch64, win-64
     - linux-64, linux-aarch64, win-64
     - linux-64, linux-aarch64 [1]_

   * - GPU hardware
     - 
     - | All hardware supported by the underlying CUDA Toolkit [5]_
       |
       | *Optional*: needed if the execution space is GPU.
     - Compute Capability 7.0+ (Volta and above)
   * - CUDA driver [2]_
     - 
     - | 450.80.02+ (Linux) / 450.39+ (Windows) with CUDA 11.x
       |
       | 525.60.13+ (Linux) / 527.41+ (Windows) with CUDA 12.x
       |
       | *Optional*: needed if the execution space is GPU or for loading any CUDA library.
     - 525.60.13+ (Linux) with CUDA 12.x
   * - Python
     - 3.9-3.12
     - 3.9-3.12
     - 3.9-3.11 [3]_
   * - pip
     - 22.3.1+
     - 
     - 
   * - setuptools
     - >=61.0.0
     - 
     - 
   * - wheel
     - >=0.34.0
     - 
     - 
   * - Cython
     - >=0.29.22,<3
     - 
     - 
   * - CUDA
     - | CUDA 11.x or 12.x
       | (only need headers from NVCC & CUDART [6]_)
     - | CUDA 11.2-11.8 or 12.x
       |
       | *Optional*: depending on the math operations in use
     - | CUDA 12.0-12.3 [7]_
       | (NVRTC, NVVM, CCCL [8]_, CUDART)
   * - NumPy
     - 
     - v1.21+
     - v1.21+
   * - | CuPy
       | (see `CuPy installation guide <https://docs.cupy.dev/en/stable/install.html>`_)
     - 
     - v10.0.0+ [4]_
     - 
   * - | PyTorch
       | (see `PyTorch installation guide <https://pytorch.org/get-started/locally/>`_)
     - 
     - v1.10+ (optional)
     - 
   * - MathDx (cuBLASDx, cuFFTDx, ...)
     -
     -
     - 24.04.0
   * - Numba
     - 
     - 
     - 0.59.1
   * - pynvjitlink
     - 
     - 
     - 0.14.1


Test Configuration
------------------

nvmath-python is tested in the following environments:

.. TODO:
   Update me

.. list-table::
   :widths: 50 50

   * - CUDA
     - 11.8, 12.4
   * - Driver
     - R450, R520, R525, R550
   * - Python
     - 3.9, 3.10, 3.11, 3.12
   * - CPU architecture
     - x86_64, ARM64
   * - Operating system
     - RHEL9, Ubuntu 22.04, Windows11


Run nvmath-python
=================

As mentioned earlier, nvmath-python can be run against all ways of CUDA installation, including wheels, conda packages, and local CTK. As a result, there is detection logic to discover shared libraries (for host APIs) and headers (for device APIs to do JIT compilation).

Shared libraries
----------------

- pip wheels: Will be auto-discovered if installed
- conda packages: Will be auto-discovered if installed, after wheel
- local CTK: On Linux one needs to ensure the DSOs are discoverable by the dynamic linker, say by setting ``LD_LIBRARY_PATH`` or updating system search paths to include the DSO locations.


Headers 
-------

This includes libraries such as CCCL and MathDx.

- pip wheels: Will be auto-discovered if installed
- conda packages: Will be auto-discovered if installed, after wheel
- local CTK: Need to set ``CUDA_HOME`` (or ``CUDA_PATH``) and ``MATHDX_HOME`` (for MathDx headers)


Host APIs
---------

This terminlogy is explained in the :ref:`host api section`.

Examples
........

See the ``examples`` directory in the repo. Currently we have:

- ``examples/fft``
- ``examples/linalg``


Tests
.....

The ``requirements/pip/tests.txt`` file lists dependencies required for ``pip``-controlled environments to run tests. These requirements are installed via the main ``requirements/pip-dev-<name>.txt`` files.


Running functionality tests
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block::

   pytest tests/example_tests tests/nvmath_tests/fft tests/nvmath_tests/linalg

Running performance tests
~~~~~~~~~~~~~~~~~~~~~~~~~

This will currently run two tests for fft and one test for linalg:

.. code-block::

   pytest -v -s -k 'perf' tests/nvmath_tests/fft/ 
   pytest -v -s -k 'perf' tests/nvmath_tests/linalg/ 


Device APIs
-----------

This terminlogy is explained in the :ref:`device api section`.

Examples
........

See the ``examples/device`` directory in the repo.


Tests
.....

Running functionality tests
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block::

   pytest tests/nvmath_tests/device examples/device


Running performance tests
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block::

   pytest -v -s -k 'perf' tests/nvmath_tests/device/


Troubleshooting
===============

For ``pip``-users, there are known limitations (many of which are nicely captured in the
`pypackaging community project <https://pypackaging-native.github.io>`_) in Python packaging tools.
For a complex library such as nvmath-python that interacts with many native libraries, there are user-visible caveats.

1. Be sure that there are no packages with both ``-cu11`` (for CUDA 11) and ``-cu12`` (for CUDA 12) suffices coexisting
   in your Python environment. For example, this is a corrupted environment:

   .. code-block:: bash

      $ pip list
      Package            Version
      ------------------ ---------
      nvidia-cublas-cu11 11.11.3.6
      nvidia-cublas-cu12 12.5.2.13
      pip                24.0
      setuptools         70.0.0
      wheel              0.43.0

   Some times such conflicts could come from a dependency of the libraries that you use, so pay extra attention to what's
   installed.
2. ``pip`` does not attempt to check if the installed packages can actually be run against the installed GPU driver (CUDA GPU
   driver cannot be installed by ``pip``), so make sure your GPU driver is new enough to support the installed ``-cuXX``
   packages [2]_. The driver version can be checked by executing ``nvidia-smi`` and inspecting the ``Driver Version`` field on the
   output table.
3. CuPy installed from ``pip`` currently (as of v13.1.0) only supports conda and system CTK, and not ``pip``-installed CUDA wheels.
   nvmath-python can help CuPy use the CUDA libraries installed to ``site-packages`` (where wheels are installed to) if ``nvmath``
   is imported. As of beta 1 (v0.1.0) the libraries are "soft-loaded" (no error is raised if a library is not installed) when
   ``import nvmath`` happens. This behavior may change in a future release.
4. Numba installed from ``pip`` currently (as of v0.59.1) only supports conda and system CTK, and not ``pip``-installed CUDA wheels.
   nvmath-python can also help Numba use the CUDA compilers installed to ``site-packages`` if ``nvmath`` is imported.  
   Same as above, this behavior may change in a future release.

In general, mixing-and-matching CTK packages from ``pip``, ``conda``, and the system is possible but can be very fragile, so
please understand what you're doing.
The nvmath-python internals are designed to work with everything installed either via ``pip``, ``conda``, or local system
(local CTK, including `tarball extractions <https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#tarball-and-zip-archive-deliverables>`_, are the fallback solution in the detection logic), but mix-n-match makes the detection logic
impossible to get right.

To help you perform an integrity check, the rule of thumb is that every single package should only come from one place (either
``pip``, or ``conda``, or local system). For example, if both ``nvidia-cufft-cu11`` (which is from ``pip``) and ``libcufft`` (from
``conda``) appear in the output of ``conda list``, something is almost certainly wrong. Below is the package name mapping between
``pip`` and ``conda``, with ``XX={11,12}`` denoting CUDA's major version:

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - pip
     - conda
   * - ``nvidia-cuda-nvcc-cuXX``
     - ``cuda-nvcc`` ``cuda-version=XX``
   * - ``nvidia-cuda-nvrtc-cuXX``	
     - ``cuda-nvrtc`` ``cuda-version=XX``
   * - ``nvidia-cuda-runtime-cuXX``	
     - ``cuda-cudart-dev`` ``cuda-version=XX``
   * - ``nvidia-cuda-cccl-cuXX``
     - ``cuda-cccl`` ``cuda-version=XX``
   * - ``pynvjitlink-cuXX``	
     - ``pynvjitlink`` ``cuda-version=XX``
   * - ``nvidia-cublas-cuXX``	
     - ``libcublas`` ``cuda-version=XX``
   * - ``nvidia-cusolver-cuXX``	
     - ``libcusolver`` ``cuda-version=XX``
   * - ``nvidia-cusparse-cuXX``	
     - ``libcusparse`` ``cuda-version=XX``
   * - ``nvidia-cufft-cuXX``	
     - ``libcufft`` ``cuda-version=XX``
   * - ``nvidia-curand-cuXX``	
     - ``libcurand`` ``cuda-version=XX``

Note that system packages by design do not show up in the output of ``conda list`` or ``pip list``. Linux users should check
the installation list from your distro package manager (``apt``, ``yum``, ``dnf``, ...). See also the `Linux Package Manager
Installation Guide <https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#package-manager-installation>`_ for
additional information.


.. rubric:: Footnotes

.. [1] Windows support will be added in a future release.
.. [2] nvmath-python relies on `CUDA minor version compatibility <https://docs.nvidia.com/deploy/cuda-compatibility/index.html#minor-version-comaptibility>`_.
.. [3] Python 3.12 support will be added in the next release.
.. [4] As of beta 1 (v0.1.0), CuPy is a required run-time dependency. In a future release it will be turned into an optional run-time dependency.
.. [5] For example, Hopper GPUs are supported starting CUDA 11.8, so they would not work with libraries from CUDA 11.7 or below.
.. [6] While we need some CUDA headers at build time, there is no limitation in the CUDA version seen at build time.
.. [7] CUDA 12.4+ are not yet supported due to a known compiler bug.
.. [8] If CCCL is installed via ``pip`` manually it needs to be constrained with ``"nvidia-cuda-cccl-cu12>=12.5.*"`` due to a packaging issue; the ``[dx]`` extras already takes care of this.
