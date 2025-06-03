Installation
***************

Install nvmath-python
=====================

nvmath-python, like most modern Python packages, provides pre-built binaries (wheels and
conda packages) to the end users. The full source code is hosted in the
`NVIDIA/nvmath-python <https://github.com/NVIDIA/nvmath-python>`_ repository.

In terms of CUDA Toolkit (CTK) choices, nvmath-python is designed and implemented to allow
building and running against 1. ``pip``-wheel, 2. ``conda``, or 3. system installation of
CTK. Having a full CTK installation at either build- or run-time is not necessary; only a
small subset, as explained below, is enough.

Host & device APIs (see :ref:`nvmath overview`) have different run-time dependencies and
requirements. Even among host APIs the needed underlying libraries are different (for
example, :func:`~nvmath.fft.fft` on GPUs only needs cuFFT and not cuBLAS). Libraries are
loaded when only needed. Therefore, nvmath-python is designed to have most of its
dependencies *optional*, but provides convenient installation commands for users to quickly
spin up a working Python environment.

The :ref:`cheatsheet <cheatsheet>` below captures nvmath-python's required and optional
build-time and run-time dependencies. Using the installation commands from the sections
below should support most of your needs.


.. _install from pypi:

Install from PyPI
-----------------

The pre-built wheels can be ``pip``-installed from the public PyPI. There are several
optional dependencies expressible in the standard "extras" bracket notation. The following
assumes that **CTK components are also installed via pip** (so no extra step from users is
needed; the dependencies are pulled via extras).

.. list-table::
   :widths: 25 50
   :header-rows: 1

   * - Command
     - Description
   * - ``pip install nvmath-python[cu11]``
     - Install nvmath-python along with all CUDA 11 optional
       dependencies (wheels for cuBLAS/cuFFT/... and CuPy) to support
       nvmath host APIs.
   * - ``pip install nvmath-python[cu12]``
     - Install nvmath-python along with all CUDA 12 optional
       dependencies (wheels for cuBLAS/cuFFT/... and CuPy) to support
       nvmath host APIs.
   * - ``pip install nvmath-python[cu12,dx]``
     - Install nvmath-python along with all CUDA 12 optional
       dependencies (wheels for cuBLAS/cuFFT/..., CuPy, Numba,
       pynvjitlink, ...) to support nvmath host & device APIs (which
       only supports CUDA 12) [8]_.
   * - ``pip install nvmath-python[cpu]``
     - Install nvmath-python along with all CPU optional dependencies
       (wheels for NVPL or MKL) to support optimized CPU FFT APIs. [1]_

       **Note**:

       1. NVPL supports only ARM architecture, while MKL or another FFTW3 [9]_
          compatible library may be substituted for x86 architecture.
       2. The environment variable ``NVMATH_FFT_CPU_LIBRARY`` may be used to
          provide the path to an alternate shared object which implements the
          FFTW3 (non-guru) API. Ensure ``LD_LIBRARY_PATH`` includes this
          library if it is not already in the PATH.

   * - ``pip install nvmath-python[cu12-distributed]``
     - Install nvmath-python along with all MGMN optional dependencies (wheels for mpi4py,
       NVSHMEM, cuFFTMp, ...) to support multi-GPU multi-node APIs.

       **Note**: Users must provide an MPI implementation.

   * - ``pip install nvmath-python[cu12,dx] 'nvidia-cuda-nvcc-cu12==12.8.*' 'nvidia-cuda-nvrtc-cu12==12.8.*' --extra-index-url https://download.pytorch.org/whl/cu128 torch``
     - Install nvmath-python along with all CUDA 12 optional dependencies to support
       nvmath.device APIs and a PyTorch built with CTK 12.8.

       **Note**: PyTorch has strict pinnings for some CUDA components, and builds of PyTorch
       lag behind the latest CUDA component releases. We must therefore explicitly require
       that the nvcc and nvrtc components installed for device extensions match; otherwise,
       pip will create a mismatched environment with CTK components from different releases.
       Verbose installation commands such as the the one here are necessary because of
       limitations of the current wheel format and Python package index. Please see the
       PyTorch installation instructions for releases built with other CTK versions.

The options below are for adventurous users who want to manage most of the dependencies
themselves. The following assumes that **system CTK is installed**.

.. list-table::
   :widths: 25 50
   :header-rows: 1

   * - Command
     - Description
   * - ``pip install nvmath-python[sysctk11]``
     - Install nvmath-python along with CuPy for CUDA 11 to support
       nvmath host APIs.

       **Note**: Set ``LD_LIBRARY_PATH`` to include the CUDA libraries.

   * - ``pip install nvmath-python[sysctk12]``
     - Install nvmath-python along with CuPy for CUDA 12 to support
       nvmath host APIs.

       **Note**: Set ``LD_LIBRARY_PATH`` to include the CUDA libraries.

   * - ``pip install nvmath-python[sysctk12-dx]``
     - Install nvmath-python along with CuPy for CUDA 12 to support
       nvmath host & device APIs.

       **Note**:

       1. Set ``LD_LIBRARY_PATH`` to include the CUDA libraries.
       2. To use :mod:`nvmath.device` APIs, set ``CUDA_HOME`` (or ``CUDA_PATH``)
          to point to the system CTK.

   * - ``pip install nvmath-python[sysctk12] mpi4py``
     - Install nvmath-python and mpi4py along with no MGMN optional dependencies to support
       multi-GPU multi-node APIs.

       **Note**: Users must provide an MPI implementation and the required cuMp libraries
       and dependencies (NVSHMEM, cuFFTMp, ...).

For system admins or expert users, ``pip install nvmath-python`` would be a bare minimal
installation (very lightweight). This allows fully explicit control of all dependencies.


Install from conda
------------------

Conda packages can be installed from the `conda-forge <https://conda-forge.org>`_ channel.

.. list-table::
   :widths: 25 50
   :header-rows: 1

   * - Command
     - Description
   * - ``conda install -c conda-forge nvmath-python cuda-version=11``
     - Install nvmath-python along with all CUDA 11 optional
       dependencies (packages for cuBLAS/cuFFT/... and CuPy) to support
       nvmath host APIs.
   * - ``conda install -c conda-forge nvmath-python cuda-version=12``
     - Install nvmath-python along with all CUDA 12 optional
       dependencies (packages for cuBLAS/cuFFT/... and CuPy) to support
       nvmath host APIs.
   * - ``conda install -c conda-forge -c rapidsai nvmath-python-dx "pynvjitlink>=0.6"
       cuda-version=12``
     - Install nvmath-python along with all CUDA 12 optional
       dependencies (packages for cuBLAS/cuFFT/..., CuPy, Numba,
       pynvjitlink, ...) to support nvmath host & device APIs (which
       only supports CUDA 12).

       **Note**:

       1. ``nvmath-python-dx`` is a metapackage for ease of installing
          ``nvmath-python`` and other dependencies.
       2. Currently, ``pynvjitlink`` is only available on the rapidsai channel,
          and not on conda-forge.
   * - ``conda install -c conda-forge nvmath-python-cpu``
     - Install nvmath-python along with all CPU optional dependencies
       (NVPL or other) to support optimized CPU FFT APIs. [1]_

       **Note**:

       1. ``nvmath-python-cpu`` is a meta-package for ease of installing
          ``nvmath-python`` and other dependencies.
       2. NVPL is for ARM architecture only. MKL or another FFTW3 [9]_ compatible
          library may be substituted for x86 architecture.
       3. The environment variable ``NVMATH_FFT_CPU_LIBRARY`` may be used to
          provide the path to an alternate shared object which implements the
          FFTW3 (non-guru) API. ``LD_LIBRARY_PATH`` should be set properly to
          include this library if it is not already in the PATH.

   * - ``conda install -c conda-forge nvmath-python-distributed``
     - Install nvmath-python along with all MGMN optional dependencies (packages for mpi4py,
       NVSHMEM, cuFFTMp, MPI, ...) to support multi-GPU multi-node APIs.

       **Note**: conda-forge provides a pass-through MPI package variant that may be used in
       order to use a system-installed MPI instead of the conda-forge-provided MPI
       implementations.

**Notes**:

- For expert users, ``conda install -c conda-forge nvmath-python=*=core*`` would be a bare
  minimal installation (very lightweight). This allows fully explicit control of all
  dependencies.
- If you installed ``conda`` from `miniforge <https://github.com/conda-forge/miniforge>`_,
  most likely the conda-forge channel is already set as the default, then the ``-c
  conda-forge`` part in the above instruction can be omitted.


Build from source
-----------------

Once you clone the repository and go into the root directory, you can build the project from
source. There are several ways to build it since we need some CUDA headers at build time.

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
     - Skip creating a build isolation (it would use CUDA headers from
       ``$CUDA_PATH/include`` instead), build the project, and install it to the current
       user environment together with the run-time dependencies. Use:

       - conda: After installing CUDA 12 conda packages, set the environment variable
         ``CUDA_PATH``

         * linux-64: ``CUDA_PATH=$CONDA_PREFIX/targets/x86_64-linux/``
         * linux-aarch64: ``CUDA_PATH=$CONDA_PREFIX/targets/sbsa-linux/``
         * win-64: ``CUDA_PATH=$CONDA_PREFIX\Library``

       - system CTK: Just set ``CUDA_PATH`` to the system CTK location.

**Notes**:

- If you add the "extras" notation after the dot ``.`` (for example ``.[cu11]``,
  ``.[cu12,dx]``, ...), it has the same meaning as explained in the :ref:`previous section
  <install from pypi>`.
- If you don't want the run-time dependencies to be automatically handled, add ``--no-deps``
  after the ``pip install`` command above; in this case, however, it's your responsibility
  to make sure that all the run-time requirements are met.
- By replacing ``install`` by ``wheel``, a wheel can be built targeting the current OS and
  CPython version.
- If you want inplace/editable install, add the ``-e`` flag to the command above (before the
  dot ``.``). This is suitable for local development with a system-installed CTK. However,
  our wheels rely on *non-editable builds* so that the RPATH hack can kick in. DO NOT pass
  the ``-e`` flag when building wheels!
- All optional run-time dependencies as listed below need to be manually installed.


.. _cheatsheet:

Cheatsheet
----------

Below we provide a summary of requirements to support all nvmath-python functionalities. A
dependency is *required* unless stated otherwise.

.. list-table::
   :widths: 25 25 25 25 25 25
   :header-rows: 1

   * -
     - When Building
     - When Running - host APIs
     - When Running - device APIs
     - When Running - host API callbacks
     - When Running - distributed APIs
   * - CPU architecture & OS
     - linux-64, linux-aarch64, win-64
     - linux-64, linux-aarch64, win-64
     - linux-64, linux-aarch64 [1]_
     - linux-64, linux-aarch64
     - linux-64, linux-aarch64
   * - GPU hardware
     -
     - | All hardware supported by the underlying CUDA Toolkit [5]_
       |
       | *Optional*: needed if the execution space is GPU.
     - Compute Capability 7.0+ (Volta and above)
     - Compute Capability 7.0+ (Volta and above)
     - `Data Center GPU <https://developer.nvidia.com/cuda-gpus>`_
       with Compute Capability 7.0+ (Volta and above).
       GPU connectivity: :cufftmp_hw:`requirements`
   * - CUDA driver [2]_
     -
     - | 450.80.02+ (Linux) / 450.39+ (Windows) with CUDA >=11.2
       |
       | 525.60.13+ (Linux) / 527.41+ (Windows) with CUDA >=12.0
       |
       | *Optional*: needed if the execution space is GPU or for loading any CUDA library.
     - 525.60.13+ (Linux) with CUDA 12.x
     - 525.60.13+ (Linux) with CUDA 12.x
     - 525.60.13+ (Linux) with CUDA 12.x
   * - Python
     - 3.10-3.12
     - 3.10-3.12
     - 3.10-3.12
     - 3.10-3.12
     - 3.10-3.12
   * - pip
     - 22.3.1+
     -
     -
     -
     -
   * - setuptools
     - >=70.0.0
     -
     -
     -
     -
   * - wheel
     - >=0.34.0
     -
     -
     -
     -
   * - Cython
     - >=0.29.22,<3
     -
     -
     -
     -
   * - CUDA
     - | CUDA >=11.2
       | (only need headers from NVCC & CUDART [6]_)
     - | CUDA >=11.2
       |
       | *Optional*: depending on the math operations in use
     - | CUDA >=12.0,!=12.4.*,!=12.5.0 [7]_
       | (NVRTC, NVVM, CCCL [8]_, CUDART)
     - CUDA 12.x
     - CUDA 12.x
   * - NumPy
     -
     - >=1.24
     - >=1.24
     - >=1.24
     - >=1.24
   * - | CuPy
       | (see `CuPy installation guide <https://docs.cupy.dev/en/stable/install.html>`_)
     -
     - >=10.0.0 [4]_
     -
     - >=10.0.0 [4]_
     - >=10.0.0 [4]_
   * - | PyTorch
       | (see `PyTorch installation guide <https://pytorch.org/get-started/locally/>`_)
     -
     - >=1.10 (optional) [10]_
     -
     - >=1.10 (optional)
     - >=1.10 (optional)
   * - libmathdx (cuBLASDx, cuFFTDx, ...)
     -
     -
     - 0.2.*
     -
     -
   * - numba-cuda
     -
     -
     - >=0.11.0
     - >=0.11.0
     -
   * - Numba
     -
     -
     - >=0.59.1
     - >=0.59.1
     -
   * - pynvjitlink
     -
     -
     - >=0.6
     -
     -
   * - Math Kernel Library (MKL)
     -
     - >=2024 (optional)
     -
     -
     -
   * - NVIDIA Performance Libraries (NVPL)
     -
     - 24.7 (optional)
     -
     -
     -


Test Configuration
------------------

nvmath-python is tested in the following environments:

.. TODO:
   Update me

.. list-table::
   :widths: 50 50

   * - CUDA
     - 11.x (latest), 12.0, 12.8
   * - Driver
     - R520, R525, R570
   * - GPU model
     - H100, B200, RTX 4090, CG1 (Grace-Hopper)
   * - Python
     - 3.10, 3.11, 3.12
   * - CPU architecture
     - x86_64, aarch64
   * - Operating system
     - Ubuntu 22.04, Ubuntu 20.04, RHEL 9, Windows 11


Run nvmath-python
=================

As mentioned earlier, nvmath-python can be run with all methods of CUDA installation,
including wheels, conda packages, and system CTK. As a result, there is detection logic to
discover shared libraries (for host APIs) and headers (for device APIs to do JIT
compilation).

Shared libraries
----------------

- pip wheels: Will be auto-discovered if installed
- conda packages: Will be auto-discovered if installed, after wheel
- system CTK: On Linux, the users needs to ensure the shared libraries are discoverable by
  the dynamic linker, say by setting ``LD_LIBRARY_PATH`` or updating system search paths to
  include the DSO locations.


Headers 
-------

This includes libraries such as CCCL and MathDx.

- pip wheels: Will be auto-discovered if installed
- conda packages: Will be auto-discovered if installed, after wheel
- system CTK: Need to set ``CUDA_HOME`` (or ``CUDA_PATH``) and ``MATHDX_HOME`` (for MathDx
  headers)


Host APIs
---------

This terminology is explained in the :ref:`host api section`.

Examples
........

See the ``examples`` directory in the repo. Currently we have:

- ``examples/fft``
- ``examples/linalg``


Tests
.....

The ``requirements/pip/tests.txt`` file lists dependencies required for ``pip``-controlled
environments to run tests. These requirements are installed via the main
``requirements/pip-dev-<name>.txt`` files.


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
`pypackaging community project <https://pypackaging-native.github.io>`_) in Python packaging
tools. For a complex library such as nvmath-python that interacts with many native
libraries, there are user-visible caveats.

1. Be sure that there are no packages with both ``-cu11`` (for CUDA 11) and ``-cu12`` (for
   CUDA 12) suffices coexisting in your Python environment. For example, this is a corrupted
   environment:

   .. code-block:: bash

      $ pip list
      Package            Version
      ------------------ ---------
      nvidia-cublas-cu11 11.11.3.6
      nvidia-cublas-cu12 12.5.2.13
      pip                24.0
      setuptools         70.0.0
      wheel              0.43.0

   Sometimes such conflicts could come from a dependency of the libraries that you use, so
   pay extra attention to what's installed.
2. ``pip`` does not attempt to check if the installed packages can actually be run against
   the installed GPU driver (CUDA GPU driver cannot be installed by ``pip``), so make sure
   your GPU driver is new enough to support the installed ``-cuXX`` packages [2]_. The
   driver version can be checked by executing ``nvidia-smi`` and inspecting the ``Driver
   Version`` field on the output table.
3. CuPy installed from ``pip`` currently (as of v13.3.0) only supports conda and system CTK,
   and not ``pip``-installed CUDA wheels. nvmath-python can help CuPy use the CUDA libraries
   installed to ``site-packages`` (where wheels are installed to) if ``nvmath`` is imported.
   From beta 2 (v0.2.0) onwards the libraries are "soft-loaded" (no error is raised if a
   library is not installed) when ``import nvmath`` happens. This behavior may change in a
   future release.
4. Numba installed from ``pip`` currently (as of v0.60.0) only supports conda and system
   CTK, and not ``pip``-installed CUDA wheels. nvmath-python can also help Numba use the
   CUDA compilers installed to ``site-packages`` if ``nvmath`` is imported. Same as above,
   this behavior may change in a future release.
5. PyTorch installed from ``pip`` pins some CUDA wheels packages to version v12.6 (or v12.8
   depending on the installation method). However, nvmath-python does not pin CUDA wheels
   packages, so they will float up the latest version. This can cause a mismatch between
   compiler components when using the ``dx`` extra. In this case, it's recommended to
   manually constrain ``cuda-cccl``, ``cuda-nvcc``, ``cuda-nvrtc``, and ``cuda-runtime``
   packages to match the variant of PyTorch installed.

In general, mixing-and-matching CTK packages from ``pip``, ``conda``, and the system is
possible but can be very fragile, so it's important to understand what you're doing. The
nvmath-python internals are designed to work with everything installed either via ``pip``,
``conda``, or local system (system CTK, including `tarball extractions
<https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html
#tarball-and-zip-archive-deliverables>`_, are the fallback solution in the detection logic),
but mix-n-match makes the detection logic impossible to get right.

To help you perform an integrity check, the rule of thumb is that every single package
should only come from one place (either ``pip``, or ``conda``, or local system). For
example, if both ``nvidia-cufft-cu11`` (which is from ``pip``) and ``libcufft`` (from
``conda``) appear in the output of ``conda list``, something is almost certainly wrong.
Below is the package name mapping between ``pip`` and ``conda``, with ``XX={11,12}``
denoting CUDA's major version:

.. list-table::
   :widths: 50 50 50
   :header-rows: 1

   * - pip
     - conda (``cuda-version>=12``)
     - conda (``cuda-version<12``)
   * - ``nvidia-cuda-nvcc-cuXX``
     - ``cuda-nvcc``
     - n/a
   * - ``nvidia-cuda-nvrtc-cuXX``
     - ``cuda-nvrtc``
     - ``cudatoolkit``
   * - ``nvidia-cuda-runtime-cuXX``
     - ``cuda-cudart-dev``
     - ``cudatoolkit``
   * - ``nvidia-cuda-cccl-cuXX``
     - ``cuda-cccl``
     - n/a
   * - ``pynvjitlink-cuXX``
     - ``pynvjitlink``
     - n/a
   * - ``nvidia-cublas-cuXX``
     - ``libcublas``
     - ``cudatoolkit``
   * - ``nvidia-cusolver-cuXX``
     - ``libcusolver``
     - ``cudatoolkit``
   * - ``nvidia-cusparse-cuXX``
     - ``libcusparse``
     - ``cudatoolkit``
   * - ``nvidia-cufft-cuXX``
     - ``libcufft``
     - ``cudatoolkit``
   * - ``nvidia-curand-cuXX``
     - ``libcurand``
     - ``cudatoolkit``

Note that system packages (by design) do not show up in the output of ``conda list`` or
``pip list``. Linux users should check the installation list from your distro package
manager (``apt``, ``yum``, ``dnf``, ...). See also the `Linux Package Manager Installation
Guide <https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html
#package-manager-installation>`_ for additional information.

For more information with regard to the new CUDA 12+ package layout on conda-forge, see the
`CUDA recipe README <https://github.com/conda-forge/cuda-feedstock/tree/main/recipe>`_.


.. rubric:: Footnotes

.. [1] Windows support will be added in a future release.
.. [2] nvmath-python relies on `CUDA minor version compatibility
    <https://docs.nvidia.com/deploy/cuda-compatibility/index.html
    #minor-version-compatibility>`_.
.. [4] As of beta 4.0 (v0.4.0), CuPy is a required run-time dependency except for CPU-only
    execution. In a future release it will be turned into an optional run-time dependency.
.. [5] For example, Hopper GPUs are supported starting CUDA 11.8, so they would not work
    with libraries from CUDA 11.7 or below.
.. [6] While we need some CUDA headers at build time, there is no limitation in the CUDA
    version seen at build time.
.. [7] These versions are not supported due to a known compiler bug; the ``[dx]`` extras
    already takes care of this.
.. [8] If CCCL is installed via ``pip`` manually it needs to be constrained with
    ``"nvidia-cuda-cccl-cu12>=12.4.127"`` due to a packaging issue; the ``[dx]`` extras
    already takes care of this.
.. [9] The library must ship FFTW3 symbols for single and double precision transforms in a
    single ``so`` file.
.. [10] To use ``matmul`` with FP8 or MXFP8 you need PyTorch version built with CUDA 12.8
    (``>=2.7.0`` or nightly version)
