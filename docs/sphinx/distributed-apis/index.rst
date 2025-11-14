****************
Distributed APIs
****************

The following modules of nvmath-python offer integration with NVIDIA's
high-performance distributed computing libraries. Distributed APIs are
called from host code but execute on a distributed (multi-node multi-GPU)
system.

Overview
========

The distributed APIs look and feel similar to their CPU and GPU counterparts,
with a few key differences:

* To use the APIs, the application is launched with multiple processes,
  currently using MPI (e.g. ``mpirun``).

* There is one process per GPU.

* The operands to the API on each process are the **local partition** of the global
  operands (as in the Single program multiple data -SPMD- model) and the user specifies
  the :doc:`distribution <distribution>` (how the data is partitioned across
  processes). This allows the user to partition once and compose across distributed APIs.

* The local operands in certain memory spaces may require **special
  allocation** considerations. For example, GPU operands to the distributed
  FFT and Reshape need to be allocated in a partitioned global address (PGAS)
  space using NVSHMEM. We offer helpers to allocate CuPy ndarrays and
  PyTorch tensors in PGAS space (refer to :doc:`Distributed API Utilities
  <utils>` for details).

.. important::
    To use the distributed APIs, you must first initialize the distributed runtime
    (see :doc:`Distributed runtime <runtime>`).

========
Contents
========

.. toctree::
   :caption: API Reference
   :maxdepth: 2

   Distributed runtime <runtime.rst>
   Operand distribution <distribution.rst>
   Linear Algebra <linalg/index.rst>
   Fast Fourier Transform <fft/index.rst>
   Reshape <reshape/index.rst>
   Distributed API Utilities <utils.rst>
