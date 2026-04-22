****************
Distributed APIs
****************

The following modules of nvmath-python offer integration with NVIDIA's
high-performance distributed computing libraries such as cuBLASMp and cuFFTMp.
Distributed APIs are called from host code and execute on a multi-node multi-GPU system.

========
Overview
========

The distributed APIs look and feel similar to their single-process host API counterparts,
with a few key differences:

* To use the APIs, the application is launched with multiple processes
  (e.g. using MPI or
  `torch.distributed <https://docs.pytorch.org/docs/stable/distributed.html>`_).

* There is one process per GPU.

* The operands on each process are the **local partition** of the global
  operands (as in the Single program, multiple data -SPMD- model) and the user
  specifies the **distribution** (:doc:`how the data is partitioned across processes
  <distribution>`). This allows the user to partition once and compose across
  distributed APIs.

* Some distributed operations like FFT and Reshape require GPU operands
  to be in NVSHMEM **symmetric memory**. We offer optional helpers
  to allocate CuPy ndarrays and PyTorch tensors in symmetric memory
  (refer to :doc:`Distributed API Utilities <utils>` for details).

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
   Distributed API Utilities <utils.rst>
