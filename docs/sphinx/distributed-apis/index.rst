****************
Distributed APIs
****************

The following modules of nvmath-python offer integration with NVIDIA's
high-performance distributed computing libraries. Distributed APIs are
called from host code but execute on a distributed (multi-node multi-GPU)
system.

Overview
--------

The distributed APIs look and feel similar to their CPU and GPU counterparts,
with a few key differences:

* To use the APIs, the application is launched with multiple processes,
  currently using MPI (e.g. ``mpirun``).

* Each process is assigned to one GPU.

* The operands to the API are the **local partition** of the global operands
  (as in the Single program multiple data -SPMD- model) and the user specifies
  the **distribution** (how the data is partitioned across processes). This
  allows the user to partition once and compose across distributed APIs.

* The local operands in certain memory spaces may require **special
  allocation** considerations. For example, GPU operands to the distributed
  FFT and Reshape need to be allocated in a partitioned global address (PGAS)
  space using NVSHMEM. We offer helpers to allocate CuPy ndarrays and
  PyTorch tensors in PGAS space (refer to :doc:`Distributed API Utilities
  <utils>` for details).

.. _distributed-api-initialize:

Initializing the distributed runtime
------------------------------------

To use the distributed APIs, you must first initialize the distributed runtime.
This is done by having each process provide a local CUDA device ID (referring
to a GPU on the host on which that process runs) and an MPI communicator:

.. code-block:: python

    import nvmath.distributed
    from mpi4py import MPI
    comm = MPI.COMM_WORLD  # can use any MPI communicator
    nvmath.distributed.initialize(device_id, comm)

.. note::

    nvmath-python uses MPI for bootstrapping, and other bootstrapping modes
    may become available in the future.

    Under the hood, the distributed math libraries use additional
    communication backends, such as NVSHMEM.

    You are free to use MPI in other parts of your application.

After initializing the distributed runtime you may use the distributed APIs.
Certain APIs such as FFT and Reshape use a PGAS model for parallelism and
require GPU operands to be allocated on the *symmetric memory heap*. Refer to
:doc:`Distributed API Utilities <utils>` for examples and details of how to manage
GPU operands on symmetric memory.

========
Contents
========

.. toctree::
   :caption: API Reference
   :maxdepth: 2

   Fast Fourier Transform <fft/index.rst>
   Reshape <reshape/index.rst>
   Distributed API Utilities <utils.rst>
