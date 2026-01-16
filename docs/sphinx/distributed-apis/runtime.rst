*******************
Distributed runtime
*******************

.. _distributed-api-initialize:

Initializing the distributed runtime
====================================

To use the distributed APIs, you must first initialize the distributed runtime.
This is done by having each process provide a local CUDA device ID (referring
to a GPU on the host on which that process runs), an MPI communicator, and the
desired communication backends:

.. code-block:: python

    import nvmath.distributed
    from mpi4py import MPI
    comm = MPI.COMM_WORLD  # can use any MPI communicator
    nvmath.distributed.initialize(device_id, comm, backends=["nvshmem", "nccl"])

.. note::

    nvmath-python uses MPI for bootstrapping, and other bootstrapping methods
    may become available in the future.

    Under the hood, the distributed math libraries use additional
    communication backends, such as NVSHMEM and NCCL.

    You are free to use MPI in other parts of your application.

.. note::

    NVSHMEM backend is required for symmetric memory operations.

After initializing the distributed runtime you may use the distributed APIs.
Certain APIs such as FFT and Reshape require GPU operands to be allocated on the
*symmetric memory heap*. Refer to :doc:`Distributed API Utilities <utils>` for
examples and details of how to manage GPU operands on symmetric memory.

API Reference
=============

.. module:: nvmath.distributed

.. autosummary::
   :toctree: generated/

   initialize
   finalize
   get_context

   :template: dataclass.rst

   DistributedContext
