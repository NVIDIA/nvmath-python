*************************
Distributed API Utilities
*************************

NVSHMEM symmetric memory management
===================================

Some distributed APIs like :class:`nvmath.distributed.fft.FFT` and
:class:`nvmath.distributed.reshape.Reshape` use a Partitioned Global Address Space (PGAS)
model for parallelism and require GPU operands to be allocated on the
`NVSHMEM symmetric memory heap <https://docs.nvidia.com/nvshmem/api/using.html>`_.
We offer helpers to allocate CuPy ndarrays and PyTorch tensors in symmetric memory.
To do so, simply specify the *local* shape, the array package and dtype:

.. code-block:: python

    import cupy
    import torch
    import nvmath.distributed

    # NVSHMEM backend required for the following symmetric memory APIs.
    nvmath.distributed.initialize(device_id, process_group, backends=["nvshmem"])

    # Allocate a CuPy array of shape (3,3) on each process
    a = nvmath.distributed.allocate_symmetric_memory((3,3), cupy, dtype=cupy.float32)

    # Allocate a torch tensor of shape (3,3) on each process
    b = nvmath.distributed.allocate_symmetric_memory((3,3), torch, dtype=torch.float64)

    # ... do distributed computations using these operands, as well as any
    # array operations supported by the array package

    nvmath.distributed.free_symmetric_memory(a, b)

.. important::
    Any symmetric memory owned by the user (e.g. allocated
    with :func:`~nvmath.distributed.allocate_symmetric_memory` or returned to the
    user by a distributed API) must be deleted explicitly using
    :func:`~nvmath.distributed.free_symmetric_memory`. You cannot rely on the Python
    garbage collector to do this, since freeing a symmetric allocation is a
    collective call which must be done by all processes, and the garbage collector
    does not free memory in a deterministic fashion.

.. note::
    The allocation size on each process must be the same (due to the symmetric
    memory requirement). This implies that, by default, the shape and dtype must
    be the same on every process. For non-uniform shapes, you can use
    ``make_symmetric=True`` to force a symmetric allocation under the hood (see
    the example below).

If the shape and dtype is not uniform across processes, you can make the allocation
symmetric by using ``make_symmetric=True``:

.. code-block:: python

    # Get my process rank.
    rank = nvmath.distributed.get_context().process_group.rank
    # Note: this will raise an error if make_symmetric is False.
    if rank == 0:
        a = nvmath.distributed.allocate_symmetric_memory((3,3), cupy, make_symmetric=True)
    else:
        a = nvmath.distributed.allocate_symmetric_memory((2,3), cupy, make_symmetric=True)
    # ...
    nvmath.distributed.free_symmetric_memory(a)

This will allocate a buffer of the same size (in bytes) on each process, with
the returned ndarray/tensor backed by that buffer, but of exactly the requested shape
on that process. The size of the buffer is determined based on the process with most
elements (rank 0 in the above example).

.. _distributed-api-util-reference:

API Reference
-------------

.. module:: nvmath.distributed
   :no-index:

nvmath-python provides host-side APIs for managing symmetric memory.

.. autosummary::
   :toctree: generated/

   allocate_symmetric_memory
   free_symmetric_memory


.. _distributed-reshape-overview:

Distributed Reshape
===================

The distributed reshape module :mod:`nvmath.distributed.reshape` in
nvmath-python leverages the NVIDIA cuFFTMp library and provides APIs that can
be directly called from the host to efficiently redistribute local operands
on multiple processes on multi-node multi-GPU systems at scale. Both stateless
function-form APIs and stateful class-form APIs are provided:

- function-form reshape using :func:`nvmath.distributed.reshape.reshape`.
- stateful reshape using :class:`nvmath.distributed.reshape.Reshape`.

Reshape is a general-purpose API to change how data is distributed or
partitioned across processes, by shuffling data among the processes.
Distributed reshape supports arbitrary data distributions in the form of
1D/2D/3D boxes (see :ref:`distribution-box` distribution).

Example
-------

To perform a distributed reshape, each process specifies its own input and output box, which
determines the distribution of the input and output, respectively.

As an example, consider a matrix that is distributed column-wise on two processes (each
process owns a contiguous chunk of columns). To redistribute the matrix row-wise, we can use
distributed reshape:

.. note::
    To use the distributed Reshape APIs you need to
    :ref:`initialize the distributed runtime <distributed-api-initialize>`
    with the NVSHMEM communication backend.

.. code-block:: python

    from nvmath.distributed.distribution import Box

    # The global dimensions of the matrix are 4x4. The matrix is distributed
    # column-wise, so each process has 4 rows and 2 columns.

    # Get my process rank.
    rank = nvmath.distributed.get_context().process_group.rank

    # Initialize the matrix on each process, as a NumPy ndarray (on the CPU).
    A = np.zeros((4, 2)) if rank == 0 else np.ones((4, 2))

    # Reshape from column-wise to row-wise.
    if rank == 0:
        input_box = Box((0, 0), (4, 2))
        output_box = Box((0, 0), (2, 4))
    else:
        input_box = Box((0, 2), (4, 4))
        output_box = Box((2, 0), (4, 4))

    # Distributed reshape returns a new operand with its own buffer.
    B = nvmath.distributed.reshape.reshape(A, input_box, output_box)

    # The result is a NumPy ndarray, distributed row-wise:
    # [0] B:
    # [[0. 0. 1. 1.]
    #  [0. 0. 1. 1.]]
    #
    # [1] B:
    # [[0. 0. 1. 1.]
    #  [0. 0. 1. 1.]]
    print(f"[{rank}] B:\n{B}")


.. _distributed-reshape-api-reference:

API Reference (:mod:`nvmath.distributed.reshape`)
-------------------------------------------------

.. module:: nvmath.distributed.reshape

.. autosummary::
   :toctree: generated/

   reshape
   Reshape

.. autosummary::
   :toctree: generated/
   :template: dataclass

   ReshapeOptions
