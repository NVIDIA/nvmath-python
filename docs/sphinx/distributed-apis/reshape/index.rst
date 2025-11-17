*******************
Distributed Reshape
*******************

.. _distributed-reshape-overview:

Overview
========

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

.. tip::
    Reminder to initialize the distributed context first as per
    :ref:`distributed-api-initialize`.

.. code-block:: python

    from nvmath.distributed.distribution import Box

    # The global dimensions of the matrix are 4x4. The matrix is distributed
    # column-wise, so each process has 4 rows and 2 columns.

    # Get process rank from mpi4py communicator.
    rank = communicator.Get_rank()

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

API Reference
=============

.. module:: nvmath.distributed.reshape


Reshape support (:mod:`nvmath.distributed.reshape`)
---------------------------------------------------

.. autosummary::
   :toctree: generated/

   reshape
   Reshape

   :template: dataclass.rst

   ReshapeOptions
