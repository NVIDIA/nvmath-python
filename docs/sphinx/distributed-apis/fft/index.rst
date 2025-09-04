**********************************
Distributed Fast Fourier Transform
**********************************

.. _distributed-fft-overview:

Overview
========

The distributed Fast Fourier Transform (FFT) module :mod:`nvmath.distributed.fft` in
nvmath-python leverages the NVIDIA cuFFTMp library and provides a powerful suite of APIs
that can be directly called from the host to efficiently perform discrete Fourier
transformations on multi-node multi-GPU systems at scale. Both stateless function-form
APIs and stateful class-form APIs are provided to support a spectrum of N-dimensional
FFT operations. These include forward and inverse complex-to-complex (C2C) transformations,
as well as complex-to-real (C2R) and real-to-complex (R2C) transforms:

- N-dimensional forward C2C FFT transform by :func:`nvmath.distributed.fft.fft`.
- N-dimensional inverse C2C FFT transform by :func:`nvmath.distributed.fft.ifft`.
- N-dimensional forward R2C FFT transform by :func:`nvmath.distributed.fft.rfft`.
- N-dimensional inverse C2R FFT transform by :func:`nvmath.distributed.fft.irfft`.
- All types of N-dimensional FFT by stateful :class:`nvmath.distributed.fft.FFT`.

.. note::

    The API :func:`~nvmath.distributed.fft.fft` and related function-form APIs perform
    **N-D FFT** operations, similar to :func:`numpy.fft.fftn`. Currently, 2-D and 3-D
    FFTs are supported.

The distributed FFT APIs are similar to their non-distributed host API counterparts, with
some key differences:

* The operands to the API are the **local partition** of the global operands and
  the user specifies the **distribution** (how the data is partitioned across
  processes). There are two types of distribution supported for FFT: ``Slab`` and custom
  ``Box`` (these are described below).

* GPU operands need to be allocated on **symmetric memory**. Refer to
  :doc:`Distributed API Utilities <../utils>` for examples and details of how to
  manage symmetric memory GPU operands. The :func:`nvmath.distributed.fft.allocate_operand`
  helper described below can also be used to allocate on symmetric memory.

* All distributed FFT operations (including R2C and C2R) are **in-place** (the result is
  stored in the same buffer as the input operand). This has special implications on the
  properties of the buffer and memory layout, due to the following: (i) in general, varying
  input and output distribution means that on a given process the input and output can have
  different shape (and size), particularly when global data does not divide evenly among
  processes; (ii) for R2C and C2R transformations, the shape and dtype of the input and
  output is different, and cuFFTMp has special requirements concerning buffer padding and
  strides. Due to the above, it's recommended to allocate FFT operands with
  :func:`nvmath.distributed.fft.allocate_operand` to ensure that the operand and
  its underlying buffer have the required characteristics for the given distributed FFT
  operation. This helper is described below.


Slab distribution
-----------------

``Slab`` is a 1D data decomposition where the data is partitioned across processes along one
dimension (currently X or Y).

.. tip::
    ``Slab`` is the most optimized distribution mode to use with distributed FFT.

To illustrate with a simple example:

.. tip::
    Reminder to initialize the distributed context first as per
    :ref:`distributed-api-initialize`.

.. code-block:: python

    # Get number of processes from mpi4py communicator.
    nranks = communicator.Get_size()

    # The global 3-D FFT size is (64, 256, 128).
    # Here, the input data is distributed across processes according to the
    # Slab distribution on the Y axis.
    shape = 64, 256 // nranks, 128

    # Create NumPy ndarray (on the CPU) on each process, with the local shape.
    a = np.random.rand(*shape) + 1j * np.random.rand(*shape)

    # Forward FFT.
    # By default, the reshape option is True, which means that the output of the
    # distributed FFT will be re-distributed to retain the same distribution as
    # the input (in this case Slab.Y).
    b = nvmath.distributed.fft.fft(a, distribution=nvmath.distributed.fft.Slab.Y)

For the purposes of the transform with ``reshape=False``, ``Slab.X``
and ``Slab.Y`` are considered complementary distributions. If ``reshape=False``, the
returned operand will use the complementary distribution. The following example illustrates
this using GPU operands:

.. code-block:: python

    # The global 3-D FFT size is (512, 256, 512).
    # Here, the input data is distributed across processes according to the
    # Slab distribution on the X axis.
    shape = 512 // nranks, 256, 512

    # cuFFTMp uses the NVSHMEM PGAS model for distributed computation, which
    # requires GPU operands to be on the symmetric heap.
    a = nvmath.distributed.allocate_symmetric_memory(shape, cp, dtype=cp.complex128)
    # a is a cupy ndarray and can be operated on using in-place cupy operations.
    with cp.cuda.Device(device_id):
        a[:] = cp.random.rand(*shape, dtype=cp.float64) + 1j * cp.random.rand(*shape, dtype=cp.float64)

    # Forward FFT.
    # Here, the forward FFT operand is distributed according to Slab.X distribution.
    # With reshape=False, the FFT result will be distributed according to Slab.Y distribution.
    b = nvmath.distributed.fft.fft(a, distribution=nvmath.distributed.fft.Slab.X, options={"reshape": False})

    # Now we can perform an inverse FFT with reshape=False and get the
    # result in Slab.X distribution (recall that `b` has Slab.Y distribution).
    c = nvmath.distributed.fft.ifft(b, distribution=nvmath.distributed.fft.Slab.Y, options={"reshape": False})

    # Synchronize the default stream
    with cp.cuda.Device(device_id):
        cp.cuda.get_current_stream().synchronize()

    # All cuFFTMp operations are inplace (a, b, and c share the same memory buffer), so
    # we take care to only free the buffer once.
    nvmath.distributed.free_symmetric_memory(a)

.. note::
    Distributed FFT operations are in-place, which needs to be taken into account
    when freeing the GPU operands on symmetric memory (as shown in the above example).

Refer to :class:`nvmath.distributed.fft.Slab` for more details.

Custom box distribution
-----------------------

Distributed FFT also supports arbitrary data distributions in the form of 2D/3D boxes.
Please refer to :ref:`distributed-reshape-box` for an overview.

.. tip::
    While efficient, ``Box`` distribution is less optimized than ``Slab``
    for distributed FFT.

To perform a distributed FFT using a custom ``Box`` distribution, each process specifies
its own input and output box, which determines the distribution of the input and output
operands, respectively (note that input and output distributions can be the same or not).

With box distribution there is also the notion of complementary distribution:
``(input_box, output_box)`` and ``(output_box, input_box)`` are complementary.

Here is an example of a distributed FFT across 4 GPUs using a custom pencil distribution:

.. code-block:: python

    # Get process rank from mpi4py communicator.
    rank = communicator.Get_rank()

    # The global 3-D FFT size is (64, 256, 128).
    # The input data is distributed across 4 processes using a custom pencil
    # distribution.
    X, Y, Z = (64, 256, 128)
    shape = X // 2, Y // 2, Z  # pencil decomposition on X and Y axes

    # NumPy ndarray, on the CPU.
    a = np.random.rand(*shape) + 1j * np.random.rand(*shape)

    # Forward FFT.
    if rank == 0:
        input_box = [(0, 0, 0), (32, 128, 128)]
    elif rank == 1:
        input_box = [(0, 128, 0), (32, 256, 128)]
    elif rank == 2:
        input_box = [(32, 0, 0), (64, 128, 128)]
    else:
        input_box = [(32, 128, 0), (64, 256, 128)]
    # Use the same pencil distribution for the output.
    output_box = input_box
    b = nvmath.distributed.fft.fft(a, distribution=[input_box, output_box])

Operand allocation helper
-------------------------

The :func:`~nvmath.distributed.fft.allocate_operand` helper can be used to allocate an
operand that meets the requirements (in terms of buffer size, padding and strides) for
the specified FFT operation . For GPU operands, the allocation will be done on the
symmetric heap.

.. important::
    Any memory on the symmetric heap that is owned by the user (including memory
    allocated with :func:`~nvmath.distributed.fft.allocate_operand`) must be deleted
    explicitly using :func:`~nvmath.distributed.free_symmetric_memory`. Refer to
    :doc:`Distributed API Utilities <../utils>` for more information.

To allocate an operand, each process specifies the local shape of its input, the array
package, dtype, distribution and FFT type. For example:

.. code-block:: python

    import cupy as cp

    # Get number of processes from mpi4py communicator.
    nranks = communicator.Get_size()

    from nvmath.distributed.fft import Slab

    # The global *real* 3-D FFT size is (512, 256, 512).
    # The input data is distributed across processes according to
    # the cuFFTMp Slab distribution on the X axis.
    shape = 512 // nranks, 256, 512

    # Allocate the operand on the symmetric heap with the required properties
    # for the specified distributed FFT R2C.
    a = nvmath.distributed.fft.allocate_operand(
        shape,
        cp,
        input_dtype=cp.float32,
        distribution=Slab.X,
        fft_type="R2C",
    )
    # a is a cupy ndarray and can be operated on using in-place cupy operations.
    with cp.cuda.Device(device_id):
        a[:] = cp.random.rand(*shape, dtype=cp.float32)

    # R2C (forward) FFT.
    # In this example, the R2C operand is distributed according to Slab.X distribution.
    # With reshape=False, the R2C result will be distributed according to Slab.Y distribution.
    b = nvmath.distributed.fft.rfft(a, distribution=Slab.X, options={"reshape": False})

    # Distributed FFT performs computations in-place. The result is stored in the same
    # buffer as operand a. Note, however, that operand b has a different dtype and shape
    # (because the output has complex dtype and Slab.Y distribution).

    # C2R (inverse) FFT.
    # The inverse FFT operand is distributed according to Slab.Y. With reshape=False,
    # the C2R result will be distributed according to Slab.X distribution.
    c = nvmath.distributed.fft.irfft(b, distribution=Slab.Y, options={"reshape": False})

    # Synchronize the default stream
    with cp.cuda.Device(device_id):
        cp.cuda.get_current_stream().synchronize()

    # The shape of c is the same as a (due to Slab.X distribution). Once again, note that
    # a, b and c are sharing the same symmetric memory buffer (distributed FFT operations
    # are in-place).
    nvmath.distributed.free_symmetric_memory(a)

.. _distributed-fft-api-reference:

API Reference
=============

.. module:: nvmath.distributed.fft


FFT support (:mod:`nvmath.distributed.fft`)
-------------------------------------------

.. autosummary::
   :toctree: generated/

   allocate_operand
   fft
   ifft
   rfft
   irfft
   FFT

   :template: dataclass.rst

   FFTOptions
   FFTDirection
   Slab
