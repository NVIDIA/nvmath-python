.. _stream-semantics-guide:

Stream Semantics
****************

.. tip::

   If all your operations use the default stream (or a single stream that you
   use consistently across all operations), you do not need to worry
   about stream ordering.
   The discussion below is only relevant when using **multiple streams**.

Because nvmath-python offers both blocking and non-blocking execution modes
(see :ref:`high-level call blocking`) as well as stateless (function-form)
and stateful (class-form) APIs, reasoning about stream semantics
requires considering both of these aspects.
The discussion below is organized along these two dimensions,
starting with the stateless API form since it is the simpler case.
While the examples use :mod:`nvmath.fft` for concreteness, the same
stream semantics apply to all nvmath-python modules
(e.g. :func:`nvmath.linalg.advanced.matmul`,
:class:`nvmath.linalg.advanced.Matmul`,
:class:`nvmath.sparse.advanced.DirectSolver`, and others).

Stateless APIs
==============

A stateless host API involves a single function call that encapsulates
the entire computation.

.. rubric:: Execution space mismatching memory space (blocking)

The simplest scenario is when the execution space does not match the operand memory
space — most commonly, CPU operands with GPU execution. In this case, the call is
always guaranteed to be blocking:
it does not return until the operation has completed, and stream ordering is
automatically enforced by the API.
You can use the result immediately.

.. tabs::

   .. code-tab:: python NumPy

      import numpy as np
      import nvmath
      from cuda.core import Device

      dev = Device(0)
      dev.set_current()

      # Create operand on the host.
      a = np.random.rand(64, 256, 128) + 1j * np.random.rand(64, 256, 128)

      s1 = dev.create_stream()
      s2 = dev.create_stream()

      # Since a is on the host, calling nvmath.fft.fft is blocking
      # regardless of the stream used.
      r1 = nvmath.fft.fft(a, axes=[0, 1], execution="cuda", stream=s1)
      # r1 is ready to use immediately.

      # Some other operation modifies a in place.

      r2 = nvmath.fft.fft(a, axes=[0, 1], execution="cuda", stream=s2)
      # r2 is ready to use immediately.

   .. code-tab:: python PyTorch

      import torch
      import nvmath

      # Create operand on the host.
      a = torch.randn(64, 256, 128, dtype=torch.complex64)

      s1 = torch.cuda.Stream()
      s2 = torch.cuda.Stream()

      # Since a is on the host, calling nvmath.fft.fft is blocking
      # regardless of the stream used.
      r1 = nvmath.fft.fft(a, axes=[0, 1], execution="cuda", stream=s1)
      # r1 is ready to use immediately.

      # Some other operation modifies a in place.

      r2 = nvmath.fft.fft(a, axes=[0, 1], execution="cuda", stream=s2)
      # r2 is ready to use immediately.


.. rubric:: Execution space matching memory space (non-blocking)

When the operands reside on the GPU and execution is on the GPU, the call is
non-blocking by default: it returns as soon as the work is
enqueued on the stream. If you use more than one stream,
you must establish ordering explicitly to avoid race conditions
when reusing operands or consuming results. For example, this can be
achieved by recording an event on one stream and
waiting for it on the other.

.. code-block:: python

   import cupy as cp
   import nvmath

   s1 = cp.cuda.Stream()
   s2 = cp.cuda.Stream()

   with s1:
       a = cp.random.rand(64, 256, 128) + 1j * cp.random.rand(64, 256, 128)

   # Non-blocking: returns immediately after enqueuing on s1.
   b = nvmath.fft.fft(a, axes=[0, 1], execution="cuda", stream=s1)

   # Before using the result for subsequent operations on s2,
   # ensure s1's work has finished.
   e1 = s1.record()
   s2.wait_event(e1)
   c = nvmath.fft.ifft(b, axes=[0, 1], execution="cuda", stream=s2)

   # Synchronize before consuming results on the host.
   s2.synchronize()

.. _stream-ordered-deallocations:

.. rubric:: Stream-ordered deallocations

A subtlety arises with **stream-ordered memory allocators** (such as those used by CuPy
and PyTorch). When a GPU array is deallocated, the deallocation is enqueued on the
*allocation* stream, not on the stream that last used the array. If another stream is
still accessing the array at that point, a race condition occurs.
The user must order the allocation stream to wait for the consuming
stream before dropping the last reference:

.. code-block:: python

   import cupy as cp
   import nvmath

   s1 = cp.cuda.Stream()
   s2 = cp.cuda.Stream()

   # Create tensor on stream s1.
   m, n = 10, 20
   with s1:
       a = cp.random.rand(m, n) + 1j * cp.random.rand(m, n)

   e1 = s1.record()
   s2.wait_event(e1)

   # Manipulate tensor on stream s2.
   with s2:
       a *= 2.

   # Run FFT on stream s2. Non-blocking call returns immediately.
   r = nvmath.fft.fft(a, axes=[-2, -1], stream=s2)

   # 'a' was allocated on s1, so dropping the last reference frees it on s1.
   # s2 may still be reading 'a', so s1 must wait for s2 first.
   e2 = s2.record()
   s1.wait_event(e2)
   a = None


Stateful APIs
=============

Stateful APIs expose multiple phases (construction, :meth:`~nvmath.fft.FFT.plan`,
:meth:`~nvmath.fft.FFT.execute`, :meth:`~nvmath.fft.FFT.reset_operand`, etc.), each
accepting an optional ``stream``. This gives fine-grained control over which stream each
step uses, but stream ordering becomes more involved when using more than one stream.

.. rubric:: Execution space mismatching memory space (blocking)

When the execution space does not match the operand memory space (e.g. the most
common use case is when the operands are on the host and the execution
is on the GPU), each method call is blocking and stream ordering is
automatically enforced, just as with the stateless API.

.. tabs::

   .. code-tab:: python NumPy

      import numpy as np
      import nvmath
      from cuda.core import Device

      dev = Device(0)
      dev.set_current()

      # Create operands on the host.
      a = np.random.rand(64, 256, 128) + 1j * np.random.rand(64, 256, 128)
      b = np.random.rand(64, 256, 128) + 1j * np.random.rand(64, 256, 128)

      s1 = dev.create_stream()

      # Create stateful FFT object with CUDA execution.
      with nvmath.fft.FFT(a, axes=[0, 1], execution="cuda") as f:
         f.plan()

         # Since a was originally allocated on the host, calling f.execute()
         # is blocking and won't return until the operation has completed.
         r1 = f.execute()
         # r1 is ready to use immediately.

         # Set new operand b and run the second FFT on s1.
         f.reset_operand(b, stream=s1)
         r2 = f.execute(stream=s1)
         # r2 is ready to use immediately.

   .. code-tab:: python PyTorch

      import torch
      import nvmath

      # Create operands on the host.
      a = torch.randn(64, 256, 128, dtype=torch.complex64)
      b = torch.randn(64, 256, 128, dtype=torch.complex64)

      s1 = torch.cuda.Stream()

      # Create stateful FFT object with CUDA execution.
      with nvmath.fft.FFT(a, axes=[0, 1], execution="cuda") as f:
         f.plan()

         # Since a was originally allocated on the host, calling f.execute()
         # is blocking and won't return until the operation has completed.
         r1 = f.execute()
         # r1 is ready to use immediately.

         # Set new operand b and run the second FFT on s1.
         f.reset_operand(b, stream=s1)
         r2 = f.execute(stream=s1)
         # r2 is ready to use immediately.


.. rubric:: Execution space matching memory space (non-blocking)

When the operands reside on the GPU and execution is on the GPU, you must ensure
correct ordering *between* API calls when using different streams. For example,
if you run :meth:`~nvmath.fft.FFT.execute` on stream ``s1`` and later call
:meth:`~nvmath.fft.FFT.reset_operand` and :meth:`~nvmath.fft.FFT.execute` on
stream ``s2``, you must order the work on ``s2`` to start only after the previous
execute on ``s1`` has completed (e.g. by having ``s2`` wait on an event recorded
on ``s1`` after the first execute).

.. code-block:: python

   import cupy as cp
   import nvmath

   shape = 512, 256, 256
   axes = 0, 1

   s1 = cp.cuda.Stream()
   s2 = cp.cuda.Stream()

   # Create operand on stream s1.
   with s1:
      a = cp.random.rand(*shape, dtype=cp.float64) + 1j * cp.random.rand(*shape, dtype=cp.float64)

   # Create stateful FFT object, initializing it using stream s1.
   with nvmath.fft.FFT(a, axes=axes, stream=s1) as f:
      f.plan(stream=s1)
      b = f.execute(stream=s1)
      e1 = s1.record()

      with s2:
         c = cp.random.rand(*shape, dtype=cp.float64) + 1j * cp.random.rand(*shape, dtype=cp.float64)

      # Order s2 after the first execute on s1 to avoid race conditions.
      s2.wait_event(e1)
      # Set new operand c and run the second FFT on s2.
      f.reset_operand(c, stream=s2)
      d = f.execute(stream=s2)
      # Ensure s2 has completed before accessing the result.
      s2.synchronize()

The same ordering requirement applies when the **current stream** changes
between method calls. When no ``stream`` argument is passed (see, e.g.,
:meth:`~nvmath.fft.FFT.plan` or :meth:`~nvmath.fft.FFT.execute`), each method
uses the operands' package's current stream. If that current stream changes
between calls, the user must add explicit ordering.
The snippet below highlights such a scenario:

.. code-block:: python

   import torch
   import nvmath

   device = 0
   a = torch.randn(64, 256, 128, dtype=torch.complex64, device=f"cuda:{device}")
   s1 = torch.cuda.Stream(device=device)

   print(f"{'device index':<36} {device}")
   print(f"{'s1.cuda_stream':<36} {s1.cuda_stream}")
   print(
      f"{'current_stream(device)':<36} {torch.cuda.current_stream(device).cuda_stream}  (before plan)"
   )

   # Create stateful FFT object.
   f = nvmath.fft.FFT(a, axes=[0, 1])

   # Calling plan() without passing a stream argument means that it uses the
   # operand package's current stream on the CUDA device detected from the
   # operand when the stateful object was initialized.
   # In this case, the device detected is gpu:0, so nvmath-python internally
   # queries the current stream using torch.cuda.current_stream(device=0).
   current_stream_at_plan_time = torch.cuda.current_stream(device)
   print(f"{'current_stream_at_plan_time':<36} {current_stream_at_plan_time.cuda_stream}")
   f.plan()

   print(
      f"{'current_stream(device)':<36} {torch.cuda.current_stream(device).cuda_stream}  (after plan)"
   )

   # Inside the following block the package's current stream becomes s1,
   # so calling execute() with no stream argument will run on s1.
   # The user must order s1 after the stream plan() used—the same object
   # returned by current_stream() above.
   s1.wait_stream(current_stream_at_plan_time)
   with s1:
      print(
         f"{'current_stream(device)':<36} {torch.cuda.current_stream(device).cuda_stream}  (inside 'with s1:')"
      )
      print(
         f"{'current_stream == s1':<36} {torch.cuda.current_stream(device) == s1}"
      )
      r = f.execute()

   print(
      f"{'current_stream(device)':<36} {torch.cuda.current_stream(device).cuda_stream}  (after 'with s1:')"
   )

   s1.synchronize()
   f.free()


.. rubric:: User responsibility for operand lifetime

The :ref:`stream-ordered deallocation <stream-ordered-deallocations>` scenario
described above also applies to stateful APIs.
In particular, :meth:`~nvmath.fft.FFT.release_operand`, :meth:`~nvmath.fft.FFT.free`,
and the context-manager exit can drop the last reference to a user operand, triggering
deallocation on the operand's *allocation* stream (see the
:cuda_doc:`CUDA stream-ordered memory allocator <04-special-topics/stream-ordered-memory-allocation.html>`).
If a different stream is still
accessing the operand, the allocation stream must wait before the reference is dropped,
as shown in the example below.

.. code-block:: python

   import cupy as cp
   import nvmath

   def make_fft(stream):
       with stream:
           a = cp.random.rand(64, 128) + 1j * cp.random.rand(64, 128)
       return nvmath.fft.FFT(a, axes=[-2, -1], stream=stream, execution="cuda")

   s_alloc = cp.cuda.Stream()
   s_exec = cp.cuda.Stream()

   # Note that the operand is created inside make_fft, the fft object
   # is the sole owner of the operand because no external reference
   # to it outside the FFT object exists.
   f = make_fft(s_alloc)
   f.plan(stream=s_alloc)

   # Since plan() enqueued work on s_alloc, s_exec must wait for it
   # to finish before we can safely execute on s_exec.
   s_exec.wait_event(s_alloc.record())
   f.execute(stream=s_exec)

   # DANGER: the operand created inside make_fft is only held by the FFT object,
   # so release_operand() will drop the last reference and trigger deallocation
   # on s_alloc. We must ensure s_exec has finished first.
   s_alloc.wait_event(s_exec.record())
   f.release_operand()

   f.free()

For a full multi-stream example, see `FFT with multiple streams
<https://github.com/NVIDIA/nvmath-python/tree/main/examples/fft/example09_streams.py>`_.
