.. _stateful_apis_guide:

Stateful APIs: Design and Usage Patterns
****************************************

nvmath-python's stateful APIs (called from host code) are a distinctive feature
that revolve around an explicit multi-phase lifecycle: initialization,
planning, execution, operand resetting, and resource release.
Separating planning from execution and operand manipulation provides fine-grained
control enabling performance optimizations that
are not possible with stateless function calls, particularly when the same
operation needs to be performed multiple times
with different (but compatible) data.
The consistent design across all modules ensures that the same concepts
are applicable across different operations (FFT, linear algebra,
sparse operations, etc.).

This page describes the design principles, lifecycle, and usage patterns
of these stateful APIs.


Multi-Phase Design
==================

nvmath-python's stateful APIs revolve around several key phases:

#. **Problem Specification**

   This phase corresponds to initializing the stateful object and is designed
   to be lightweight. It involves providing input operands, defining the
   operation, and setting options that affect its execution.
   The operands are validated to ensure they are suitable for the operation.

#. **Planning**

   The planning phase (``plan()``) is generally the most resource-intensive step.
   It selects the optimal strategy for the defined operation and may include
   autotuning when available. This expensive work is performed once and the
   resulting plan can be reused across multiple executions.

#. **Execution and operand manipulation**

   This phase (``execute()``) allows for repeated execution, e.g. when the operand
   is modified in-place or explicitly reset using the ``reset_operands()`` or
   ``reset_operands_unchecked()`` methods (or ``reset_operand()`` and
   ``reset_operand_unchecked()`` for single-operand APIs such as
   :class:`nvmath.fft.FFT`). The costs associated with initializing and planning
   can therefore be amortized over multiple executions.

   .. note::

      When resetting operands, the new ones must be compatible with those
      provided during initialization. Two sets of constraints apply. The
      first is enforced across all modules: the new operands must come from
      the same package and memory space as the originals (see
      :ref:`package-compatibility` and :ref:`one-object-one-memory-space`).
      The second is more relaxed: operands with the same shape, dtype, and
      strides as the originals are always accepted, and some modules
      additionally allow these to differ in some cases. Refer to each class's
      ``reset_operand()`` or ``reset_operands()`` method for the
      module-specific details.

   .. note::

       Some stateful APIs have specialized execution methods instead of a
       generic ``execute()``. For example,
       :class:`~nvmath.sparse.advanced.DirectSolver` has
       :meth:`~nvmath.sparse.advanced.DirectSolver.factorize` and
       :meth:`~nvmath.sparse.advanced.DirectSolver.solve`. The same principles
       apply: these methods can be called multiple times to amortize planning
       costs, and operands can be reset between executions.

#. **Resource Management**

   Users are advised to use stateful objects from within a context using the
   `with statement
   <https://docs.python.org/3/reference/compound_stmts.html#the-with-statement>`_,
   which automatically handles the release of internal resources upon exit. If
   the object is not used as a context manager using ``with``, it is necessary
   to explicitly call the ``free()`` method to ensure all resources are properly
   released.

   .. note::

       Requiring that an explicit ``free()`` is called for resource release is
       motivated by the fact that Python's garbage collector may delay freeing
       object resources when the object goes out of scope or its reference count
       drops to zero. For details, refer to the `__del__ method Python
       documentation
       <https://docs.python.org/3/reference/datamodel.html#object.__del__>`_.

   Stateful objects also support **partial resource release** through
   ``release_operands()`` (or ``release_operand()`` for single-operand APIs
   such as :class:`nvmath.fft.FFT`). This drops references to user-provided
   operands while preserving the planned state, which is useful for reducing
   peak memory usage. See :meth:`nvmath.fft.FFT.release_operand` for full
   semantics.

.. _package-compatibility:

Operands' Package Compatibility
===============================

Each stateful API defines which external package combinations are accepted
for its tensor operands. Across all dense APIs (e.g.,
:class:`~nvmath.tensor.BinaryContraction`,
:class:`~nvmath.tensor.TernaryContraction`,
:class:`nvmath.linalg.advanced.Matmul`, and
:class:`nvmath.distributed.linalg.advanced.Matmul`), all operands must belong to the
**same** package (NumPy, CuPy, or PyTorch). The :mod:`nvmath.sparse` module
(e.g., :class:`~nvmath.sparse.advanced.DirectSolver`
and :class:`nvmath.sparse.Matmul`) is an exception because it supports operands
to come from different but compatible packages (for example, SciPy--NumPy
or CuPy.sparse--CuPy).

The package combination established at initialization remains fixed for the
lifetime of the object: any operands supplied after initialization, for
example when resetting operands for repeated executions, must preserve the
original package relationship.

Refer to each module's documentation for more details on this topic.

.. _one-object-one-memory-space:

One Stateful Object, One Memory Space
=====================================

A critical requirement for stateful APIs is **memory space consistency**. Once a stateful
object is initialized with operands in a particular memory space (CPU or specific GPU
device), all subsequent operations *must* use operands in the same memory space:

- If initialized with CPU arrays, all reset operands must also be CPU arrays
- If initialized with GPU arrays on device 0, all reset operands must be on device 0
- Mixing memory spaces is not allowed and will raise an error

This design choice ensures:

- Users explicitly control when and where data movement occurs (no hidden
  device-to-device copies)
- Consistency with the prepared execution plan (no hidden replanning)

If you need to work with operands on different devices, create separate stateful objects
for each device. This is why the initialization is meant to be lightweight.

Memory space consistency requirements are documented in the API reference for each
stateful class (e.g., :class:`nvmath.fft.FFT`, :class:`nvmath.linalg.advanced.Matmul`,
:class:`nvmath.sparse.advanced.DirectSolver`, and
:class:`nvmath.distributed.linalg.advanced.Matmul`).
Refer to the module-specific documentation for details on how these requirements apply
to each operation.

One Stateful Object, One Execution Device
=========================================

Closely related to memory space consistency is the concept of a **fixed execution device**.
When a stateful object is initialized, the execution device is determined based on the
operands and execution options, and this device remains fixed for the lifetime of the
object. All subsequent operations must be consistent with the same device.

The execution device determination and its immutability are documented in the API
reference for each stateful class. Refer to the module-specific documentation for
details on execution options and device selection.

.. _one-object-one-stream:

One Stateful Object, One Stream At A Time
==========================================

A stateful object must only be used by **one CUDA stream at a time**.
Simultaneous use from multiple streams is not supported.
If you need to pass different streams to different method calls, see
:ref:`stream-semantics-guide` for ordering requirements and examples.


Key Takeaways
=============

- Follow the lifecycle: initialize -> plan -> execute (repeatedly) -> free
- Use ``reset_operands()`` to reuse plans with new compatible data
- One memory space and one execution device, fixed at initialization
- Only one CUDA stream may use a stateful object at a time
- Use context managers (``with`` statement) for automatic resource cleanup


Getting Started with Stateful APIs
==================================

Once you are familiar with the pattern for one API, the consistent design across
modules makes it straightforward to apply the same concepts elsewhere. The links
below provide module-specific documentation and examples.

**API Documentation:**

- :ref:`host api types`: Overview of stateless vs. stateful APIs
- :class:`nvmath.fft.FFT`: FFT stateful API class documentation
- :class:`nvmath.linalg.advanced.Matmul`:
  Advanced matrix multiplication stateful API class documentation
- :class:`nvmath.sparse.advanced.DirectSolver`:
  Sparse direct solver stateful API class documentation
- :class:`nvmath.distributed.fft.FFT`:
  Distributed FFT stateful API class documentation
- :class:`nvmath.distributed.linalg.advanced.Matmul`:
  Distributed advanced matrix multiplication stateful API class documentation

**Notebooks:**

- :doc:`Introduction to GEMM with nvmath-python <tutorials/notebooks/matmul/01_introduction>`

**Examples on GitHub:**

Basic stateful usage:
- `FFT <fft-stateful-example_>`_
- `Matrix multiplication <matmul-stateful-example_>`_
- `Sparse direct solver <solver-stateful-example_>`_
- `Distributed matrix multiplication <distributed-matmul-stateful-example_>`_

In-place operand modification:
- `FFT <fft-inplace-example_>`_
- `Matrix multiplication <matmul-inplace-example_>`_

Resetting operands:
- `FFT <fft-reset-example_>`_
- `Sparse direct solver <solver-reset-example_>`_
- `Tensor contraction <tensor-reset-example_>`_

Resource management: `FFT <fft-resource-example_>`_

.. _fft-stateful-example:
   https://github.com/NVIDIA/nvmath-python/tree/main/examples/fft/example02_stateful_cupy.py
.. _matmul-stateful-example:
   https://github.com/NVIDIA/nvmath-python/tree/main/examples/linalg/advanced/matmul/example04_stateful_cupy.py
.. _solver-stateful-example:
   https://github.com/NVIDIA/nvmath-python/tree/main/examples/sparse/advanced/direct_solver/example04_stateful_cupy.py
.. _distributed-matmul-stateful-example:
   https://github.com/NVIDIA/nvmath-python/tree/main/examples/distributed/linalg/advanced/matmul/example05_stateful_cupy.py
.. _fft-inplace-example:
   https://github.com/NVIDIA/nvmath-python/tree/main/examples/fft/example05_stateful_inplace.py
.. _matmul-inplace-example:
   https://github.com/NVIDIA/nvmath-python/tree/main/examples/linalg/advanced/matmul/example05_stateful_inplace.py
.. _fft-reset-example:
   https://github.com/NVIDIA/nvmath-python/tree/main/examples/fft/example05_stateful_reset.py
.. _solver-reset-example:
   https://github.com/NVIDIA/nvmath-python/tree/main/examples/sparse/advanced/direct_solver/example05_reset_operands.py
.. _tensor-reset-example:
   https://github.com/NVIDIA/nvmath-python/tree/main/examples/tensor/contraction/example06_stateful_reset.py
.. _fft-resource-example:
   https://github.com/NVIDIA/nvmath-python/tree/main/examples/fft/example11_resource_mgmt.py
