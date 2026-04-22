*****************
Tensor Operations
*****************

.. _tensor-overview:

Overview
========

The tensor module :mod:`nvmath.tensor` in nvmath-python provides APIs for tensor
operations powered by the high-performance NVIDIA cuTENSOR library. We currently
offer binary and ternary contraction APIs supporting the CUDA execution space.

For contracting a tensor network, refer to the `Network
<https://docs.nvidia.com/cuda/cuquantum/latest/python/generated/
cuquantum.tensornet.Network.html>`_
API from the cuQuantum library. While network contraction can be used for
binary and ternary contraction, the focus here is on the optimal contraction of a *tensor
network* and therefore not all options pertinent to each pairwise
contraction are available to the user. The generalized binary
:math:`\alpha \; a \cdot b + \beta \; c` and ternary
:math:`\alpha \; a \cdot b \cdot c + \beta \; d` contraction operations
(where :math:`\cdot` represents tensor contraction)
in this module are fused, and support options specific to efficient
execution of these operations.

.. code-block:: python

    import cupy as cp
    from cupyx.profiler import benchmark

    import nvmath

    a = cp.random.rand(64, 8, 8, 6, 6)
    b = cp.random.rand(64, 8, 8, 6, 6)

    # Create a stateful BinaryContraction object 'contraction'.
    with nvmath.tensor.BinaryContraction("pijkl,pjiab->lakbp", a, b) as contraction:
        # Get the handle to the plan preference object
        plan_preference = contraction.plan_preference
        # update the kernel rank to the third best for the underlying algorithm
        plan_preference.kernel_rank = 2

        for algo in (
            nvmath.tensor.ContractionAlgo.DEFAULT_PATIENT,
            nvmath.tensor.ContractionAlgo.GETT,
            nvmath.tensor.ContractionAlgo.TGETT,
            nvmath.tensor.ContractionAlgo.TTGT,
            nvmath.tensor.ContractionAlgo.DEFAULT,
        ):
            print(f"Algorithm: {algo.name}")
            plan_preference.algo = algo
            # Plan the Contraction to activate the updated plan preference
            contraction.plan()
            print(benchmark(contraction.execute, n_repeat=20))

More examples of tensor operations can be found on our
`GitHub <https://github.com/NVIDIA/nvmath-python/tree/main/examples/tensor>`_ repository.

.. _tensor-api-reference:

Host API Reference
==================

.. module:: nvmath.tensor


Tensor Operations (:mod:`nvmath.tensor`)
----------------------------------------

.. autosummary::
   :toctree: generated/

   binary_contraction
   ternary_contraction
   tensor_qualifiers_dtype
   BinaryContraction
   TernaryContraction
   ContractionAlgo
   ContractionAutotuneMode
   ContractionJitMode
   ContractionCacheMode
   ComputeDesc
   ContractionPlanPreference
   Operator

.. autosummary::
   :toctree: generated/
   :template: dataclass

   ContractionOptions
   ExecutionCUDA
