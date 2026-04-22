**************************
Distributed Linear Algebra
**************************

.. _distributed-linalg-overview:

Overview
========

The distributed Linear Algebra module :mod:`nvmath.distributed.linalg.advanced` in
nvmath-python leverages the NVIDIA cuBLASMp library and provides a powerful suite
of APIs that can be directly called from the host to efficiently perform matrix
multiplications on multi-node multi-GPU systems at scale. Both stateless
function-form APIs and stateful class-form APIs are provided.

The distributed matrix multiplication APIs are similar to their non-distributed host
API counterparts, with the key difference that the operands to the API on each process
are the **local partition** of the global operands, and the user specifies the
**distribution** (how the data is partitioned across processes). The APIs natively
support the block-cyclic distribution (see :ref:`distribution-block`).

Operand distribution
--------------------

To perform a distributed operation, first you have to specify how the operand is
distributed across processes. Distributed matrix multiply natively supports the
block-cyclic distribution (see :ref:`distribution-block`), therefore you must
provide a distribution compatible with block-cyclic. Compatible distributions
include :ref:`distribution-block-cyclic`, :ref:`distribution-block-non-cyclic`,
and :ref:`distribution-slab` (with uniform partition sizes).

Memory layout
-------------

cuBLASMp requires operands to use Fortran-order memory layout, while Python libraries
such as NumPy and PyTorch use C-order by default.
See :ref:`distribution-mem-layout` for guidelines on memory layout conversion
for distributed operands and potential implications on distribution.

Matrix qualifiers
-----------------

Matrix qualifiers are used to indicate whether an input matrix is transposed or not.

For example, for ``A.T @ B`` you have to specify:

.. code-block:: python

    from nvmath.distributed.linalg.advanced import matrix_qualifiers_dtype, matmul

    qualifiers = np.zeros((3,), dtype=matrix_qualifiers_dtype)
    qualifiers[0]["is_transpose"] = True  # a is transposed
    qualifiers[1]["is_transpose"] = False  # b is not transposed (optional)

    ...

    result = matmul(a, b, distributions=distributions, qualifiers=qualifiers)

.. caution::
    A common strategy to convert memory layout to Fortran-order (required by cuBLASMp)
    is to transpose the input matrices, as explained in :ref:`distribution-mem-layout`.
    Remember to set the matrix qualifiers accordingly.


Distributed algorithm
---------------------

cuBLASMp implements efficient communication-overlap algorithms that are suited for
distributed machine learning scenarios with tensor parallelism.
Algorithms include AllGather+GEMM and GEMM+ReduceScatter.
These algorithms have special requirements in terms of how each of the operands
is distributed and their transpose qualifiers.

Currently, to be able to use these algorithms, cuBLASMp requires: the matrices to be
distributed using a 1D partitioning scheme without the cyclic distribution and
the partition sizes to be uniform (:ref:`distribution-block-non-cyclic`
and :ref:`distribution-slab` are valid distributions for this use case).

Please refer to
`cuBLASMp documentation <https://docs.nvidia.com/cuda/cublasmp/usage/tp.html>`_
for full details.

Epilogue input and output distribution
--------------------------------------

Generally, the distribution of an epilogue input or output is the same as the distribution
of the matrix associated with that input/output. For example, the bias vector is applied
to the matmul result and has the same distribution: if the result matrix is partitioned on
the M dimension, the bias is similarly partitioned; if the result is partitioned on the N
dimension (but not M) the bias is replicated on every process.

For bias gradient epilogues, the epilogue output distribution is as follows:

- For BGRAD and BGRADA it follows how matrix A is partitioned on M.

- For BGRADB the epilogue output is always replicated.

Example
-------

The following example performs :math:`\alpha A @ B + \beta C` with inputs distributed
according to a :ref:`distribution-slab` distribution (partitioning on a single dimension):

.. note::
    To use the distributed Matmul APIs you need to
    :ref:`initialize the distributed runtime <distributed-api-initialize>`
    with the NCCL communication backend.

.. code-block:: python

    import cupy as cp
    from nvmath.distributed.distribution import Slab
    from nvmath.distributed.linalg.advanced import matrix_qualifiers_dtype

    # Get my process rank.
    rank = nvmath.distributed.get_context().process_group.rank

    # The global problem size m, n, k
    m, n, k = 128, 512, 1024

    # Prepare sample input data.
    with cp.cuda.Device(device_id):
        a = cp.random.rand(*Slab.X.shape(rank, (m, k)))
        b = cp.random.rand(*Slab.X.shape(rank, (n, k)))
        c = cp.random.rand(*Slab.Y.shape(rank, (n, m)))

    # Get transposed views with Fortran-order memory layout
    a = a.T  # a is now (k, m) with Slab.Y
    b = b.T  # b is now (k, n) with Slab.Y
    c = c.T  # c is now (m, n) with Slab.X

    distributions = [Slab.Y, Slab.Y, Slab.X]

    qualifiers = np.zeros((3,), dtype=matrix_qualifiers_dtype)
    qualifiers[0]["is_transpose"] = True  # a is transposed

    alpha = 0.45
    beta = 0.67

    # Perform the distributed GEMM.
    result = nvmath.distributed.linalg.advanced.matmul(
        a,
        b,
        c=c,
        alpha=alpha,
        beta=beta,
        distributions=distributions,
        qualifiers=qualifiers,
    )

    # Synchronize the default stream, since by default the execution
    # is non-blocking for GPU operands.
    cp.cuda.get_current_stream().synchronize()

    # result is distributed row-wise
    assert result.shape == Slab.X.shape(rank, (m, n))


You can find many more examples `here
<https://github.com/NVIDIA/nvmath-python/tree/main/examples/distributed/
linalg/advanced/matmul>`_.


.. _distributed-linalg-api-reference:

API Reference
=============

.. module:: nvmath.distributed.linalg.advanced

Distributed Linear Algebra APIs (:mod:`nvmath.distributed.linalg.advanced`)
---------------------------------------------------------------------------

.. autosummary::
   :toctree: generated/

   matmul
   matrix_qualifiers_dtype
   Matmul
   MatmulComputeType
   MatmulEpilog
   MatmulAlgoType

.. autosummary::
   :toctree: generated/
   :template: dataclass

   MatmulEpilogPreferences
   MatmulOptions
   MatmulPlanPreferences
   MatmulQuantizationScales

Helpers
^^^^^^^

The module :mod:`nvmath.linalg.advanced.helpers.matmul` provides helper
functions that facilitate working with the narrow-precision features of
:mod:`nvmath.distributed.linalg.advanced` module.
