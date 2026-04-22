*********
Host APIs
*********

The following modules of nvmath-python offer integration with NVIDIA's
high-performance computing libraries such as cuBLAS, cuDSS, cuFFT, and
cuTENSOR (and their NVPL counterparts) through host APIs.  Host APIs
are called from host code but can execute in any supported execution
space (CPU or GPU).

============
Key Concepts
============

.. _matrix-tensor-qualifiers:

----------------------------
Matrix and Tensor Qualifiers
----------------------------

nvmath-python :ref:`interoperates <host api interop>` with operands from
NumPy, CuPy, and PyTorch, whose array types do not inherently encode all
the metadata an operation may require (e.g., triangular structure or lazy
conjugation). We therefore provide such auxiliary information via the notion
of qualifiers on the tensor operands, which is supplied as a NumPy ndarray
of the same length as the number of operands with the appropriate qualifiers
dtype. Each qualifier in the array provides auxiliary information about the
corresponding operand.

The following example shows a matrix multiplication between two matrices,
:math:`a` and :math:`b`, where :math:`a` should be treated as a regular
dense matrix while :math:`b` as a lower triangular matrix.
Note how the qualifier is used to inform the API of :math:`b`'s triangular structure.

.. code-block:: python

    import numpy as np
    import nvmath

    # Prepare sample input data.
    m, k = 123, 789
    a = np.random.rand(m, k).astype(np.float32)
    b = np.tril(np.random.rand(k, k).astype(np.float32))

    # We can choose the execution space for the matrix multiplication using ExecutionCUDA or
    # ExecutionCPU. By default, the execution space matches the operands, so in order to execute
    # a matrix multiplication on NumPy arrays using CUDA we need to specify ExecutionCUDA.
    # Tip: use help(nvmath.linalg.ExecutionCUDA) to see available options.
    execution = nvmath.linalg.ExecutionCUDA()

    # We can use structured matrices as inputs by providing the corresponding qualifier which
    # describes the matrix. By default, all inputs are assumed to be general matrices.
    # MatrixQualifiers are provided as an array of custom NumPy dtype,
    # nvmath.linalg.matrix_qualifiers_dtype.
    qualifiers = np.full(
       (2,), nvmath.linalg.GeneralMatrixQualifier.create(),
       dtype=nvmath.linalg.matrix_qualifiers_dtype
    )
    qualifiers[1] = nvmath.linalg.TriangularMatrixQualifier.create(
       uplo=nvmath.linalg.FillMode.LOWER
    )

    result = nvmath.linalg.matmul(a, b, execution=execution, qualifiers=qualifiers)

The following example shows how a qualifier is used to conjugate a CuPy tensor
operand as part of the contraction operation.  Since complex-conjugation is a
memory-bound operation, this fusion improves performance compared to the
alternative of performing the conjugation *a priori* using CuPy.

.. code-block:: python

    import cupy as cp
    import numpy as np
    import nvmath

    a = cp.random.rand(8, 8, 8, 8) + 1j * cp.random.rand(8, 8, 8, 8)
    b = cp.random.rand(8, 8, 8, 8) + 1j * cp.random.rand(8, 8, 8, 8)
    c = cp.random.rand(8, 8, 8, 8) + 1j * cp.random.rand(8, 8, 8, 8)
    d = cp.random.rand(8, 8, 8, 8) + 1j * cp.random.rand(8, 8, 8, 8)

    # create an array of qualifiers (of length # of operands) with the default identity operator
    qualifiers = np.full(
       4, nvmath.tensor.Operator.OP_IDENTITY,
       dtype=nvmath.tensor.tensor_qualifiers_dtype
    )
    # set the qualifier for operand b to conjugate
    qualifiers[1] = nvmath.tensor.Operator.OP_CONJ

    # result[i,j,p,q] = \sum_{k,l,m,n} a[i,j,k,l] * b[k,l,m,n].conj() * c[m,n,p,q] + d[i,j,p,q]
    result = nvmath.tensor.ternary_contraction(
       "ijkl,klmn,mnpq->ijpq",
       a, b, c, d=d, qualifiers=qualifiers, beta=1
    )

.. seealso::

   :class:`nvmath.linalg.matrix_qualifiers_dtype`,
   :class:`nvmath.linalg.advanced.matrix_qualifiers_dtype`,
   :class:`nvmath.distributed.linalg.advanced.matrix_qualifiers_dtype`,
   :class:`nvmath.tensor.tensor_qualifiers_dtype`

Examples using qualifers can be found in the
`examples <https://github.com/NVIDIA/nvmath-python/tree/main/examples>`_
directory on GitHub.

========
Contents
========

.. toctree::
   :caption: API Reference
   :maxdepth: 2

   Linear Algebra <linalg/index.rst>
   Sparse Linear Algebra <sparse/index.rst>
   Fast Fourier Transform <fft/index.rst>
   Tensor Operations <tensor/index.rst>
   Host API Utilities <utils.rst>
