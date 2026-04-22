# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

__all__ = ["Matmul", "matmul"]

import functools
import logging
import operator
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import IntEnum

try:
    import cuda.core as cc
except ImportError:
    import cuda.core.experimental as cc

import numpy as np

from nvmath import memory
from nvmath._internal.layout import check_monotonic_strides, is_contiguous_in_memory
from nvmath.bindings import cusparse
from nvmath.internal import formatters, tensor_wrapper, typemaps, utils
from nvmath.sparse._internal import common_utils as sp_utils
from nvmath.sparse._internal import cusparse_utils
from nvmath.sparse._internal.sparse_ust_ifc import USTensorHolder
from nvmath.sparse._internal.utils import calculate_strides
from nvmath.sparse.ust._kernel import KernelGen
from nvmath.sparse.ust.tensor import Tensor as UST

from ._configuration import ComputeType, ExecutionCUDA, MatmulOptions, matrix_qualifiers_dtype
from ._helpers import compile_prolog
from ._thunks import default_prolog

# TODO: make sure KernelGen is thread-safe.
KERNEL_CACHE = KernelGen(False)


CUSPARSE_FORMATS = ["CSR", "CSC", "BSR", "BSC", "COO"]

CODEGEN_FORMATS = ["UST", "DIA"] + CUSPARSE_FORMATS

# Supported index and data types.

CUSPARSE_INDEX_TYPES = {"int32", "int64"}

CODEGEN_INDEX_TYPES = {"int32", "int64"}

VALID_INDEX_TYPES = CUSPARSE_INDEX_TYPES | CODEGEN_INDEX_TYPES

CUSPARSE_DTYPES = {"float16", "bfloat16", "float32", "float64", "complex32", "complex64", "complex128"}

CODEGEN_DTYPES = {"float16", "bfloat16", "float32", "float64", "complex64", "complex128"}

VALID_DTYPES = CUSPARSE_DTYPES | CODEGEN_DTYPES

CUSPARSE_COMPUTE_TYPES = {"float32", "float64", "complex64", "complex128"}

CODEGEN_COMPUTE_TYPES = {"float32", "float64", "complex64", "complex128"}

VALID_COMPUTE_TYPES = CUSPARSE_COMPUTE_TYPES | CODEGEN_COMPUTE_TYPES


# TODO:
# For the dense tensors forming the sparse tensor representation, check that they are
# dense and contiguous in C-order.


class BackendSupport(IntEnum):
    """An IntEnum class for capturing dispatch and codegen support for the MM."""

    NOT_SUPPORTED = 0
    PROVISIONALLY_SUPPORTED = 1
    SUPPORTED = 2


class Api(IntEnum):
    """An IntEnum class for capturing the codegen or dispatch API."""

    NONE = 0
    CODEGEN = 1
    MM = 2
    MM_OP = 3


class InvalidMatmulState(Exception):
    pass


# TODO: generic (semantic) compatibility check.
# dispatch or codegen specific support checks (dtype, compute type, batching,
# num dense dim) etc.


def _get_cusparse_op_code(qualifier):
    if qualifier["is_transpose"] and qualifier["is_conjugate"]:
        return cusparse.Operation.CONJUGATE_TRANSPOSE

    if qualifier["is_transpose"]:
        return cusparse.Operation.TRANSPOSE

    assert not qualifier["is_conjugate"], "Internal error."

    return cusparse.Operation.NON_TRANSPOSE


def _is_regular_precision(dtype_name):
    e = r"(complex|b?float)(\d+)"
    m = re.search(e, dtype_name)
    assert m is not None, "Internal error."
    size = int(m.group(2))
    is_complex = m.group(1) == "complex"
    if is_complex:
        size //= 2
    return size >= 32


# Require that conjugate and transpose be specified together.
@dataclass
class DenseLayoutTraits:
    """An internal data class for capturing dense matrix traits."""

    order: cusparse.Order
    ld: int
    mm_shape: Sequence[int]
    batch_shape: Sequence[int]
    batch_count: int
    batch_offset: int  # Based on strides


@dataclass
class SparseLayoutTraits:
    """An internal data class for capturing dense matrix traits."""

    sparse_format: str
    mm_shape: Sequence[int]
    batch_count: int
    batch_broadcast: bool = False


@dataclass
class SpMMTraits:
    """An internal data class for capturing the matrix multiplication traits."""

    M: int
    N: int
    K: int
    a_layout_traits: SparseLayoutTraits
    b_layout_traits: DenseLayoutTraits
    c_layout_traits: DenseLayoutTraits
    batch_count: int
    batch_shape: Sequence[int]


def get_dense_matrix_layout_traits(
    mm_shape: Sequence[int],
    mm_strides: Sequence[int],
    batch_strides: Sequence[int],
    ordering: cusparse.Order | None = None,
    orientation: cusparse.Order | None = None,
) -> tuple[cusparse.Order, int, int]:
    """
    The 'ordering' option specifies the layout order, if it's not None, as in the case of
    the D matrix whose layout is determined by the other operands' layout. It is required if
    the matrix is degenerate (a vector or scalar: len(mm_shape) < 2).

    The 'orientation' option (ROW or COL) is required to infer the correct leading dimension
    for degenerate matrices (a vector or scalar: len(mm_shape) < 2).
    """
    if len(mm_shape) < 2:  # The result D can be a scalar or vector.
        assert ordering is not None, "Internal Error: 'ordering' must be specified for degenerate matrices."
        assert orientation is not None, "Internal Error: 'orientation' must be specified for degenerate matrices."
        batch_offset = min(batch_strides) if batch_strides else 0
        order = ordering
        if len(mm_shape) < 1:
            ld = 1
        elif order != orientation:
            # For a ROW vector in COL order, the LD should be 1 since we promote the row
            # vector to a matrix as (1, M). Similarly, for a COL vector in ROW order, the LD
            # should be 1 as well since we promote the column vector to a matrix as (M, 1).
            ld = 1
        else:
            ld = max(mm_strides[0], mm_shape[0])
        return order, ld, batch_offset

    M, N = mm_shape

    if ordering is not None:
        order = ordering
        message = f"Internal Error: incompatible ordering '{ordering}' and strides {mm_strides}"
        if order == cusparse.Order.ROW:
            assert mm_strides[0] >= mm_strides[1] and mm_strides[1] == 1, message
        else:
            assert mm_strides[1] >= mm_strides[0] and mm_strides[0] == 1, message
    else:
        # Important: start with the first dimension so that cases like (M, 1) : (1, 1) or
        # (1, M) : (1, 1) in CuTe notation map to COL.
        if mm_strides[0] == 1:
            order = cusparse.Order.COL
        elif mm_strides[1] == 1:
            order = cusparse.Order.ROW
        else:
            if M == 1:
                order = cusparse.Order.COL
            elif N == 1:
                order = cusparse.Order.ROW
            else:
                raise ValueError("Unsupported layout.")

    ld = max(M, mm_strides[1]) if order == cusparse.Order.COL else max(N, mm_strides[0])

    # Batch dimensions should be contiguous in memory, which we have already checked. The
    # batch_offset should be based on the lowest stride in the batch dimension to account
    # for embedded matrices.
    batch_offset = min(batch_strides) if batch_strides else 0

    return order, ld, batch_offset


def get_spmm_traits(a, b, c, *, qualifiers, inplace, logger):
    """
    First check A and B compatibility:

    1. The sparse operand is always 2D or higher. Gives M, K.
    2. The dense operand B can be a vector or matrix. If it's
       a vector, an implicit singleton extent is added (but it
       doesn't appear in the result. Implicitly N=1 in this case.
    3. C can be a vector or matrix compatible with (M,N).
    4. The dense matrix batch dimensions must be in C order.
    5. M, K, N must be determined considering the transpose flag.

    1. Check MM compatibility (K):
        a. First pad A and/or B MM dimensions if 1-D according to NumPy convention.
        b. The padding is used to determine M, N, and K but should not appear in the output
           dimensions.
        c. If both A and B are N-D, the dimensions must match.
    2. Check batch dimensions:
        a. One of A or B can have missing batch extents, in which case it is broadcast,
           otherwise
        b. A and B must have the same batch ordering.
        c. In addition, the batch dimensions must be tileable (contiguous in memory).

    Then check C:

    C can be None. If C is passed in, it must be matrix. Batching rule is the
    same as above.
    """

    # TODO: remove this guard once we add the number of sparse/dense dimension attributes
    # to UST.
    if not isinstance(a, USTensorHolder):
        a_num_dense_dim = a.attr_name_map["num_dense_dim"](a.tensor)
        if a_num_dense_dim > 0:
            raise ValueError(
                "The sparse matrix contains tensor elements, which is not supported for sparse matrix multiplication."
            )

    a_shape = list(a.shape)
    a_batch_shape, a_mm_shape = a_shape[:-2], a_shape[-2:]

    b_shape, b_strides = list(b.shape), list(b.strides)
    b_batch_shape, b_mm_shape = b_shape[:-2], b_shape[-2:]
    b_batch_strides, b_mm_strides = b_strides[:-2], b_strides[-2:]

    # Handle 1D `a` and `b` according to our semantics.
    d_mm_shape = []
    a_vector = False
    if len(a_mm_shape) == 1:
        a_mm_shape = [1] + a_mm_shape
        a_vector = True
    else:
        d_mm_shape.append(a_mm_shape[0])  # The first mode for d applies only when a is not a vector.

    if len(b_mm_shape) == 1:
        s, d = b_mm_shape[0], b_mm_strides[0]
        b_mm_shape = b_mm_shape + [1]
        b_mm_strides = b_mm_strides + [s * d]
    else:
        d_mm_shape.append(b_mm_shape[1])  # The second mode for d applies only when b is not a vector.

    logger.debug(f"The MM shape for operand A is {a_mm_shape}.")
    logger.debug(f"The MM shape for operand B is {b_mm_shape} with strides {b_mm_strides}.")
    logger.debug(f"The MM shape for operand D is {d_mm_shape}.")

    a_qualifiers, b_qualifiers, c_qualifiers = qualifiers
    M0, K0 = a_mm_shape
    if a_qualifiers["is_transpose"]:
        M0, K0 = K0, M0

    K1, N0 = b_mm_shape
    if b_qualifiers["is_transpose"]:
        K1, N0 = N0, K1

    if K0 != K1:
        raise ValueError(
            f"The 'K' extent must match for the operands: K={K0} in operand A is not equal to K={K1} in operand B. \
The transpose qualifiers for A and B are {a_qualifiers['is_transpose'].item(), b_qualifiers['is_transpose'].item()}."
        )

    # The batch dimensions of all component dense arrays of the sparse tensor `a` should
    # be tileable in C-order.
    batch_shape = []
    if len(a_batch_shape) > 0:
        batch_shape = a_batch_shape

    # Check if the dense operand `b` batch dimensions are tileable as well as compatible
    # with `a`.
    if len(b_batch_shape) > 0:
        if not (
            is_contiguous_in_memory(b_batch_shape, b_batch_strides) and check_monotonic_strides(b_batch_strides, reverse=True)
        ):
            message = (
                f"The batch layout for B corresponding to shape = {b_batch_shape} and strides = {b_batch_strides} "
                "is currently not supported because it is not tileable and in C-order."
            )
            raise ValueError(message)
        logger.debug(
            f"The batch layout for B corresponding to shape = {b_batch_shape} and strides = {b_batch_strides} \
IS tileable in C-order."
        )

        if not batch_shape:
            batch_shape = b_batch_shape

    if len(b_batch_shape) > 0 and b_batch_shape != batch_shape:
        raise ValueError(f"The batch dimensions of operands A {batch_shape} and B {b_batch_shape} must match.")

    num_batch_dim = len(batch_shape)
    logger.debug(f"The batch shape is {batch_shape} with batch axis in C-order.")

    batch_count = functools.reduce(operator.mul, batch_shape, 1)

    # Create sparse matrix layout traits for `a`.
    a_layout_traits = SparseLayoutTraits(
        sparse_format=a.format_name, mm_shape=a_mm_shape, batch_count=batch_count, batch_broadcast=len(a_batch_shape) == 0
    )

    # Create dense matrix layout traits for `b`.
    b_order, b_ld, b_batch_offset = get_dense_matrix_layout_traits(b_mm_shape, b_mm_strides, b_batch_strides)
    b_layout_traits = DenseLayoutTraits(
        order=b_order,
        ld=b_ld,
        mm_shape=b_mm_shape,
        batch_shape=b_batch_shape,
        batch_offset=b_batch_offset,
        batch_count=batch_count,
    )

    # Process matrix c, if provided.
    c_layout_traits = None
    if c is not None:
        # 1. C can be a vector, as long as it's columns don't need to be broadcast.
        # 2. C can be a matrix of dimension (M, N).
        # 3. C can be batched matrices of dimension (..., M, N).
        c_shape, c_strides = list(c.shape), list(c.strides)

        c_batch_shape, c_mm_shape = c_shape[:-2], c_shape[-2:]
        c_batch_strides, c_mm_strides = c_strides[:-2], c_strides[-2:]
        if len(c_mm_shape) == 0:
            c_mm_shape = [1, 1]
            c_mm_strides = [1, 1]
        elif len(c_mm_shape) == 1:
            # raise ValueError(f"C cannot be a vector. C shape: {c_mm_shape}")
            s, d = c_mm_shape[0], c_mm_strides[0]
            c_mm_shape = [1] + c_mm_shape if a_vector else c_mm_shape + [1]
            c_mm_strides = c_mm_strides + [s * d]
        logger.debug(f"The MM shape for operand C is {c_mm_shape} with strides {c_mm_strides}.")

        Mc, Nc = c_mm_shape
        if Mc != M0:
            raise ValueError(
                f"The M dimension of the C matrix ({Mc}) must match the M dimension of A ({M0}). \
The transpose qualifier for A is {a_qualifiers['is_transpose']}."
            )

        if Nc != N0:
            raise ValueError(
                f"The N dimension of the C matrix ({Nc}) must match the N dimension of B ({N0}). \
The transpose qualifier for B is {b_qualifiers['is_transpose']}."
            )

        # For the inplace option, C must be batched if an operand is batched.
        if inplace or len(c_batch_shape) > 0:
            if c_batch_shape != batch_shape:
                raise ValueError(
                    f"The batch dimension of operand C {c_batch_shape} must match with that of the other operands "
                    f"{batch_shape}."
                )

            if c_batch_shape and not (
                is_contiguous_in_memory(c_batch_shape, c_batch_strides)
                and check_monotonic_strides(c_batch_strides, reverse=True)
            ):
                raise ValueError("The batch axes of operand C must be in C-order.")

        c_order, c_ld, c_batch_offset = get_dense_matrix_layout_traits(c_mm_shape, c_mm_strides, c_batch_strides)
    else:
        # Compute the shape and strides for the result `c`.
        c_mm_shape = d_mm_shape
        c_shape = batch_shape + c_mm_shape

        c_axis_order = list(range(0, num_batch_dim))
        # Use the result_ordering from b's ordering.
        if b_order == cusparse.Order.ROW:
            c_axis_order += [num_batch_dim + 1, num_batch_dim]
        elif b_order == cusparse.Order.COL:
            c_axis_order += [num_batch_dim, num_batch_dim + 1]

        c_strides = calculate_strides(c_shape, c_axis_order)

        c_batch_shape, _ = c_shape[:-2], c_shape[-2:]
        c_batch_strides, c_mm_strides = c_strides[:-2], c_strides[-2:]

        # For degenerate matrices, we need to specify the result orientation.
        result_orientation = None
        if len(c_mm_shape) < 2:
            if M0 == 1:
                result_orientation = cusparse.Order.ROW
            elif N0 == 1:
                result_orientation = cusparse.Order.COL

        c_order, c_ld, c_batch_offset = get_dense_matrix_layout_traits(
            c_mm_shape, c_mm_strides, c_batch_strides, ordering=cusparse.Order.ROW, orientation=result_orientation
        )

    logger.debug(f"The layout order for operand C is {c_order.name}, with LD {c_ld}, and batch offset {c_batch_offset}.")
    # Create dense matrix layout traits for `c`.
    c_layout_traits = DenseLayoutTraits(
        order=c_order,
        ld=c_ld,
        mm_shape=c_mm_shape,
        batch_shape=c_batch_shape,
        batch_offset=c_batch_offset,
        batch_count=batch_count,
    )

    M, N, K = M0, N0, K0
    logger.debug(f"The SpMM operand dimensions are M={M}, N={N}, K={K}.")

    # Create the SpMM traits.
    spmm_traits = SpMMTraits(
        M=M,
        N=N,
        K=K,
        a_layout_traits=a_layout_traits,
        b_layout_traits=b_layout_traits,
        c_layout_traits=c_layout_traits,
        batch_count=batch_count,
        batch_shape=batch_shape,
    )

    return spmm_traits


SPARSE_MM_DOCUMENTATION = utils.COMMON_SHARED_DOC_MAP.copy()
SPARSE_MM_DOCUMENTATION.update(
    {
        "a": """\
A *sparse* tensor representing the first operand ``a`` in the sparse matrix multiplication
(SpMM) from one of the supported sparse packages: SciPy, CuPy, PyTorch, or a
:class:`universal sparse tensor (UST) <nvmath.sparse.ust.Tensor>` object
(see :ref:`semantics <spmm_semantics>`). The sparse representation may be in any of the formats supported by
the sparse package (CSR, BSC, COO, ...), including novel formats defined using the UST DSL.
""".replace("\n", " "),
        #
        "b": """\
A *dense* tensor representing the second operand ``b`` in the SpMM (see :ref:`semantics <spmm_semantics>`).
The currently supported types are :class:`numpy.ndarray`, :class:`cupy.ndarray`,
:class:`torch.Tensor`, and :class:`nvmath.sparse.ust.Tensor`.""".replace("\n", " "),
        "c": """\
A *dense* tensor representing the addend ``c`` in the SpMM (see :ref:`semantics <spmm_semantics>`). The currently
supported types are :class:`numpy.ndarray`, :class:`cupy.ndarray`, :class:`torch.Tensor`,
and :class:`nvmath.sparse.ust.Tensor`.""".replace("\n", " "),
        #
        "alpha": """\
The scale factor for the matrix multiplication term as a real or complex number. The default is
:math:`1.0`.""".replace("\n", " "),
        #
        "beta": """\
The scale factor for the addend term in the matrix multiplication as a real or complex number.
The default is :math:`1.0`.""".replace("\n", " "),
        #
        "qualifiers": """\
If desired, specify the matrix qualifiers as a :class:`numpy.ndarray` of
:data:`~nvmath.sparse.matmul_matrix_qualifiers_dtype` objects of length 3 corresponding to
the operands ``a``, ``b``, and ``c``. See :ref:`matrix-tensor-qualifiers` for the motivation behind
qualifiers.""".replace("\n", " "),
        #
        "options": """\
Specify options for the sparse matrix multiplication as a :class:`~nvmath.sparse.MatmulOptions`
object. Alternatively, a `dict` containing the parameters for the ``MatmulOptions`` constructor
can also be provided. If not specified, the value will be set to the default-constructed
``MatmulOptions`` object.""".replace("\n", " "),
        #
        "execution": """\
Specify execution space options for the SpMM as a :class:`ExecutionCUDA` object (the only
execution space currently supported). If not specified, a :class:`ExecutionCUDA` object will
be default-constructed.""".replace("\n", " "),
        #
        "prologs": """\
A dict mapping an operand label (``"a"``, ``"b"``, ``"c"``) to its prolog operation in LTO-IR
format (as a :class:`bytes` object). The prolog is a user-written unary function in Python
that returns the transformed value, which has the data type of the operand to which it is
applied. This function can be compiled to LTO-IR using the helper
:func:`~nvmath.sparse.compile_matmul_prolog` or your own compiler of choice. If not
specified, no prolog will be applied to the operands.""".replace("\n", " "),
        #
        "epilog": """\
The epilog operation in LTO-IR format (as a :class:`bytes` object). The epilog is a
user-written unary function in Python that returns the transformed value, which has the
data type of the SpMM result. This function can be compiled to LTO-IR using the helper
:func:`~nvmath.sparse.compile_matmul_epilog` or your own compiler of choice. If not
specified, no epilog will be applied to the SpMM result.""".replace("\n", " "),
        #
        "semiring": """\
A dict mapping the semiring operations (``"mul"``, ``"add"``, ``"atomic_add"``) to LTO-IR
code (as a :class:`bytes` object). Each semiring operation is a binary function in Python
that returns a value.  These function can be compiled to LTO-IR using the helpers
:func:`~nvmath.sparse.compile_matmul_mul`, :func:`~nvmath.sparse.compile_matmul_add`, or
:func:`~nvmath.sparse.compile_matmul_atomic_add` or your own compiler of choice. If not
specified, the standard definitions of these operations from elementary algebra
will be used.""".replace("\n", " "),
        #
        "compute_capability": """\
The target compute capability, specified as a string (``'80'``, ``'89'``, ...). The
default is the compute capability of the current device.""".replace("\n", " "),
        #
        "result": """\
The result of the sparse matrix multiplication (epilog applied). Currently only in-place
SpMM is supported (the result of the computation is written into the addend ``c``).
""".replace("\n", " "),
        #
        "release_operands": utils._release_operand_docstring(True),
        #
        "reset_operands_unchecked": utils._reset_operand_unchecked_docstring(True),
        #
        "stream": """\
Provide the CUDA stream to use for executing the operation. Acceptable inputs include
``cudaStream_t`` (as Python :class:`int`), :class:`cupy.cuda.Stream`, and
:class:`torch.cuda.Stream`. If a stream is not provided, the current stream for the
operand device will be queried from the dense operand ``b`` (and ``c``) package.
""".replace("\n", " "),
        #
        "semantics": """\
        .. _spmm_semantics:

        The semantics of the matrix multiplication follows :external:py:data:`numpy.matmul` semantics, with some restrictions on
        broadcasting. In addition, the semantics for the fused matrix addition are described below.

        * For in-place matrix multiplication (where the result is written into ``c``) the result has the same shape as ``c``.
        * The operand ``a`` must be a sparse matrix or batched sparse matrix. Popular named formats like BSC, BSR, COO,
          CSR, ...  are supported in addition to custom formats defined using the UST DSL.
        * The operands ``b`` and ``c`` must be "dense" matrices (that is, their layout is strided).
        * If the operands ``a`` and ``b`` are matrices, they are multiplied according to the rules of matrix
          multiplication.
        * If argument ``b`` is 1-D, it is promoted to a matrix by appending ``1`` to its dimensions. After matrix
          multiplication, the appended ``1`` is removed from the result's dimensions if the operation is not in-place.
        * If ``a`` or ``b`` is N-D (N > 2), then the operand is treated as a batch of matrices. If both ``a`` and ``b``
          are N-D, their batch dimensions must match. If exactly one of ``a`` or ``b`` is N-D, the other operand is
          broadcast.
        * The operand for the matrix addition ``c`` must be a matrix of shape (M, N), or the batched equivalent
          (..., M, N). Here M and N are the dimensions of the result of the matrix multiplication. If batch dimensions
          are not present, ``c`` is broadcast across batches as needed. If the operation is in-place, ``c`` cannot be
          broadcast since it must be large enough to hold the result.
""".strip(),
    }
)


@utils.docstring_decorator(SPARSE_MM_DOCUMENTATION, skip_missing=False)
class Matmul:
    """
    Create a stateful object encapsulating the specified matrix multiplication computation,
    which is one of :math:`epilog(\\alpha \\, op_h(a) \\, @ \\, op_h(b) + \\beta \\, c)` or
    :math:`epilog(prolog_a(op_t(a)) \\, @ \\, prolog_b(op_t(b)) + prolog_c(c))`, along with
    the required resources to perform the operation. The :math:`op_h` and :math:`op_t`
    operators optionally specify transpose/hermitian or transpose operations respectively
    via the ``qualifiers`` argument. In addition, the scalar multiplication and addition
    operators ("semiring") can be customized by the user, if desired.

    .. note::
        The complex conjugate operation is mutually exclusive with prolog since it can be
        absorbed into the prolog.

    .. note::
        Currently only in-place sparse matrix multiplication is supported, so operand ``c``
        must be provided. This restriction will be removed in a future release.

    A stateful object can be used to amortize the cost of preparation (planning in the
    case of matrix multiplication) across multiple executions (also see the
    :ref:`Stateful APIs <host api types>` section). Prolog, epilog, and semiring
    operations can be specified in :meth:`plan`.

    The function-form API :func:`matmul` is a convenient alternative to using stateful
    objects for *single* use (the user needs to perform just one matrix multiplication, for
    example), in which case there is no possibility of amortizing preparatory costs. The
    function-form APIs are just convenience wrappers around the stateful object APIs.

    Using the stateful object typically involves the following steps:

    1. **Problem Specification**: Initialize the object with a defined operation and
       options.
    2. **Preparation**: Use :meth:`plan` to determine the best algorithmic implementation
       for this specific matrix multiplication operation.
    3. **Execution**: Perform the matrix multiplication computation with :meth:`execute`.
    4. **Resource Management**: Ensure all resources are released either by explicitly
       calling :meth:`free` or by managing the stateful object within a context manager.

    Detailed information on what's happening in the various phases described above can be
    obtained by passing in a :class:`logging.Logger` object to :class:`MatmulOptions` or by
    setting the appropriate options in the root logger object, which is used by default:

        >>> import logging
        >>> logging.basicConfig(
        ...     level=logging.INFO,
        ...     format="%(asctime)s %(levelname)-8s %(message)s",
        ...     datefmt="%m-%d %H:%M:%S",
        ... )

    A user can select the desired logging level and, in general, take advantage of all of
    the functionality offered by the Python `logging` module.

    Args:
        a: {a}

        b: {b}

        c: {c}

        alpha: {alpha}

        beta: {beta}

        qualifiers: {qualifiers}

        options: {options}

        execution: {execution}

        stream: {stream}

    Semantics:
        {semantics}

    .. seealso::
        :class:`MatmulOptions`, :class:`ExecutionCUDA`, :meth:`plan`,
        :meth:`release_operands`, :meth:`reset_operands`,
        :meth:`reset_operands_unchecked`, :meth:`execute`, :func:`matmul`.

    Examples:

        >>> import cupy as cp
        >>> import cupyx.scipy.sparse as sp
        >>> import nvmath

        The problem parameters.

        >>> m, n, k = 4, 2, 4
        >>> index_type, dtype = "int32", "float64"

        Create a sparse float64 ndarray in CSR format on the GPU.

        >>> crow_indices = cp.array([0, 2, 4, 6, 8], dtype=index_type)
        >>> col_indices = cp.array([0, 1, 0, 1, 2, 3, 2, 3], dtype=index_type)
        >>> values = cp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=dtype)

        >>> a = sp.csr_matrix((values, col_indices, crow_indices), shape=(m, k))

        Create the dense operands ``b`` and ``c``.

        >>> b = cp.ones((k, n), dtype=dtype)
        >>> c = cp.zeros((m, n), dtype=dtype)

        We will define a sparse matrix multiplication (SpMM) operation
        :math:`c := \\alpha \\, a \\, @ \\, b + \\beta \\, c` using the generic sparse matrix
        multiplication interface.

        >>> mm = nvmath.sparse.Matmul(a, b, c=c, alpha=1.2, beta=1.0)

        Options can be provided above to control the behavior of the operation using the
        `options` argument (see :class:`MatmulOptions`).

        Next, plan the operation. Optionally, user-defined prologs, epilog, and semiring
        can be provided.

        >>> mm.plan()

        Now execute the matrix multiplication. The result ``r`` is also a CuPy ndarray, and
        specifically in this example, it is the same as operand ``c`` since the operation
        is in-place.

        >>> r = mm.execute()
        >>> assert r is c, "Error: the operation is not in-place."

        Finally, free the object's resources. To avoid having to explicitly make this
        call, it's recommended to use the Matmul object as a context manager as shown below,
        if possible.

        >>> mm.free()

        .. note:: All :class:`Matmul` methods execute on the package current stream by default.
            Alternatively, the ``stream`` argument can be used to run a method on a specified
            stream.

        Let's now look at the SpMM :math:`c := a.T \\, @ \\, b.H + c` with PyTorch sparse/dense
        tensors on the CPU.

        >>> import numpy as np
        >>> import torch

        >>> m, n, k = 2, 2, 2
        >>> index_type, dtype = torch.int32, torch.float32

        Create and coalesce a sparse COO tensor on the CPU.

        >>> indices = torch.tensor([[0, 1], [0, 1]], dtype=index_type)
        >>> values = torch.tensor([2.0, 4.0], dtype=dtype) + 1.0j
        >>> a = torch.sparse_coo_tensor(indices, values, (k, m))
        >>> a = a.coalesce()

        Create the dense operands ``b`` and ``c``.

        >>> b = torch.ones(n, k, dtype=dtype) + 1.0j
        >>> c = torch.zeros(m, n, dtype=dtype) + 0.0j

        The transpose/hermitian operations on ``a`` and ``b`` will be specified using
        :ref:`qualifiers <matrix-tensor-qualifiers>`.

        >>> qualifiers = np.zeros((3,), dtype=nvmath.sparse.matmul_matrix_qualifiers_dtype)
        >>> qualifiers[0]["is_transpose"] = 1
        >>> qualifiers[1]["is_transpose"] = qualifiers[1]["is_conjugate"] = 1

        Create the SpMM operation and use it as a context manager.

        >>> with nvmath.sparse.Matmul(a, b, c=c, qualifiers=qualifiers) as mm:
        ...     # Plan the operation.
        ...     mm.plan()
        ...
        ...     # Execute it.
        ...     r = mm.execute()

        All the resources used by the object are released at the end of the block.

        Finally, let's see how to perform a matrix multiplication on a novel format
        using UST operands.

        >>> device_id = 0
        >>> dtype = torch.float64
        >>> m, n, k = 3, 2, 8

        Create a dense torch tensor and view it as UST.

        >>> a = torch.tensor(
        ...     [[1, 0, 0, 0, 0, 0, 0, 2], [0, 0, 3, 4, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 5]],
        ...     dtype=dtype,
        ...     device=device_id,
        ... )
        >>> a = nvmath.sparse.ust.Tensor.from_package(a)

        Create a delta-compression format, using the predefined type or directly with
        the UST DSL.

        >>> delta = nvmath.sparse.ust.NamedFormats.DELTA(2)

        Convert the torch tensor to the delta-compressed format.

        >>> a = a.convert(tensor_format=delta)

        Create dense operands ``b`` and ``c`` for the SpMM.

        >>> b = torch.ones(k, n, dtype=dtype, device=device_id)
        >>> b = nvmath.sparse.ust.Tensor.from_package(b)
        >>> c = torch.zeros(m, n, dtype=dtype, device=device_id)
        >>> c = nvmath.sparse.ust.Tensor.from_package(c)

        Create, plan, and execute the SpMM operation.

        >>> with nvmath.sparse.Matmul(a, b, c=c) as mm:
        ...     # Plan the SpMM.
        ...     mm.plan()
        ...
        ...     # Execute it.
        ...     r = mm.execute()

        View the UST result as a torch tensor.

        >>> r = r.to_package()

        Examples sampling the vast space of possibilities can be found in the
        `nvmath/examples/sparse/generic/matmul
        <https://github.com/NVIDIA/nvmath-python/tree/main/examples/sparse/generic/matmul>`_
        directory.
    """  # noqa: W505

    def __init__(
        self,
        a,
        b,
        /,
        c=None,
        *,
        alpha=None,
        beta=None,
        qualifiers=None,
        options=None,
        execution: ExecutionCUDA | None = None,
        stream: utils.AnyStream | int | None = None,
    ):
        options = utils.check_or_create_options(MatmulOptions, options, "matrix multiplication options")
        assert options is not None
        self.options = options

        self.execution = execution = utils.check_or_create_options(
            ExecutionCUDA, execution, "matrix multiplication execution options"
        )

        if c is None:
            raise NotImplementedError("The operand C is currently required since only inplace SpMM is supported.")

        self.logger = options.logger if options.logger is not None else logging.getLogger()

        # Wrap operand 'a'.
        try:
            self.a = a = sp_utils.wrap_sparse_operands(a)
        except Exception as e:
            raise TypeError(
                """The operand 'a' must be an N-D sparse array/tensor from one of the supported packages: nvmath.sparse.ust,
CuPy, PyTorch, or SciPy."""
            ) from e

        self.logger.info(f"The sparse format for operand A is '{a.format_name}'.")

        # Note the sparse wrapper type for the fast path in reset_operands_unchecked().
        self.a_wrapper_type = self.a.__class__
        self.a_attr_name_map = self.a.attr_name_map
        self.logger.debug(f"The sparse wrapper type for operand A is '{self.a_wrapper_type}'.")

        if sp_utils.sparse_or_dense(b) != "dense":
            raise ValueError(f"The operand `b` {type(b)} must be dense.")

        self.ust_operands = []
        # TODO: create unified sparse-dense abstraction for UST later. For now we'll
        # extract the dense tensor from the dense UST for `b` and `c`.
        if (b_package := utils.infer_object_package(b)) == "nvmath":
            self.ust_operands.append(b)
            # Though b.wrapped_operand is the TensorHolder
            # we look for, create a new one, so that
            # matmul is free to modify it in place if needed.
            self.b = b = tensor_wrapper.wrap_operand(b.wrapped_operand.tensor)
        else:
            # Wrap operand 'b' (currently limit to dense operand).
            self.b = b = tensor_wrapper.wrap_operand(b)  # type:ignore

        self.operands = operands = [a, b]

        # For now, assume 'c' is provided.
        if sp_utils.sparse_or_dense(c) != "dense":
            raise ValueError(f"The operand `c` {type(c)} must be dense.")

        if (c_package := utils.infer_object_package(c)) == "nvmath":
            self.ust_operands.append(c)
            # Though c.wrapped_operand is the TensorHolder
            # we look for, create a new one, so that
            # matmul is free to modify it in place if needed.
            self.c = c = tensor_wrapper.wrap_operand(c.wrapped_operand.tensor)
        else:
            self.c = c = tensor_wrapper.wrap_operand(c)
        operands.append(c)

        # The number of operands, since `c` is optional. If `c` is not provided,
        # set Beta = 0.
        self.num_operands = len(operands)

        if self.num_operands == 3 and b_package != c_package:
            raise TypeError(f"The operands 'b' ({b_package}) and 'c' ({c_package}) do NOT belong to the same package.")

        self.logger.info(f"The data type of operand A is '{a.dtype}', and that of operand B is '{b.dtype}'.")
        if c is not None:
            self.logger.info(f"The data type of operand C is '{c.dtype}'.")
        self.logger.info(f"The index type of operand A is '{a.index_type}'.")

        # Currently, only SpMM is supported and so the index type is obtained from `a`.
        self.index_type_name = a.index_type
        self.index_type = typemaps.NAME_TO_DATA_TYPE[a.index_type]
        if self.index_type_name not in VALID_INDEX_TYPES:
            raise TypeError(
                f"The index type {self.index_type_name} is not supported. The supported index types are {VALID_INDEX_TYPES}."
            )

        # Currently we require `c` and only inplace update of `c` is supported.
        assert self.num_operands == 3, "Internal Error."
        self.inplace = True  # Currently only inplace update of `c` is supported.
        if self.inplace:
            self.logger.info("The operation will be performed inplace with operand C.")

        # Determine the data types for a and b.
        self.a_dtype = typemaps.NAME_TO_DATA_TYPE[a.dtype]
        self.b_dtype = typemaps.NAME_TO_DATA_TYPE[b.dtype]
        self.a_dtype_name = a.dtype
        self.b_dtype_name = b.dtype

        self.is_complex = "complex" in self.a_dtype_name or "complex" in self.b_dtype_name

        # Determine the data type for c.
        if self.num_operands == 3:
            self.c_dtype = typemaps.NAME_TO_DATA_TYPE[c.dtype]
        elif self.num_operands == 2:
            self.c_dtype = self.a_dtype
        self.c_dtype_name = typemaps.DATA_TYPE_TO_NAME[self.c_dtype]

        # The common (semantic) checks for the problem such as compatibility,
        # tileability etc.
        value_type_names = {self.a_dtype_name, self.b_dtype_name, self.c_dtype_name}
        if len(value_type_names) != 1:
            raise NotImplementedError(
                f"The dtype for the operands {self.a_dtype_name}, {self.b_dtype_name}, and \
{self.c_dtype_name} don't match. Mixing operands of different precisions is currently not supported."
            )
        self.value_type_name = value_type_names.pop()

        if self.value_type_name not in VALID_DTYPES:
            raise TypeError(
                f"The dtype (value type) {self.value_type_name} is not supported. The supported dtypes are {VALID_DTYPES}."
            )
        self.value_type = typemaps.NAME_TO_DATA_TYPE[self.a_dtype_name]
        self.regular_precision = _is_regular_precision(self.value_type_name)

        # Set compute type.
        if self.options.compute_type is None:
            default_compute_type = ComputeType.CUDA_C_32F if self.is_complex else ComputeType.CUDA_R_32F
            self.compute_type = self.a_dtype if self.regular_precision else default_compute_type
        else:
            try:
                self.compute_type = ComputeType(self.options.compute_type)
            except Exception as e:
                message = f"The specified compute type {self.options.compute_type} is not a valid compute type."
                raise TypeError(message) from e
        self.compute_type_name = typemaps.DATA_TYPE_TO_NAME[self.compute_type]

        if self.is_complex and "complex" not in self.compute_type_name:
            raise ValueError(f"The specified compute type {self.compute_type_name} is complex.")
        if not self.is_complex and "complex" in self.compute_type_name:
            raise ValueError(
                f"The specified compute type {self.compute_type_name} is complex, while the \
operands' dtype {self.value_type} is real."
            )

        self.logger.info(f"The compute type for the matrix multiplication is {self.compute_type_name}.")

        # Set alpha and beta: note that the currently-supported compute types are valid
        # NumPy dtypes.
        self.alpha = np.zeros((1,), dtype=self.compute_type_name)
        try:
            self.alpha[0] = alpha if alpha is not None else 1
        except (ValueError, TypeError) as e:
            raise ValueError(f"The value provided for alpha {alpha} is not convertible to dtype '{self.alpha.dtype}'.") from e

        self.beta = np.zeros((1,), dtype=self.compute_type_name)
        if beta is not None and self.num_operands == 2:
            self.logger.warning(f"Matmul: The provided beta value {beta} is ignored since operand C is not specified.")
        try:
            default_beta = 1.0 if self.inplace else 0.0
            self.beta[0] = beta if beta is not None and self.num_operands == 3 else default_beta
        except (ValueError, TypeError) as e:
            raise ValueError(f"The value provided for beta {beta} is not convertible to dtype '{self.beta.dtype}'.") from e

        # Check for compatible operand packages.
        self.sparse_package = utils.infer_object_package(self.a.tensor)

        self.dense_package: str
        if self.ust_operands:
            orig_dense_package = {utils.infer_object_package(o) for o in self.ust_operands}
            if len(orig_dense_package) != 1:
                raise TypeError(f"The dense operands `b` and `c` don't belong to the same package: {orig_dense_package}.")
            orig_dense_package = orig_dense_package.pop()  # type: ignore[assignment]
            # Use the wrapped operands to get the dense package for dense UST.
            self.dense_package = utils.get_operands_package(operands[1:])
        else:
            self.dense_package = orig_dense_package = utils.get_operands_package(operands[1:])  # type: ignore

        if (self.sparse_package, orig_dense_package) not in cusparse_utils.COMPATIBLE_SP_DN_PACKAGES:
            raise TypeError(
                f"""The sparse operand package {self.sparse_package} and dense operand(s) package {orig_dense_package} are
not part of the compatible choices: {cusparse_utils.COMPATIBLE_SP_DN_PACKAGES}."""
            )

        a_dense_package = self.a.dense_tensorholder_type.name
        addendum = ""
        if self.sparse_package == "nvmath":
            addendum = f", with the dense representation package {a_dense_package}"
        self.logger.info(f"The sparse operand package is {self.sparse_package}{addendum}.")
        self.logger.info(f"The dense operand(s) package is {orig_dense_package}{addendum}.")

        # The dense package must match also match the dense package used to represent the
        # sparse tensor to correctly handle stream ordering etc.
        if self.sparse_package == "nvmath" and a_dense_package != self.dense_package:
            raise TypeError(
                f"The UST operands `a` uses a different representation package ({a_dense_package}) from that \
of the other operand(s) ({self.dense_package})."
            )

        # Memory space.
        self.memory_space = "cuda"
        self.device_id = utils.get_operands_device_id(operands)
        if self.device_id == "cpu":
            if self.dense_package == "numpy":
                self.dense_package = "cuda"
            self.memory_space = "cpu"
            self.device_id = self.execution.device_id  # type: ignore[union-attr]

        # An UST operand backed by CPU-only packages like NumPy is not supported.
        if self.memory_space == "cpu" and self.ust_operands and self.dense_package == "cuda":
            raise NotImplementedError("The Matmul operation does not currently support UST operands backed by NumPy.")

        # Execution space.
        self.execution_space = "cuda"
        self.logger.info(
            f"The input operands' memory space is {self.memory_space}, and the execution space is on device {self.device_id}."
        )

        # Set stream.
        stream_holder = utils.get_or_create_stream(self.device_id, stream, self.dense_package)
        self.logger.info(f"The specified stream for the Matmul ctor is {stream_holder.obj}.")

        # Copy operands to device if needed.
        self.cpu_c_ref = None
        if self.memory_space == "cpu":
            if self.inplace:
                self.cpu_c_ref = self.ust_operands[1] if self.ust_operands else self.operands[2]

            # Note that UST *code generation* is currently not supported for CPU memory
            # space while dispatch is, it's checked later in plan().

            self.operands = [o.to(self.device_id, stream_holder) for o in self.operands]

            # Update the operand aliases to point to the mirrors.
            if self.num_operands == 2:
                self.a, self.b = self.operands
            else:
                self.a, self.b, self.c = self.operands

            # Replace the CPU UST operands with new device UST by *creating* them from the
            # dense operands.
            if self.ust_operands:
                self.ust_operands[0] = UST.from_package(self.b.tensor, stream_holder.external)
                if self.num_operands == 3:
                    self.ust_operands[1] = UST.from_package(self.c.tensor, stream_holder.external)

        # Set qualifiers.
        self.qualifiers = qualifiers if qualifiers is not None else np.zeros((3,), dtype=matrix_qualifiers_dtype)
        if self.qualifiers.dtype != matrix_qualifiers_dtype:
            raise ValueError(
                "The qualifiers must be specified as a NumPy array of length 3 corresponding to the operands A, B, and "
                "C of type 'matrix_qualifiers_dtype'."
            )

        # Set qualifiers based on torch lazy conjugation flag if not provided.
        self.qualifiers[1]["is_conjugate"] = self.qualifiers[1]["is_conjugate"] ^ self.operands[1].is_conjugate
        if self.num_operands == 3:
            self.qualifiers[2]["is_conjugate"] = self.qualifiers[2]["is_conjugate"] ^ self.operands[2].is_conjugate
        if self.qualifiers[2]["is_conjugate"]:
            raise NotImplementedError("The conjugate flag is currently not supported for operand C.")
        if self.qualifiers[2]["is_transpose"]:
            raise NotImplementedError("The transpose flag is currently not supported for operand C.")

        # The SpMM traits.
        self.spmm_traits = get_spmm_traits(a, b, c, qualifiers=self.qualifiers, inplace=self.inplace, logger=self.logger)

        # Set blocking or non-blocking behavior.
        self.blocking = self.options.blocking is True or self.memory_space == "cpu"
        if self.blocking:
            self.call_prologue = "This call is blocking and will return only after the operation is complete."
        else:
            self.call_prologue = (
                "This call is non-blocking and will return immediately after the operation is launched on the device."
            )

        # The result class is that of the wrapped dense device operand 'b'.
        self.result_class = self.operands[1].__class__

        # Set memory allocator.
        self.allocator = (
            options.allocator
            if options.allocator is not None
            else memory._MEMORY_MANAGER[self.dense_package](self.device_id, self.logger)
        )

        # Set memory limit.
        self.memory_limit = utils.get_memory_limit_from_device_id(self.options.memory_limit, self.device_id)
        self.logger.info(f"The memory limit is {formatters.MemoryStr(self.memory_limit)}.")

        # Determine provisional support for dispatch and codegen here, finalize later
        # in plan.
        if self.options.codegen:
            self.logger.info(
                "The Matmul operation will use a custom kernel if possible or raise an \
error since the codegen option is True."
            )

        if (
            sp_utils.sparse_or_dense(a.tensor) == "sparse"
            and self.index_type_name in CUSPARSE_INDEX_TYPES
            and self.value_type_name in CUSPARSE_DTYPES
            and a.format_name in cusparse_utils.SUPPORTED_NAMED_FORMATS
        ):
            self.dispatch_init = BackendSupport.PROVISIONALLY_SUPPORTED
        else:
            self.dispatch_init = BackendSupport.NOT_SUPPORTED
        self.logger.debug(f"The dispatch viability of the Matmul operation is {self.dispatch_init.name}.")

        if self.options.codegen and self.sparse_package != "nvmath":
            raise NotImplementedError("The code generation path is only available for UST operands.")

        if (
            self.index_type_name in CODEGEN_INDEX_TYPES
            and self.value_type_name in CODEGEN_DTYPES
            and self.sparse_package == "nvmath"
            and a.format_name in CODEGEN_FORMATS
            and self.memory_space == "cuda"
            and not (self.spmm_traits.a_layout_traits.batch_broadcast and len(self.spmm_traits.b_layout_traits.batch_shape) > 0)
            and a.num_dimensions <= 4
        ):
            self.codegen_init = BackendSupport.PROVISIONALLY_SUPPORTED
        else:
            self.codegen_init = BackendSupport.NOT_SUPPORTED
        self.logger.debug(f"The codegen viability of the Matmul operation is {self.codegen_init.name}.")

        if self.dispatch_init == BackendSupport.NOT_SUPPORTED and self.codegen_init == BackendSupport.NOT_SUPPORTED:
            raise NotImplementedError("The matrix multiplication is not currently supported for the specified operands.")

        # The finalization of dispatch vs codegen, which we'll determine later in plan().
        self.dispatch: BackendSupport | None = None
        self.codegen: BackendSupport | None = None

        # The actual API to use, which we'll determine later in plan().
        self.api = Api.NONE

        # Capture operand extents, strides, and lazy conjugation for consistency check
        # when resetting operands.
        self.operand_extents = tuple(o.shape for o in self.operands)
        self.operand_strides = (None,) + tuple(o.strides for o in self.operands[1:])

        # PyTorch currently doesn't support lazy conjugation for sparse tensors.
        assert not self.operands[0].values.is_conjugate, "Internal error."
        self.lazy_conjugation = (None, self.operands[1].is_conjugate, False)

        # Library attributes.
        self.handle = None
        self.own_handle: bool | None = None

        # Plan attributes.
        self.a_ifc = None  # type: ignore
        self.b_ifc = None  # type: ignore
        self.c_ifc = None  # type: ignore
        # For dispatch to the SpMMOp API.
        self.mm_op_plan = None

        self.mm_planned = False

        # Workspace attributes.
        self.workspace_ptr: None | memory.MemoryPointer = None
        self.workspace_size = 0
        self.workspace_allocated_size = 0
        self.workspace_allocated_here = False

        # Attributes to establish stream ordering.
        self.workspace_stream: cc.Stream | None = None
        self.last_compute_event: cc.Event | None = None

        # Track whether the operands have been released.
        self.operands_released = False

        self.valid_state = True
        self.logger.info("The Matmul operation has been created.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.free()

    def _check_valid_matmul(self, *args, **kwargs):
        """
        Check if the Matmul object is alive and well.
        """
        if not self.valid_state:
            raise InvalidMatmulState("The Matmul object cannot be used after resources are free'd")

    def _check_valid_operands(self, *args, **kwargs):
        """
        Check if the operands are available for the operation.
        """
        what = kwargs["what"]
        if self.operands_released:
            raise RuntimeError(
                f"{what} cannot be performed if the operands have been set to None. Use reset_operands() to set the "
                f"desired input before performing the {what.lower()}."
            )

    def _free_plan_resources(self, exception: Exception | None = None) -> bool:
        """
        Free resources allocated in planning.
        """

        # Destroy plan.
        if self.mm_op_plan is not None:
            cusparse.sp_mm_op_destroy_plan(self.mm_op_plan)
            self.mm_op_plan = None

        # Destroy matrix layouts.
        if self.a_ifc is not None and self.a_ifc.descriptor is not None:
            cusparse.destroy_sp_mat(self.a_ifc.descriptor)
            self.a_ifc.descriptor = None
        if self.b_ifc is not None and self.b_ifc.descriptor is not None:
            cusparse.destroy_dn_mat(self.b_ifc.descriptor)
            self.b_ifc.descriptor = None
        if self.c_ifc is not None and self.c_ifc.descriptor is not None:
            cusparse.destroy_dn_mat(self.c_ifc.descriptor)
            self.c_ifc.descriptor = None

        self.mm_planned = False
        return True

    def _check_planned(self, *args, **kwargs):
        what = kwargs["what"]
        if not self.mm_planned:
            raise RuntimeError(f"{what} cannot be performed before plan() has been called.")

    def _free_workspace_memory(self, exception: Exception | None = None) -> bool:
        """
        Free workspace by releasing the MemoryPointer object.
        """
        if self.workspace_ptr is None:
            return True

        self.workspace_ptr = None
        self.workspace_allocated_size = 0
        self.logger.debug("[_free_workspace_memory] The workspace has been released.")

        return True

    def _reset_workspace_allocation_tracking(self):
        """
        Reset workspace allocation tracking attributes to False at the end of the methods
        where workspace memory is potentially allocated. This is necessary to prevent any
        exceptions raised before method entry from using stale tracking values.
        """
        self.workspace_allocated_here = False

    @utils.precondition(_check_valid_matmul)
    def _release_workspace_memory_perhaps(self, release_workspace):
        """
        Free workspace memory if it's larger than the specified limit.
        """
        if not release_workspace:
            return True

        # Establish ordering wrt the computation and free workspace if requested.
        if self.last_compute_event is not None:
            self.workspace_stream.wait(self.last_compute_event)
            self.logger.debug("Established ordering with respect to the computation before releasing the workspace.")
            self.last_compute_event = None

        self.logger.debug("[_release_workspace_memory_perhaps] The workspace memory will be released.")
        return self._free_workspace_memory()

    def _release_workspace_memory_perhaps_wrapper(self, exception: Exception | None = None) -> bool:
        """
        This is used in @atomic.
        """
        self._release_workspace_memory_perhaps(release_workspace=self.workspace_allocated_here)
        self._reset_workspace_allocation_tracking()
        return True

    @utils.precondition(_check_valid_matmul)
    @utils.precondition(_check_planned, "Workspace memory allocation")
    @utils.atomic(_free_workspace_memory, method=True)
    def _allocate_workspace_memory(self, stream_holder: utils.StreamHolder):
        """
        Allocate workspace memory using the specified allocator.
        """

        assert self.workspace_size is not None, "Internal Error."
        assert self.workspace_allocated_here is False, "Internal Error."

        if self.workspace_size == 0:  # For performance, bypass allocator for workspace size == 0.
            self.workspace_ptr = memory.MemoryPointer(0, 0, finalizer=None)
        else:
            self.logger.debug("Allocating workspace for performing the matrix multiplication...")
            with utils.device_ctx(self.device_id), stream_holder.ctx:
                try:
                    if isinstance(self.allocator, memory.BaseCUDAMemoryManagerAsync):
                        self.workspace_ptr = self.allocator.memalloc_async(self.workspace_size, stream_holder.obj)
                    else:
                        self.workspace_ptr = self.allocator.memalloc(self.workspace_size)
                    self.workspace_allocated_here = True
                except TypeError as e:
                    message = (
                        "The method 'memalloc' in the allocator object must conform to the interface in the "
                        "'BaseCUDAMemoryManager' protocol."
                    )
                    raise TypeError(message) from e

        self.workspace_allocated_size = self.workspace_size
        self.workspace_stream = stream_holder.obj
        self.logger.debug(
            f"Finished allocating device workspace of size {formatters.MemoryStr(self.workspace_size)} in the context "
            f"of stream {self.workspace_stream}."
        )

    def _allocate_workspace_memory_perhaps(self, stream_holder: utils.StreamHolder):
        """
        Allocate workspace memory using the specified allocator, if it hasn't already been
        done.
        """

        if self.workspace_ptr is not None and self.workspace_allocated_size >= self.workspace_size:
            return

        return self._allocate_workspace_memory(stream_holder)

    @utils.precondition(_check_valid_matmul)
    @utils.precondition(_check_valid_operands, "Planning")
    @utils.atomic(_free_plan_resources, method=True)
    def plan(
        self, *, prologs=None, epilog=None, semiring=None, compute_capability=None, stream: utils.AnyStream | int | None = None
    ):
        r"""
        Plan the sparse matrix multiplication operation, considering the prolog(s), epilog, and
        semiring operations.

        Args:
            prologs: {prologs}

            epilog: {epilog}

            semiring: {semiring}

            compute_capability: {compute_capability}

            stream: {stream}

        Examples:

            We'll see how to use prologs specify the SpMM
            :math:`c := 3.14 \, sin(a) \, @ \, b + c`, where the elementwise transformations
            are fully-fused into the matrix multiplication.

            >>> import math
            >>> import cupy as cp
            >>> import cupyx.scipy.sparse as sp
            >>> import nvmath


            Create a sparse operand in DIA format and view it as UST.

            >>> n = 4
            >>> values = cp.array(
            ...     [[0.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0], [-1.0, -2.0, -3.0, 0.0]]
            ... )
            >>> offsets = cp.array([1, 0, -1], dtype=cp.int32)
            >>> a = sp.dia_matrix((values, offsets), shape=(n, n))
            >>> a = nvmath.sparse.ust.Tensor.from_package(a)

            Dense ``b`` and ``c``, also viewed as UST.

            >>> b = cp.ones((n, n))
            >>> b = nvmath.sparse.ust.Tensor.from_package(b)
            >>> c = cp.zeros((n, n))
            >>> c = nvmath.sparse.ust.Tensor.from_package(c)

            Define the prolog for ``a``, and compile to LTO-IR using the helper (or
            your own compiler).

            >>> def transform_a(a):
            ...     return 3.14 * math.sin(a)
            >>>
            >>> prolog_a = nvmath.sparse.compile_matmul_prolog(
            ...     transform_a, operand_label="a", dtype="float64"
            ... )

            Create, plan, and execute the SpMM.

            >>> with nvmath.sparse.Matmul(a, b, c, beta=1.0) as mm:
            ...     # Plan the SpMM operation with the prologs.
            ...     mm.plan(prologs={{"a": prolog_a}})
            ...
            ...     # Execute the SpMM.
            ...     r = mm.execute()

        Further examples can be found in the `nvmath/examples/sparse/generic/matmul
        <https://github.com/NVIDIA/nvmath-python/tree/main/examples/sparse/generic/matmul>`_
        directory.
        """  # noqa: W505

        # Set self.dispatch and self.codegen from the corresponding initial values. This
        # is to enable replanning.
        self.dispatch = self.dispatch_init
        self.codegen = self.codegen_init

        self.logger.info("= PLANNING PHASE =")

        # Create stream holder.
        stream_holder = utils.get_or_create_stream(self.device_id, stream, self.dense_package)
        self.logger.info(f"The specified stream for the Matmul plan is {stream_holder.obj}.")

        if prologs is not None:
            if not isinstance(prologs, Mapping):
                raise TypeError(
                    f"The prologs argument must be a mapping from operand name ('a', 'b', 'c') \
to the LTO-IR buffers generated by the compile helper functions. The specified type is {type(prologs)}."
                )
            elif not prologs:
                # Treat empty prologs as if it were None.
                prologs = None
                self.logger.warning(f"Matmul plan(): The specified prologs {prologs} is ignored since it is an empty mapping.")

        if semiring is not None:
            if not isinstance(semiring, Mapping):
                raise TypeError(
                    f"The semiring argument must be a mapping from operation name ('add', 'mul') \
to the LTO-IR buffers generated by the compile helper functions. The specified type is {type(semiring)}."
                )

            if "add" not in semiring or "mul" not in semiring:
                raise ValueError(
                    f"Both 'add' and 'mul' operations should be specified to define the semiring \
structure: specified operations = {semiring.keys()}."
                )
            self.logger.info("The user-specified semiring operations will be used instead of the default.")

        # Check if it still can be codegen'd. Provisional code generation becomes final.
        if self.sparse_package == "nvmath" and (
            (
                self.alpha[0] != 1.0
                or self.beta[0] != 1.0
                or self.qualifiers[0]["is_conjugate"]
                or self.qualifiers[1]["is_conjugate"]
                or self.compute_type_name not in CODEGEN_COMPUTE_TYPES
            )
            and prologs is not None
        ):
            self.codegen = BackendSupport.NOT_SUPPORTED
            self.logger.debug(
                f"The Matmul operation is currently not supported for the code generation path: \
alpha = {self.alpha[0]}, beta = {self.beta[0]}, a_conjugate = {bool(self.qualifiers[0]['is_conjugate'])}, b_conjugate = \
{bool(self.qualifiers[1]['is_conjugate'])}, compute type = {self.compute_type_name}"
            )

        # Check if it still can be dispatched and if it can which function to dispatch
        # it to. Provisional dispatch becomes final.
        if (
            (self.qualifiers[0]["is_conjugate"] and not self.qualifiers[0]["is_transpose"])
            or (self.qualifiers[1]["is_conjugate"] and not self.qualifiers[1]["is_transpose"])
            or self.compute_type_name not in CUSPARSE_COMPUTE_TYPES
            or prologs is not None
        ):
            self.dispatch = BackendSupport.NOT_SUPPORTED
            self.logger.debug(
                f"The Matmul operation is currently not supported for dispatch: a_transpose = \
{bool(self.qualifiers[0]['is_transpose'])}, a_conjugate = {bool(self.qualifiers[0]['is_conjugate'])}, \
b_transpose = {bool(self.qualifiers[1]['is_transpose'])}, b_conjugate = {bool(self.qualifiers[1]['is_conjugate'])}, \
compute type = {self.compute_type_name}, prologs specified = {prologs is not None}."
            )

        # User-specified code generation, if true, takes precedence.
        if self.codegen == BackendSupport.NOT_SUPPORTED and self.options.codegen:
            message = ""
            if self.qualifiers[0]["is_conjugate"] or self.qualifiers[1]["is_conjugate"]:
                message += "The conjugate operation cannot be specified if prologs are provided \
for the code generation path."
            if self.alpha[0] != 1.0 or self.beta[0] != 1.0:
                message += "The scalars alpha and/or beta cannot be specified if prologs are provided \
for the code generation path."
            raise NotImplementedError(
                f"The code generation option is requested (options.codegen=True) but the code generation path is \
currently not supported.\n{message}"
            )

        # User-specified codegen option > dispatch > codegen.
        if self.options.codegen:
            if self.codegen != BackendSupport.PROVISIONALLY_SUPPORTED:
                raise NotImplementedError(
                    "The matrix multiplication is not currently supported for the code generation \
path, and options.codegen is set to True ."
                )
            self.codegen = BackendSupport.SUPPORTED
            self.api = Api.CODEGEN

        if not self.options.codegen and self.dispatch == BackendSupport.PROVISIONALLY_SUPPORTED:
            if semiring is None and epilog is None:
                self.logger.debug("MM plan (dispatch): neither semiring or epilog is specified.")
                self.api = Api.MM
                self.dispatch = BackendSupport.SUPPORTED
            elif semiring is not None and epilog is not None:
                self.logger.debug(
                    "MM plan (dispatch): both semiring and epilog are specified. No support due \
to 2-argument epilog for now."
                )
                self.api = Api.MM_OP
                self.dispatch = BackendSupport.NOT_SUPPORTED
            else:
                self.logger.debug(
                    f"MM plan (dispatch): exactly one of semiring {semiring is not None} or epilog \
{epilog is not None} is specified."
                )
                self.dispatch = BackendSupport.NOT_SUPPORTED

        if self.dispatch == BackendSupport.NOT_SUPPORTED and self.codegen == BackendSupport.PROVISIONALLY_SUPPORTED:
            self.codegen = BackendSupport.SUPPORTED
            self.api = Api.CODEGEN

        if self.dispatch == BackendSupport.NOT_SUPPORTED and self.codegen == BackendSupport.NOT_SUPPORTED:
            raise NotImplementedError("The matrix multiplication is not currently supported for the specified operands.")

        if self.api == Api.CODEGEN:
            assert self.codegen == BackendSupport.SUPPORTED, "Internal error."
            if semiring is not None and "atomic_add" not in semiring:
                raise ValueError(
                    "The 'atomic_add' operation needs to be specified in the semiring for the \
code generation path."
                )
            self.logger.info("The Matmul kernel will be generated by the UST library and compiled JIT.")
        elif self.api in [Api.MM, Api.MM_OP]:
            assert self.dispatch == BackendSupport.SUPPORTED, "Internal error."
            self.logger.info("The Matmul operation will be dispatched to the cuSPARSE library.")
        else:
            raise AssertionError("Internal error: API not available.")

        if self.api == Api.CODEGEN:
            from nvmath.sparse.ust._emitter import populate_matmul_parameters

            assert self.memory_space == "cuda", "Internal error."

            if self.beta[0] != 1.0:
                raise NotImplementedError("The code generation path is not supported for beta != 1.")

            # Create prolog LTO-IR for alpha, beta, and conjugate operation.
            if prologs is None:
                prologs = {}

            if self.alpha[0] != 1.0 or self.qualifiers[0]["is_conjugate"]:
                prolog_a = default_prolog(self.alpha[0], is_conjugate=self.qualifiers[0]["is_conjugate"])
                prolog_a = compile_prolog(
                    prolog_a, operand_label="a", dtype=self.compute_type_name, compute_capability=compute_capability
                )
                prologs["a"] = prolog_a

            if self.qualifiers[1]["is_conjugate"]:
                prolog_b = default_prolog(1.0, is_conjugate=self.qualifiers[1]["is_conjugate"])
                prolog_b = compile_prolog(
                    prolog_b, operand_label="b", dtype=self.compute_type_name, compute_capability=compute_capability
                )
                prologs["b"] = prolog_b

            a, b, c = self.a.tensor, *self.ust_operands
            self.kernel = KERNEL_CACHE.generate_matmul(
                a,
                b,
                c,
                compute_type=self.compute_type_name,
                prologs=prologs,
                epilog=epilog,
                semiring=semiring,
                transpose_a=self.qualifiers[0]["is_transpose"] > 0,
                transpose_b=self.qualifiers[1]["is_transpose"] > 0,
            )

            self.kernel_parameters, self.kernel_problem_size = populate_matmul_parameters(a, b, c, dense_bc=True)

            self.mm_planned = True
            self.logger.info("The Matmul planning is complete for the code generation path.")
            return

        # Capture epilog and semiring LTO-IR if specified.
        assert (semiring is None and epilog is None) ^ (semiring is not None and epilog is not None), "Internal error."
        if semiring is not None and epilog is not None:
            if "add" not in semiring or "mul" not in semiring:
                raise ValueError("The semiring option must provide LTO-IR code for both 'add' and 'mul' operations.")

            if not isinstance(semiring["add"], bytes) or not isinstance(semiring["mul"], bytes):
                raise ValueError("The LTO-IR code for 'add' and 'mul' semiring operation must be bytestring objects.")

        # Create handle.
        with utils.device_ctx(self.device_id):
            self.own_handle = True
            self.handle = cusparse.create()
            self.logger.info(f"The library handle has been created: {self.handle}.")

        # Set stream.
        cusparse.set_stream(self.handle, stream_holder.ptr)

        if (v := cusparse.get_version(self.handle)) < 12501 and self.a.format_name == "CSC" and self.spmm_traits.N == 1:
            raise NotImplementedError(f"This version ({v}) of cuSPARSE does not support CSC with n == 1.")

        # Set the pointer mode.
        cusparse.set_pointer_mode(self.handle, cusparse.PointerMode.HOST)
        self.a_ifc = getattr(cusparse_utils, self.a.format_name + "Ifc")(self.a, self.spmm_traits.a_layout_traits)
        self.b_ifc = cusparse_utils.DenseMatrixIfc(self.b, self.spmm_traits.b_layout_traits)  # type: ignore[assignment]
        self.c_ifc = cusparse_utils.DenseMatrixIfc(self.c, self.spmm_traits.c_layout_traits)  # type: ignore[assignment]

        # Transpose
        self.op_a = _get_cusparse_op_code(self.qualifiers[0])
        self.op_b = _get_cusparse_op_code(self.qualifiers[1])

        # Create matrix descriptors.
        self.a_ifc.create()  # type: ignore[attr-defined]
        self.b_ifc.create()  # type: ignore[attr-defined]
        self.c_ifc.create()  # type: ignore[attr-defined]

        if self.api == Api.MM:
            # Compute workspace.
            self.workspace_size = cusparse.sp_mm_buffer_size(
                self.handle,
                self.op_a,
                self.op_b,
                self.alpha.ctypes.data,
                self.a_ifc.descriptor,  # type: ignore[attr-defined]
                self.b_ifc.descriptor,  # type: ignore[attr-defined]
                self.beta.ctypes.data,
                self.c_ifc.descriptor,  # type: ignore[attr-defined]
                self.compute_type,
                cusparse.SpMMAlg.DEFAULT,
            )
        elif self.api == Api.MM_OP:
            # TODO: Fix `sp_mm_op_create_plan` to accept bytes objects.
            add_buffer = np.frombuffer(semiring["add"], dtype=np.int8)
            mul_buffer = np.frombuffer(semiring["mul"], dtype=np.int8)
            epilog_buffer = np.frombuffer(epilog, dtype=np.int8)

            # Plan and compute workspace.
            with utils.device_ctx(self.device_id):
                self.mm_op_plan, self.workspace_size = cusparse.sp_mm_op_create_plan(
                    self.handle,
                    self.op_a,
                    self.op_b,
                    self.a_ifc.descriptor,  # type: ignore[attr-defined]
                    self.b_ifc.descriptor,  # type: ignore[attr-defined]
                    self.c_ifc.descriptor,  # type: ignore[attr-defined]
                    self.compute_type,
                    cusparse.SpMMAlg.DEFAULT,
                    add_buffer.ctypes.data,
                    len(semiring["add"]),
                    mul_buffer.ctypes.data,
                    len(semiring["mul"]),
                    epilog_buffer.ctypes.data,
                    len(epilog),
                )
        else:
            raise AssertionError("Internal error.")

        self.logger.info(f"The memory limit is {formatters.MemoryStr(self.memory_limit)}.")
        if self.workspace_size > self.memory_limit:
            raise RuntimeError(
                f"The memory required for the computation is {self.workspace_size} \
({formatters.MemoryStr(self.workspace_size)}), while the specified memory limit is {self.memory_limit} \
({formatters.MemoryStr(self.memory_limit)})."
            )

        # TODO: preprocessing based on format and algorithm.

        self.mm_planned = True
        self.logger.info("The Matmul planning is complete for the library dispatch path.")

    @utils.precondition(_check_valid_matmul)
    def _check_and_set_dense_operand(self, operand, operand_name, operand_index, operand_ifc, stream_holder):
        assert self.operands is not None, "Internal Error."
        assert 0 < operand_index < 3, "Internal Error."

        package = utils.infer_object_package(operand.tensor)

        # Conjugate flag of the provided operands must match the original qualifiers
        if self.lazy_conjugation[operand_index] != operand.is_conjugate:
            raise ValueError(f"The provided operand {operand_name} has different conjugate flag than the original operand")

        memory_space = operand.device
        if memory_space != self.memory_space:
            raise TypeError(
                f"The memory space for '{operand_name}' ({memory_space}) doesn't match the original one ({self.memory_space})."
            )

        device_id = operand.device_id
        if device_id != "cpu" and device_id != self.device_id:
            raise TypeError(
                f"The device id for '{operand_name}' ({device_id}) doesn't match the original one ({self.device_id})."
            )

        value_type = operand.dtype
        shape = operand.shape
        strides = operand.strides

        # Handle cupy <> numpy asymmetry.
        if package == "numpy":
            package = "cuda"

        # Check package, device ID, shape, strides, and dtype.
        if package != self.dense_package:
            message = f"The package for '{operand_name}' ({package}) doesn't match the original one ({self.dense_package})."
            raise TypeError(message)

        if value_type != self.value_type_name:
            message = f"The dtype for '{operand_name}' ({value_type}) doesn't match the original one ({self.value_type_name})."
            raise TypeError(message)

        required_shape = self.operand_extents[operand_index]
        if shape != required_shape:
            message = f"The shape of '{operand_name}' ({shape}) doesn't match the original one ({required_shape})."
            raise TypeError(message)

        required_strides = self.operand_strides[operand_index]
        if strides != required_strides:
            message = f"The strides of '{operand_name}' ({strides}) don't match the original one ({required_strides})."
            raise TypeError(message)

        if device_id == "cpu":
            # Copy operand in the original buffer if it exists or create a new one
            # otherwise.
            o = getattr(self, operand_name)
            if o is None:
                o = operand.to(self.device_id, stream_holder)
            else:
                o.copy_(operand, stream_holder)
        else:
            o = operand
        setattr(self, operand_name, o)

        # Update alias.
        self.operands[operand_index] = o

        # Update the pointer values for dispatch. For codegen update it in
        # release_operands since individual operand pointers can't be updated.
        if self.api in [Api.MM, Api.MM_OP]:
            operand_ifc.update(o)
        elif self.api != Api.CODEGEN:
            raise AssertionError(f"Internal error: unsupported backend {self.api}.")

    @utils.precondition(_check_valid_matmul)
    def reset_operands(
        self,
        *,
        a=None,
        b=None,
        c=None,
        alpha=None,
        beta=None,
        stream: utils.AnyStream | int | None = None,
    ):
        """
        Reset one or more operands held by this :class:`Matmul` instance.
        Only the operands explicitly passed are updated; omitted operands retain
        their current values.

        This method will perform various checks on the new operands to make sure:

        - The shapes, index and data types match those of the old ones.

        - The packages that the operands belong to match those of the old ones.

        - If input tensors are on GPU, the device must match.

        Args:
            a: {a}

            b: {b}

            c: {c}

            alpha: {alpha}

            beta: {beta}

            stream: {stream}

        Examples:


            >>> import torch
            >>> import nvmath

            Prepare sample input data.

            >>> device_id = 0
            >>> dtype = torch.float64
            >>> m, n, k = 4, 2, 8

            Create a torch CSR tensor.

            >>> a = torch.ones(m, k, dtype=dtype, device=device_id)
            >>> a = a.to_sparse_csr()

            Dense ``b`` and ``c``.

            >>> b = torch.ones(k, n, dtype=dtype, device=device_id)
            >>> c = torch.ones(m, n, dtype=dtype, device=device_id)

            Create a stateful object that specifies the operation
            :math:`c := \\alpha \\, a \\, @ \\, b + \\beta \\, c`.

            >>> alpha, beta = 1.2, 2.4
            >>> with nvmath.sparse.Matmul(a, b, c, alpha=alpha, beta=beta) as mm:
            ...     # Plan the operation.
            ...     mm.plan()
            ...
            ...     # The first execution.
            ...     r = mm.execute()
            ...
            ...     # The operands can be reset in-place using the array library.
            ...     # Note that since `c` has been updated in-place, it also needs to
            ...     # be reset unless accumulating into it is desired.
            ...     b *= 3.14
            ...     c[:] = 1.0
            ...
            ...     # Execute the operation with updated `b`.
            ...     r = mm.execute()
            ...
            ...     # The operands can also be reset using the reset_operand[_unchecked]
            ...     # methods. This is needed when the memory space doesn't match the
            ...     # execution space or if the operands have not been updated in-place.
            ...     # Any operands that are not reset retain their prior values.
            ...
            ...     # The sparse matrix needs to be compatible (have the same sparse
            ...     # format, shape, dtype, NNZ etc.).
            ...     a = 2.718 * torch.ones(m, k, dtype=dtype, device=device_id)
            ...     a = a.to_sparse_csr()
            ...
            ...     c = 6.28 * torch.ones(m, n, dtype=dtype, device=device_id)
            ...
            ...     mm.reset_operands(a=a, c=c)
            ...
            ...     # Execute the operation with the reset `a` and `c`, `b` retains
            ...     # its previous value.
            ...     r = mm.execute()

            For more details, please refer to `the reset operand examples here
            <https://github.com/NVIDIA/nvmath-python/tree/main/examples/sparse/generic/matmul/>`_.
        """  # noqa: W505

        self.logger.info("Resetting operands...")

        # If operands have been released, all required operands must be provided
        if self.operands_released and (a is None or b is None or c is None):
            raise ValueError("After release_operands(), 'a', 'b', and 'c' must be provided to reset_operands().")

        if a is None and b is None and c is None and alpha is None and beta is None:
            msg = "Calling reset_operands() with all arguments set to None is not allowed. "
            msg += "Use release_operands() to release all operands."
            raise ValueError(msg)

        # Update alpha.
        if alpha is not None:
            if self.api == Api.CODEGEN:
                raise NotImplementedError("The value of `alpha` cannot be currently reset for the code generation path.")

            try:
                self.alpha[0] = alpha
            except (ValueError, TypeError) as e:
                raise ValueError(
                    f"The value provided for alpha {alpha} is not convertible to dtype '{self.alpha.dtype}'."
                ) from e
            self.logger.info("The factor `alpha` has been reset to %s.", alpha)

        # Update beta.
        if beta is not None:
            if self.api == Api.CODEGEN:
                raise NotImplementedError("The value of `beta` cannot be currently reset for the code generation path.")

            if self.num_operands == 2:
                self.logger.warning(f"Matmul: The provided beta value {beta} is ignored since operand C is not specified.")
            else:
                try:
                    self.beta[0] = beta
                except (ValueError, TypeError) as e:
                    raise ValueError(
                        f"The value provided for beta {beta} is not convertible to dtype '{self.beta.dtype}'."
                    ) from e
            self.logger.info("The factor `beta` has been reset to %s.", beta)

        stream_holder = utils.get_or_create_stream(self.device_id, stream, self.dense_package)

        # Update a.
        if a is not None:
            # Wrap A.
            try:
                a = sp_utils.wrap_sparse_operands(a)
            except Exception as e:
                raise TypeError(
                    "The operand 'a' must be an N-D sparse array/tensor tensor from one of the supported  \
packages: nvmath, CuPy, PyTorch, or SciPy."
                ) from e

            sparse_package = utils.infer_object_package(a.tensor)
            device_id = a.device_id
            memory_space = a.device
            value_type = a.dtype
            index_type = a.index_type

            shape = a.shape
            a_shape = self.operand_extents[0]

            # Check package, device ID, dtype, and index type.
            if sparse_package != self.sparse_package:
                raise TypeError(
                    f"The package for 'a' ({sparse_package}) doesn't match the original one ({self.sparse_package})."
                )

            if sparse_package == "nvmath" and a.dense_tensorholder_type.name != self.dense_package:
                raise TypeError(
                    f"The UST operand 'a' representation package ({a.dense_tensorholder_type.name}) doesn't match the "
                    f"original one ({self.dense_package})."
                )

            if memory_space != self.memory_space:
                raise TypeError(
                    f"The memory space for 'a' ({memory_space}) doesn't match the original one ({self.memory_space})."
                )

            if device_id != "cpu" and device_id != self.device_id:
                raise TypeError(f"The device id for 'a' ({device_id}) doesn't match the original one ({self.device_id}).")

            if value_type != self.value_type_name:
                raise TypeError(f"The dtype for 'a' ({value_type}) doesn't match the original one ({self.value_type_name}).")

            if index_type != self.index_type_name:
                raise TypeError(
                    f"The index type for 'a' ({index_type}) doesn't match the original one ({self.index_type_name})."
                )

            if shape != a_shape:
                raise TypeError(f"The shape of 'a' ({shape}) doesn't match the original one ({a_shape}).")

            # Copy operand in the original buffer if it exists or create a new one.
            if device_id == "cpu":
                if self.a is not None:
                    self.a.copy_(a, stream_holder)
                else:
                    self.a = a.to(self.device_id, stream_holder)
            else:
                self.a = a

            # Update alias.
            self.operands[0] = self.a

            # Update the pointer values in the existing device buffers.
            if self.api in [Api.MM, Api.MM_OP]:
                self.a_ifc.update(self.a)  # type: ignore[attr-defined]
            elif self.api != Api.CODEGEN:
                raise AssertionError(f"Internal error: unsupported backend {self.api}.")

            self.logger.info("The operand `a` has been reset.")

        # TODO: unify treatment of TensorHolder and UST for dense operands.
        # Update `b`.
        if b is not None:
            if sp_utils.sparse_or_dense(b) != "dense":
                raise ValueError(f"The operand `b` {type(b)} must be dense.")
            package = utils.infer_object_package(b)
            if (package == "nvmath") != (self.sparse_package == "nvmath"):
                raise TypeError(f"The package for 'b' ({package}) doesn't match the original one ({self.sparse_package}).")
            b_wrapped = tensor_wrapper.wrap_operand(b.wrapped_operand.tensor if package == "nvmath" else b)
            # Check and update self.operands and the alias.
            self._check_and_set_dense_operand(b_wrapped, "b", 1, self.b_ifc, stream_holder)
            if package == "nvmath":
                if self.memory_space == "cuda":
                    self.ust_operands[0] = b
                else:
                    self.ust_operands[0] = UST.from_package(self.b.tensor, stream_holder.external)
            self.logger.info("The operand `b` has been reset.")

        # Update `c`.
        if c is not None:
            if sp_utils.sparse_or_dense(c) != "dense":
                raise ValueError(f"The operand `c` {type(c)} must be dense.")
            package = utils.infer_object_package(c)
            if (package == "nvmath") != (self.sparse_package == "nvmath"):
                raise TypeError(f"The package for 'c' ({package}) doesn't match the original one ({self.sparse_package}).")
            c_wrapped = tensor_wrapper.wrap_operand(c.wrapped_operand.tensor if package == "nvmath" else c)
            # Check and update self.operands and the alias.
            self._check_and_set_dense_operand(c_wrapped, "c", 2, self.c_ifc, stream_holder)
            if package == "nvmath":
                if self.memory_space == "cuda":
                    self.ust_operands[1] = c
                else:
                    self.ust_operands[1] = UST.from_package(self.c.tensor, stream_holder.external)
            if self.memory_space == "cpu" and self.inplace:
                self.cpu_c_ref = c if package == "nvmath" else c_wrapped
            self.logger.info("The operand `c` has been reset.")

        # For codegen, all pointers will have to be reset since individual pointers
        # can't be updated.
        if (a is not None or b is not None or c is not None) and self.api == Api.CODEGEN:
            from nvmath.sparse.ust._emitter import populate_matmul_parameters

            assert self.memory_space == "cuda", "Internal error."

            a, b, c = self.a.tensor, *self.ust_operands
            self.kernel_parameters, self.kernel_problem_size = populate_matmul_parameters(a, b, c, dense_bc=True)

        # Update release operands state.
        self.operands_released = False

    def reset_operands_unchecked(
        self,
        *,
        a=None,
        b=None,
        c=None,
        alpha=None,
        beta=None,
        stream: utils.AnyStream | int | None = None,
    ):
        """
        {reset_operands_unchecked}
        """

        if alpha is not None:
            self.alpha[0] = alpha

        if beta is not None:
            self.beta[0] = beta

        if self.memory_space == "cuda":
            if a is not None:
                self.operands[0].reset_unchecked(a)
                self.a = self.operands[0]

                if self.api in [Api.MM, Api.MM_OP]:
                    self.a_ifc.update(self.a)  # type: ignore[attr-defined]

            if b is not None:
                if self.ust_operands:
                    self.ust_operands[0] = b
                    # Recall `b` is the wrapped dense non-UST operand.
                    self.operands[1].tensor = b.wrapped_operand.tensor
                else:
                    self.operands[1].tensor = b

                # Update the alias.
                self.b = self.operands[1]

                if self.api in [Api.MM, Api.MM_OP]:
                    self.b_ifc.update(self.b)  # type: ignore[attr-defined]

            if c is not None:
                if self.ust_operands:
                    self.ust_operands[1] = c
                    # Recall `c` is the wrapped dense non-UST operand.
                    self.operands[2].tensor = c.wrapped_operand.tensor
                else:
                    self.operands[2].tensor = c

                # Update the alias.
                self.c = self.operands[2]

                if self.api in [Api.MM, Api.MM_OP]:
                    self.c_ifc.update(self.c)  # type: ignore[attr-defined]

        else:
            # Handle CPU memory space.

            stream_holder = utils.get_or_create_stream(self.device_id, stream, self.dense_package)

            if a is not None:
                a_wrapped = self.a_wrapper_type.create_from_tensor(a, attr_name_map=self.a_attr_name_map)
                self.a = self.operands[0] = a_wrapped.to(self.device_id, stream_holder)

                if self.api in [Api.MM, Api.MM_OP]:
                    self.a_ifc.update(self.a)  # type: ignore[attr-defined]

            # We can't use the dense wrapper type `self.[b,c].__class__` when going across
            # memory spaces due to asymmetries (NumPyTensor <> NDBufferTensor, etc).
            if b is not None:
                b_wrapped = tensor_wrapper.wrap_operand(b.wrapped_operand.tensor if self.sparse_package == "nvmath" else b)
                self.b = self.operands[1] = b_wrapped.to(self.device_id, stream_holder)
                if self.ust_operands:
                    self.ust_operands[0] = UST.from_package(self.b.tensor, stream_holder.external)

                if self.api in [Api.MM, Api.MM_OP]:
                    self.b_ifc.update(self.b)  # type: ignore[attr-defined]

            if c is not None:
                c_wrapped = tensor_wrapper.wrap_operand(c.wrapped_operand.tensor if self.sparse_package == "nvmath" else c)
                self.c = self.operands[2] = c_wrapped.to(self.device_id, stream_holder)
                if self.ust_operands:
                    self.ust_operands[1] = UST.from_package(self.c.tensor, stream_holder.external)

                if self.api in [Api.MM, Api.MM_OP]:
                    self.c_ifc.update(self.c)  # type: ignore[attr-defined]

                if self.inplace:
                    self.cpu_c_ref = c if self.sparse_package == "nvmath" else c_wrapped

        if self.api == Api.CODEGEN:
            from nvmath.sparse.ust._emitter import populate_matmul_parameters

            a, b, c = self.a.tensor, *self.ust_operands
            self.kernel_parameters, self.kernel_problem_size = populate_matmul_parameters(a, b, c, dense_bc=True)

        # Update release operands state.
        self.operands_released = False

    @utils.precondition(_check_valid_matmul)
    def release_operands(self):
        """
        {release_operands}
        """

        # Fast exit.
        if self.operands_released:
            return

        # Set the sparse operand to None.
        self.operands[0].release()

        # For the dense operands, keep the wrapper but set the internal references to None.
        self.operands[1].tensor = None
        if self.ust_operands:
            self.ust_operands[0] = None

        # Set the aliases to None as well.
        self.a = self.b = None

        if self.num_operands == 3:
            self.operands[2].tensor = self.c = None
            if self.ust_operands:
                self.ust_operands[1] = None

        if self.inplace:
            self.cpu_c_ref = None

        self.operands_released = True
        self.logger.info("The user-provided operands have been released.")

    @utils.precondition(_check_valid_matmul)
    @utils.precondition(_check_planned, "Execution")
    @utils.precondition(_check_valid_operands, "Execution")
    @utils.atomic(_release_workspace_memory_perhaps_wrapper, method=True)
    def execute(self, *, release_workspace=False, stream: utils.AnyStream | int | None = None):
        """
        Execute a prepared (planned) sparse matrix multiplication.

        Args:
            release_workspace: {release_workspace}

            stream: {stream}

        Returns:
           {result}
        """
        log_info = self.logger.isEnabledFor(logging.INFO)

        if log_info:
            self.logger.info("= EXECUTION PHASE =")
        stream_holder = utils.get_or_create_stream(self.device_id, stream, self.dense_package)
        if log_info:
            self.logger.info(f"The specified stream for execute() is {stream_holder.obj}.")

        # TODO: create empty tensor for the result.

        if log_info:
            message = ""
            if self.api == Api.CODEGEN:
                message = "(codegen path)"
            elif self.api == Api.MM:
                message = "(library dispatch path to SpMM)"
            elif self.api == Api.MM_OP:
                message = "(library dispatch path to SpMMOp)"
            else:
                raise AssertionError("Internal error.")
            self.logger.info(f"Starting matrix multiplication {message}...")
            self.logger.info(f"{self.call_prologue}")
        # TODO: clean this up.
        if self.api == Api.CODEGEN:
            from nvmath.sparse.ust._jit import launch_kernel

            launch_kernel(
                self.kernel,
                self.kernel_parameters,
                self.kernel_problem_size,
                device_id=self.device_id,
                stream_holder=stream_holder,
                blocking=True,
            )
            self.last_compute_event = None
        else:
            # Set stream for library execution.
            cusparse.set_stream(self.handle, stream_holder.ptr)

            # Allocate workspace if needed.
            self._allocate_workspace_memory_perhaps(stream_holder)

            raw_workspace_ptr = utils.get_ptr_from_memory_pointer(self.workspace_ptr)
            with utils.cuda_call_ctx(stream_holder, self.blocking, timing=log_info) as (
                self.last_compute_event,
                elapsed,
            ):
                if self.api == Api.MM:
                    cusparse.sp_mm(
                        self.handle,
                        self.op_a,
                        self.op_b,
                        self.alpha.ctypes.data,
                        self.a_ifc.descriptor,  # type: ignore[attr-defined]
                        self.b_ifc.descriptor,  # type: ignore[attr-defined]
                        self.beta.ctypes.data,
                        self.c_ifc.descriptor,  # type: ignore[attr-defined]
                        self.compute_type,
                        cusparse.SpMMAlg.DEFAULT,
                        raw_workspace_ptr,
                    )
                elif self.api == Api.MM_OP:
                    cusparse.sp_mm_op(self.mm_op_plan, raw_workspace_ptr)
                else:
                    raise AssertionError("Internal error.")

            if log_info and elapsed.data is not None:
                self.logger.info(f"The matrix multiplication calculation took {elapsed.data:.3f} ms to complete.")

        # Establish ordering wrt the computation and free workspace if requested.
        if release_workspace:
            self._release_workspace_memory_perhaps(True)

        # Return the result.
        if self.memory_space == "cpu":
            # Currently only inplace is supported.
            if self.ust_operands:
                self.cpu_c_ref.copy_(self.ust_operands[1], stream_holder)  # type: ignore
                out = self.cpu_c_ref
            else:
                self.cpu_c_ref.copy_(self.c, stream_holder)  # type: ignore
                out = self.cpu_c_ref.tensor  # type: ignore
        else:
            if self.ust_operands:
                out = self.ust_operands[1]
            else:
                out = self.c.tensor

        # Release internal reference to the result to permit recycling of memory.
        # TODO: handle result allocated internally.
        self._reset_workspace_allocation_tracking()

        return out

    def free(self):
        """Free Matmul resources.

        It is recommended that the :class:`Matmul` object be used within a context, but if
        it is not possible then this method must be called explicitly to ensure that the
        matrix multiplication resources (especially internal library objects) are properly
        cleaned up.
        """

        if not self.valid_state:
            return

        try:
            # Future operations on the workspace stream should be ordered after the
            # computation.
            if self.last_compute_event is not None:
                if self.workspace_stream is not None:
                    self.workspace_stream.wait(self.last_compute_event)
                self.last_compute_event = None

            self._free_workspace_memory()

            self._free_plan_resources()

            # Free handle if we own it.
            if self.handle is not None and self.own_handle:
                cusparse.destroy(self.handle)
                self.handle, self.own_handle = None, False

            # Set all attributes to None except for logger and valid_state.
            for attr in list(vars(self)):
                if attr not in {"logger", "valid_state"}:
                    setattr(self, attr, None)

        except Exception as e:
            self.logger.critical("Internal error: only part of the Matmul object's resources have been released.")
            self.logger.critical(str(e))
            raise e
        finally:
            self.valid_state = False

        self.logger.info("The Matmul object's resources have been released.")


@utils.docstring_decorator(SPARSE_MM_DOCUMENTATION, skip_missing=False)
def matmul(
    a,
    b,
    /,
    c=None,
    *,
    alpha=None,
    beta=None,
    qualifiers=None,
    prologs=None,
    epilog=None,
    semiring=None,
    compute_capability=None,
    options=None,
    execution: ExecutionCUDA | None = None,
    stream: utils.AnyStream | int | None = None,
):
    """
    Perform the specified sparse matrix multiplication computation, which is one of
    :math:`epilog(\\alpha \\, op_h(a) \\, @ \\, op_h(b) + \\beta \\, c)` or
    :math:`epilog(prolog_a(op_t(a)) \\, @ \\, prolog_b(op_t(b)) + prolog_c(c))`. The
    :math:`op_h` and :math:`op_t` operators optionally specify transpose/hermitian or
    transpose operations respectively via the ``qualifiers`` argument. In addition, the
    scalar multiplication and addition operators ("semiring") can be customized by the
    user, if desired.

    .. note::
        The complex conjugate operation is mutually exclusive with prolog since it can be
        absorbed into the prolog.

    .. note::
        Currently only in-place sparse matrix multiplication is supported, so operand ``c``
        must be provided. This restriction will be removed in a future release.

    This function-form is a wrapper around the stateful :class:`Matmul` object APIs and is
    meant for *single* use (the user needs to perform just one sparse matrix multiplication,
    for example), in which case there is no possibility of amortizing preparatory costs.

    Detailed information on what's happening within this function can be obtained by passing
    in a :class:`logging.Logger` object to :class:`MatmulOptions` or by setting the
    appropriate options in the root logger object, which is used by default:

        >>> import logging
        >>> logging.basicConfig(
        ...     level=logging.INFO,
        ...     format="%(asctime)s %(levelname)-8s %(message)s",
        ...     datefmt="%m-%d %H:%M:%S",
        ... )

    A user can select the desired logging level and, in general, take advantage of all of
    the functionality offered by the Python `logging` module.

    Args:
        a: {a}

        b: {b}

        c: {c}

        alpha: {alpha}

        beta: {beta}

        qualifiers: {qualifiers}

        prologs: {prologs}

        epilog: {epilog}

        semiring: {semiring}

        compute_capability: {compute_capability}

        options: {options}

        execution: {execution}

        stream: {stream}

    Returns:
        {result}

    Semantics:
        {semantics}

    .. seealso::
        :class:`Matmul`, :class:`MatmulOptions`, :class:`ExecutionCUDA`

    Examples:

        >>> import torch
        >>> import nvmath

        Prepare sample data.

        >>> index_type, dtype = torch.int32, torch.float32
        >>> device_id = 0
        >>> shape = 2, 2

        Create a torch COO tensor, and view it as UST.

        >>> indices = torch.tensor([[0, 1], [0, 1]], dtype=index_type)
        >>> values = torch.tensor([2.0, 4.0], dtype=dtype)
        >>> a = torch.sparse_coo_tensor(indices, values, shape, device=device_id)
        >>> a = a.coalesce()
        >>> a = nvmath.sparse.ust.Tensor.from_package(a)

        Dense 'b' and 'c', also viewed as UST objects.

        >>> b = torch.ones(*shape, dtype=dtype, device=device_id)
        >>> b = nvmath.sparse.ust.Tensor.from_package(b)
        >>> c = torch.zeros(*shape, dtype=dtype, device=device_id)
        >>> c = nvmath.sparse.ust.Tensor.from_package(c)

        Solve :math:`c := a @ b + c`.

        >>> r = nvmath.sparse.matmul(a, b, c, beta=1.0)

        The result can also be viewed as a torch tensor.

        >>> r = nvmath.sparse.ust.Tensor.to_package(r)

    .. note:: This function is a convenience wrapper around :class:`Matmul` and is
          specifically meant for *single* use.

    Further examples can be found in the `nvmath/examples/sparse/generic/matmul
    <https://github.com/NVIDIA/nvmath-python/tree/main/examples/sparse/generic/matmul>`_
    directory.
    """

    with Matmul(
        a, b, c=c, alpha=alpha, beta=beta, qualifiers=qualifiers, options=options, execution=execution, stream=stream
    ) as mm:
        mm.plan(prologs=prologs, epilog=epilog, semiring=semiring, compute_capability=compute_capability, stream=stream)
        r = mm.execute(stream=stream)

    return r
