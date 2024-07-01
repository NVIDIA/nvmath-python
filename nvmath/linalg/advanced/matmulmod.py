# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

__all__ = ['MatmulComputeType', 'Matmul', 'matmul']

import copy
from collections import namedtuple
from dataclasses import dataclass
import functools
import logging
import operator
from typing import Sequence

import cupy as cp
import numpy as np

from nvmath import memory

from nvmath.linalg.advanced import algorithmmod
from nvmath.linalg.advanced import configuration
from nvmath.bindings import cublas
from nvmath.bindings import cublasLt as cublaslt

from nvmath._internal import formatters
from nvmath._internal import tensor_wrapper
from nvmath._internal import typemaps
from nvmath._internal import utils

from nvmath.linalg._internal import matmul_desc_ifc, matmul_pref_ifc, matrix_layout_ifc
from nvmath.linalg._internal.typemaps import NAME_TO_DEFAULT_SCALE_TYPE, NAME_TO_DEFAULT_COMPUTE_TYPE
from nvmath.linalg._internal.utils import axis_order_in_memory, calculate_strides, check_batch_tileable, create_handle, destroy_handle, get_handle, pointer_aligned_to
from nvmath.linalg._internal.epilog_protocol import EPILOG_INPUT_HANDLERS_MAP, EPILOG_OUTPUT_HANDLERS_MAP

MatmulComputeType = cublas.ComputeType

EpilogInputTraits = namedtuple("EpilogInputTraits", ["dtype", "extents", "strides"])

@dataclass
class MMTraits:
    """An internal data class for capturing the matrix multiplication traits. The
       result traits are captured separately, because we need to wait for the
       epilog to be provided.
    """
    M : int = None
    N : int = None
    K : int = None
    d_mm_shape : Sequence[int] = None
    a_layout_traits : object = None
    b_layout_traits : object = None
    c_layout_traits : object = None
    batch_count : int = None
    batch_shape : Sequence[int] = None
    batch_axis_order: Sequence[int] = None

@dataclass
class ResultTraits:
    """An internal data class for capturing the result matrix's traits.
    """
    d_layout_traits : object = None
    result_shape : Sequence[int] = None
    result_strides : Sequence[int] = None

@dataclass
class MatrixLayout:
    """An internal data class for capturing the tensor layout.
    """
    shape : Sequence[int] = None
    strides : Sequence[int] = None
    is_conjugate : bool = None    # Used to support is_conjugate via conjugate_transpose.

@dataclass
class LayoutTraits:
    """An internal data class for capturing the matrix multiplication traits.
    """
    order : int = None            # cublaslt.Order.{ROW, COL}
    ld : int = None
    batch_offset : int = None     # Based on strides
    is_conjugate : bool = None    # Used to support is_conjugate via conjugate_transpose.
    mm_shape : Sequence[int] = None
    mm_strides : Sequence[int] = None

    def get_mm_layout(self, transpose=False):
        if self.is_conjugate:
            transpose = True
        if not transpose:
            return *self.mm_shape, self.ld, self.order

        # Use of transpose is supported only for A and B for two specific use cases till the C library directly supports these use cases:
        #   1. When A or B has the conjugate qualifier, we transpose it internally and then use conjugate transpose in the MM (A @ B.conj() == A @ B.T.H).
        #   2. When the epilog is BGRADB, we transpose B internally and use transpose in the MM since this epilog requires B to be transposed (A @ B == A @ B.T.T).
        # This requires that the layout order be ROW or COL (no special layouts such as structured or hierarchical).
        assert self.mm_shape is not None and self.mm_strides is not None, "Internal Error."
        assert self.ld != 0, "Internal Error."

        mm_shape = self.mm_shape[1], self.mm_shape[0]
        if self.order == cublaslt.Order.ROW:
            order = cublaslt.Order.COL
            ld = max(self.mm_shape[1], self.mm_strides[0])
        elif self.order == cublaslt.Order.COL:
            order = cublaslt.Order.ROW
            ld = max(self.mm_shape[0], self.mm_strides[1])
        else:
            assert False, "Internal Error. Invalid layout order."

        return  *mm_shape, ld, order


def get_matrix_layout_traits(mm_shape, mm_strides, batch_strides, col_bcast):
    if len(mm_shape) < 2:   # The result D can be a scalar or vector.
        batch_offset = min(batch_strides) if batch_strides else 0
        order = cublaslt.Order.COL
        ld = max(mm_shape[0], mm_strides[0]) if len(mm_shape) == 1 else 1
        return order, ld, batch_offset

    M, N = mm_shape

    # Important: start with the first dimension so that cases like (M, 1) : (1, 1) or (1, M) : (1, 1) in CuTe notation map to COL.
    if mm_strides[0] == 1:
        order = cublaslt.Order.COL
    elif mm_strides[1] == 1:
        order = cublaslt.Order.ROW
    else:
        if M == 1:
            order = cublaslt.Order.COL
        elif N == 1:
            order = cublaslt.Order.ROW
        else:
            raise ValueError("Unsupported layout.")

    # We need to handle broadcast dimensions with zero-stride for the c matrix.
    if col_bcast and N == 1:
        ld = 0
    else:
        ld = max(M, mm_strides[1]) if order == cublaslt.Order.COL else max(N, mm_strides[0])

    # Batch dimensions should be contiguous in memory, which we have already checked.
    # The batch_offset should be based on the lowest stride in the batch dimension to account for embedded matrices.
    batch_offset = min(batch_strides) if batch_strides else 0

    return order, ld, batch_offset

def get_mm_traits(a_layout, b_layout, c_layout, logger):
    """
    First check A and B compatibility:

    1. Check MM compatibility (K):
        a. First pad A and/or B MM dimensions if 1-D according to NumPy convention.
        b. The padding is used to determine M, N, and K but should not appear in the output dimensions.
        c. If both A and B are N-D, the dimensions must match.
    2. Check batch dimensions:
        a. One of A or B can have missing batch extents, in which case it is broadcast, otherwise
        b. A and B must have the same batch ordering.
        c. In addition, the batch dimensions must be tileable (contiguous in memory).

    Then check C:

    C can be None. If C is passed in, it must be a vector or matrix. Batching rule is the same as above.
    """
    a_shape, a_strides = list(a_layout.shape), list(a_layout.strides)
    b_shape, b_strides = list(b_layout.shape), list(b_layout.strides)

    a_batch_shape, a_mm_shape = a_shape[:-2], a_shape[-2:]
    b_batch_shape, b_mm_shape = b_shape[:-2], b_shape[-2:]

    a_batch_strides, a_mm_strides = a_strides[:-2], a_strides[-2:]
    b_batch_strides, b_mm_strides = b_strides[:-2], b_strides[-2:]

    d_mm_shape = []
    if len(a_mm_shape) == 1:
        s, d = a_mm_shape[0], a_mm_strides[0]
        a_mm_shape = [1] + a_mm_shape
        a_mm_strides = [s * d] + a_mm_strides
    else:
        d_mm_shape.append(a_mm_shape[0])    # The first mode for d applies only when a is not a vector.

    if len(b_mm_shape) == 1:
        s, d = b_mm_shape[0], b_mm_strides[0]
        b_mm_shape = b_mm_shape + [1]
        b_mm_strides = b_mm_strides + [s * d]
    else:
        d_mm_shape.append(b_mm_shape[1])    # The second mode for d applies only when b is not a vector.

    logger.debug(f"The MM shape for operand A is {a_mm_shape} with strides {a_mm_strides}.")
    logger.debug(f"The MM shape for operand B is {b_mm_shape} with strides {b_mm_strides}.")
    logger.debug(f"The MM shape for operand D is {d_mm_shape}.")

    M0, K0 = a_mm_shape
    K1, N0 = b_mm_shape
    if K0 != K1:
        raise ValueError(f"The 'K' extent must match for the operands: K={K0} in operand A is not equal to K={K1} in operand B.")

    # Check if batch dimensions of A and B are tileable as well as compatible.
    batch_shape, batch_axis_order = [], ()
    if len(a_batch_shape) > 0:
        if not check_batch_tileable(a_batch_shape, a_batch_strides):
            message = f"The batch layout for A corresponding to shape = {a_batch_shape} and strides = {a_batch_strides} is currently not supported because it is not tileable."
            raise ValueError(message)
        logger.debug(f"The batch layout for A corresponding to shape = {a_batch_shape} and strides = {a_batch_strides} IS tileable.")
        batch_shape = a_batch_shape
        batch_axis_order = a_batch_axis_order = axis_order_in_memory(a_batch_strides)

    if len(b_batch_shape) > 0:
        if not check_batch_tileable(b_batch_shape, b_batch_strides):
            message = f"The batch layout for B corresponding to shape = {b_batch_shape} and strides = {b_batch_strides} is currently not supported because it is not tileable."
            raise ValueError(message)
        logger.debug(f"The batch layout for B corresponding to shape = {b_batch_shape} and strides = {b_batch_strides} IS tileable.")
        batch_shape = b_batch_shape
        batch_axis_order = b_batch_axis_order = axis_order_in_memory(b_batch_strides)

    if len(a_batch_shape) > 0 and len(b_batch_shape) > 0:
        if a_batch_shape != b_batch_shape:
            raise ValueError(f"The batch dimensions of operands A {a_batch_shape} and B {b_batch_shape} must match.")
        if a_batch_axis_order != b_batch_axis_order:
            raise ValueError(f"The batch order of operands A {a_batch_axis_order} and B {b_batch_axis_order} must match.")

    logger.debug(f"The batch shape is {batch_shape} with batch axis order {batch_axis_order}.")

    batch_count = functools.reduce(operator.mul, batch_shape, 1)

    # Create matrix layout traits.
    a_order, a_ld, a_batch_offset = get_matrix_layout_traits(a_mm_shape, a_mm_strides, a_batch_strides, col_bcast=False)
    a_layout_traits = LayoutTraits(order=a_order, ld=a_ld, batch_offset=a_batch_offset, is_conjugate=a_layout.is_conjugate, mm_shape=a_mm_shape, mm_strides=a_mm_strides)
    logger.debug(f"The layout order for operand A is {a_order.name}, with LD {a_ld}, and batch offset {a_batch_offset}.")

    b_order, b_ld, b_batch_offset = get_matrix_layout_traits(b_mm_shape, b_mm_strides, b_batch_strides, col_bcast=False)
    b_layout_traits = LayoutTraits(order=b_order, ld=b_ld, batch_offset=b_batch_offset, is_conjugate=b_layout.is_conjugate, mm_shape=b_mm_shape, mm_strides=b_mm_strides)
    logger.debug(f"The layout order for operand B is {b_order.name}, with LD {b_ld}, and batch offset {b_batch_offset}.")

    # Process matrix c, if provided.
    c_layout_traits = None
    if c_layout is not None:
        # C can be a vector of dimension M, which is broadcast.
        # C can be a matrix of dimension (M, N) or (M, 1), broadcast in the latter case and has to have contiguous strides.
        # C can be batched matrices of dimension (..., M, N) or (..., M, 1), broadcast in the latter case and has to have contiguous strides.
        c_shape, c_strides = list(c_layout.shape), list(c_layout.strides)

        c_batch_shape, c_mm_shape = c_shape[:-2], c_shape[-2:]
        c_batch_strides, c_mm_strides = c_strides[:-2], c_strides[-2:]
        if len(c_mm_shape) == 1:
            s, d = c_mm_shape[0], c_mm_strides[0]
            c_mm_shape = c_mm_shape + [1]
            c_mm_strides = c_mm_strides + [s * d]
        logger.debug(f"The MM shape for operand C is {c_mm_shape} with strides {c_mm_strides}.")

        Mc, Nc = c_mm_shape
        if Mc != M0:
            raise ValueError(f"The M dimension of the C matrix ({Mc}) must match the M dimension of A.")

        if Nc != 1 and Nc != N0:
            raise ValueError(f"The N dimension of the C matrix ({Nc}) must match the N dimension of B.")

        if len(c_batch_shape) > 0 and c_batch_shape != batch_shape:
            raise ValueError(f"The batch dimension of operand C {c_batch_shape} must match with that of the other operands {batch_shape}.")
        if len(c_batch_shape) > 0:
            c_batch_axis_order = axis_order_in_memory(c_batch_strides)
            if c_batch_axis_order != batch_axis_order:
                raise ValueError(f"The batch axis order of operand C {c_batch_axis_order} must match with that of the other operands {batch_axis_order}.")

        if len(c_batch_shape) > 0:
            if not check_batch_tileable(c_batch_shape, c_batch_strides):
                message = f"The batch layout for C corresponding to shape = {c_batch_shape} and strides = {c_batch_strides} is currently not supported because it is not tileable."
                raise ValueError(message)

        c_order, c_ld, c_batch_offset = get_matrix_layout_traits(c_mm_shape, c_mm_strides, c_batch_strides, col_bcast=True)
        c_layout_traits = LayoutTraits(order=c_order, ld=c_ld, batch_offset=c_batch_offset, is_conjugate=c_layout.is_conjugate)
        logger.debug(f"The layout order for operand C is {c_order.name}, with LD {c_ld}, and batch offset {c_batch_offset}.")

    return MMTraits(M=M0, N=N0, K=K0, d_mm_shape=d_mm_shape, batch_count=batch_count, batch_shape=batch_shape, batch_axis_order=batch_axis_order,
                    a_layout_traits=a_layout_traits, b_layout_traits=b_layout_traits, c_layout_traits=c_layout_traits)

def get_result_traits(mm_traits, epilog_ordering, logger):
    """
    epilog_ordering = value of type cublaslt.Order or None.

    The result layout is determined from:
    - the ordering of operand c, if it is provided, or
    - the epilog requirement, if it exists, or
    - the ordering of operand a.

    The result batch dimensions must have the same extents and axis order as the inputs. The MM layout can be C or F.
    """
    # The result shape is the batch shape + d_mm_shape.
    result_shape = mm_traits.batch_shape + mm_traits.d_mm_shape

    if mm_traits.c_layout_traits is not None:
        result_ordering = mm_traits.c_layout_traits.order
    elif epilog_ordering is not None:
        result_ordering = epilog_ordering
    else:
        result_ordering = mm_traits.a_layout_traits.order

    if result_ordering == cublaslt.Order.ROW:
        d_order = list(range(len(mm_traits.d_mm_shape) - 1, -1, -1))
    elif result_ordering == cublaslt.Order.COL:
        d_order = list(range(len(mm_traits.d_mm_shape)))
    else:
        assert False, "Internal Error."

    result_axis_order = [len(mm_traits.batch_axis_order) + a for a in d_order] + list(mm_traits.batch_axis_order)

    # Calculate the result strides.
    result_strides = calculate_strides(result_shape, result_axis_order)

    # The result's traits.
    d_batch_strides, d_mm_strides = result_strides[:len(mm_traits.batch_shape)], result_strides[len(mm_traits.batch_shape):]
    d_order, d_ld, d_batch_offset = get_matrix_layout_traits(mm_traits.d_mm_shape, d_mm_strides, d_batch_strides, col_bcast=False)
    d_layout_traits = LayoutTraits(order=d_order, ld=d_ld, batch_offset=d_batch_offset, is_conjugate=False)
    logger.debug(f"The layout order for operand D is {d_order.name}, with LD {d_ld}, and batch offset {d_batch_offset}.")

    return ResultTraits(result_shape=result_shape, result_strides=result_strides, d_layout_traits=d_layout_traits)

SHARED_MM_DOCUMENTATION = utils.COMMON_SHARED_DOC_MAP.copy()
SHARED_MM_DOCUMENTATION.update({
    'a':
        "A tensor representing the first operand to the matrix multiplication (see `Semantics`). The currently supported types are "
        ":class:`numpy.ndarray`, :class:`cupy.ndarray`, and :class:`torch.Tensor`.",
    'b':
        "A tensor representing the second operand to the matrix multiplication (see `Semantics`). The currently supported types are "
        ":class:`numpy.ndarray`, :class:`cupy.ndarray`, and :class:`torch.Tensor`.",
    'c':
        "(Optional) A tensor representing the operand to add to the matrix multiplication result (see `Semantics`). The currently supported "
        "types are :class:`numpy.ndarray`, :class:`cupy.ndarray`, and :class:`torch.Tensor`.",
    'alpha':
        "The scale factor for the matrix multiplication term as a real or complex number. The default is :math:`1.0`.",
    'beta':
        "The scale factor for the matrix addition term as a real or complex number. A value for `beta` must be provided if operand `c` is specified.",
    'algorithms':
        "A sequence of :class:`Algorithm` objects that can be directly provided to bypass planning. The algorithm objects must "
        "be compatible with the matrix multiplication. A typical use for this option is to provide algorithms serialized (pickled) "
        "from a previously planned and autotuned matrix multiplication.",
    'epilog':
        r"Specify an epilog :math:`F` as an object of type :class:`MatmulEpilog` to apply to the result of the matrix "
        r"multiplication: :math:`F(\alpha A @ B + \beta C`). The default is no epilog.",
    'epilog_inputs':
        "Specify the additional inputs needed for the selected epilog as a dictionary, where the key is the epilog "
        "input name and the value is the epilog input. The epilog input must be a tensor with the same package and in the same "
        "memory space as the operands (see the constructor for more information on the operands). If the required epilog inputs "
        "are not provided, an exception is raised that lists the required epilog inputs.",
    'qualifiers':
        "If desired, specify the matrix qualifiers as a :class:`numpy.ndarray` of :class:`~nvmath.linalg.advanced.matrix_qualifiers_dtype` objects "
        "of length 3 corresponding to the operands `a`, `b`, and `c`.",
    'options':
        "Specify options for the matrix multiplication as a :class:`~nvmath.linalg.advanced.MatmulOptions` object. Alternatively, a `dict` containing the parameters for the "
        "``MatmulOptions`` constructor can also be provided. If not specified, the value will be set to the default-constructed ``MatmulOptions`` object.",
    'preferences':
        "This parameter specifies the preferences for planning as a :class:`MatmulPlanPreferences` object. Alternatively, a "
        "dictionary containing the parameters for the ``MatmulPlanPreferences`` constructor can also be provided. If not specified, the "
        "value will be set to the default-constructed ``MatmulPlanPreferences`` object.",
    'result':
        "The result of the specified matrix multiplication (epilog applied), which remains on the same device and belong to the same package as the input operands. If an epilog "
        "(like :attr:`nvmath.linalg.advanced.MatmulEpilog.RELU_AUX`) that results in extra output is used, a tuple is returned with the first element being the matrix multiplication result (epilog applied) and the second "
        "element being the auxiliary output provided by the selected epilog as a `dict`.",
    'semantics':
        """The semantics of the matrix multiplication follows :func:`numpy.matmul` semantics, with some restrictions on broadcasting. In addition, the
        semantics for the fused matrix addition are described below:

            * If arguments `a` and `b` are matrices, they are multiplied according to the rules of matrix multiplication.
            * If argument `a` is 1-D, it is promoted to a matrix by prefixing ``1`` to its dimensions. After matrix multiplication, the prefixed ``1``
              is removed from the result's dimensions.
            * If argument `b` is 1-D, it is promoted to a matrix by appending ``1`` to its dimensions. After matrix multiplication, the appended ``1``
              is removed from the result's dimensions.
            * If `a` or `b` is N-D (N > 2), then the operand is treated as a batch of matrices. If both `a` and `b` are N-D, their batch dimensions
              must match. If exactly one of `a` or `b` is N-D, the other operand is broadcast.
            * The operand for the matrix addition `c` may be a vector of length M, a matrix of shape (M, 1) or (M, N), or batched versions of the
              latter (..., M, 1) or (..., M, N). Here M and N are the dimensions of the result of the matrix multiplication. If a vector is provided
              or N = 1, the columns of `c` are broadcast for the addition. If batch dimensions are not present, `c` is broadcast across batches as
              needed."""
})

class InvalidMatmulState(Exception):
    pass

@utils.docstring_decorator(SHARED_MM_DOCUMENTATION, skip_missing=False)
class Matmul:
    r"""
    Matmul(a, b, /, c=None, *, alpha=None, beta=None, qualifiers=None, options=None, stream=None)

    Create a stateful object encapsulating the specified matrix multiplication computation :math:`\alpha a @ b + \beta c` and the required
    resources to perform the operation.  A stateful object can be used to amortize the cost of preparation (planning in the case of matrix
    multiplication) across multiple executions (also see the :ref:`Stateful APIs <host api types>` section).

    The function-form API :func:`matmul` is a convenient alternative to using stateful objects for *single* use (the user needs to
    perform just one matrix multiplication, for example), in which case there is no possibility of amortizing preparatory costs. The
    function-form APIs are just convenience wrappers around the stateful object APIs.

    Using the stateful object typically involves the following steps:

    1. **Problem Specification**: Initialize the object with a defined operation and options.
    2. **Preparation**: Use :meth:`plan` to determine the best algorithmic implementation for this specific matrix multiplication operation.
    3. **Execution**: Perform the matrix multiplication computation with :meth:`execute`.
    4. **Resource Management**: Ensure all resources are released either by explicitly calling :meth:`free` or by managing the stateful object within a context manager.

    Detailed information on what's happening in the various phases described above can be obtained by passing in a :class:`logging.Logger` object
    to :class:`MatmulOptions` or by setting the appropriate options in the root logger object, which is used by default:

        >>> import logging
        >>> logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%m-%d %H:%M:%S')

    A user can select the desired logging level and, in general, take advantage of all of the functionality offered by the Python `logging` module.

    Args:
        a: {a}
        b: {b}
        c: {c}
        alpha: {alpha}
        beta: {beta}
        qualifiers: {qualifiers}
        options: {options}
        stream: {stream}

    Semantics:
        {semantics}

    See Also:
        :meth:`autotune`, :meth:`plan`, :meth:`reset_operands`, :meth:`execute`

    Examples:

        >>> import numpy as np
        >>> import nvmath

        Create two 2-D float64 ndarrays on the CPU:

        >>> M, N, K = 1024, 1024, 1024
        >>> a = np.random.rand(M, K)
        >>> b = np.random.rand(K, N)

        We will define a matrix multiplication operation followed by a RELU epilog function using the specialized matrix multiplication inteface.

        Create a Matmul object encapsulating the problem specification above:

        >>> mm = nvmath.linalg.advanced.Matmul(a, b)

        Options can be provided above to control the behavior of the operation using the `options` argument (see :class:`MatmulOptions`).

        Next, plan the operation. The epilog is specified, and optionally, preferences can be specified for planning:

        >>> epilog = nvmath.linalg.advanced.MatmulEpilog.RELU
        >>> mm.plan(epilog=epilog)

        Certain epilog choices (like :attr:`nvmath.linalg.advanced.MatmulEpilog.BIAS`) require additional input provided using the `epilog_inputs` argument to :meth:`plan`.

        Now execute the matrix multiplication, and obtain the result `r1` as a NumPy ndarray.

        >>> r1 = mm.execute()

        Finally, free the object's resources. To avoid having to explicitly making this call, it's recommended to use the Matmul object as
        a context manager as shown below, if possible.

        >>> mm.free()

        Note that all :class:`Matmul` methods execute on the current stream by default. Alternatively, the `stream` argument can be used to run a method on a specified stream.

        Let's now look at the same problem with CuPy ndarrays on the GPU.

        Create a 3-D complex128 CuPy ndarray on the GPU:

        >>> import cupy as cp
        >>> a = cp.random.rand(M, K)
        >>> b = cp.random.rand(K, N)

        Create an Matmul object encapsulating the problem specification described earlier and use it as a context manager.

        >>> with nvmath.linalg.advanced.Matmul(a, b) as mm:
        ...    mm.plan(epilog=epilog)
        ...
        ...    # Execute the operation to get the first result.
        ...    r1 = mm.execute()
        ...
        ...    # Update operands A and B in-place (see reset_operands() for an alternative).
        ...    a[:] = cp.random.rand(M, K)
        ...    b[:] = cp.random.rand(K, N)
        ...
        ...    # Execute the operation to get the new result.
        ...    r2 = mm.execute()


        All the resources used by the object are released at the end of the block.

        Further examples can be found in the `nvmath/examples/linalg/advanced/matmul <https://github.com/NVIDIA/nvmath-python/tree/main/examples/linalg/advanced/matmul>`_ directory.
    """

    def __init__(self, a, b, /, c=None, *, alpha=None, beta=None, qualifiers=None, options=None, stream=None):

        options = utils.check_or_create_options(configuration.MatmulOptions, options, "Matrix multiplication options")
        self.options = options

        self.logger = options.logger if options.logger is not None else logging.getLogger()

        # The matrix multiplication has two required operands 'a' and 'b', and one optional operand 'c'.
        a = tensor_wrapper.wrap_operand(a)
        b = tensor_wrapper.wrap_operand(b)
        self.logger.info(f"= SPECIFICATION PHASE =")
        self.logger.info(f"The data type of operand A is '{a.dtype}', and that of operand B is '{b.dtype}'.")

        self.num_operands = 2
        if c is not None:
            self.num_operands = 3
            c = tensor_wrapper.wrap_operand(c)
            self.logger.info(f"The data type of operand C is {c.dtype}.")

        if c is not None and beta is None:
            raise ValueError("A value for beta must be provided if operand C is provided.")

        if a.dtype != b.dtype:
            raise ValueError(f"The dtype of operands A {a.dtype} and B {b.dtype} must be the same.")
        self.ab_dtype_name = a.dtype

        assert self.num_operands == 2 or self.num_operands == 3, "Internal Error."

        # Infer the library package & device ID the operands belong to.
        operands = [a, b]
        if self.num_operands == 3:
            operands.append(c)
        self.operands = operands

        self.package = utils.get_operands_package(operands)
        self.memory_space = 'cuda'
        self.device_id = utils.get_operands_device_id(operands)
        if self.device_id is None:
            if self.package == 'numpy':
                self.package = 'cupy'
            self.memory_space = 'cpu'
            self.device_id = options.device_id
        self.logger.info(f"The input operands' memory space is {self.memory_space}, and the execution space is on device {self.device_id}.")

        # Allocate device memory (in stream context) if needed.
        stream_holder = utils.get_or_create_stream(self.device_id, stream, self.package)
        self.logger.info(f"The specified stream for the Matmul ctor is {stream_holder.obj}.")

        # Copy operands to device if needed.
        if self.memory_space == 'cpu':
            self.operands = tensor_wrapper.to(self.operands, self.device_id, stream_holder)

        # Set qualifiers.
        self.qualifiers = qualifiers if qualifiers is not None else np.zeros((3,), dtype=configuration.matrix_qualifiers_dtype)
        if self.qualifiers.dtype != configuration.matrix_qualifiers_dtype:
            raise ValueError("The qualifiers must be specified as a NumPy array of length 3 corresponding to the operands A, B, and C of type 'configuration.matrix_qualifiers_dtype'.")
        if self.qualifiers[2]['is_conjugate']:
            raise ValueError("The conjugate flag is currently not supported for operand C.")
        # Set qualifiers based on torch lazy conjugation flag if not provided.
        if self.package == 'torch' and qualifiers is None:
            self.qualifiers[0]['is_conjugate'] = self.operands[0].tensor.is_conj()
            self.qualifiers[1]['is_conjugate'] = self.operands[1].tensor.is_conj()
            if len(self.operands) > 2 and self.operands[2].tensor.is_conj():
                raise ValueError("The conjugate flag is currently not supported for operand C.")
            self.lazy_conjugation = True
        else:
            self.lazy_conjugation = False

        # Set blocking or non-blocking behavior.
        self.blocking = self.options.blocking is True or self.memory_space == 'cpu'
        if self.blocking:
            self.call_prologue = "This call is blocking and will return only after the operation is complete."
        else:
            self.call_prologue = "This call is non-blocking and will return immediately after the operation is launched on the device."

        # The result class is that of the first wrapped device operand.
        self.result_class = self.operands[0].__class__

        # Set memory allocator.
        self.allocator = options.allocator if options.allocator is not None else memory._MEMORY_MANAGER[self.package](self.device_id, self.logger)

        # Set memory limit.
        self.memory_limit = utils.get_memory_limit_from_device_id(self.options.memory_limit, self.device_id)
        self.logger.info(f"The memory limit is {formatters.MemoryStr(self.memory_limit)}.")

        # Set handle. We don't destroy handles we create.
        if options.handle is not None:
            self.handle = options.handle
        else:
            self.handle = get_handle(self.device_id)

        # Determine the scale type.
        if options.scale_type is None:
            self.scale_type      = NAME_TO_DEFAULT_SCALE_TYPE[self.ab_dtype_name]
            self.scale_type_name = typemaps.DATA_TYPE_TO_NAME[self.scale_type]
        else:
            self.scale_type = options.scale_type
            if self.scale_type not in typemaps.DATA_TYPE_TO_NAME:
                message = f"Unsupported scale type. The data type '{self.scale_type}' is currently not supported."
                raise ValueError(message)
            self.scale_type_name = typemaps.DATA_TYPE_TO_NAME[self.scale_type]
        self.logger.info(f"The scale type is '{self.scale_type_name}'.")

        # Determine the data types for a and b.
        self.a_dtype = typemaps.NAME_TO_DATA_TYPE[a.dtype]
        self.b_dtype = typemaps.NAME_TO_DATA_TYPE[b.dtype]

        # Determine the data type for c (if not provided) and d. The two must match.
        if self.num_operands == 2:
            self.d_dtype_name, self.d_dtype = self.ab_dtype_name, typemaps.NAME_TO_DATA_TYPE[self.ab_dtype_name]
            self.c_dtype = self.d_dtype
            self.c_dtype_name = self.d_dtype_name
        else:
            self.c_dtype_name = c.dtype
            self.c_dtype = typemaps.NAME_TO_DATA_TYPE[self.c_dtype_name]
            self.d_dtype = self.c_dtype
            self.d_dtype_name = typemaps.DATA_TYPE_TO_NAME[self.d_dtype]
        self.logger.info(f"The data type for the result D is '{self.d_dtype_name}'.")

        # Determine the compute type.
        self.compute_type = options.compute_type if options.compute_type is not None else NAME_TO_DEFAULT_COMPUTE_TYPE[self.ab_dtype_name]
        if self.compute_type not in cublas.ComputeType:
            message = f"Unsupported compute type. The compute type '{self.compute_type}' is currently not supported."
            raise ValueError(message)
        self.logger.info(f"The compute type is {self.compute_type.name}.")

        # Set alpha and beta.
        self.alpha = np.zeros((1,), dtype=self.scale_type_name)
        try:
            self.alpha[0] = alpha if alpha is not None else 1
        except (ValueError, TypeError) as e:
            raise ValueError(f"The value provided for alpha {alpha} is not convertible to dtype '{self.alpha.dtype}'.") from e

        self.beta = np.zeros((1,), dtype=self.scale_type_name)
        if beta is not None and self.num_operands == 2:
            self.logger.warning(f"Matmul: The provided beta value {beta} is ignored since operand C is not specified.")
        try:
            self.beta[0] = beta if beta is not None and self.num_operands == 3 else 0
        except (ValueError, TypeError) as e:
            raise ValueError(f"The value provided for beta {beta} is not convertible to dtype '{self.beta.dtype}'.") from e

        # Capture operand extents and strides for consistency check when resetting operands.
        self.operand_extents = tuple(o.shape for o in self.operands)
        self.operand_strides = tuple(o.strides for o in self.operands)

        # Create operand layouts.
        a_layout = MatrixLayout(self.operands[0].shape, self.operands[0].strides, self.qualifiers[0]['is_conjugate'])
        b_layout = MatrixLayout(self.operands[1].shape, self.operands[1].strides, self.qualifiers[1]['is_conjugate'])
        c_layout = MatrixLayout(self.operands[2].shape, self.operands[2].strides) if self.num_operands == 3 else None

        # Get the operation traits.
        self.mm_traits     = get_mm_traits(a_layout, b_layout, c_layout, self.logger)
        self.result_traits = None    # Wait till planning to determine this based on the epilog.
        self.logger.info(f"The matrix multiplication attributes are M = {self.mm_traits.M}, N = {self.mm_traits.N}, and K = {self.mm_traits.K}.")
        self.logger.info(f"The batch count is {self.mm_traits.batch_count}, and the batch shape is {self.mm_traits.batch_shape} with batch axis order {self.mm_traits.batch_axis_order}.")

        # Create and set the operation descriptor.
        self.mm_desc = cublaslt.matmul_desc_create(self.compute_type, self.scale_type)

        mm_desc_ifc = matmul_desc_ifc.MatmulDescInterface(self.mm_desc)
        mm_desc_ifc.compute_type    = self.compute_type
        mm_desc_ifc.scale_type      = self.scale_type

        # Guard SM count target and fast accumulation flag.
        version = cublaslt.get_version()
        if options.sm_count_target > 0:
            if version < 111103:
                raise ValueError(f"The 'sm_count_target' option is not supported in cuBLASLt version {version}.")
            mm_desc_ifc.sm_count_target = options.sm_count_target
            self.logger.info(f"The SM count target is {options.sm_count_target}.")

        if options.fast_accumulation:
            if version < 111103:
                raise ValueError(f"The 'fast_accumulation' option is not supported in cuBLASLt version {version}.")
            mm_desc_ifc.fast_accum      = options.fast_accumulation
            self.logger.info(f"The flag for fast accumulation mode is {options.fast_accumulation}.")

        # Epilog attributes.
        self.epilog = None

        # Epilog attributes: name-to-operand.
        self.epilog_operands = dict()

        # Epilog attributes: epilog input name-to-handler.
        self.epilog_input_name_to_handler = dict()

        # Epilog attributes: name-to-output tensor.
        self.epilog_outputs = dict()

        # Keep track of epilog input traits for resetting operands.
        self.epilog_inputs_traits = dict()

        # Keep track of epilog output handlers to allocate output in execute().
        self.epilog_output_handlers = []

        # Plan attributes.
        self.preference_ptr  = None
        self.a_layout_ptr, self.b_layout_ptr, self.c_layout_ptr, self.d_layout_ptr = None, None, None, None
        self.flop_count = 0
        self.mm_planned = False

        # Algorithm attributes.
        self.algorithms_buffer = None
        self.algorithm_objects = None

        # Workspace attributes.
        self.workspace_ptr, self.workspace_size = None, None
        self.workspace_allocated_size = 0
        self.workspace_allocated_here = False

        # Attributes to establish stream ordering.
        self.workspace_stream = None
        self.last_compute_event = None

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
        what = kwargs['what']
        if self.operands is None:
            raise RuntimeError(f"{what} cannot be performed if the operands have been set to None. Use reset_operands() to set the desired input before using performing the {what.lower()}.")

    def _free_plan_resources(self, exception=None):
        """
        Free resources allocated in planning.
        """

        # Destroy matrix layouts.
        if self.a_layout_ptr is not None:
            cublaslt.matrix_layout_destroy(self.a_layout_ptr)
            self.a_layout_ptr = None
        if self.b_layout_ptr is not None:
            cublaslt.matrix_layout_destroy(self.b_layout_ptr)
            self.b_layout_ptr = None
        if self.num_operands == 3 and self.c_layout_ptr is not None:   # Note that c layout aliases with that of d.
            cublaslt.matrix_layout_destroy(self.c_layout_ptr)
        self.c_layout_ptr = None
        if self.d_layout_ptr is not None:
            cublaslt.matrix_layout_destroy(self.d_layout_ptr)
            self.d_layout_ptr = None

        if self.preference_ptr is not None:
            cublaslt.matmul_preference_destroy(self.preference_ptr)
            self.preference_ptr = None

        self.mm_planned   = False
        return True

    def _check_planned(self, *args, **kwargs):
        what = kwargs['what']
        if not self.mm_planned:
            raise RuntimeError(f"{what} cannot be performed before plan() has been called.")

    def _free_workspace_memory(self, exception=None):
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
        Reset workspace allocation tracking attributes to False at the end of the methods where workspace memory is
        potentially allocated. This is necessary to prevent any exceptions raised before method entry from using
        stale tracking values.
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
            self.workspace_stream.wait_event(self.last_compute_event)
            self.logger.debug("Established ordering with respect to the computation before releasing the workspace.")

        self.logger.debug("[_release_workspace_memory_perhaps] The workspace memory will be released.")
        return self._free_workspace_memory()

    def _release_workspace_memory_perhaps_wrapper(self, exception=None):
        """
        This is used in @atomic.
        """
        self._release_workspace_memory_perhaps(release_workspace=self.workspace_allocated_here)
        return True

    @utils.precondition(_check_valid_matmul)
    @utils.precondition(_check_planned, "Workspace memory allocation")
    @utils.atomic(_free_workspace_memory, method=True)
    def _allocate_workspace_memory_perhaps(self, stream_holder):
        """
        Allocate workspace memory using the specified allocator, if it hasn't already been done.
        """

        assert self.workspace_size is not None, "Internal Error."
        assert self.workspace_allocated_here is False, "Internal Error."

        if self.workspace_ptr is not None and self.workspace_allocated_size >= self.workspace_size:
            return

        if self.workspace_size == 0:    # For performance, bypass allocator for workspace size == 0.
            self.workspace_ptr = memory.MemoryPointer(0, 0, finalizer=None)
        else:
            self.logger.debug(f"Allocating workspace for performing the matrix multiplication...")
            with utils.device_ctx(self.device_id), stream_holder.ctx:
                try:
                    self.workspace_ptr = self.allocator.memalloc(self.workspace_size)
                except TypeError as e:
                    message = "The method 'memalloc' in the allocator object must conform to the interface in the "\
                              "'BaseCUDAMemoryManager' protocol."
                    raise TypeError(message) from e

        self.workspace_allocated_size = self.workspace_size
        self.workspace_stream = stream_holder.obj
        self.logger.debug(f"Finished allocating device workspace of size {formatters.MemoryStr(self.workspace_size)} in the context of stream {self.workspace_stream}.")

    @utils.precondition(_check_valid_matmul)
    def applicable_algorithm_ids(self, limit=8):
        """Obtain the algorithm IDs that are applicable to this matrix multiplication.

        Args:
            limit: The maximum number of applicable algorithm IDs that is desired

        Returns:
            A sequence of algorithm IDs that are applicable to this matrix multiplication problem specification, in random order.
        """
        ...
        algo_ids = cublaslt.matmul_algo_get_ids(self.handle, self.compute_type, self.scale_type, self.a_dtype, self.b_dtype, self.c_dtype, self.d_dtype, limit)
        return algo_ids

    @utils.precondition(_check_valid_matmul)
    @utils.atomic(_free_plan_resources, method=True)
    def plan(self, *, preferences=None, algorithms=None, epilog=None, epilog_inputs=None, stream=None): # Epilog inputs require as many inputs (with specific shapes etc) as required by the epilogue. It's a dict.
        """
        Plan the matrix multiplication operation, considering the epilog (if provided).

        Args:
            preferences: {preferences}
            algorithms: {algorithms}
            epilog: {epilog}
            epilog_inputs: {epilog_inputs}
            stream: {stream}

        Returns:
            A sequence of :class:`nvmath.linalg.advanced.Algorithm` objects that are applicable to this matrix multiplication problem
            specification, heuristically ordered from fastest to slowest.

        Notes:
            Epilogs that have ``BIAS`` in their name need an epilog input with the key ``'bias'``.
            Epilogs that have ``DRELU`` need an epilog input with the key ``'relu_aux'``, which is produced in a "forward pass" epilog like ``RELU_AUX`` or ``RELU_AUX_BIAS``.
            Similarly, epilogs with ``DGELU`` in their name require an epilog input with the key ``'gelu_aux'``, produced in the corresponding forward pass operation.

        Examples:

            >>> import numpy as np
            >>> import nvmath

            Create two 3-D float64 ndarrays on the CPU representing batched matrices, along with a bias vector:

            >>> batch = 32
            >>> M, N, K = 1024, 1024, 1024
            >>> a = np.random.rand(batch, M, K)
            >>> b = np.random.rand(batch, K, N)
            >>> bias = np.random.rand(M)   # The bias vector will be broadcast along the columns, as well as along the batch dimension.

            We will define a matrix multiplication operation followed by a :attr:`nvmath.linalg.advanced.MatmulEpilog.RELU_BIAS` epilog function.

            >>> with nvmath.linalg.advanced.Matmul(a, b) as mm:
            ...
            ...     # Plan the operation with RELU_BIAS epilog and corresonding epilog input.
            ...     p = nvmath.linalg.advanced.MatmulPlanPreferences(limit=8)
            ...     epilog = nvmath.linalg.advanced.MatmulEpilog.RELU_BIAS
            ...     epilog_inputs = {{'bias': bias}}
            ...     mm.plan(preferences=p, epilog=epilog, epilog_inputs=epilog_inputs)    # The preferences can also be provided as a dict: {{'limit': 8}}
            ...
            ...     # Execute the matrix multiplication, and obtain the result `r` as a NumPy ndarray.
            ...     r = mm.execute()

            Some epilogs like :attr:`nvmath.linalg.advanced.MatmulEpilog.RELU_AUX` produce auxiliary output.

            >>> with nvmath.linalg.advanced.Matmul(a, b) as mm:
            ...
            ...     # Plan the operation with RELU_AUX epilog>
            ...     epilog = nvmath.linalg.advanced.MatmulEpilog.RELU_AUX
            ...     mm.plan(epilog=epilog)
            ...
            ...     # Execute the matrix multiplication, and obtain the result `r` along with the auxiliary output.
            ...     r, auxiliary = mm.execute()

            The auxiliary output is a Python `dict` with the names of each auxiliary output as keys.

        Further examples can be found in the `nvmath/examples/linalg/advanced/matmul <https://github.com/NVIDIA/nvmath-python/tree/main/examples/linalg/advanced/matmul>`_ directory.
        """
        self.logger.info(f"= PLANNING PHASE =")

        # Clear epilog operands, since different epilogs can be provided in different calls.
        # We don't need to worry about ordering, since it's the user's responsibility to order calls that accept a stream argument.
        # This applies to CPU operands as well, even though we move them to the GPU, since the execution is blocking.
        self.epilog_operands = dict()                 # Clear operands in case of repeated planning.
        self.epilog_input_name_to_handler = dict()    # Clear input name to handler map as well,
        self.epilog_inputs_traits = dict()            # ... and the input traits as well.

        mm_traits = self.mm_traits
        mm_desc_ifc = matmul_desc_ifc.MatmulDescInterface(self.mm_desc)

        stream_holder = utils.get_or_create_stream(self.device_id, stream, self.package)
        self.logger.info(f"The specified stream for the matrix multiplication plan is {stream_holder.obj}.")

        # Base FLOP count.
        self.flop_count = 2 * mm_traits.M * mm_traits.N * mm_traits.K
        self.logger.info(f"The base matrix multiplication FLOP count is {formatters.FLOPSStr(self.flop_count, 'FLOP')}.")

        if epilog is None and epilog_inputs is not None:
            self.logger.warning(f"Matmul: The provided epilog inputs {epilog_inputs.keys()} are ignored since an epilog is not specified.")

        self.epilog = epilog
        epilog_ordering = None
        if epilog is not None:
            assert epilog in EPILOG_INPUT_HANDLERS_MAP, "Not supported."
            self.logger.info(f"The specified epilog is {epilog.name}.")

            epilog_input_handler_types = EPILOG_INPUT_HANDLERS_MAP[epilog]
            if epilog_input_handler_types:
                epilog_input_handlers = [handler_type(self.logger, mm_traits, epilog, self.d_dtype_name) for handler_type in epilog_input_handler_types]

                # Check if the epilog requires a specific result layout, and if the requirement is consistent for all the handlers.
                epilog_input_handlers_ordering = set(h.order for h in epilog_input_handlers)
                assert len(epilog_input_handlers_ordering) == 1, "Internal error."
                epilog_ordering = epilog_input_handlers_ordering.pop()

                required_epilog_input_names = set(h.name for h in epilog_input_handlers)

                self.logger.info(f"The epilog requires the following additional inputs: {required_epilog_input_names}.")
                if required_epilog_input_names and epilog_inputs is None:
                    raise ValueError(f"The epilog {epilog.name} requires the following input tensors: {required_epilog_input_names}.")

                if required_epilog_input_names != set(epilog_inputs.keys()):
                    raise ValueError(f"The epilog {epilog.name} requires the following input tensors: {required_epilog_input_names}. The provided tensor names are: {epilog_inputs.keys()}")

                # Wrap epilog inputs. Take a copy of the user-provided dict.
                epilog_inputs = epilog_inputs.copy()
                for name in epilog_inputs:
                    epilog_inputs[name] = tensor_wrapper.wrap_operand(epilog_inputs[name])

                # Check if epilog inputs all belong to the same package, which is the same as the package of the MM operands.
                epilog_package = utils.get_operands_package(list(epilog_inputs.values()))
                epilog_package = 'cupy' if epilog_package == 'numpy' else epilog_package   # Handle the NumPy <=> CuPy asymmetry.
                if self.package != epilog_package:
                    message = f"Library package mismatch for epilog: '{self.package}' => '{epilog_package}'"
                    raise TypeError(message)

                # Check if all epilog inputs all are on the same device, which is the device of the operands.
                device_id = utils.get_operands_device_id(list(epilog_inputs.values()))
                if device_id is not None and self.device_id != device_id:
                    raise ValueError(f"The epilog inputs must be on the same device ({device_id}) as the operands ({self.device_id}).")

                # Move epilog inputs to the GPU, if needed.
                if device_id is None:
                    for e in required_epilog_input_names:
                        self.logger.debug(f"The epilog input {e} will be copied to device{self.device_id}.")
                        self.epilog_operands[e] = tensor_wrapper.to(epilog_inputs[e], self.device_id, stream_holder)
                else:
                    for e in required_epilog_input_names:
                        self.epilog_operands[e] = epilog_inputs[e]

                # First validate all epilog inputs. Use the GPU tensors in case metadata has changed.
                for handler in epilog_input_handlers:
                    handler.validate(epilog_inputs[handler.name])

                # Finally, update the MM descriptor. Note that we pass in self.epilog_operands (which are on the GPU).
                for handler in epilog_input_handlers:
                    handler.update(mm_desc_ifc, self.epilog_operands[handler.name])
                    self.epilog_input_name_to_handler[handler.name] = handler

                # Capture the epilog operands traits for consistency checks when resetting operands.
                self.epilog_inputs_traits = {name: EpilogInputTraits(dtype=self.epilog_operands[name].dtype, extents=self.epilog_operands[name].shape, strides=self.epilog_operands[name].strides) for name in self.epilog_operands}

            epilog_output_handler_types = EPILOG_OUTPUT_HANDLERS_MAP[epilog]
            if epilog_output_handler_types:
                self.epilog_output_handlers = epilog_output_handlers = [handler_type(self.logger, mm_traits, epilog, self.d_dtype_name) for handler_type in epilog_output_handler_types]

                # Check if the epilog requires a specific result layout, and if the requirement is consistent for all the handlers.
                epilog_output_handlers_ordering = set(h.order for h in epilog_output_handlers)
                assert len(epilog_output_handlers_ordering) == 1, "Internal error."
                op_epilog_ordering = epilog_output_handlers_ordering.pop()
                if epilog_ordering is None:
                    epilog_ordering = op_epilog_ordering
                else:
                    assert epilog_ordering == op_epilog_ordering, "Internal error."

                # Update the MM descriptor, except for the device pointer.
                for handler in epilog_output_handlers:
                    handler.update(mm_desc_ifc)

            # Set the epilog. At this point, we're sure that the epilog inputs, if any, are valid and have been set.
            mm_desc_ifc.epilogue = epilog

        # Fill the result traits, now that we know the epilog.
        self.result_traits = result_traits = get_result_traits(mm_traits, epilog_ordering, self.logger)
        self.logger.info(f"The layout order for the result D is {self.result_traits.d_layout_traits.order.name}, with LD {self.result_traits.d_layout_traits.ld}, and batch offset {self.result_traits.d_layout_traits.batch_offset}.")

        preferences = utils.check_or_create_options(configuration.MatmulPlanPreferences, preferences, "Matrix multiplication plan preferences")

        # Internally transpose operand A if required (conjugate flag) and create layout.
        transpose = False
        if mm_traits.a_layout_traits.is_conjugate and 'complex' in self.ab_dtype_name:
            mm_desc_ifc.transa = cublas.Operation.C
            transpose = True
            self.logger.debug(f"To conjugate A, the operand A will be internally transposed and the matrix multiplication will be performed with OP_C for operand A.")
        m, n, ld, a_order = mm_traits.a_layout_traits.get_mm_layout(transpose=transpose)
        self.a_layout_ptr = cublaslt.matrix_layout_create(self.a_dtype, rows=m, cols=n, ld=ld)
        self.logger.debug(f"Layout for A: rows = {m}, cols = {n}, ld = {ld}.")

        # Internally transpose operand B if required (conjugate flag, or epilog is BGRADB) and create layout.
        transpose = False
        if mm_traits.b_layout_traits.is_conjugate and 'complex' in self.ab_dtype_name:
            mm_desc_ifc.transb = cublas.Operation.C
            transpose = True
            self.logger.debug(f"To conjugate B, the operand B will be internally transposed and the matrix multiplication will be performed with OP_C for operand B.")
        elif epilog == configuration.MatmulEpilog.BGRADB:
            mm_desc_ifc.transb = cublas.Operation.T
            transpose = True
            self.logger.debug(f"For BGRADB epilog, the operand B will be internally transposed and the matrix multiplication will be performed with OP_T for operand B.")
        m, n, ld, b_order = mm_traits.b_layout_traits.get_mm_layout(transpose=transpose)
        self.b_layout_ptr = cublaslt.matrix_layout_create(self.b_dtype, rows=m, cols=n, ld=ld)
        self.logger.debug(f"Layout for B: rows = {m}, cols = {n}, ld = {ld}.")

        self.d_layout_ptr = cublaslt.matrix_layout_create(self.d_dtype, rows=mm_traits.M, cols=mm_traits.N, ld=result_traits.d_layout_traits.ld)

        layout_a_ifc = matrix_layout_ifc.MatrixLayoutInterface(self.a_layout_ptr)
        layout_a_ifc.order = a_order
        layout_a_ifc.batch_count = mm_traits.batch_count
        layout_a_ifc.strided_batch_offset = mm_traits.a_layout_traits.batch_offset

        layout_b_ifc = matrix_layout_ifc.MatrixLayoutInterface(self.b_layout_ptr)
        layout_b_ifc.order = b_order
        layout_b_ifc.batch_count = mm_traits.batch_count
        layout_b_ifc.strided_batch_offset = mm_traits.b_layout_traits.batch_offset

        layout_d_ifc = matrix_layout_ifc.MatrixLayoutInterface(self.d_layout_ptr)
        layout_d_ifc.order = result_traits.d_layout_traits.order
        layout_d_ifc.batch_count = mm_traits.batch_count
        layout_d_ifc.strided_batch_offset = result_traits.d_layout_traits.batch_offset

        if self.num_operands == 2:
            self.c_layout_ptr = self.d_layout_ptr
        else:
            self.c_layout_ptr = cublaslt.matrix_layout_create(self.c_dtype, rows=mm_traits.M, cols=mm_traits.N, ld=mm_traits.c_layout_traits.ld)
            layout_c_ifc = matrix_layout_ifc.MatrixLayoutInterface(self.c_layout_ptr)
            layout_c_ifc.order = mm_traits.c_layout_traits.order
            layout_c_ifc.batch_count = mm_traits.batch_count
            layout_c_ifc.strided_batch_offset = mm_traits.c_layout_traits.batch_offset

        limit = preferences.limit
        if algorithms is None:
            num_algorithms = np.empty((1,), dtype=np.int32)
            self.algorithms_buffer  = np.zeros((limit,), dtype=algorithmmod.algorithm_dtype)
        else:
            assert all(isinstance(algo, algorithmmod.Algorithm) for algo in algorithms), "The algorithms passed to plan() are of wrong type."
            num_algorithms = len(algorithms)

        if self.preference_ptr is None:
            self.preference_ptr = cublaslt.matmul_preference_create()
        else:
            # We need to create a new preferences object to avoid preferences being set in a cumulative manner if plan() is called multiple times.
            cublaslt.matmul_preference_destroy(self.preference_ptr)
            self.preference_ptr = cublaslt.matmul_preference_create()

        if algorithms is None:
            # Set preferences.
            preference_ifc = matmul_pref_ifc.MatmulPreferenceInterface(self.preference_ptr)
            preference_ifc.max_workspace_bytes = self.memory_limit
            preference_ifc.reduction_scheme_mask = preferences.reduction_scheme_mask
            preference_ifc.max_waves_count = preferences.max_waves_count
            preference_ifc.impl_mask = preferences.numerical_impl_mask

            # Set minimum aligments.
            a_ptr, b_ptr = self.operands[0].data_ptr, self.operands[1].data_ptr
            preference_ifc.min_alignment_a_bytes = min(256, pointer_aligned_to(a_ptr))
            preference_ifc.min_alignment_b_bytes = min(256, pointer_aligned_to(b_ptr))
            self.logger.debug(f"The minimum alignment for operand A is {preference_ifc.min_alignment_a_bytes} bytes.")
            self.logger.debug(f"The minimum alignment for operand B is {preference_ifc.min_alignment_b_bytes} bytes.")
            if self.num_operands == 3:
                c_ptr = self.operands[2].data_ptr
                preference_ifc.min_alignment_c_bytes = min(256, pointer_aligned_to(c_ptr))
                self.logger.debug(f"The minimum alignment for operand C is {preference_ifc.min_alignment_c_bytes} bytes.")
            # The result alignment should be 256 bytes.
            self.logger.debug(f"The minimum alignment for the result D is the default 256 bytes.")

            timing =  bool(self.logger and self.logger.handlers)
            self.logger.info("Starting matrix multiplication planning...")
            with utils.device_ctx(self.device_id), utils.cuda_call_ctx(stream_holder, blocking=True, timing=timing) as (_, elapsed):
                cublaslt.matmul_algo_get_heuristic(self.handle, self.mm_desc, self.a_layout_ptr, self.b_layout_ptr, self.c_layout_ptr, self.d_layout_ptr, self.preference_ptr, limit, self.algorithms_buffer.ctypes.data, num_algorithms.ctypes.data)

            num_algorithms = num_algorithms[0]
            if num_algorithms == 0:
                raise RuntimeError("Planning failed to find any suitable algorithm.")
            self.algorithms_buffer = self.algorithms_buffer[:num_algorithms]

            # Create algorithm objects.
            self.algorithm_objects = tuple(algorithmmod.Algorithm(a) for a in self.algorithms_buffer)
        else:
            self.algorithm_objects = tuple(algorithms)
            self.algorithms_buffer = np.array([algo.algorithm for algo in algorithms], dtype=algorithmmod.algorithm_dtype)

        # Create the map from object to buffer.
        self.algorithm_object_to_buffer = dict(zip(self.algorithm_objects, self.algorithms_buffer))

        self.workspace_size = int(np.max(self.algorithms_buffer["workspace_size"]))

        if algorithms is None:
            self.logger.info(f"The plan found {num_algorithms} suitable algorithms within the requested limit of {limit} algorithms, with a workspace requirement of {formatters.MemoryStr(self.workspace_size)}.")
        else:
            self.logger.info(f"The plan is using {num_algorithms} algorithm passed through the algorithms argument, with a workspace requirement of {formatters.MemoryStr(self.workspace_size)}.")

        self.mm_planned = True
        if algorithms is None and elapsed.data is not None:
                self.logger.info(f"The matrix multiplication planning phase took {elapsed.data:.3f} ms to complete.")

        return self.algorithm_objects

    @property
    def algorithms(self):
        """
        After planning using :meth:`plan()`, get the sequence of algorithm objects to inquire their capabilities, configure them, or serialize them for later use.

        Returns:
            A sequence of :class:`nvmath.linalg.advanced.Algorithm` objects that are applicable to this matrix multiplication problem specification.
        """
        return self.algorithm_objects

    def _check_and_set_operand(self, operand, operand_name, mm_desc_ifc, stream_holder, *, operand_index=None, epilog_name=None, package=None, dtype=None, extents=None, strides=None):
        """
        Check to make sure that the provided operand is consistent with the one it's updating, and update it.
        """
        assert (operand_index is None) ^ (epilog_name is None), "Internal Error."

        # Make sure that the data type and extents match.
        utils.check_attribute_match(dtype, operand.dtype, "data type")
        utils.check_attribute_match(extents, operand.shape, "extents")

        package = utils.infer_object_package(operand.tensor)

        # Conjugate flag of the provided operands must match the original qualifiers
        if operand_index is not None and package == 'torch' and self.lazy_conjugation:
            if self.qualifiers[operand_index]['is_conjugate'] != operand.tensor.is_conj():
                raise ValueError(f"The provided operand {operand_name} has different conjugate flag than the original operand")

        device_id = operand.device_id
        if device_id is None:
            package = 'cupy' if package == 'numpy' else package   # Handle the NumPy <=> CuPy asymmetry.
            if self.package != package:
                message = f"Library package mismatch: '{self.package}' => '{package}'"
                raise TypeError(message)

            # Check if we have a GPU buffer to update into.
            if operand_index is not None:
                o = self.operands[operand_index]
            else:
                o = self.epilog_operands[epilog_name]
            if o is None:    # No buffer, create one.
                # Copy operand across memory spaces (CPU to GPU).
                o = tensor_wrapper.to(operand, self.device_id, stream_holder)
                if operand_index is not None:
                    self.operands[operand_index] = o
                else:
                    self.epilog_operands[epilog_name] = o
                    # Update the epilog pointer, since we're starting afresh.
                    self.epilog_input_name_to_handler[epilog_name].update(mm_desc_ifc, o)
            else:
                # In-place copy to existing device pointer because the new operand is on the CPU.
                tensor_wrapper.copy_([operand], [o], stream_holder)
        else:
            if self.package != package:
                message = f"Library package mismatch: '{self.package}' => '{package}'"
                raise TypeError(message)

            utils.check_attribute_match(strides, operand.strides, "strides")

            if self.device_id != device_id:
                raise ValueError(f"The operand {operand_name} must be on the same device ({device_id}) as the original operand "
                                 f"({self.device_id}).")

            # Finally, replace the original operand by the new one.
            if operand_index is not None:
                self.operands[operand_index] = operand
            else:
                self.epilog_operands[epilog_name] = operand
                # Update the epilog pointer, since we're starting afresh.
                self.epilog_input_name_to_handler[epilog_name].update(mm_desc_ifc, operand)

        self.logger.info(f"Operand '{operand_name}' has been reset to the new value.")

        return

    @utils.precondition(_check_valid_matmul)
    def reset_operands(self, a=None, b=None, c=None, *, alpha=None, beta=None, epilog_inputs=None, stream=None):
        """
        Reset the operands held by this :class:`Matmul` instance.

        This method has two use cases: (1) it can be used to provide new operands for execution when the original operands are on the CPU,
        or (2) it can be used to release the internal reference to the previous operands and make their memory available for other use by
        passing ``None`` for *all* arguments. In this case, this method must be called again to provide the desired operands before another
        call to execution APIs like :meth:`autotune` or :meth:`execute`.

        This method is not needed when the operands reside on the GPU and in-place operations are used to update the operand values.

        This method will perform various checks on the new operands to make sure:

            - The shapes, strides, datatypes match those of the old ones.
            - The packages that the operands belong to match those of the old ones.
            - If input tensors are on GPU, the device must match.

        Args:
            a: {a}
            b: {b}
            c: {c}
            alpha: {alpha}
            beta: {beta}
            epilog_inputs: {epilog_inputs}
            stream: {stream}

        Examples:

            >>> import cupy as cp
            >>> import nvmath

            Create two 3-D float64 ndarrays on the GPU:

            >>> M, N, K = 128, 128, 256
            >>> a = cp.random.rand(M, K)
            >>> b = cp.random.rand(K, N)

            Create an matrix multiplication object as a context manager

            >>> with nvmath.linalg.advanced.Matmul(a, b) as mm:
            ...    # Plan the operation.
            ...    mm.plan()
            ...
            ...    # Execute the MM to get the first result.
            ...    r1 = mm.execute()
            ...
            ...    # Reset the operands to new CuPy ndarrays.
            ...    c = cp.random.rand(M, K)
            ...    d = cp.random.rand(K, N)
            ...    mm.reset_operands(c, d)
            ...
            ...    # Execute to get the new result corresponding to the updated operands.
            ...    r2 = mm.execute()

            Note that if only a subset of operands are reset, the operands that are not reset hold their original values.

            With :meth:`reset_operands`, minimal overhead is achieved as problem specification and planning are only performed once.

            For the particular example above, explicitly calling :meth:`reset_operands` is equivalent to updating the operands in-place, i.e, replacing ``mm.reset_operand(c, d)`` with ``a[:]=c`` and ``b[:]=d``.
            Note that updating the operand in-place should be adopted with caution as it can only yield the expected result under the additional constraint below:
            
                - The operand is on the GPU (more precisely, the operand memory space should be accessible from the execution space).

            For more details, please refer to `inplace update example <https://github.com/NVIDIA/nvmath-python/tree/main/examples/linalg/advanced/example05_stateful_inplace.py>`_.
        """

        if c is not None and self.num_operands == 2:
                raise ValueError("The matrix multiplication problem specification does not include operand C, so it cannot be reset.")

        if a is None and b is None and c is None and epilog_inputs is None and alpha is None and beta is None:
            self.operands = None
            self.epilog_operands = dict()
            self.logger.info(f"The operands have been reset to None.")
            return

        # If the operands have been reset to None, then all required operands (a, b, c, and epilog_inputs need to be provided).
        if self.operands is None:
            if a is None or b is None or (c is None and self.num_operands == 3):
                op_names = "A, B"
                if c is None and self.num_operands == 3:
                    op_names += ", C"
                raise ValueError(f"Operands {op_names} must be provided.")
            epilog_names = self.epilog_inputs_traits.keys()
            if epilog_inputs is None:
                if epilog_names:
                    raise ValueError(f"The epilog inputs {epilog_names} must be provided.")
            else:
                # Check that all required epilog inputs names are provided.
                if epilog_names != epilog_inputs.keys():
                    raise ValueError(f"The epilog inputs {epilog_names} are required. The provided epilog input names are {epilog_inputs.keys()}.")
            self.operands = [None] * self.num_operands
            self.epilog_operands = {name : None for name in epilog_names}

        # Future operations on the workspace stream should be ordered after the computation.
        if self.last_compute_event is not None:
            self.workspace_stream.wait_event(self.last_compute_event)

        # Update alpha.
        if alpha is not None:
            try:
                self.alpha[0] = alpha
            except (ValueError, TypeError) as e:
                raise ValueError(f"The value provided for alpha {alpha} is not convertible to dtype '{self.alpha.dtype}'.") from e

        # Update beta.
        if beta is not None:
            if self.num_operands == 2:
                self.logger.warning(f"Matmul: The provided beta value {beta} is ignored since operand C is not specified.")
            else:
                try:
                    self.beta[0] = beta
                except (ValueError, TypeError) as e:
                    raise ValueError(f"The value provided for beta {beta} is not convertible to dtype '{self.beta.dtype}'.") from e

        mm_desc_ifc = matmul_desc_ifc.MatmulDescInterface(self.mm_desc)

        stream_holder = utils.get_or_create_stream(self.device_id, stream, self.package)

        # Reset the provided operands.
        if a is not None:
            a = tensor_wrapper.wrap_operand(a)
            index = 0
            self._check_and_set_operand(a, 'A', mm_desc_ifc, stream_holder, operand_index=index, dtype=self.ab_dtype_name, extents=self.operand_extents[index], strides=self.operand_strides[index])

        if b is not None:
            b = tensor_wrapper.wrap_operand(b)
            index = 1
            self._check_and_set_operand(b, 'B', mm_desc_ifc, stream_holder, operand_index=index, dtype=self.ab_dtype_name, extents=self.operand_extents[index], strides=self.operand_strides[index])

        if c is not None:    # If we get here, we know that C is one of the operands in the problem specification.
            c = tensor_wrapper.wrap_operand(c)
            index = 2
            self._check_and_set_operand(c, 'C', mm_desc_ifc, stream_holder, operand_index=index, dtype=self.c_dtype_name, extents=self.operand_extents[index], strides=self.operand_strides[index])

        # Reset the provided epilog inputs.
        if epilog_inputs is not None:
            for name in epilog_inputs:
                epilog_input = tensor_wrapper.wrap_operand(epilog_inputs[name])
                self._check_and_set_operand(epilog_input, name, mm_desc_ifc, stream_holder, epilog_name=name, dtype=self.epilog_inputs_traits[name].dtype, extents=self.epilog_inputs_traits[name].extents, strides=self.epilog_inputs_traits[name].strides)

    @utils.precondition(_check_valid_matmul)
    @utils.precondition(_check_planned, "Autotuning")
    @utils.precondition(_check_valid_operands, "Autotuning")
    @utils.atomic(_release_workspace_memory_perhaps, method=True)
    def autotune(self, iterations=3, prune=None, release_workspace=False, stream=None): # Prune means keep top N of the algorithms only.
        """
        Autotune the matrix multiplication to order the algorithms from the fastest measured execution time to the slowest. Once autotuned,
        the optimally-ordered algorithm sequence can be accessed using :py:attr:`algorithms`.

        Args:
            iterations: The number of autotuning iterations to perform.
            prune: An integer N, specifying the top N fastest algorithms to retain after autotuning. The default is to retain all algorithms.
            release_workspace: {release_workspace}
            stream: {stream}
        """
        self.logger.info(f"= AUTOTUNING PHASE =")
        # Measure time taken for autotuning.
        from timeit import default_timer as timer

        self.logger.info(f"Starting autotuning...")
        start = timer()

        num_algorithms = len(self.algorithm_objects)
        limit = min(prune, num_algorithms) if prune is not None else num_algorithms
        self.logger.info(f"The number of algorithms in the plan is {num_algorithms}, from which the top {limit} will be retained.")
        self.logger.info(f"The requested number of iterations is {iterations}.")

        # Autotune setup.
        stream_holder = utils.get_or_create_stream(self.device_id, stream, self.package)

        # Allocate workspace if needed.
        self._allocate_workspace_memory_perhaps(stream_holder)

        # Create empty tensors for auxiliary output.
        mm_desc_ifc = matmul_desc_ifc.MatmulDescInterface(self.mm_desc)
        epilog_outputs = dict()
        for handler in self.epilog_output_handlers:
            name = handler.name
            shape, strides, dtype_name = handler.attributes()
            epilog_outputs[name] = aux = utils.create_empty_tensor(self.result_class, shape, dtype_name, self.device_id, stream_holder, strides)

            # Update the data pointer in the MM descriptor.
            handler.update_ptr(mm_desc_ifc, aux.data_ptr)

        # Create empty tensor for the result.
        result = utils.create_empty_tensor(self.result_class, self.result_traits.result_shape, self.d_dtype_name, self.device_id, stream_holder, self.result_traits.result_strides)
        result_ptr = result.data_ptr

        c_ptr = self.operands[2].data_ptr if self.num_operands == 3 else result_ptr
        a, b = self.operands[0], self.operands[1]
        raw_workspace_ptr = utils.get_ptr_from_memory_pointer(self.workspace_ptr)
        alpha_ptr, a_ptr, b_ptr, beta_ptr = self.alpha.ctypes.data, a.data_ptr, b.data_ptr, self.beta.ctypes.data

        def execute_matmul(algorithm_ptr):
            cublaslt.matmul(self.handle, self.mm_desc, alpha_ptr, a_ptr, self.a_layout_ptr, b_ptr, self.b_layout_ptr, beta_ptr, c_ptr, self.c_layout_ptr, result_ptr, self.d_layout_ptr, algorithm_ptr, raw_workspace_ptr, self.workspace_size, stream_holder.ptr)

        # Tune.
        with utils.device_ctx(self.device_id), utils.cuda_call_ctx(stream_holder, blocking=False, timing=False) as (self.last_compute_event, elapsed):
            gpu_times = list()
            for algorithm_struct in self.algorithms_buffer:
                algorithm_ptr = algorithm_struct['algorithm'].ctypes.data
                t = list()
                for i in range(iterations):
                   start0 = stream_holder.obj.record()
                   execute_matmul(algorithm_ptr=algorithm_ptr)
                   end0 = stream_holder.obj.record()
                   end0.synchronize()
                   t.append(cp.cuda.get_elapsed_time(start0, end0))
                gpu_times.append(min(t))

        # Establish ordering wrt the computation and free workspace if requested.
        self._release_workspace_memory_perhaps(release_workspace=release_workspace)
        self._reset_workspace_allocation_tracking()

        # Get the sort order based on the GPU times.
        sorted_gpu_times, sort_order = zip(*sorted(zip(gpu_times, range(num_algorithms))))

        # Reorder the algorithms buffer and algorithm objects according to the sort order, and prune it.
        algorithms_buffer = np.zeros((limit,), dtype=algorithmmod.algorithm_dtype)
        algorithm_objects = np.zeros((limit,), dtype=algorithmmod.Algorithm)
        for i in range(limit):
            algorithms_buffer[i] = self.algorithms_buffer[sort_order[i]]
            algorithm_objects[i] = self.algorithm_objects[sort_order[i]]
        self.algorithms_buffer = algorithms_buffer
        self.algorithm_objects = algorithm_objects

        # Create the map from object to buffer.
        self.algorithm_object_to_buffer = dict(zip(self.algorithm_objects, self.algorithms_buffer))

        gpu_times_str = ", ".join(f"{t:0.3f}" for t in gpu_times)
        self.logger.info(f"The autotuned GPU times (in milliseconds) are: {gpu_times_str}.")
        self.logger.info(f"The corresponding sort order is: {sort_order}.")
        orig_flop_rate = self.flop_count / gpu_times[0] * 1000
        if sort_order[0] != 0:
            self.logger.info(f"Autotuning found that the algorithm originally ranked {sort_order[0]} is the best out of the {num_algorithms} in the plan, and moved it to rank 0.")
            new_flop_rate = self.flop_count / sorted_gpu_times[0] * 1000
            self.logger.info(f"Autotuning has improved performance from {formatters.FLOPSStr(orig_flop_rate, 'FLOP/s')} to {formatters.FLOPSStr(new_flop_rate, 'FLOP/s')}.")
        else:
            self.logger.info(f"Autotuning found that the algorithm ranked best by the plan heuristics remains the best out of the {num_algorithms} algorithms in the plan.")
            self.logger.info(f"The best performance remains at {formatters.FLOPSStr(orig_flop_rate, 'FLOP/s')}.")

        end = timer()
        self.logger.info(f"The autotuning took {(end - start) * 1000.:.3f} ms to complete.")

    @utils.precondition(_check_valid_matmul)
    @utils.precondition(_check_planned, "Execution")
    @utils.precondition(_check_valid_operands, "Execution")
    @utils.atomic(_release_workspace_memory_perhaps, method=True)
    def execute(self, *, algorithm=None, release_workspace=False, stream=None):
        """
        Execute a prepared (planned and possibly autotuned) matrix multiplication.

        Args:
            algorithm: (Experimental) An algorithm chosen from the sequence returned by :meth:`plan` or :py:attr:`algorithms`. By default,
                the first algorithm in the sequence is used.
            release_workspace: {release_workspace}
            stream: {stream}

        Returns:
           {result}
        """
        log_info  = self.logger.isEnabledFor(logging.INFO)
        log_debug = self.logger.isEnabledFor(logging.DEBUG)

        if log_info: self.logger.info(f"= EXECUTION PHASE =")
        stream_holder = utils.get_or_create_stream(self.device_id, stream, self.package)
        if log_info: self.logger.info(f"The specified stream for execute() is {stream_holder.obj}.")

        # Allocate workspace if needed.
        self._allocate_workspace_memory_perhaps(stream_holder)

        # Create empty tensors for auxiliary output.
        mm_desc_ifc = matmul_desc_ifc.MatmulDescInterface(self.mm_desc)
        for handler in self.epilog_output_handlers:
            name = handler.name
            shape, strides, dtype_name = handler.attributes()
            if log_debug: self.logger.debug(f"Beginning auxiliary output tensor '{name}' creation...")
            if log_debug: self.logger.debug(f"The '{name}' tensor shape = {shape} with strides = {strides} and data type '{dtype_name}'.")
            self.epilog_outputs[name] = aux = utils.create_empty_tensor(self.result_class, shape, dtype_name, self.device_id, stream_holder, strides)
            if log_debug: self.logger.debug(f"The auxiliary output tensor '{name}' has been created.")

            # Update the data pointer in the MM descriptor.
            handler.update_ptr(mm_desc_ifc, aux.data_ptr)

        # Create empty tensor for the result.
        if log_debug: self.logger.debug("Beginning output (empty) tensor creation...")
        if log_debug: self.logger.debug(f"The output tensor shape = {self.result_traits.result_shape} with strides = {self.result_traits.result_strides} and data type '{self.d_dtype_name}'.")
        self.result = utils.create_empty_tensor(self.result_class, self.result_traits.result_shape, self.d_dtype_name, self.device_id, stream_holder, self.result_traits.result_strides)
        if log_debug: self.logger.debug("The output (empty) tensor has been created.")

        result_ptr = self.result.data_ptr

        # Select the first (best) algorithm if one is not provided.
        if algorithm is None:
            algorithm_struct = self.algorithms_buffer[0]['algorithm']
            if log_info: self.logger.info(f"The highest ranked algorithm in the plan (algorithm id = {self.algorithm_objects[0].algorithm_id}) will be used.")
        else:
            if algorithm not in self.algorithm_objects:
                raise ValueError("Algorithm passed to execute() has to be included in the plan() algorithms")
            algorithm_struct = self.algorithm_object_to_buffer[algorithm]['algorithm']
            if log_info: self.logger.info(f"The specified algorithm (algorithm id = {algorithm.algorithm_id}) will be used.")

        c_ptr = self.operands[2].data_ptr if self.num_operands == 3 else self.result.data_ptr
        a, b = self.operands[0], self.operands[1]
        raw_workspace_ptr = utils.get_ptr_from_memory_pointer(self.workspace_ptr)
        timing =  bool(self.logger and self.logger.handlers)
        if log_info: self.logger.info(f"Starting matrix multiplication...")
        if log_info: self.logger.info(f"{self.call_prologue}")
        with utils.device_ctx(self.device_id), utils.cuda_call_ctx(stream_holder, self.blocking, timing) as (self.last_compute_event, elapsed):
            cublaslt.matmul(self.handle, self.mm_desc, self.alpha.ctypes.data, a.data_ptr, self.a_layout_ptr, b.data_ptr, self.b_layout_ptr, self.beta.ctypes.data, c_ptr, self.c_layout_ptr, self.result.data_ptr, self.d_layout_ptr, algorithm_struct.ctypes.data, raw_workspace_ptr, self.workspace_size, stream_holder.ptr)

        if elapsed.data is not None:
            if log_info: self.logger.info(f"The matrix multiplication calculation took {elapsed.data:.3f} ms to complete.")

        # Establish ordering wrt the computation and free workspace if requested.
        self._release_workspace_memory_perhaps(release_workspace=release_workspace)

        # Return the result and auxiliary outputs, if present.
        aux = dict()
        if self.memory_space == 'cpu':
            out = self.result.to('cpu', stream_holder=stream_holder)
            # Copy auxiliary output to CPU.
            for name in self.epilog_outputs:
                aux[name] = self.epilog_outputs[name].to('cpu', stream_holder=stream_holder)
        else:
            out = self.result.tensor
            # Return the unwrapped epilog output tensor(s).
            aux = {name: self.epilog_outputs[name].tensor for name in self.epilog_outputs}

        # Release internal reference to the result to permit recycling of memory.
        self.result = None
        self.epilog_outputs = dict()
        self._reset_workspace_allocation_tracking()

        if aux:
            return out, aux

        return out

    def free(self):
        """Free Matmul resources.

        It is recommended that the :class:`Matmul` object be used within a context, but if it is not possible then this
        method must be called explicitly to ensure that the matrix multiplication resources (especially internal library objects) are
        properly cleaned up.
        """

        if not self.valid_state:
            return

        try:
            # Future operations on the workspace stream should be ordered after the computation.
            if self.last_compute_event is not None:
                self.workspace_stream.wait_event(self.last_compute_event)

            self._free_workspace_memory()

            self._free_plan_resources()

            # We won't destroy the handle.

        except Exception as e:
            self.logger.critical("Internal error: only part of the Matmul object's resources have been released.")
            self.logger.critical(str(e))
            raise e
        finally:
            self.valid_state = False

        self.logger.info("The Matmul object's resources have been released.")


@utils.docstring_decorator(SHARED_MM_DOCUMENTATION, skip_missing=False)
def matmul(a, b, /, c=None, *, alpha=None, beta=None, epilog=None, epilog_inputs=None, qualifiers=None, options=None, preferences=None, algorithm=None, stream=None):
    r"""
    Perform the specified matrix multiplication computation :math:`F(\alpha a @ b + \beta c)`, where :math:`F` is the epilog. This function-form is a wrapper around the
    stateful :class:`Matmul` object APIs and is meant for *single* use (the user needs to perform just one matrix multiplication, for
    example), in which case there is no possibility of amortizing preparatory costs.

    Detailed information on what's happening within this function can be obtained by passing in a :class:`logging.Logger` object
    to :class:`MatmulOptions` or by setting the appropriate options in the root logger object, which is used by default:

        >>> import logging
        >>> logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%m-%d %H:%M:%S')

    A user can select the desired logging level and, in general, take advantage of all of the functionality offered by the Python `logging` module.

    Args:
        a: {a}
        b: {b}
        c: {c}
        alpha: {alpha}
        beta: {beta}
           from a previously planned and autotuned matrix multiplication.
        epilog: {epilog}
        epilog_inputs: {epilog_inputs}
        qualifiers: {qualifiers}
        options: {options}
        preferences: {preferences}
        algorithm: An object of type :class:`Algorithm` objects can be directly provided to bypass planning, if desired. The algorithm object must
            be compatible with the matrix multiplication. A typical use for this option is to provide an algorithm that has been serialized
            (pickled) from a previously planned and autotuned matrix multiplication.
        stream: {stream}

    Semantics:
        {semantics}

    See Also:
        :class:`Matmul`, :class:`MatmulOptions`, :class:`MatmulEpilog`, :class:`MatmulPlanPreferences`

    Examples:

        >>> import cupy as cp
        >>> import nvmath

        Create three float64 ndarrays on the GPU:

        >>> M, N, K = 128, 64, 256
        >>> a = cp.random.rand(M, K)
        >>> b = cp.random.rand(K, N)
        >>> c = cp.random.rand(M, N)

        Perform the operation :math:`\alpha A @ B + \beta C` using :func:`matmul`. The result `r` is also a CuPy float64 ndarray:

        >>> r = nvmath.linalg.advanced.matmul(a, b, c, alpha=1.23, beta=0.74)

        An epilog can be used as well. Here we perform :math:`RELU(\alpha A @ B + \beta C)`:

        >>> epilog = nvmath.linalg.advanced.MatmulEpilog.RELU
        >>> r = nvmath.linalg.advanced.matmul(a, b, c, alpha=1.23, beta=0.74, epilog=epilog)

        Options can be provided to customize the operation:

        >>> compute_type = nvmath.linalg.advanced.MatmulComputeType.COMPUTE_32F_FAST_TF32
        >>> o = nvmath.linalg.advanced.MatmulOptions(compute_type=compute_type)
        >>> r = nvmath.linalg.advanced.matmul(a, b, options=o)

        See `MatmulOptions` for the complete list of available options.

        The package current stream is used by default, but a stream can be explicitly provided to the Matmul operation. This can be done if the
        operands are computed on a different stream, for example:

        >>> s = cp.cuda.Stream()
        >>> with s:
        ...    a = cp.random.rand(M, K)
        ...    b = cp.random.rand(K, N)
        >>> nvmath.linalg.advanced.matmul(a, b, stream=s)

        The operation above runs on stream `s` and is ordered with respect to the input computation.

        Create  NumPy ndarrays on the CPU.

        >>> import numpy as np
        >>> a = np.random.rand(M, K)
        >>> b = np.random.rand(K, N)

        Provide the NumPy ndarrays to :func:`matmul`, with the result also being a NumPy ndarray:

        >>> r = nvmath.linalg.advanced.matmul(a, b)

    Notes:
        - This function is a convenience wrapper around :class:`Matmul` and and is specifically meant for *single* use.

    Further examples can be found in the `nvmath/examples/linalg/advanced/matmul <https://github.com/NVIDIA/nvmath-python/tree/main/examples/linalg/advanced/matmul>`_ directory.
    """

    # Set algorithm limit to 1, but take a copy first if needed.
    if isinstance(preferences, configuration.MatmulPlanPreferences):
        preferences = copy.copy(preferences)

    preferences = utils.check_or_create_options(configuration.MatmulPlanPreferences, preferences, "Matrix multiplication plan preferences")
    preferences.limit = 1

    if algorithm is None:
        algorithms = None
    else:
        algorithms = [algorithm] # The type of algorithm should be algorithm.Algorithm and will be checked in plan()

    with Matmul(a, b, c=c, alpha=alpha, beta=beta, qualifiers=qualifiers, options=options, stream=stream) as mm:

        mm.plan(preferences=preferences, epilog=epilog, epilog_inputs=epilog_inputs, stream=stream, algorithms=algorithms)

        r = mm.execute(stream=stream)

    return r
