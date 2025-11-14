"""
Matrix multiplication function selection and matching for BLAS operations.

This module provides functionality to select and configure the appropriate BLAS matrix
multiplication functions based on matrix qualifiers, batch traits, and execution context.
It supports both CPU (NVPL) and GPU (cuBLAS) backends and handles various matrix types
including general, symmetric, hermitian, triangular, and diagonal matrices.
"""

import logging
import typing

import numpy as np

from nvmath._internal.templates import ExecutionCPU, ExecutionCUDA
from nvmath.internal import tensor_wrapper, typemaps, utils
from nvmath.linalg._internal.batch import BatchTraits
from nvmath.linalg._internal.layout import BLASMMTraits
import nvmath.bindings.cublas as cublas

from .qualifiers import (
    DiagonalMatrixQualifier,
    GeneralMatrixQualifier,
    HermitianMatrixQualifier,
    MatrixQualifier,
    SymmetricMatrixQualifier,
    TriangularMatrixQualifier,
)
from .wrap import (
    cublas_enum_mapper,
    cublas_mm_function,
    get_address_zeroth_element,
    get_value_zeroth_element,
    nvpl_enum_mapper,
    nvpl_mm_function,
)

"""A matmul function returned by this module."""
WrappedMMFunction: typing.TypeAlias = typing.Callable[
    [
        tensor_wrapper.TensorHolder,
        tensor_wrapper.TensorHolder,
        tensor_wrapper.TensorHolder,
        np.ndarray,
        np.ndarray,
        utils.StreamHolder | None,
    ],
    None,
]

"""A function which returns a BLAS implementation's matching matmul function."""
MMFunctionGetter: typing.TypeAlias = typing.Callable[
    [
        ExecutionCUDA | ExecutionCPU,
        typemaps.cudaDataType,
        str,
        logging.Logger,
        typing.Literal["", "stride", "group"],
    ],
    typing.Callable,
]


def select_blas_mm_function(
    batch_traits: tuple[BatchTraits, BatchTraits, BatchTraits],
    mm_traits: BLASMMTraits,
    qualifiers: MatrixQualifier,
    logger: logging.Logger,
    execution: ExecutionCUDA | ExecutionCPU,
) -> WrappedMMFunction:
    """Return a matrix multiplication function which matches the provided arguments."""

    # At this level, we only select the appropriate library
    match execution:
        case ExecutionCPU():
            mm_function_getter = nvpl_mm_function
            mm_enum_mapper = nvpl_enum_mapper
            # NOTE: the BLAS APIs take values for real alpha/beta, pointers for complex
            if mm_traits.a_layout_traits.dtype.name.startswith("CUDA_C"):
                mm_alpha_beta_picker = get_address_zeroth_element
            else:
                mm_alpha_beta_picker = get_value_zeroth_element
        case ExecutionCUDA():
            mm_function_getter = cublas_mm_function
            mm_enum_mapper = cublas_enum_mapper
            mm_alpha_beta_picker = get_address_zeroth_element
        case _:
            raise ValueError("Only ExectionCUDA and ExecutionCPU are supported.")

    return _select_blas_mm_function_from_qualifiers(
        batch_traits,
        mm_traits,
        qualifiers,
        logger,
        execution,
        mm_function_getter,
        mm_enum_mapper,
        mm_alpha_beta_picker,
    )


def _select_blas_mm_function_from_qualifiers(
    batch_traits: tuple[BatchTraits, BatchTraits, BatchTraits],
    mm_traits: BLASMMTraits,
    qualifiers: MatrixQualifier,
    logger: logging.Logger,
    execution: ExecutionCUDA | ExecutionCPU,
    mm_function_getter: MMFunctionGetter,
    mm_enum_mapper: typing.Callable,
    mm_alpha_beta_picker: typing.Callable,
) -> WrappedMMFunction:
    """Match and wrap a matrix multiplication function based on the provided arguments."""
    # NOTE: The parameters of this function are only the operands that will be resettable by
    # reset_operands(); the rest of the parameters should be unchanged by reset_operands.
    # Therefore we can amortize the cost of those bits.
    batchCount = batch_traits[2].count
    if batchCount >= 0 and GeneralMatrixQualifier.is_valid(qualifiers):
        operationA = mm_traits.a_layout_traits.operation
        operationB = mm_traits.b_layout_traits.operation
        m, n, k = mm_traits.M, mm_traits.N, mm_traits.K
        lda, ldb, ldc = mm_traits.a_layout_traits.ld, mm_traits.b_layout_traits.ld, mm_traits.c_layout_traits.ld
        assert mm_traits.c_layout_traits.operation == cublas.Operation.N
        strideA, strideB, strideC = (t.stride for t in batch_traits)

        if mm_traits.is_swapped_AB:
            strideA, strideB = strideB, strideA

        operationA = mm_enum_mapper(operationA)
        operationB = mm_enum_mapper(operationB)

        func = mm_function_getter(
            execution,
            mm_traits.a_layout_traits.dtype,
            GeneralMatrixQualifier.abbreviation,
            logger,
            "stride" if batchCount > 1 else "",
        )

        def wrapped(
            A: tensor_wrapper.TensorHolder,
            B: tensor_wrapper.TensorHolder,
            C: tensor_wrapper.TensorHolder,
            alpha: np.ndarray,
            beta: np.ndarray,
            stream_holder: utils.StreamHolder | None,
        ) -> None:
            if mm_traits.is_swapped_AB:
                A, B = B, A
            logger.debug(
                "Calling %s(operationA=%s, operationB=%s, m=%d, n=%d, k=%d, alpha=%s, lda=%d, strideA=%d, "
                "ldb=%d, strideB=%d, beta=%s, ldc=%d, strideC=%d, batchCount=%d)",
                func.__name__,
                operationA,
                operationB,
                m,
                n,
                k,
                alpha[0],
                lda,
                strideA,
                ldb,
                strideB,
                beta[0],
                ldc,
                strideC,
                batchCount,
            )
            if batchCount > 1:
                func(
                    operationA,
                    operationB,
                    m,
                    n,
                    k,
                    mm_alpha_beta_picker(alpha),
                    A.data_ptr,
                    lda,
                    strideA,
                    B.data_ptr,
                    ldb,
                    strideB,
                    mm_alpha_beta_picker(beta),
                    C.data_ptr,
                    ldc,
                    strideC,
                    batchCount,
                    stream_holder=stream_holder,
                )
            else:
                func(
                    operationA,
                    operationB,
                    m,
                    n,
                    k,
                    mm_alpha_beta_picker(alpha),
                    A.data_ptr,
                    lda,
                    B.data_ptr,
                    ldb,
                    mm_alpha_beta_picker(beta),
                    C.data_ptr,
                    ldc,
                    stream_holder=stream_holder,
                )
    elif (
        batchCount >= 0
        and GeneralMatrixQualifier.is_valid(qualifiers[2])
        and (
            (
                (HermitianMatrixQualifier.is_valid(qualifiers[0]) or SymmetricMatrixQualifier.is_valid(qualifiers[0]))
                and GeneralMatrixQualifier.is_valid(qualifiers[1])
            )
            or (
                (HermitianMatrixQualifier.is_valid(qualifiers[1]) or SymmetricMatrixQualifier.is_valid(qualifiers[1]))
                and GeneralMatrixQualifier.is_valid(qualifiers[0])
            )
        )
    ):
        if HermitianMatrixQualifier.is_valid(qualifiers[0]) or SymmetricMatrixQualifier.is_valid(qualifiers[0]):
            qualifierS = qualifiers[0]
            # qualifierG = qualifiers[1]
            # qualifierC = qualifiers[2]
            traitsS = mm_traits.a_layout_traits
            traitsG = mm_traits.b_layout_traits
            traitsC = mm_traits.c_layout_traits
            strideS = batch_traits[0].stride
            strideG = batch_traits[1].stride
            strideC = batch_traits[2].stride
            is_left_side_symmetric = True
            # Parameter A must always be the symmetric matrix, so if the user provided a
            # SymmetricMatrixQualifier as qualifiers[1], we must move the corresponding
            # matrix to the A input in the BLAS API.
            swapABinputs = False
        elif HermitianMatrixQualifier.is_valid(qualifiers[1]) or SymmetricMatrixQualifier.is_valid(qualifiers[1]):
            # qualifierG = qualifiers[0]
            qualifierS = qualifiers[1]
            # qualifierC = qualifiers[2]
            traitsS = mm_traits.b_layout_traits
            traitsG = mm_traits.a_layout_traits
            traitsC = mm_traits.c_layout_traits
            strideS = batch_traits[1].stride
            strideG = batch_traits[0].stride
            strideC = batch_traits[2].stride
            is_left_side_symmetric = False
            swapABinputs = True
        else:
            raise ValueError("Internal Error: At least one must be a HermitianMatrixQualifier | SymmetricMatrixQualifier!")

        if mm_traits.is_swapped_AB:
            traitsS, traitsG = traitsG, traitsS
            is_left_side_symmetric = not is_left_side_symmetric

        func = mm_function_getter(
            execution,
            traitsS.dtype,
            qualifierS["abbreviation"],
            logger,
            "stride" if batchCount > 1 else "",
        )

        if traitsG.operation != cublas.Operation.N or traitsC.operation != cublas.Operation.N:
            raise ValueError(
                f"Operations on the non-hermitian/non-symmetric operands B,C are not supported for {func.__name__}"
            )

        if traitsS.operation == cublas.Operation.C and SymmetricMatrixQualifier.is_valid(qualifierS):
            raise ValueError(f"Conjugate-Transpose on operand A is not supported for {func.__name__}")

        if traitsS.operation == cublas.Operation.T and HermitianMatrixQualifier.is_valid(qualifierS):
            raise ValueError(f"Transpose on operand A is not supported for {func.__name__}")

        side = cublas.SideMode.LEFT if is_left_side_symmetric else cublas.SideMode.RIGHT
        uplo = cublas.FillMode.LOWER if traitsS.is_lower else cublas.FillMode.UPPER
        m, n = traitsG.shape
        lda, ldb, ldc = traitsS.ld, traitsG.ld, traitsC.ld

        side = mm_enum_mapper(side)
        uplo = mm_enum_mapper(uplo)

        def wrapped(
            A: tensor_wrapper.TensorHolder,
            B: tensor_wrapper.TensorHolder,
            C: tensor_wrapper.TensorHolder,
            alpha: np.ndarray,
            beta: np.ndarray,
            stream_holder: utils.StreamHolder | None,
        ) -> None:
            if swapABinputs:
                A, B = B, A
            logger.debug(
                "Calling %s(side=%s, uplo=%s, m=%d, n=%d, alpha=%s, lda=%d, strideA=%d, "
                "ldb=%d, strideB=%d, beta=%s, ldc=%d, strideC=%d, batchCount=%d)",
                func.__name__,
                side,
                uplo,
                m,
                n,
                alpha[0],
                lda,
                strideS,
                ldb,
                strideG,
                beta[0],
                ldc,
                strideC,
                batchCount,
            )
            if batchCount > 1:
                func(
                    side,
                    uplo,
                    m,
                    n,
                    mm_alpha_beta_picker(alpha),
                    A.data_ptr,
                    lda,
                    strideS,
                    B.data_ptr,
                    ldb,
                    strideG,
                    mm_alpha_beta_picker(beta),
                    C.data_ptr,
                    ldc,
                    strideC,
                    batchCount,
                    stream_holder=stream_holder,
                )
            else:
                func(
                    side,
                    uplo,
                    m,
                    n,
                    mm_alpha_beta_picker(alpha),
                    A.data_ptr,
                    lda,
                    B.data_ptr,
                    ldb,
                    mm_alpha_beta_picker(beta),
                    C.data_ptr,
                    ldc,
                    stream_holder=stream_holder,
                )
    elif (
        batchCount >= 0
        and GeneralMatrixQualifier.is_valid(qualifiers[2])
        and (
            (TriangularMatrixQualifier.is_valid(qualifiers[0]) and GeneralMatrixQualifier.is_valid(qualifiers[1]))
            or (TriangularMatrixQualifier.is_valid(qualifiers[1]) and GeneralMatrixQualifier.is_valid(qualifiers[0]))
        )
    ):
        func = mm_function_getter(
            execution,
            mm_traits.a_layout_traits.dtype,
            TriangularMatrixQualifier.abbreviation,
            logger,
            "stride" if batchCount > 1 else "",
        )
        if TriangularMatrixQualifier.is_valid(qualifiers[0]):
            qualifierT = qualifiers[0]
            # qualifierG = qualifiers[1]
            # qualifierC = qualifiers[2]
            traitsT = mm_traits.a_layout_traits
            traitsG = mm_traits.b_layout_traits
            traitsC = mm_traits.c_layout_traits
            strideT = batch_traits[0].stride
            strideG = batch_traits[1].stride
            strideC = batch_traits[2].stride
            is_left_side_triangle = True
            # Parameter A must always be the triangular matrix, so if the user provided a
            # TriangularMatrixQualifier as qualifiers[1], we must move the corresponding
            # matrix to the A input in the BLAS API.
            swapABinputs = False
        elif TriangularMatrixQualifier.is_valid(qualifiers[1]):
            # qualifierG = qualifiers[0]
            qualifierT = qualifiers[1]
            # qualifierC = qualifiers[2]
            traitsT = mm_traits.b_layout_traits
            traitsG = mm_traits.a_layout_traits
            traitsC = mm_traits.c_layout_traits
            strideT = batch_traits[1].stride
            strideG = batch_traits[0].stride
            strideC = batch_traits[2].stride
            is_left_side_triangle = False
            swapABinputs = True
        else:
            raise ValueError("Internal Error: At least one must be a TriangularMatrixQualifier!")

        if mm_traits.is_swapped_AB:
            traitsT, traitsG = traitsG, traitsT
            is_left_side_triangle = not is_left_side_triangle

        if traitsG.operation != cublas.Operation.N or traitsC.operation != cublas.Operation.N:
            raise ValueError(f"Operations on the non-triangular operands B,C are not supported for {func.__name__}")

        side = cublas.SideMode.LEFT if is_left_side_triangle else cublas.SideMode.RIGHT
        uplo = cublas.FillMode.LOWER if traitsT.is_lower else cublas.FillMode.UPPER
        operation = traitsT.operation
        diag = qualifierT["diag"]
        m, n = traitsG.shape
        lda, ldb, ldc = traitsT.ld, traitsG.ld, traitsC.ld

        if (
            mm_traits.a_layout_traits.dtype == typemaps.cudaDataType.CUDA_C_64F
            and n >= 256
            and m == 1
            and side == cublas.SideMode.RIGHT
        ):
            raise ValueError("This configuration is unsupported for CTK <13.")

        side = mm_enum_mapper(side)
        uplo = mm_enum_mapper(uplo)
        operation = mm_enum_mapper(operation)
        diag = mm_enum_mapper(diag)

        def wrapped(
            A: tensor_wrapper.TensorHolder,
            B: tensor_wrapper.TensorHolder,
            C: tensor_wrapper.TensorHolder,
            alpha: np.ndarray,
            beta: np.ndarray,
            stream_holder: utils.StreamHolder | None,
        ) -> None:
            if swapABinputs:
                A, B = B, A
            logger.debug(
                "Calling %s(side=%s, uplo=%s, operation=%s, diag=%s, m=%d, n=%d, alpha=%s, lda=%d, strideA=%d, "
                "ldb=%d, strideB=%d, ldc=%d, strideC=%d, batchCount=%d)",
                func.__name__,
                side,
                uplo,
                operation,
                diag,
                m,
                n,
                alpha[0],
                lda,
                strideT,
                ldb,
                strideG,
                ldc,
                strideC,
                batchCount,
            )
            if batchCount > 1:
                func(
                    side,
                    uplo,
                    operation,
                    diag,
                    m,
                    n,
                    mm_alpha_beta_picker(alpha),
                    A.data_ptr,
                    lda,
                    strideT,
                    B.data_ptr,
                    ldb,
                    strideG,
                    C.data_ptr,
                    ldc,
                    strideC,
                    batchCount,
                    stream_holder=stream_holder,
                )
            else:
                func(
                    side,
                    uplo,
                    operation,
                    diag,
                    m,
                    n,
                    mm_alpha_beta_picker(alpha),
                    A.data_ptr,
                    lda,
                    B.data_ptr,
                    ldb,
                    C.data_ptr,
                    ldc,
                    stream_holder=stream_holder,
                )
    elif (
        batchCount >= 0
        and GeneralMatrixQualifier.is_valid(qualifiers[2])
        and (
            (DiagonalMatrixQualifier.is_valid(qualifiers[0]) and GeneralMatrixQualifier.is_valid(qualifiers[1]))
            or (DiagonalMatrixQualifier.is_valid(qualifiers[1]) and GeneralMatrixQualifier.is_valid(qualifiers[0]))
        )
    ):
        func = mm_function_getter(
            execution,
            mm_traits.a_layout_traits.dtype,
            DiagonalMatrixQualifier.abbreviation,
            logger,
            "stride" if batchCount > 1 else "",
        )
        if DiagonalMatrixQualifier.is_valid(qualifiers[0]):
            qualifierX = qualifiers[0]
            # qualifierG = qualifiers[1]
            # qualifierC = qualifiers[2]
            traitsX = mm_traits.a_layout_traits
            traitsG = mm_traits.b_layout_traits
            traitsC = mm_traits.c_layout_traits
            strideX = batch_traits[0].stride
            strideG = batch_traits[1].stride
            strideC = batch_traits[2].stride
            is_left_side_diagonal = True
            # Parameter X must always be the diagonal matrix, so if the user provided a
            # DiagonalMatrixQualifier as qualifiers[0], we must move the corresponding
            # matrix to the X input in the BLAS API.
            swapABinputs = True
        elif DiagonalMatrixQualifier.is_valid(qualifiers[1]):
            # qualifierG = qualifiers[0]
            qualifierX = qualifiers[1]
            # qualifierC = qualifiers[2]
            traitsX = mm_traits.b_layout_traits
            traitsG = mm_traits.a_layout_traits
            traitsC = mm_traits.c_layout_traits
            strideX = batch_traits[1].stride
            strideG = batch_traits[0].stride
            strideC = batch_traits[2].stride
            is_left_side_diagonal = False
            swapABinputs = False
        else:
            raise ValueError("Internal Error: At least one must be a DiagonalMatrixQualifier!")

        if mm_traits.is_swapped_AB:
            traitsX, traitsG = traitsG, traitsX
            is_left_side_diagonal = not is_left_side_diagonal

        if traitsG.operation != cublas.Operation.N or traitsC.operation != cublas.Operation.N:
            raise ValueError(f"Operations on the non-diagonal operands A,C are not supported for {func.__name__}")

        # Operation.T is allowed for diagonal matrix because the transpose is a no-op
        if traitsX.operation == cublas.Operation.C:
            raise ValueError(f"Conjugate-Transpose on operand X is not supported for {func.__name__}")

        side = cublas.SideMode.LEFT if is_left_side_diagonal else cublas.SideMode.RIGHT
        m, n = traitsG.shape
        lda, ldc = traitsG.ld, traitsC.ld
        incx = qualifierX["incx"] * max(traitsX.strides)

        side = mm_enum_mapper(side)

        def wrapped(
            A: tensor_wrapper.TensorHolder,
            B: tensor_wrapper.TensorHolder,
            C: tensor_wrapper.TensorHolder,
            alpha: np.ndarray,
            beta: np.ndarray,
            stream_holder: utils.StreamHolder | None,
        ) -> None:
            if swapABinputs:
                A, B = B, A
            logger.debug(
                "Calling %s(side=%s, m=%d, n=%d, lda=%d, strideA=%d, incx=%d, strideX=%d, ldc=%d, strideC=%d, batchCount=%d)",
                func.__name__,
                side,
                m,
                n,
                lda,
                strideG,
                incx,
                strideX,
                ldc,
                strideC,
                batchCount,
            )
            if batchCount > 1:
                func(
                    side,
                    m,
                    n,
                    A.data_ptr,
                    lda,
                    strideG,
                    B.data_ptr,
                    incx,
                    strideX,
                    C.data_ptr,
                    ldc,
                    strideC,
                    batchCount,
                    stream_holder=stream_holder,
                )
            else:
                func(
                    side,
                    m,
                    n,
                    A.data_ptr,
                    lda,
                    B.data_ptr,
                    incx,
                    C.data_ptr,
                    ldc,
                    stream_holder=stream_holder,
                )
    else:
        msg = f"No available generic matrix multiplication matches the provided matrices: {qualifiers}."
        raise ValueError(msg)

    return wrapped
