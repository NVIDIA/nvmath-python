# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0


from .common_axes import (
    DType,
    DenseRHS,
    ExecutionSpace,
    SparseArrayType,
)
from nvmath.sparse.advanced import DirectSolverMatrixType


supported_dtypes = (DType.float32, DType.float64, DType.complex64, DType.complex128)
supported_index_dtypes = (DType.int32,)

supported_sparse_array_types = (SparseArrayType.CSR,)

supported_exec_space_dense_rhs = {
    ExecutionSpace.cudss_cuda: [DenseRHS.vector, DenseRHS.matrix, DenseRHS.batch],
    ExecutionSpace.cudss_hybrid: [DenseRHS.vector],
}

supported_sparse_type_dtype = {
    DirectSolverMatrixType.GENERAL: [DType.float32, DType.float64, DType.complex64, DType.complex128],
    DirectSolverMatrixType.SYMMETRIC: [DType.float32, DType.float64],
    DirectSolverMatrixType.HERMITIAN: [DType.complex64, DType.complex128],
    DirectSolverMatrixType.SPD: [DType.float32, DType.float64],
    DirectSolverMatrixType.HPD: [DType.complex64, DType.complex128],
}

supported_index_dtype = (DType.int32,)
