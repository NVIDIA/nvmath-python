# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from numba import cuda

from nvmath.device import (
    CholeskySolver,
    LeastSquaresSolver,
    LQFactorize,
    LQMultiply,
    LUPivotSolver,
    LUSolver,
    QRFactorize,
    QRMultiply,
    Solver,
    TriangularSolver,
)
from nvmath.device.cusolverdx import _OrthogonalFactorizerProperties, _SolverProperties
from nvmath.device.cusolverdx_backend import ALLOWED_CUSOLVERDX_FUNCTIONS

from .helpers import random_complex, random_real

# ======================================
# Test Data
# ======================================

MINIMAL_KWARGS = {
    "size": (32, 32, 2),
    "precision": np.float64,
    "execution": "Block",
}

ADDITIONAL_REQUIRED_KWARGS = {
    "potrf": {
        "fill_mode": "upper",
    },
    "potrs": {
        "fill_mode": "upper",
    },
    "posv": {
        "fill_mode": "upper",
    },
    "trsm": {
        "fill_mode": "upper",
        "diag": "unit",
        "side": "left",
    },
    "unmqr": {
        "side": "left",
    },
    "unmlq": {
        "side": "left",
    },
    "htev": {"job": "no_vectors"},
    "heev": {"job": "no_vectors", "fill_mode": "upper"},
}

CHOLESKY_BASIC_EXAMPLE_KWARGS = {
    "function": "potrf",
    "size": (32, 32),
    "precision": np.float64,
    "execution": "Block",
    "fill_mode": "upper",
    "leading_dimensions": (33, 32),
}

SIDE_SUPPORTED_FUNCS = ["unmqr", "unmlq", "trsm"]
SIDE_UNSUPPORTED_FUNCS = [f for f in ALLOWED_CUSOLVERDX_FUNCTIONS if f not in SIDE_SUPPORTED_FUNCS]
SIDE_SUPPORTED_VALUES = ["left", "right"]

DIAG_SUPPORTED_FUNCS = ["trsm"]
DIAG_UNSUPPORTED_FUNCS = [f for f in ALLOWED_CUSOLVERDX_FUNCTIONS if f not in DIAG_SUPPORTED_FUNCS]
DIAG_SUPPORTED_VALUES = ["non_unit", "unit"]

FILL_MODE_SUPPORTED_FUNCS = ["potrf", "potrs", "posv", "trsm", "heev"]

FILL_MODE_UNSUPPORTED_FUNCS = [f for f in ALLOWED_CUSOLVERDX_FUNCTIONS if f not in FILL_MODE_SUPPORTED_FUNCS]
FILL_MODE_SUPPORTED_VALUES = ["upper", "lower"]

TRANSPOSE_MODE_FULL_SUPPORT_FUNCS = [
    "getrs_no_pivot",
    "gesv_no_pivot",
    "getrs_partial_pivot",
    "gesv_partial_pivot",
    "unmqr",
    "unmlq",
    "trsm",
    "gels",
]
TRANSPOSE_MODE_FULL_SUPPORT_VALUES = ["non_transposed", "conj_transposed", "transposed"]
TRANSPOSE_MODE_PARTIAL_SUPPORT_FUNCS = ["getrf_no_pivot", "geqrf", "gelqf", "getrf_partial_pivot"]
TRANSPOSE_MODE_PARTIAL_SUPPORT_VALUES = ["non_transposed"]
TRANSPOSE_MODE_PARTIAL_SUPPORT_UNSUPPORTED_VALUES = ["conj_transposed", "transposed"]
TRANSPOSE_MODE_UNSUPPORTED_FUNCS = [
    f
    for f in ALLOWED_CUSOLVERDX_FUNCTIONS
    if f not in (TRANSPOSE_MODE_FULL_SUPPORT_FUNCS + TRANSPOSE_MODE_PARTIAL_SUPPORT_FUNCS)
]

SOLVER_FUNCTION_TO_CLASS = {
    "potrf": CholeskySolver,
    "potrs": CholeskySolver,
    "getrf_no_pivot": LUSolver,
    "getrs_no_pivot": LUSolver,
    "trsm": TriangularSolver,
    "getrf_partial_pivot": LUPivotSolver,
    "getrs_partial_pivot": LUPivotSolver,
    "geqrf": QRFactorize,
    "gelqf": LQFactorize,
    "unmqr": QRMultiply,
    "unmlq": LQMultiply,
    "gels": LeastSquaresSolver,
}

# ======================================
# Data gen functions
# ======================================


def format_solver(solver: Solver | _SolverProperties | _OrthogonalFactorizerProperties) -> str:
    if isinstance(solver, _OrthogonalFactorizerProperties):
        return (
            f"Solver(size=({solver.size}), lda={solver.lda}, "
            + f"arr_a={solver.a_arrangement}, data_type={solver.data_type}, "
            + f"value_type={solver.value_type}, precision={solver.precision}, block_dim={solver.block_dim})"
        )

    return (
        f"Solver(size=({solver.size}), lda={solver.lda}, ldb={solver.ldb}, "
        + f"arr_a={solver.a_arrangement}, arr_b={solver.b_arrangement}, data_type={solver.data_type}, "
        + f"value_type={solver.value_type}, precision={solver.precision}, block_dim={solver.block_dim})"
    )


def verify_relative_error(data_test, data_ref, rel_err: float, abs_error: float, solver: Solver | _SolverProperties) -> bool:
    diff_norm = np.linalg.norm(data_test - data_ref)
    ref_norm = np.linalg.norm(data_ref)

    if diff_norm > ref_norm * rel_err + abs_error:
        solver_msg = format_solver(solver)
        msg = (
            "Failed: | data_test - data_ref | > | data_ref | * rel_err + abs_error with: "
            + f"| data_test - data_ref | = {diff_norm}, | data_ref | = {ref_norm}, "
            + f"| data_ref | * rel_err + abs_error = {ref_norm * rel_err + abs_error}"
            + f" with solver: {solver_msg}"
        )
        raise RuntimeError(msg)


def verify_triangular_relative_error(
    data_test, data_ref, rel_err: float, abs_error: float, solver: _SolverProperties, fill_mode: str = "upper"
) -> bool:
    assert fill_mode in {"upper", "lower"}
    assert data_test.shape == data_ref.shape
    assert data_test.ndim == 2 and data_ref.ndim == 2

    rows, cols = data_test.shape
    if fill_mode == "upper":
        mask = np.triu(np.ones((rows, cols), dtype=bool))
    else:
        mask = np.tril(np.ones((rows, cols), dtype=bool))

    data_test_tri = np.where(mask, data_test, 0)
    data_ref_tri = np.where(mask, data_ref, 0)

    verify_relative_error(
        data_test_tri,
        data_ref_tri,
        rel_err,
        abs_error,
        solver,
    )


def prepare_random_matrix(
    shape: tuple[int, int, int],
    dtype=np.float64,
    is_complex=False,
    is_hermitian=False,
    is_diag_dom=False,
    is_positive_definite=False,
):
    assert len(shape) == 3

    data = random_complex(shape, dtype, "C") if is_complex else random_real(shape, dtype, "C")
    if is_positive_definite:
        is_hermitian = True

    _, m, n = shape

    if is_hermitian:
        hermitian_size = min(m, n)
        row_idx, col_idx = np.triu_indices(hermitian_size, k=1)
        data[:, col_idx, row_idx] = np.conj(data[:, row_idx, col_idx])

        if is_complex:
            diag_idx = np.arange(hermitian_size)
            data[:, diag_idx, diag_idx] = data[:, diag_idx, diag_idx].real

    if is_positive_definite:
        data = data @ np.ascontiguousarray(data.conj().transpose(0, 2, 1))

    if is_diag_dom:
        diag_size = min(m, n)
        diag_idx = np.arange(diag_size)
        row_abs_sum = np.sum(np.abs(data[:, :diag_size, :]), axis=-1)
        data[:, diag_idx, diag_idx] = 2.0 * row_abs_sum + 5.0

    return data


# ======================================
# Device functions
# ======================================


@cuda.jit(device=True, forceinline=True)
def load_to_shared_strided(matrix, smem, shape, strides):
    start = cuda.threadIdx.x
    step = cuda.blockDim.x

    if len(shape) == 2:
        stop = shape[0] * shape[1]

        for index in range(start, stop, step):
            col = index % shape[1]
            row = index // shape[1]

            smem[row * strides[0] + col * strides[1]] = matrix[row, col]
    else:
        stop = shape[0] * shape[1] * shape[2]

        for index in range(start, stop, step):
            col = index % shape[2]
            temp = index // shape[2]
            row = temp % shape[1]
            sample_idx = temp // shape[1]

            smem[sample_idx * strides[0] + row * strides[1] + col * strides[2]] = matrix[sample_idx, row, col]


@cuda.jit(device=True, forceinline=True)
def store_from_shared_strided(smem, matrix, shape, strides):
    start = cuda.threadIdx.x
    step = cuda.blockDim.x

    if len(shape) == 2:
        stop = shape[0] * shape[1]

        for index in range(start, stop, step):
            col = index % shape[1]
            row = index // shape[1]

            matrix[row, col] = smem[row * strides[0] + col * strides[1]]
    else:
        stop = shape[0] * shape[1] * shape[2]

        for index in range(start, stop, step):
            col = index % shape[2]
            temp = index // shape[2]
            row = temp % shape[1]
            sample_idx = temp // shape[1]

            matrix[sample_idx, row, col] = smem[sample_idx * strides[0] + row * strides[1] + col * strides[2]]
