# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import itertools
from collections.abc import Sequence

import numpy as np
import pytest
import scipy
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
    TriangularSolver,
)

from .cusolverdx_common import (
    load_to_shared_strided,
    prepare_random_matrix,
    store_from_shared_strided,
    verify_relative_error,
    verify_triangular_relative_error,
)
from .helpers import requires_ctk

pytestmark = requires_ctk((12, 6, 85))  # CTK 12.6 Update 3

np.random.seed(43)

_PREC_TABLE = {
    np.float32: 1e-4,
    np.float64: 1e-6,
}
_ABS_ERR = 1e-8
_BATCHES_PER_BLOCK = 3

# ======================================
# Helpers
# ======================================


def reconstruct_qr(a: np.ndarray, tau: np.ndarray, n_rows: int, tau_size: int) -> np.ndarray:
    r = np.triu(a)
    q = np.identity(n_rows, dtype=a.dtype)

    for i in range(tau_size):
        v = np.zeros((n_rows, 1), dtype=a.dtype)
        v[i, 0] = 1.0
        v[i + 1 :, 0] = a[i + 1 :, i]

        q = q @ (np.identity(n_rows, dtype=a.dtype) - tau[i] * (v @ v.T.conj()))

    return q, q @ r


def reconstruct_lq(a: np.ndarray, tau: np.ndarray, n_cols: int, tau_size: int) -> np.ndarray:
    lower = np.tril(a)
    q = np.identity(n_cols, dtype=a.dtype)

    for i in range(tau_size - 1, -1, -1):
        v = np.zeros((n_cols, 1), dtype=a.dtype)
        v[i, 0] = 1.0
        v[i + 1 :, 0] = a[i, i + 1 :].T.conj()

        q = q @ (np.identity(n_cols, dtype=a.dtype) - tau[i] * (v @ v.T.conj())).T.conj()

    return q, lower @ q


# ======================================
# Load/Store tests
# ======================================


@pytest.mark.parametrize(
    "storage",
    [
        (32, 32, 32, "col_major"),
        (17, 21, 23, "col_major"),
        (8, 8, 9, "row_major"),
        (3, 3, 5, "row_major"),
        (20, 24, 33, "col_major"),
    ],
)
def test_load_store(storage):
    m, n, ld, arr = storage
    batches = 1

    a = prepare_random_matrix(shape=(batches, m, n), dtype=np.float32, is_hermitian=True, is_diag_dom=True)
    a_d = cuda.to_device(a)
    b = np.zeros((batches, m, n), dtype=np.float32)
    b_d = cuda.to_device(b)

    @cuda.jit
    def f(a, b):
        smem = cuda.shared.array(shape=(0,), dtype=np.byte, alignment=16)
        a_shared = smem.view(np.float32)
        if arr == "col_major":
            strides = (m * ld, 1, ld)
        else:
            strides = (n * ld, ld, 1)
        load_to_shared_strided(a, a_shared, (batches, m, n), strides)
        store_from_shared_strided(a_shared, b, (batches, m, n), strides)

    f[1, m * n, 0, m * n * 4 * 3](a_d, b_d)
    cuda.synchronize()
    b_h = b_d.copy_to_host()
    assert np.linalg.norm(b_h - a) == 0.0


# ======================================
# Cholesky
# ======================================


@pytest.mark.parametrize("api", ["runtime_ld", "compiletime_ld"])
@pytest.mark.parametrize("storage", [((5, 5), None), ((5, 5), (7, 9))])
@pytest.mark.parametrize("precision", [np.float64, np.float32])
@pytest.mark.parametrize("arrangement", [("col_major", "col_major"), ["col_major", "row_major"]])
@pytest.mark.parametrize("data_type", ["real", "complex"])
@pytest.mark.parametrize("fill_mode", ["upper", "lower"])
@pytest.mark.parametrize("block_dim", [None, "suggested"])
def test_func_potrf(api, storage, precision, arrangement, data_type, fill_mode, block_dim):
    size, leading_dimensions = storage
    solver = CholeskySolver(
        size=size,
        execution="Block",
        precision=precision,
        arrangement=arrangement,
        leading_dimensions=leading_dimensions,
        data_type=data_type,
        fill_mode=fill_mode,
        batches_per_block=_BATCHES_PER_BLOCK,
        block_dim=block_dim,
    )
    assert solver._factorize.block_dim == solver._solve.block_dim

    n_a = solver.a_size()

    @cuda.jit
    def f(a, info):
        a_shared = cuda.shared.array(n_a, dtype=solver.value_type)

        load_to_shared_strided(a, a_shared, solver.a_shape, solver.a_strides())
        cuda.syncthreads()

        if api == "compiletime_ld":
            solver.factorize(a_shared, info)
        else:
            solver.factorize(a_shared, info, lda=solver.lda)
        cuda.syncthreads()

        store_from_shared_strided(a_shared, a, solver.a_shape, solver.a_strides())

    a = prepare_random_matrix(
        solver.a_shape,
        dtype=solver.precision,
        is_complex=(solver.data_type == "complex"),
        is_diag_dom=True,
        is_positive_definite=True,
    )

    a_d = cuda.to_device(a)
    info_d = cuda.device_array(solver.info_shape, dtype=solver.info_type)

    f[1, solver.block_dim](a_d, info_d)
    cuda.synchronize()

    info_result = info_d.copy_to_host()
    a_result = a_d.copy_to_host()

    for batch in range(solver.batches_per_block):
        assert info_result[batch] == 0
        numpy_result = scipy.linalg.cholesky(a[batch], lower=(solver.fill_mode == "lower"))
        verify_triangular_relative_error(
            a_result[batch], numpy_result, _PREC_TABLE[solver.precision], _ABS_ERR, solver, solver.fill_mode
        )


@pytest.mark.parametrize("api", ["runtime_ld", "compiletime_ld"])
@pytest.mark.parametrize("storage", [((9, 9, 9), None), ((9, 9, 9), (11, 13))])
@pytest.mark.parametrize("precision", [np.float64, np.float32])
@pytest.mark.parametrize("arrangement", [("col_major", "col_major"), ["col_major", "row_major"]])
@pytest.mark.parametrize("data_type", ["real", "complex"])
@pytest.mark.parametrize("fill_mode", ["upper", "lower"])
@pytest.mark.parametrize("block_dim", [None, "suggested"])
def test_func_potrs(api, storage, precision, arrangement, data_type, fill_mode, block_dim):
    size, leading_dimensions = storage
    solver = CholeskySolver(
        size=size,
        execution="Block",
        precision=precision,
        arrangement=arrangement,
        leading_dimensions=leading_dimensions,
        data_type=data_type,
        fill_mode=fill_mode,
        batches_per_block=_BATCHES_PER_BLOCK,
        block_dim=block_dim,
    )
    assert solver._factorize.block_dim == solver._solve.block_dim

    n_a = solver.a_size()
    n_b = solver.b_size()

    @cuda.jit
    def f(a, b):
        smem_a = cuda.shared.array(n_a, dtype=solver.value_type)
        smem_b = cuda.shared.array(n_b, dtype=solver.value_type)

        load_to_shared_strided(a, smem_a, solver.a_shape, solver.a_strides())
        load_to_shared_strided(b, smem_b, solver.b_shape, solver.b_strides())
        cuda.syncthreads()

        if api == "compiletime_ld":
            solver.solve(smem_a, smem_b)
        else:
            solver.solve(smem_a, smem_b, lda=solver.lda, ldb=solver.ldb)
        cuda.syncthreads()

        store_from_shared_strided(smem_b, b, solver.b_shape, solver.b_strides())

    a = prepare_random_matrix(
        solver.a_shape,
        dtype=solver.precision,
        is_complex=(solver.data_type == "complex"),
        is_diag_dom=True,
        is_positive_definite=True,
    )
    b = prepare_random_matrix(
        solver.b_shape,
        dtype=solver.precision,
        is_complex=(solver.data_type == "complex"),
    )

    a_factorized = a.copy()
    for batch in range(solver.batches_per_block):
        a_factorized[batch] = scipy.linalg.cholesky(a[batch], lower=(solver.fill_mode == "lower"))

    a_d = cuda.to_device(a_factorized)
    b_d = cuda.to_device(b)
    f[1, solver.block_dim](a_d, b_d)
    cuda.synchronize()
    b_result = b_d.copy_to_host()

    for batch in range(solver.batches_per_block):
        scipy_sol = scipy.linalg.cho_solve((a_factorized[batch], solver.fill_mode == "lower"), b[batch])
        verify_relative_error(b_result[batch], scipy_sol, _PREC_TABLE[solver.precision], _ABS_ERR, solver)


# ======================================
# LU
# ======================================


@pytest.mark.parametrize("api", ["runtime_ld", "compiletime_ld"])
@pytest.mark.parametrize("size", [(7, 5), (11, 7), (11, 3), (7, 13), (17, 21), (23, 33), (2, 8), (8, 2)])
def test_lu_solver_factorize_only_solve_fail(api, size):
    solver = LUSolver(
        size=size,
        execution="Block",
        precision=np.float32,
    )
    n_a = solver.a_size()
    n_b = solver.b_size()

    @cuda.jit
    def solve(a, b, info):
        smem_a = cuda.shared.array(n_a, dtype=solver.value_type)
        smem_b = cuda.shared.array(n_b, dtype=solver.value_type)

        load_to_shared_strided(a, smem_a, solver.a_shape, solver.a_strides())
        load_to_shared_strided(b, smem_b, solver.b_shape, solver.b_strides())
        cuda.syncthreads()

        solver.factorize(smem_a, info)
        cuda.syncthreads()

        if api == "compiletime_ld":
            solver.solve(smem_a, smem_b)
        else:
            solver.solve(smem_a, smem_b, lda=solver.lda, ldb=solver.ldb)
        cuda.syncthreads()

        store_from_shared_strided(smem_b, b, solver.b_shape, solver.b_strides())

    a = prepare_random_matrix(
        solver.a_shape,
        dtype=solver.precision,
        is_complex=(solver.data_type == "complex"),
        is_diag_dom=True,
    )
    b = prepare_random_matrix(
        solver.b_shape,
        dtype=solver.precision,
        is_complex=(solver.data_type == "complex"),
    )

    a_d = cuda.to_device(a)
    b_d = cuda.to_device(b)
    info_d = cuda.device_array(solver.info_shape, dtype=solver.info_type)

    with pytest.raises(
        RuntimeError,
        match=(
            "Device function: solve is not available with this configuration: Operation is permitted only for square matrices"
        ),
    ):
        solve[1, solver.block_dim](a_d, b_d, info_d)
        cuda.synchronize()


@pytest.mark.parametrize("api", ["runtime_ld", "compiletime_ld"])
@pytest.mark.parametrize("storage", [((8, 12), None), ((8, 12), (13, 13))])
@pytest.mark.parametrize("precision", [np.float64, np.float32])
@pytest.mark.parametrize("arrangement", [("col_major", "col_major"), ["row_major", "row_major"]])
@pytest.mark.parametrize("data_type", ["real", "complex"])
@pytest.mark.parametrize("block_dim", [None, "suggested"])
def test_lu_solver_factorize(api, storage, precision, arrangement, data_type, block_dim):
    size, leading_dimensions = storage

    solver = LUSolver(
        execution="Block",
        size=size,
        precision=precision,
        arrangement=arrangement,
        leading_dimensions=leading_dimensions,
        data_type=data_type,
        batches_per_block=_BATCHES_PER_BLOCK,
        transpose_mode="non_transposed",
        block_dim=block_dim,
    )
    n_a = solver.a_size()

    a = prepare_random_matrix(
        solver.a_shape,
        dtype=solver.precision,
        is_diag_dom=True,
        is_complex=(solver.data_type == "complex"),
    )

    @cuda.jit
    def factorize(a, info):
        a_shared = cuda.shared.array(n_a, dtype=solver.value_type)

        load_to_shared_strided(a, a_shared, solver.a_shape, solver.a_strides())
        cuda.syncthreads()

        if api == "compiletime_ld":
            solver.factorize(a_shared, info)
        else:
            solver.factorize(a_shared, info, lda=solver.lda)
        cuda.syncthreads()

        store_from_shared_strided(a_shared, a, solver.a_shape, solver.a_strides())

    a_d = cuda.to_device(a)
    info_d = cuda.device_array(solver.info_shape, dtype=solver.info_type)

    factorize[1, solver.block_dim](a_d, info_d)
    cuda.synchronize()

    info_result = info_d.copy_to_host()
    a_result = a_d.copy_to_host()

    lower = np.tril(a_result)[:, :, : solver.m]
    u = np.triu(a_result)[:, : solver.n, :]
    idx = np.arange(min(solver.m, solver.n))
    lower[:, idx, idx] = 1
    a_recreated = lower @ u

    for batch in range(solver.batches_per_block):
        assert info_result[batch] == 0
        verify_relative_error(a_recreated[batch], a[batch], _PREC_TABLE[solver.precision], _ABS_ERR, solver)


@pytest.mark.parametrize("api", ["runtime_ld", "compiletime_ld"])
@pytest.mark.parametrize("storage", [((11, 11, 7), None), ((11, 11, 7), (13, 13))])
@pytest.mark.parametrize("precision", [np.float64, np.float32])
@pytest.mark.parametrize("arrangement", [("col_major", "col_major"), ["col_major", "row_major"]])
@pytest.mark.parametrize("data_type", ["real", "complex"])
@pytest.mark.parametrize("transpose_mode", ["non_transposed", "transposed"])
@pytest.mark.parametrize("block_dim", [None, "suggested"])
def test_lu_solver(api, storage, precision, arrangement, data_type, transpose_mode, block_dim):
    size, leading_dimensions = storage

    if transpose_mode == "transposed" and data_type == "complex":
        transpose_mode = "conj_transposed"

    solver = LUSolver(
        size=size,
        execution="Block",
        precision=precision,
        arrangement=arrangement,
        leading_dimensions=leading_dimensions,
        data_type=data_type,
        transpose_mode=transpose_mode,
        batches_per_block=_BATCHES_PER_BLOCK,
        block_dim=block_dim,
    )
    assert solver._factorize.block_dim == solver._solve.block_dim

    n_a = solver.a_size()
    n_b = solver.b_size()

    @cuda.jit
    def factorize(a, info):
        a_shared = cuda.shared.array(n_a, dtype=solver.value_type)

        load_to_shared_strided(a, a_shared, solver.a_shape, solver.a_strides())
        cuda.syncthreads()

        if api == "compiletime_ld":
            solver.factorize(a_shared, info)
        else:
            solver.factorize(a_shared, info, lda=solver.lda)
        cuda.syncthreads()

        store_from_shared_strided(a_shared, a, solver.a_shape, solver.a_strides())

    @cuda.jit
    def solve(a, b):
        smem_a = cuda.shared.array(n_a, dtype=solver.value_type)
        smem_b = cuda.shared.array(n_b, dtype=solver.value_type)

        load_to_shared_strided(a, smem_a, solver.a_shape, solver.a_strides())
        load_to_shared_strided(b, smem_b, solver.b_shape, solver.b_strides())
        cuda.syncthreads()

        if api == "compiletime_ld":
            solver.solve(smem_a, smem_b)
        else:
            solver.solve(smem_a, smem_b, lda=solver.lda, ldb=solver.ldb)
        cuda.syncthreads()

        store_from_shared_strided(smem_b, b, solver.b_shape, solver.b_strides())

    a = prepare_random_matrix(
        solver.a_shape,
        dtype=solver.precision,
        is_complex=(solver.data_type == "complex"),
        is_diag_dom=True,
    )
    b = prepare_random_matrix(
        solver.b_shape,
        dtype=solver.precision,
        is_complex=(solver.data_type == "complex"),
    )

    if transpose_mode != "non_transposed":
        a_transposed = np.ascontiguousarray(a.conj().transpose(0, 2, 1))
    else:
        a_transposed = a.copy()

    a_d = cuda.to_device(a_transposed)
    info_d = cuda.device_array(solver.info_shape, dtype=solver.info_type)

    factorize[1, solver.block_dim](a_d, info_d)
    cuda.synchronize()

    info_result = info_d.copy_to_host()
    a_result = a_d.copy_to_host()

    lower = np.tril(a_result)
    u = np.triu(a_result)
    idx = np.arange(min(solver.m, solver.n))
    lower[:, idx, idx] = 1
    a_recreated = lower @ u

    if transpose_mode != "non_transposed":
        a_recreated = np.ascontiguousarray(a_recreated.conj().transpose(0, 2, 1))

    for batch in range(solver.batches_per_block):
        assert info_result[batch] == 0
        verify_relative_error(a_recreated[batch], a[batch], _PREC_TABLE[solver.precision], _ABS_ERR, solver)

    b_d = cuda.to_device(b)
    solve[1, solver.block_dim](a_d, b_d)
    cuda.synchronize()
    b_result = b_d.copy_to_host()

    for batch in range(solver.batches_per_block):
        b_recreated = np.matmul(a[batch], b_result[batch])
        verify_relative_error(b_recreated, b[batch], _PREC_TABLE[solver.precision], _ABS_ERR, solver)


# ======================================
# Triangular
# ======================================


@pytest.mark.parametrize("api", ["runtime_ld", "compiletime_ld"])
@pytest.mark.parametrize("storage", [((17, 19), None), ((7, 9), (11, 13))])
@pytest.mark.parametrize("precision", [np.float64, np.float32])
@pytest.mark.parametrize("arrangement", [("col_major", "col_major"), ["col_major", "row_major"]])
@pytest.mark.parametrize("data_type", ["real", "complex"])
@pytest.mark.parametrize("fill_mode", ["upper", "lower"])
@pytest.mark.parametrize("transpose_mode", ["non_transposed", "transposed"])
@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("diag", ["non_unit", "unit"])
@pytest.mark.parametrize("block_dim", [None, "suggested"])
def test_triangular_solver(api, storage, precision, arrangement, data_type, fill_mode, transpose_mode, side, diag, block_dim):
    size, leading_dimensions = storage

    if transpose_mode == "transposed" and data_type == "complex":
        transpose_mode = "conj_transposed"

    solver = TriangularSolver(
        size=size,
        execution="Block",
        precision=precision,
        arrangement=arrangement,
        leading_dimensions=leading_dimensions,
        data_type=data_type,
        batches_per_block=_BATCHES_PER_BLOCK,
        side=side,
        fill_mode=fill_mode,
        diag=diag,
        transpose_mode=transpose_mode,
        block_dim=block_dim,
    )

    n_a = solver.a_size()
    n_b = solver.b_size()

    @cuda.jit
    def f(a, b):
        smem_a = cuda.shared.array(n_a, dtype=solver.value_type)
        smem_b = cuda.shared.array(n_b, dtype=solver.value_type)

        load_to_shared_strided(a, smem_a, solver.a_shape, solver.a_strides())
        load_to_shared_strided(b, smem_b, solver.b_shape, solver.b_strides())
        cuda.syncthreads()

        if api == "compiletime_ld":
            solver.solve(smem_a, smem_b)
        else:
            solver.solve(smem_a, smem_b, lda=solver.lda, ldb=solver.ldb)
        cuda.syncthreads()

        store_from_shared_strided(smem_b, b, solver.b_shape, solver.b_strides())

    a = prepare_random_matrix(
        solver.a_shape,
        dtype=solver.precision,
        is_complex=(solver.data_type == "complex"),
    )
    b = prepare_random_matrix(
        solver.b_shape,
        dtype=solver.precision,
        is_complex=(solver.data_type == "complex"),
    )

    a_d = cuda.to_device(a)
    b_d = cuda.to_device(b)
    f[1, solver.block_dim](a_d, b_d)

    cuda.synchronize()
    b_result = b_d.copy_to_host()

    scipy_trans = 0 if solver.transpose_mode == "non_transposed" else 1 if solver.transpose_mode == "transposed" else 2
    for batch in range(solver.batches_per_block):
        if side == "left":
            scipy_sol = scipy.linalg.solve_triangular(
                a[batch],
                b[batch],
                lower=(solver.fill_mode == "lower"),
                trans=scipy_trans,
                unit_diagonal=(solver.diag == "unit"),
            )
        else:
            # x @ A = B
            # <=>
            # A^T @ x^T = B^T
            scipy_sol = scipy.linalg.solve_triangular(
                a[batch].T,
                b[batch].T,
                lower=(solver.fill_mode == "upper"),  # Note: A is transposed!
                trans=scipy_trans,
                unit_diagonal=(solver.diag == "unit"),
            ).T

        verify_relative_error(b_result[batch], scipy_sol, _PREC_TABLE[solver.precision], _ABS_ERR, solver)


# ======================================
# LU with partial pivoting
# ======================================


@pytest.mark.parametrize("api", ["runtime_ld", "compiletime_ld"])
@pytest.mark.parametrize("size", [(7, 5), (11, 7), (11, 3), (7, 13), (17, 21), (23, 33), (2, 8), (8, 2)])
def test_lu_partial_pivot_solver_factorize_only_solve_fail(api, size):
    solver = LUPivotSolver(
        size=size,
        execution="Block",
        precision=np.float32,
    )
    n_a = solver.a_size()
    n_b = solver.b_size()

    @cuda.jit
    def solve(a, b, ipiv, info):
        smem_a = cuda.shared.array(n_a, dtype=solver.value_type)
        smem_b = cuda.shared.array(n_b, dtype=solver.value_type)

        load_to_shared_strided(a, smem_a, solver.a_shape, solver.a_strides())
        load_to_shared_strided(b, smem_b, solver.b_shape, solver.b_strides())
        cuda.syncthreads()

        solver.factorize(smem_a, ipiv, info)
        cuda.syncthreads()

        if api == "compiletime_ld":
            solver.solve(smem_a, ipiv, smem_b)
        else:
            solver.solve(smem_a, ipiv, smem_b, lda=solver.lda, ldb=solver.ldb)
        cuda.syncthreads()

        store_from_shared_strided(smem_b, b, solver.b_shape, solver.b_strides())

    a = prepare_random_matrix(
        solver.a_shape,
        dtype=solver.precision,
        is_complex=(solver.data_type == "complex"),
        is_diag_dom=True,
    )
    b = prepare_random_matrix(
        solver.b_shape,
        dtype=solver.precision,
        is_complex=(solver.data_type == "complex"),
    )

    a_d = cuda.to_device(a)
    b_d = cuda.to_device(b)
    ipiv_d = cuda.device_array(solver.ipiv_shape, dtype=solver.ipiv_type)
    info_d = cuda.device_array(solver.info_shape, dtype=solver.info_type)

    with pytest.raises(
        RuntimeError,
        match=(
            "Device function: solve is not available with this configuration: Operation is permitted only for square matrices"
        ),
    ):
        solve[1, solver.block_dim](a_d, b_d, ipiv_d, info_d)
        cuda.synchronize()


@pytest.mark.parametrize("api", ["runtime_ld", "compiletime_ld"])
@pytest.mark.parametrize("storage", [((3, 5), None), ((3, 5), (7, 7))])
@pytest.mark.parametrize("precision", [np.float64, np.float32])
@pytest.mark.parametrize("arrangement", [("col_major", "col_major"), ["row_major", "row_major"]])
@pytest.mark.parametrize("data_type", ["real", "complex"])
@pytest.mark.parametrize("block_dim", [None, "suggested"])
def test_lu_pivot_solver_factorize(api, storage, precision, arrangement, data_type, block_dim):
    size, leading_dimensions = storage

    solver = LUPivotSolver(
        execution="Block",
        size=size,
        precision=precision,
        arrangement=arrangement,
        leading_dimensions=leading_dimensions,
        data_type=data_type,
        batches_per_block=_BATCHES_PER_BLOCK,
        transpose_mode="non_transposed",
        block_dim=block_dim,
    )
    n_a = solver.a_size()

    a = prepare_random_matrix(
        solver.a_shape,
        dtype=solver.precision,
        is_complex=(solver.data_type == "complex"),
    )

    @cuda.jit
    def factorize(a, info, ipiv):
        a_shared = cuda.shared.array(n_a, dtype=solver.value_type)

        load_to_shared_strided(a, a_shared, solver.a_shape, solver.a_strides())
        cuda.syncthreads()

        if api == "compiletime_ld":
            solver.factorize(a_shared, ipiv, info)
        else:
            solver.factorize(a_shared, ipiv, info, lda=solver.lda)
        cuda.syncthreads()

        store_from_shared_strided(a_shared, a, solver.a_shape, solver.a_strides())

    a_d = cuda.to_device(a)
    info_d = cuda.device_array(solver.info_shape, dtype=solver.info_type)
    ipiv_d = cuda.device_array(solver.ipiv_shape, dtype=solver.ipiv_type)

    factorize[1, solver.block_dim](a_d, info_d, ipiv_d)
    cuda.synchronize()

    info_result = info_d.copy_to_host()
    a_result = a_d.copy_to_host()
    ipiv_result = ipiv_d.copy_to_host()

    lower = np.tril(a_result)[:, :, : solver.m]
    u = np.triu(a_result)[:, : solver.n, :]
    idx = np.arange(min(solver.m, solver.n))
    lower[:, idx, idx] = 1
    a_recreated = lower @ u

    for batch in range(solver.batches_per_block):
        assert info_result[batch] == 0

        for i in range(min(solver.m, solver.n) - 1, -1, -1):
            pivot = ipiv_result[batch, i] - 1

            if pivot != i:
                temp = a_recreated[batch, i, :].copy()
                a_recreated[batch, i, :] = a_recreated[batch, pivot, :]
                a_recreated[batch, pivot, :] = temp

        verify_relative_error(a_recreated[batch], a[batch], _PREC_TABLE[solver.precision], _ABS_ERR, solver)


@pytest.mark.parametrize("api", ["runtime_ld", "compiletime_ld"])
@pytest.mark.parametrize("storage", [((7, 7), None), ((7, 7, 3), (11, 11))])
@pytest.mark.parametrize("precision", [np.float64, np.float32])
@pytest.mark.parametrize("arrangement", [("col_major", "col_major"), ["row_major", "row_major"]])
@pytest.mark.parametrize("data_type", ["real", "complex"])
@pytest.mark.parametrize("transpose_mode", ["non_transposed", "transposed"])
@pytest.mark.parametrize("block_dim", [None, "suggested"])
def test_lu_pivot_solver(api, storage, precision, arrangement, data_type, transpose_mode, block_dim):
    size, leading_dimensions = storage

    if transpose_mode == "transposed" and data_type == "complex":
        transpose_mode = "conj_transposed"

    solver = LUPivotSolver(
        execution="Block",
        size=size,
        precision=precision,
        arrangement=arrangement,
        leading_dimensions=leading_dimensions,
        data_type=data_type,
        batches_per_block=_BATCHES_PER_BLOCK,
        transpose_mode=transpose_mode,
        block_dim=block_dim,
    )
    assert solver._factorize.block_dim == solver._solve.block_dim

    assert solver.m == solver.n
    n_a = solver.a_size()
    n_b = solver.b_size()

    @cuda.jit
    def factorize(a, info, ipiv):
        a_shared = cuda.shared.array(n_a, dtype=solver.value_type)

        load_to_shared_strided(a, a_shared, solver.a_shape, solver.a_strides())
        cuda.syncthreads()

        if api == "compiletime_ld":
            solver.factorize(a_shared, ipiv, info)
        else:
            solver.factorize(a_shared, ipiv, info, lda=solver.lda)
        cuda.syncthreads()

        store_from_shared_strided(a_shared, a, solver.a_shape, solver.a_strides())

    @cuda.jit
    def solve(a, b, ipiv):
        smem_a = cuda.shared.array(n_a, dtype=solver.value_type)
        smem_b = cuda.shared.array(n_b, dtype=solver.value_type)

        load_to_shared_strided(a, smem_a, solver.a_shape, solver.a_strides())
        load_to_shared_strided(b, smem_b, solver.b_shape, solver.b_strides())
        cuda.syncthreads()

        if api == "compiletime_ld":
            solver.solve(smem_a, ipiv, smem_b)
        else:
            solver.solve(smem_a, ipiv, smem_b, lda=solver.lda, ldb=solver.ldb)
        cuda.syncthreads()

        store_from_shared_strided(smem_b, b, solver.b_shape, solver.b_strides())

    a = prepare_random_matrix(
        solver.a_shape,
        dtype=solver.precision,
        is_complex=(solver.data_type == "complex"),
        is_diag_dom=True,
    )
    b = prepare_random_matrix(
        solver.b_shape,
        dtype=solver.precision,
        is_complex=(solver.data_type == "complex"),
    )

    if transpose_mode != "non_transposed":
        a_transposed = np.ascontiguousarray(a.conj().transpose(0, 2, 1))
    else:
        a_transposed = a.copy()

    a_d = cuda.to_device(a_transposed)
    info_d = cuda.device_array(solver.info_shape, dtype=solver.info_type)
    ipiv_d = cuda.device_array(solver.ipiv_shape, dtype=solver.ipiv_type)

    factorize[1, solver.block_dim](a_d, info_d, ipiv_d)
    cuda.synchronize()

    info_result = info_d.copy_to_host()
    a_result = a_d.copy_to_host()
    ipiv_result = ipiv_d.copy_to_host()

    lower = np.tril(a_result)
    u = np.triu(a_result)
    idx = np.arange(solver.m)
    lower[:, idx, idx] = 1
    a_recreated = lower @ u

    for batch in range(solver.batches_per_block):
        assert info_result[batch] == 0

        for i in range(solver.m - 1, -1, -1):
            pivot = ipiv_result[batch, i] - 1

            if pivot != i:
                temp = a_recreated[batch, i, :].copy()
                a_recreated[batch, i, :] = a_recreated[batch, pivot, :]
                a_recreated[batch, pivot, :] = temp

    if transpose_mode != "non_transposed":
        a_recreated = np.ascontiguousarray(a_recreated.conj().transpose(0, 2, 1))

    for batch in range(solver.batches_per_block):
        verify_relative_error(a_recreated[batch], a[batch], _PREC_TABLE[solver.precision], _ABS_ERR, solver)

    b_d = cuda.to_device(b)
    solve[1, solver.block_dim](a_d, b_d, ipiv_d)
    cuda.synchronize()
    b_result = b_d.copy_to_host()

    for batch in range(solver.batches_per_block):
        b_recreated = np.matmul(a[batch], b_result[batch])
        verify_relative_error(b_recreated, b[batch], _PREC_TABLE[solver.precision], _ABS_ERR, solver)


# ======================================
# Ortoghonal Factorizer Tests
# ======================================


def get_size_multiplty_qrlq(size: Sequence[int], operation: str, side: str) -> tuple[int, int, int]:
    m, n, k = size

    if operation == "qr":
        size_multiply = (m, k, n) if side == "left" else (k, m, n)
    elif operation == "lq":
        size_multiply = (n, k, m) if side == "left" else (k, n, m)

    return size_multiply


def is_proper_spec(spec) -> bool:
    size, leading_dimensions, operation, side = spec
    size_multiply = get_size_multiplty_qrlq(size, operation, side)

    return (
        not (leading_dimensions is not None and max(size) > min(leading_dimensions))
        and not (side == "left" and size_multiply[2] > size_multiply[0])
        and not (side == "right" and size_multiply[2] > size_multiply[1])
    )


@pytest.mark.parametrize("api", ["runtime_ld", "compiletime_ld"])
@pytest.mark.parametrize(
    "spec",
    [
        v
        for v in itertools.product([(14, 13, 19), (21, 13, 11)], [None, (23, 23), (24, 24)], ["qr", "lq"], ["left", "right"])
        if is_proper_spec(v)
    ],
)
@pytest.mark.parametrize("precision", [np.float64, np.float32])
@pytest.mark.parametrize("arrangement", [("col_major", "col_major"), ["row_major", "row_major"]])
@pytest.mark.parametrize("data_type", ["real", "complex"])
@pytest.mark.parametrize("transpose_mode", ["non_transposed", "transposed"])
def test_qrlq(api, spec, precision, arrangement, data_type, transpose_mode):
    size, leading_dimensions, operation, side = spec

    size_multiply = get_size_multiplty_qrlq(size, operation, side)
    if transpose_mode == "transposed" and data_type == "complex":
        transpose_mode = "conj_transposed"

    factorizer_class = QRFactorize if operation == "qr" else LQFactorize
    factorizer = factorizer_class(
        execution="Block",
        size=size,
        precision=precision,
        arrangement=arrangement[0] if arrangement is not None else None,
        leading_dimension=leading_dimensions[0] if leading_dimensions is not None else None,
        data_type=data_type,
        batches_per_block=_BATCHES_PER_BLOCK,
    )

    multiplier_class = QRMultiply if operation == "qr" else LQMultiply
    multiplier = multiplier_class(
        size=size_multiply,
        execution="Block",
        precision=precision,
        arrangement=arrangement,
        leading_dimensions=leading_dimensions,
        data_type=data_type,
        batches_per_block=_BATCHES_PER_BLOCK,
        transpose_mode=transpose_mode,
        side=side,
        block_dim=factorizer.block_dim,
    )

    n_a = factorizer.a_size()
    n_c = multiplier.c_size()
    n_tau = factorizer.tau_size

    a = prepare_random_matrix(
        factorizer.a_shape,
        dtype=factorizer.precision,
        is_complex=(factorizer.data_type == "complex"),
    )

    c = prepare_random_matrix(
        multiplier.c_shape,
        dtype=factorizer.precision,
        is_complex=(factorizer.data_type == "complex"),
    )

    @cuda.jit
    def kernel(a, c, tau):
        smem_a = cuda.shared.array(n_a, dtype=factorizer.value_type)
        smem_c = cuda.shared.array(n_c, dtype=factorizer.value_type)
        smem_tau = cuda.shared.array(n_tau, dtype=factorizer.value_type)

        load_to_shared_strided(a, smem_a, factorizer.a_shape, factorizer.a_strides())
        load_to_shared_strided(c, smem_c, multiplier.c_shape, multiplier.c_strides())
        cuda.syncthreads()

        if api == "compiletime_ld":
            factorizer.factorize(smem_a, smem_tau)
        elif api == "runtime_ld":
            factorizer.factorize(smem_a, smem_tau, lda=factorizer.lda)
        cuda.syncthreads()

        store_from_shared_strided(smem_a, a, factorizer.a_shape, factorizer.a_strides())
        store_from_shared_strided(smem_tau, tau, factorizer.tau_shape, factorizer.tau_strides)
        cuda.syncthreads()

        if api == "compiletime_ld":
            multiplier.multiply(smem_a, smem_tau, smem_c)
        else:
            multiplier.multiply(smem_a, smem_tau, smem_c, lda=multiplier.lda, ldc=multiplier.ldb)
        cuda.syncthreads()

        store_from_shared_strided(smem_c, c, multiplier.c_shape, multiplier.c_strides())

    a_d = cuda.to_device(a)
    c_d = cuda.to_device(c)
    tau_d = cuda.device_array(factorizer.tau_shape, dtype=a.dtype)

    kernel[1, factorizer.block_dim](a_d, c_d, tau_d)
    cuda.synchronize()

    a_result = a_d.copy_to_host()
    c_result = c_d.copy_to_host()
    tau_result = tau_d.copy_to_host()

    for batch in range(factorizer.batches_per_block):
        assert operation in {"qr", "lq"}

        if operation == "qr":
            q, a_recreated = reconstruct_qr(a_result[batch], tau_result[batch], factorizer.m, factorizer.tau_shape[1])
        else:
            q, a_recreated = reconstruct_lq(a_result[batch], tau_result[batch], factorizer.n, factorizer.tau_shape[1])

        verify_relative_error(a_recreated, a[batch], _PREC_TABLE[factorizer.precision], _ABS_ERR, factorizer)

        if transpose_mode != "non_transposed":
            q = q.T.conj()
        host_c_result = q @ c[batch] if side == "left" else c[batch] @ q

        verify_relative_error(c_result[batch], host_c_result, _PREC_TABLE[factorizer.precision], _ABS_ERR, multiplier)


# ======================================
# Least squares (gels)
# ======================================


@pytest.mark.parametrize("api", ["runtime_ld", "compiletime_ld"])
@pytest.mark.parametrize(
    "storage",
    [
        ((17, 11, 5), (18, 18)),
        ((11, 17, 5), None),
    ],
)
@pytest.mark.parametrize("precision", [np.float64, np.float32])
@pytest.mark.parametrize("arrangement", [("col_major", "col_major"), ["row_major", "row_major"]])
@pytest.mark.parametrize("data_type", ["real", "complex"])
@pytest.mark.parametrize("transpose_mode", ["non_transposed", "transposed"])
@pytest.mark.parametrize("block_dim", [None, "suggested"])
def test_func_gels(api, storage, precision, arrangement, data_type, transpose_mode, block_dim):
    size, leading_dimensions = storage

    if transpose_mode == "transposed" and data_type == "complex":
        transpose_mode = "conj_transposed"

    solver = LeastSquaresSolver(
        size=size,
        execution="Block",
        precision=precision,
        arrangement=arrangement,
        leading_dimensions=leading_dimensions,
        data_type=data_type,
        transpose_mode=transpose_mode,
        batches_per_block=_BATCHES_PER_BLOCK,
        block_dim=block_dim,
    )

    n_a = solver.a_size()
    n_bx = solver.bx_size()
    n_tau = solver.tau_size

    is_transposed = transpose_mode != "non_transposed"
    is_overdetermined = (solver.m >= solver.n and not is_transposed) or (solver.m < solver.n and is_transposed)

    a = prepare_random_matrix(
        solver.a_shape,
        dtype=precision,
        is_complex=(data_type == "complex"),
        is_diag_dom=True,
    )

    a_pre_op = np.ascontiguousarray(a.conj().transpose(0, 2, 1)) if is_transposed else a

    b = prepare_random_matrix(
        (solver.batches_per_block, max(solver.m, solver.n), solver.k),
        dtype=precision,
        is_complex=(data_type == "complex"),
    )

    @cuda.jit
    def kernel(a, tau, b):
        smem_a = cuda.shared.array(n_a, dtype=solver.value_type)
        smem_bx = cuda.shared.array(n_bx, dtype=solver.value_type)
        smem_tau = cuda.shared.array(n_tau, dtype=solver.value_type)

        load_to_shared_strided(a, smem_a, solver.a_shape, solver.a_strides())
        load_to_shared_strided(b, smem_bx, solver.b_shape, solver.bx_strides())
        cuda.syncthreads()

        if api == "compiletime_ld":
            solver.solve(smem_a, smem_tau, smem_bx)
        else:
            solver.solve(smem_a, smem_tau, smem_bx, lda=solver.lda, ldb=solver.ldb)
        cuda.syncthreads()

        store_from_shared_strided(smem_a, a, solver.a_shape, solver.a_strides())
        store_from_shared_strided(smem_bx, b, solver.x_shape, solver.bx_strides())
        store_from_shared_strided(smem_tau, tau, solver.tau_shape, solver.tau_strides)

    a_d = cuda.to_device(a)
    b_d = cuda.to_device(b)
    tau_d = cuda.device_array(solver.tau_shape, dtype=a.dtype)

    kernel[1, solver.block_dim](a_d, tau_d, b_d)
    cuda.synchronize()

    a_result = a_d.copy_to_host()
    b_result = b_d.copy_to_host()
    tau_result = tau_d.copy_to_host()

    rhs = b[:, : solver.b_shape[1], :]
    sol = b_result[:, : solver.x_shape[1], :]
    if is_transposed:
        a_result = np.ascontiguousarray(a_result.conj().transpose(0, 2, 1))

    for sample_idx in range(solver.batches_per_block):
        if is_overdetermined:
            _, a_reconstruct = reconstruct_qr(
                a_result[sample_idx], tau_result[sample_idx], solver.n if is_transposed else solver.m, solver.tau_shape[1]
            )
        else:
            _, a_reconstruct = reconstruct_lq(
                a_result[sample_idx], tau_result[sample_idx], solver.m if is_transposed else solver.n, solver.tau_shape[1]
            )
        verify_relative_error(a_reconstruct, a_pre_op[sample_idx], _PREC_TABLE[precision], _ABS_ERR, solver)

        if is_overdetermined:
            normal_eq_residual = a_pre_op[sample_idx].T.conj() @ ((a_pre_op[sample_idx] @ sol[sample_idx]) - rhs[sample_idx])
            verify_relative_error(
                normal_eq_residual,
                np.zeros_like(normal_eq_residual),
                _PREC_TABLE[precision],
                5e-4,
                solver,
            )
        else:
            verify_relative_error(
                rhs[sample_idx],
                a_pre_op[sample_idx] @ sol[sample_idx],
                _PREC_TABLE[precision],
                _ABS_ERR,
                solver,
            )
