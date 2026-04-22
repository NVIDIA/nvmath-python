# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

#
# Mirrors https://github.com/NVIDIA/CUDALibrarySamples/blob/main/MathDx/cuSolverDx/02_Least_Squares/gels_batched.cu
#

import warnings

import numpy as np
import scipy.linalg
from common import prepare_random_matrix, verify_relative_error
from common_numba import load_to_shared_strided, store_from_shared_strided
from numba import cuda

from nvmath.device import LeastSquaresSolver

warnings.filterwarnings("ignore", module="numba")

np.random.seed(43)

_PREC_TABLE = {
    np.float32: 1e-4,
    np.float64: 1e-6,
}
_ABS_ERR = 1e-8

BLOCKS = 3


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


def main():
    solver = LeastSquaresSolver(
        size=(20, 16, 5),
        precision=np.float64,
        data_type="complex",
        arrangement=("col_major", "row_major"),
        transpose_mode="conj_transposed",
        execution="Block",
        batches_per_block="suggested",
        block_dim="suggested",
    )

    samples_count = BLOCKS * solver.batches_per_block
    n_a = solver.a_size()
    n_bx = solver.bx_size()
    n_tau = solver.tau_size

    print(f"solver.batches_per_block={solver.batches_per_block}")
    print(f"solver.block_dim={solver.block_dim}")
    print(f"samples_count={samples_count}")

    @cuda.jit
    def kernel(a, tau, b):
        smem_a = cuda.shared.array(n_a, dtype=solver.value_type)
        smem_bx = cuda.shared.array(n_bx, dtype=solver.value_type)
        smem_tau = cuda.shared.array(n_tau, dtype=solver.value_type)

        base_sample_idx = cuda.blockIdx.x * solver.batches_per_block

        load_to_shared_strided(a[base_sample_idx:], smem_a, solver.a_shape, solver.a_strides())
        load_to_shared_strided(b[base_sample_idx:], smem_bx, solver.b_shape, solver.bx_strides())
        cuda.syncthreads()

        solver.solve(smem_a, smem_tau, smem_bx)
        cuda.syncthreads()

        store_from_shared_strided(smem_a, a[base_sample_idx:], solver.a_shape, solver.a_strides())
        store_from_shared_strided(smem_bx, b[base_sample_idx:], solver.x_shape, solver.bx_strides())
        store_from_shared_strided(smem_tau, tau[base_sample_idx:], solver.tau_shape, solver.tau_strides)

    max_mn = max(solver.m, solver.n)
    a = prepare_random_matrix(
        (samples_count, solver.m, solver.n),
        dtype=solver.precision,
        is_complex=(solver.data_type == "complex"),
        is_diag_dom=True,
    )
    b = prepare_random_matrix(
        (samples_count, max_mn, solver.k),
        dtype=solver.precision,
        is_complex=(solver.data_type == "complex"),
    )

    a_d = cuda.to_device(a)
    b_d = cuda.to_device(b)
    tau_d = cuda.device_array((samples_count, solver.tau_shape[1]), dtype=a.dtype)

    kernel[BLOCKS, solver.block_dim](a_d, tau_d, b_d)
    cuda.synchronize()

    a_result = a_d.copy_to_host()
    b_result = b_d.copy_to_host()
    tau_result = tau_d.copy_to_host()

    # For conj_transposed, device returns A^H factorized. Logical A for least squares is A^H
    is_transposed = solver.transpose_mode != "non_transposed"
    a_pre_op = np.ascontiguousarray(a.conj().transpose(0, 2, 1)) if is_transposed else a

    rhs = b[:, : solver.b_shape[1], :]
    sol = b_result[:, : solver.x_shape[1], :]
    if is_transposed:
        a_result = np.ascontiguousarray(a_result.conj().transpose(0, 2, 1))

    is_overdetermined = (solver.m >= solver.n and not is_transposed) or (solver.m < solver.n and is_transposed)

    error_sum_a = 0.0
    error_sum_x = 0.0

    for sample_idx in range(samples_count):
        # 1. Check A factorization (QR/LQ) reconstruction
        if is_overdetermined:
            _, a_reconstruct = reconstruct_qr(
                a_result[sample_idx],
                tau_result[sample_idx],
                solver.n if is_transposed else solver.m,
                solver.tau_shape[1],
            )
        else:
            _, a_reconstruct = reconstruct_lq(
                a_result[sample_idx],
                tau_result[sample_idx],
                solver.m if is_transposed else solver.n,
                solver.tau_shape[1],
            )

        error_sum_a += verify_relative_error(
            a_reconstruct, a_pre_op[sample_idx], _PREC_TABLE[solver.precision], _ABS_ERR, solver
        )

        # 2. Compare X with scipy reference
        A_op = a_pre_op[sample_idx]
        B_rhs = rhs[sample_idx]
        x_ref, _, _, _ = scipy.linalg.lstsq(A_op, B_rhs)
        error_sum_x += verify_relative_error(sol[sample_idx], x_ref, _PREC_TABLE[solver.precision], _ABS_ERR, solver)

    print("GELS: relative error of A (factorization) vs reconstructed: ", error_sum_a)
    print("GELS: relative error of X vs scipy.linalg.lstsq: ", error_sum_x)
    print("Successfully validated batched GELS against scipy reference.")


if __name__ == "__main__":
    main()
