# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

#
# Mirrors https://github.com/NVIDIA/CUDALibrarySamples/blob/main/MathDx/cuSolverDx/01_Linear_Solve/gesv_batched_partial_pivot.cu
#

import warnings

import numpy as np
import scipy
from common import prepare_random_matrix, verify_relative_error
from common_numba import load_to_shared_strided, store_from_shared_strided
from numba import cuda

from nvmath.device import LUPivotSolver

warnings.filterwarnings("ignore", module="numba")

np.random.seed(43)

_PREC_TABLE = {
    np.float32: 1e-4,
    np.float64: 1e-6,
}
_ABS_ERR = 1e-8

BLOCKS = 512


def main():
    solver = LUPivotSolver(
        size=(3, 3),
        precision=np.float64,
        data_type="complex",
        arrangement=("col_major", "col_major"),
        transpose_mode="non_transposed",
        leading_dimensions=(4, 4),
        execution="Block",
        batches_per_block="suggested",
        block_dim=(64,),
    )

    n_a = solver.a_size()
    n_b = solver.b_size()
    n_ipiv = solver.ipiv_size
    batch_count = BLOCKS * solver.batches_per_block

    print(f"solver.batches_per_block={solver.batches_per_block}")
    print(f"solver.block_dim={solver.block_dim}")
    print(f"batch_count={batch_count}")

    @cuda.jit
    def kernel(a, b, info, ipiv):
        smem_a = cuda.shared.array(n_a, dtype=solver.value_type)
        smem_b = cuda.shared.array(n_b, dtype=solver.value_type)
        smem_ipiv = cuda.shared.array(n_ipiv, dtype=np.int32)

        base_sample_idx = cuda.blockIdx.x * solver.batches_per_block

        load_to_shared_strided(a[base_sample_idx:], smem_a, solver.a_shape, solver.a_strides())
        load_to_shared_strided(b[base_sample_idx:], smem_b, solver.b_shape, solver.b_strides())
        cuda.syncthreads()

        solver.factorize(smem_a, smem_ipiv, info[base_sample_idx:])
        cuda.syncthreads()
        solver.solve(smem_a, smem_ipiv, smem_b)
        cuda.syncthreads()

        store_from_shared_strided(smem_a, a[base_sample_idx:], solver.a_shape, solver.a_strides())
        store_from_shared_strided(smem_b, b[base_sample_idx:], solver.b_shape, solver.b_strides())
        store_from_shared_strided(smem_ipiv, ipiv[base_sample_idx:], solver.ipiv_shape, solver.ipiv_strides)

    a = prepare_random_matrix(
        (batch_count, solver.a_shape[1], solver.a_shape[2]),
        dtype=solver.precision,
        is_complex=(solver.data_type == "complex"),
    )
    b = prepare_random_matrix(
        (batch_count, solver.b_shape[1], solver.b_shape[2]),
        dtype=solver.precision,
        is_complex=(solver.data_type == "complex"),
    )

    a_d = cuda.to_device(a)
    b_d = cuda.to_device(b)
    info_d = cuda.device_array(batch_count, dtype=solver.info_type)
    ipiv_d = cuda.device_array((BLOCKS * solver.ipiv_shape[0], solver.ipiv_shape[1]), dtype=solver.ipiv_type)

    kernel[BLOCKS, solver.block_dim](a_d, b_d, info_d, ipiv_d)
    cuda.synchronize()

    a_result = a_d.copy_to_host()
    b_result = b_d.copy_to_host()
    info_result = info_d.copy_to_host()
    ipiv_result = ipiv_d.copy_to_host()

    for sample_idx in range(0, batch_count):
        if info_result[sample_idx] != 0:
            raise RuntimeError(f"{info_result[sample_idx]}-th parameter is wrong for sample with idx={sample_idx}")

    lower = np.tril(a_result)[:, :, : solver.m]
    u = np.triu(a_result)[:, : solver.n, :]
    idx = np.arange(min(solver.a_shape[1:]))
    lower[:, idx, idx] = 1
    a_recreated = lower @ u

    error_sum_reconstruct = 0.0
    error_sum_solution = 0.0

    for sample_idx in range(0, batch_count):
        if info_result[sample_idx] != 0:
            raise RuntimeError(f"{info_result[sample_idx]}-th parameter is wrong for sample with idx={sample_idx}")

        # 1. Check A factorization
        for i in range(solver.ipiv_shape[1] - 1, -1, -1):
            pivot = ipiv_result[sample_idx, i] - 1

            if pivot != i:
                temp = a_recreated[sample_idx, i, :].copy()
                a_recreated[sample_idx, i, :] = a_recreated[sample_idx, pivot, :]
                a_recreated[sample_idx, pivot, :] = temp

        error_sum_reconstruct += verify_relative_error(
            a_recreated[sample_idx], a[sample_idx], _PREC_TABLE[solver.precision], _ABS_ERR, solver
        )

        # 2. Compare solutions with scipy

        x_scipy = scipy.linalg.solve(a[sample_idx], b[sample_idx])
        error_sum_solution += verify_relative_error(
            b_result[sample_idx], x_scipy, _PREC_TABLE[solver.precision], _ABS_ERR, solver
        )

    print(f"Successfully validated LU reconstruction, with accumulated error: {error_sum_reconstruct}")
    print(f"Successfully validated against scipy reference, with accumulated error: {error_sum_solution}")


if __name__ == "__main__":
    main()
