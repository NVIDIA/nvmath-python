# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

#
# Mirrors https://github.com/NVIDIA/CUDALibrarySamples/tree/main/MathDx/cuSolverDx/07_BLAS/trsm_batched.cu
#

import warnings

import numpy as np
import scipy
from common import prepare_random_matrix, verify_relative_error
from common_numba import load_to_shared_strided, store_from_shared_strided
from numba import cuda

from nvmath.device import TriangularSolver

warnings.filterwarnings("ignore", module="numba")

np.random.seed(43)

_PREC_TABLE = {
    np.float32: 1e-4,
    np.float64: 1e-6,
}
_ABS_ERR = 1e-8

BLOCKS = 1024


def main():
    solver = TriangularSolver(
        size=(4, 4),
        precision=np.float64,
        data_type="real",
        side="right",
        fill_mode="lower",
        transpose_mode="non_transposed",
        diag="non_unit",
        arrangement=("col_major", "col_major"),
        execution="Block",
        batches_per_block="suggested",
        block_dim="suggested",
    )

    n_a = solver.a_size()
    n_b = solver.b_size()
    batch_count = BLOCKS * solver.batches_per_block

    print(f"solver.batches_per_block={solver.batches_per_block}")
    print(f"solver.block_dim={solver.block_dim}")
    print(f"batch_count={batch_count}")

    @cuda.jit
    def kernel(a, b):
        smem_a = cuda.shared.array(n_a, dtype=solver.value_type)
        smem_b = cuda.shared.array(n_b, dtype=solver.value_type)

        base_sample_idx = cuda.blockIdx.x * solver.batches_per_block

        load_to_shared_strided(a[base_sample_idx:], smem_a, solver.a_shape, solver.a_strides())
        load_to_shared_strided(b[base_sample_idx:], smem_b, solver.b_shape, solver.b_strides())
        cuda.syncthreads()

        solver.solve(smem_a, smem_b)
        cuda.syncthreads()

        store_from_shared_strided(smem_a, a[base_sample_idx:], solver.a_shape, solver.a_strides())
        store_from_shared_strided(smem_b, b[base_sample_idx:], solver.b_shape, solver.b_strides())

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

    kernel[BLOCKS, solver.block_dim](a_d, b_d)
    cuda.synchronize()

    b_result = b_d.copy_to_host()

    error_sum = 0.0
    for sample_idx in range(0, batch_count):
        # x @ A = B
        # <=>
        # A^T @ x^T = B^T

        scipy_trans = 0 if solver.transpose_mode == "non_transposed" else 1 if solver.transpose_mode == "transposed" else 2
        scipy_sol = scipy.linalg.solve_triangular(
            a[sample_idx].T,
            b[sample_idx].T,
            lower=(solver.fill_mode == "upper"),  # Note: A is transposed!
            trans=scipy_trans,
            unit_diagonal=(solver.diag == "unit"),
        ).T

        error_sum += verify_relative_error(b_result[sample_idx], scipy_sol, _PREC_TABLE[solver.precision], _ABS_ERR, solver)

    print(f"Successfully validated against scipy reference, with accumulated error: {error_sum}")


if __name__ == "__main__":
    main()
