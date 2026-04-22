# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

#
# Mirrors https://github.com/NVIDIA/CUDALibrarySamples/blob/main/MathDx/cuSolverDx/01_Linear_Solve/getrf_partial_pivot.cu
#

# This example demonstrates using cuSolverDx API to perform
# LU factorization with partial pivoting on a general MxN matrix.

import warnings

import numpy as np
from common import prepare_random_matrix, verify_relative_error
from common_numba import load_to_shared_strided, store_from_shared_strided
from numba import cuda

from nvmath.device import LUPivotSolver

warnings.filterwarnings("ignore", module="numba")

_PREC_TABLE = {
    np.float32: 1e-4,
    np.float64: 1e-6,
}
_ABS_ERR = 1e-8


def main():
    solver = LUPivotSolver(
        size=(48, 32),
        precision=np.float32,
        data_type="complex",
        execution="Block",
        block_dim=(33,),
        arrangement=("col_major", "col_major"),
    )

    n_a = solver.a_size()
    n_ipiv = solver.ipiv_size

    @cuda.jit
    def getrf_kernel(a, info, ipiv):
        smem_a = cuda.shared.array(n_a, dtype=solver.value_type)
        smem_ipiv = cuda.shared.array(n_ipiv, dtype=np.int32)

        load_to_shared_strided(a, smem_a, solver.a_shape, solver.a_strides())
        cuda.syncthreads()

        solver.factorize(smem_a, smem_ipiv, info)
        cuda.syncthreads()

        store_from_shared_strided(smem_a, a, solver.a_shape, solver.a_strides())
        store_from_shared_strided(smem_ipiv, ipiv, solver.ipiv_shape, solver.ipiv_strides)

    a = prepare_random_matrix(
        solver.a_shape,
        solver.precision,
        is_complex=True,
        is_hermitian=False,
        is_diag_dom=False,
    )

    a_d = cuda.to_device(a)
    ipiv_d = cuda.device_array(solver.ipiv_shape, dtype=solver.ipiv_type)
    info_d = cuda.device_array(solver.info_shape, dtype=solver.info_type)

    getrf_kernel[1, solver.block_dim](a_d, info_d, ipiv_d)
    cuda.synchronize()

    info_result = info_d.copy_to_host()
    ipiv_result = ipiv_d.copy_to_host()
    a_result = a_d.copy_to_host()

    print(f"After cuSolverDx getrf kernel: info = {info_result[0]}")

    if info_result[0] != 0:
        raise RuntimeError(f"{info_result[0]}-th parameter is wrong")

    lower = np.tril(a_result)[:, :, : solver.m]
    u = np.triu(a_result)[:, : solver.n, :]
    idx = np.arange(min(solver.a_shape[1:]))
    lower[:, idx, idx] = 1
    a_recreated = lower @ u

    for i in range(solver.ipiv_shape[1] - 1, -1, -1):
        pivot = ipiv_result[0, i] - 1

        if pivot != i:
            temp = a_recreated[0, i, :].copy()
            a_recreated[0, i, :] = a_recreated[0, pivot, :]
            a_recreated[0, pivot, :] = temp

    error = verify_relative_error(a_recreated[0], a[0], _PREC_TABLE[solver.precision], _ABS_ERR, solver)

    print(f"Success: LU factorization completed without errors: error={error}")


if __name__ == "__main__":
    main()
