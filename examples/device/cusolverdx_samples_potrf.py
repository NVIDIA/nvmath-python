# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

#
# Mirrors https://github.com/NVIDIA/CUDALibrarySamples/tree/main/MathDx/cuSolverDx/01_Linear_Solve/potrf.cu
#

import warnings

import numpy as np
import scipy
from common import prepare_random_matrix, verify_triangular_relative_error
from common_numba import load_to_shared_strided, store_from_shared_strided
from numba import cuda

from nvmath.device import CholeskySolver

warnings.filterwarnings("ignore", module="numba")

_PREC_TABLE = {
    np.float32: 1e-4,
    np.float64: 1e-6,
}
_ABS_ERR = 1e-8


def main():
    solver = CholeskySolver(
        size=(32, 32),
        precision=np.float64,
        data_type="complex",
        execution="Block",
        leading_dimensions=(33, 33),
        block_dim=(256, 1, 1),
        fill_mode="upper",
    )

    n_a = solver.a_size()

    @cuda.jit
    def kernel(a, info):
        smem_a = cuda.shared.array(n_a, dtype=solver.value_type)

        load_to_shared_strided(a, smem_a, solver.a_shape, solver.a_strides())
        cuda.syncthreads()

        solver.factorize(smem_a, info)
        cuda.syncthreads()

        store_from_shared_strided(smem_a, a, solver.a_shape, solver.a_strides())

    print(f"Use compile-time leading dimension LDA for shared memory = {solver.lda}")
    a = prepare_random_matrix(
        solver.a_shape,
        solver.precision,
        is_complex=True,
        is_hermitian=False,  # input A is not symmetric
        is_diag_dom=True,
    )

    a_d = cuda.to_device(a)
    info_d = cuda.device_array(solver.info_shape, dtype=solver.info_type)

    kernel[1, solver.block_dim](a_d, info_d)
    cuda.synchronize()

    a_result = a_d.copy_to_host()
    info_result = info_d.copy_to_host()

    print(f"after cuSolverDx potrf kernel: info = {info_result[0]}")
    if info_result[0] != 0:
        raise RuntimeError(f"{info_result[0]}-th parameter is wrong")

    scipy_result = scipy.linalg.cholesky(a[0], lower=(solver.fill_mode == "lower"))
    error = verify_triangular_relative_error(
        a_result[0], scipy_result, _PREC_TABLE[solver.precision], _ABS_ERR, solver, solver.fill_mode
    )
    print(f"Successfully validated against scipy reference, with error: {error}")


if __name__ == "__main__":
    main()
