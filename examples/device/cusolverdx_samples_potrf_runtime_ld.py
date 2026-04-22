# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

#
# Mirrors https://github.com/NVIDIA/CUDALibrarySamples/tree/main/MathDx/cuSolverDx/01_Linear_Solve/potrf_runtime_ld.cu
#

# In some workflows, the matrix A may have already been in the shared memory,
# padded to avoid bank conflict, and being updated by other operations.
# This example shows how to use runtime leading dimensions in cuSolverDx API,
# compute Cholesky factorization for a Hermitian
# positive-definite matrix, and compare the result factors with scipy.

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
        data_type="real",
        execution="Block",
        fill_mode="lower",
        block_dim=(256, 1, 1),
    )

    lda_smem = 34
    print(f"Use runtime leading dimension LDA for shared memory = {lda_smem}")

    n_a = solver.a_size(lda=lda_smem)

    @cuda.jit
    def potrf_kernel(a, lda, info):
        smem_a = cuda.shared.array(n_a, dtype=solver.value_type)

        load_to_shared_strided(a, smem_a, solver.a_shape, solver.a_strides(lda=lda))
        cuda.syncthreads()

        solver.factorize(smem_a, info, lda=lda)
        cuda.syncthreads()

        store_from_shared_strided(smem_a, a, solver.a_shape, solver.a_strides(lda=lda))

    a = prepare_random_matrix(
        solver.a_shape,
        solver.precision,
        is_complex=False,
        is_hermitian=False,  # input A is not symmetric
        is_diag_dom=True,
    )

    a_d = cuda.to_device(a)
    info_d = cuda.device_array(solver.info_shape, dtype=solver.info_type)

    potrf_kernel[1, solver.block_dim](a_d, lda_smem, info_d)
    cuda.synchronize()

    A_result = a_d.copy_to_host()
    info_result = info_d.copy_to_host()

    print(f"after cuSolverDx potrf kernel: info = {info_result[0]}")
    if info_result[0] != 0:
        raise RuntimeError(f"{info_result[0]}-th parameter is wrong")

    scipy_result = scipy.linalg.cholesky(a[0], lower=(solver.fill_mode == "lower"))

    error = verify_triangular_relative_error(
        A_result[0], scipy_result, _PREC_TABLE[solver.precision], _ABS_ERR, solver, solver.fill_mode
    )
    print(f"Successfully validated against scipy reference, with error: {error}")


if __name__ == "__main__":
    main()
