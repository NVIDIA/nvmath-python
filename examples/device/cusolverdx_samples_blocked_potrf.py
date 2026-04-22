# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

#
# Mirrors https://github.com/NVIDIA/CUDALibrarySamples/tree/main/MathDx/cuSolverDx/10_Advanced/blocked_potrf.cu
#

# This is a basic implementation of Cholesky factorization using a single
# CTA with matrices too large for cuSolverDx's shared memory API.
# The code uses an out-of-core style, left-looking formulation
# so that it never needs more than 4 blocks in shared memory at a given point.

# There are a few limitations of this implementation:
# * Only real-valued types are supported
# * The matrix size, N, must be a multiple of the block size, NB
# * Only upper-triangular storage is implemented

#### Helper functions to load and store the diagonal blocks ###
# Only the upper triangular part is transferred

import numpy as np
import scipy
from common import prepare_random_matrix, verify_triangular_relative_error
from common_numba import time_numba_prep_args as time_numba
from numba import cuda

from nvmath.device import CholeskySolver, Matmul, TriangularSolver

_PREC_TABLE = {
    np.float32: 1e-4,
    np.float64: 1e-6,
}
_ABS_ERR = 1e-8

N = 512  # Must be a power of 2
BATCH_COUNT = 400
NB = 32  # Must be a power of 2
NT = 128
ARRANGE = "row_major"
INPUT_SIZE = N * N
SMEM_SIZE = 4 * NB * NB * 8 + 8
REPEATS = 5

assert NT % NB == 0


# https://github.com/icl-utk-edu/magma/blob/master/testing/flops.h
def get_flops_potrf(data_type, n):
    fmuls = n * (((1.0 / 6.0) * n + 0.5) * n + (1.0 / 3.0))
    fadds = n * (((1.0 / 6.0) * n) * n - (1.0 / 6.0))

    if data_type == "complex":
        return 6.0 * fmuls + 2.0 * fadds
    return fmuls + fadds


def main():
    # ==========================
    # Solver configuration
    # ==========================

    cholesky = CholeskySolver(
        fill_mode="upper",
        size=(NB, NB),
        precision=np.float64,
        data_type="real",
        arrangement=(ARRANGE, ARRANGE),
        execution="Block",
        block_dim=(NT, 1, 1),
    )

    triangular = TriangularSolver(
        size=(NB, NB, NB),
        precision=np.float64,
        data_type="real",
        side="left",
        diag="non_unit",
        transpose_mode="transposed",
        arrangement=(ARRANGE, ARRANGE),
        fill_mode="upper",
        execution="Block",
        block_dim=(NT, 1, 1),
    )

    gemm_arrange = (
        ("row_major", "col_major", "col_major") if ARRANGE == "col_major" else ("col_major", "row_major", "row_major")
    )
    alignment = 16 if (8 * NB * NB) % 16 == 0 else 8
    gemm = Matmul(
        size=(NB, NB, NB),
        precision=np.float64,
        data_type="real",
        arrangement=gemm_arrange,
        execution="Block",
        block_size=NT,
        alignment=(alignment, alignment, alignment),
    )

    # ==========================
    # Kernel implementation
    # ==========================

    @cuda.jit(device=True, forceinline=True)
    def load_diagonal_block(a, a_shared, ld_shared):
        tid = cuda.threadIdx.x

        stride_jj = NT // NB
        i = tid % NB
        j = tid // NB

        for jj in range(0, NB, stride_jj):
            if i <= j + jj:
                a_shared[i * ld_shared + (jj + j)] = a[i, jj + j]

        cuda.syncthreads()

    @cuda.jit(device=True, forceinline=True)
    def store_diagonal_block(a_shared, ld_shared, a):
        cuda.syncthreads()

        tid = cuda.threadIdx.x

        stride_jj = NT // NB
        i = tid % NB
        j = tid // NB

        for jj in range(0, NB, stride_jj):
            if i <= j + jj:
                a[i, jj + j] = a_shared[i * ld_shared + (jj + j)]

    @cuda.jit(device=True, forceinline=True)
    def copy_2d(src_gmem, dst, ld_dst):
        tid = cuda.threadIdx.x

        stride = NT // NB
        i = tid % NB
        j = tid // NB

        for jj in range(0, NB, stride):
            dst[i * ld_dst + (j + jj)] = src_gmem[i, j + jj]

    @cuda.jit(device=True, forceinline=True)
    def copy_2d_back(dst_gmem, src, ld_src):
        tid = cuda.threadIdx.x

        stride = NT // NB
        i = tid % NB
        j = tid // NB

        for jj in range(0, NB, stride):
            dst_gmem[i, j + jj] = src[i * ld_src + (j + jj)]

    @cuda.jit()
    def kernel(a, info):
        smem = cuda.shared.array(shape=(0,), dtype=np.float64, alignment=16)

        block_size = NB * NB
        a_shared = smem[:block_size]
        b_shared = smem[block_size : 2 * block_size]
        c_shared = smem[2 * block_size : 3 * block_size]
        d_shared = smem[3 * block_size : 4 * block_size]

        sinfo = d_shared[4 * block_size :].view(np.int32)[0:1]

        n_tiles = N // NB
        sample_idx = cuda.blockIdx.x
        r_info = 0
        lds = NB

        for k in range(0, n_tiles):
            a_kk = a[sample_idx, k * NB :, k * NB :]
            load_diagonal_block(a_kk, a_shared, lds)

            for i in range(0, k):
                a_ik = a[sample_idx, i * NB :, k * NB :]

                copy_2d(a_ik, c_shared, lds)

                cuda.syncthreads()
                gemm.execute(-1.0, c_shared, c_shared, 1.0, a_shared)
                cuda.syncthreads()

            cholesky.factorize(a_shared, sinfo, lda=lds)
            store_diagonal_block(a_shared, lds, a_kk)

            if cuda.threadIdx.x == 0 and r_info == 0 and sinfo[0] != 0:
                r_info = sinfo[0] + k * NB

            for j in range(k + 1, n_tiles):
                a_kj = a[sample_idx, k * NB :, j * NB :]
                copy_2d(a_kj, b_shared, lds)

                for i in range(0, k):
                    a_ik = a[sample_idx, i * NB :, k * NB :]
                    a_ij = a[sample_idx, i * NB :, j * NB :]

                    copy_2d(a_ik, c_shared, lds)
                    copy_2d(a_ij, d_shared, lds)

                    cuda.syncthreads()
                    gemm.execute(-1.0, c_shared, d_shared, 1.0, b_shared)
                    cuda.syncthreads()

                cuda.syncthreads()
                triangular.solve(a_shared, b_shared, lda=lds, ldb=lds)
                cuda.syncthreads()

                copy_2d_back(a_kj, b_shared, lds)

        if cuda.threadIdx.x == 0:
            info[sample_idx] = r_info

    # ==========================
    # Prepare data
    # ==========================

    a = prepare_random_matrix(
        (BATCH_COUNT, N, N),
        np.float64,
        is_complex=False,
        is_hermitian=False,
        is_diag_dom=True,
    )

    # ==========================
    # Check performance
    # ==========================

    def prep_args():
        A_flat_d = cuda.to_device(a)
        info_d = cuda.device_array(BATCH_COUNT, dtype=cholesky.info_type)

        return [A_flat_d, info_d]

    ms = time_numba(kernel, BATCH_COUNT, NT, SMEM_SIZE, REPEATS, prep_args)
    seconds_per_giga_batch = ms / 1e3 / BATCH_COUNT * 1e9
    gb_s = INPUT_SIZE * 8 * 2 / seconds_per_giga_batch
    gflops = get_flops_potrf("real", N) / seconds_per_giga_batch

    print(
        f"{'blocked_potrf':<30} {BATCH_COUNT:>10} {N:>5} {N:>5} {0:>5}  {gflops:>7.2f} GFLOP/s,"
        f" {gb_s:>7.2f} GB/s, {ms:>7.2f} ms, {NT} blockDim"
    )

    # ==========================
    # Check results
    # ==========================

    a_d = cuda.to_device(a)
    info_d = cuda.device_array(BATCH_COUNT, dtype=cholesky.info_type)
    kernel[BATCH_COUNT, NT, 0, SMEM_SIZE](a_d, info_d)
    cuda.synchronize()

    a_result = a_d.copy_to_host()
    info_result = info_d.copy_to_host()

    error_sum = 0.0
    for sample_idx in range(BATCH_COUNT):
        if info_result[sample_idx] != 0:
            raise RuntimeError(f"{info_result[sample_idx]}-th parameter is wrong for {sample_idx} sample")

        scipy_result = scipy.linalg.cholesky(a[sample_idx], lower=False)
        error = verify_triangular_relative_error(
            a_result[sample_idx], scipy_result, _PREC_TABLE[cholesky.precision], _ABS_ERR, cholesky, "upper"
        )
        error_sum += error

    print(f"Sum of relative errors: {error_sum}")

    if error_sum > 1e-6:
        raise RuntimeError("Failure...")
    print("Success...")


if __name__ == "__main__":
    main()
