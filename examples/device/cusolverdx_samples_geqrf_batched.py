# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

#
# Mirrors https://github.com/NVIDIA/CUDALibrarySamples/blob/main/MathDx/cuSolverDx/03_Orthogonal_Factors/geqrf_batched.cu
#

import warnings

import numpy as np
from common import prepare_random_matrix, verify_relative_error
from common_numba import load_to_shared_strided, store_from_shared_strided
from numba import cuda

from nvmath.device import QRFactorize

warnings.filterwarnings("ignore", module="numba")

np.random.seed(43)

_PREC_TABLE = {
    np.float32: 1e-4,
    np.float64: 1e-6,
}
_ABS_ERR = 1e-8

BLOCKS = 32


def reconstruct_qr(factorizer: QRFactorize, a: np.ndarray, tau: np.ndarray) -> np.ndarray:
    r = np.triu(a)
    q = np.identity(factorizer.m, dtype=a.dtype)

    for i in range(factorizer.tau_shape[1]):
        v = np.zeros((factorizer.m, 1), dtype=a.dtype)
        v[i, 0] = 1.0
        v[i + 1 :, 0] = a[i + 1 :, i]

        q = q @ (np.identity(factorizer.m, dtype=a.dtype) - tau[i] * (v @ v.T.conj()))

    return q @ r


def main():
    solver = QRFactorize(
        size=(3, 4),
        precision=np.float32,
        data_type="real",
        arrangement="row_major",
        execution="Block",
        batches_per_block="suggested",
        block_dim="suggested",
    )

    n_a = solver.a_size()
    n_tau = solver.tau_size
    batch_count = BLOCKS * solver.batches_per_block

    print(f"solver.batches_per_block={solver.batches_per_block}")
    print(f"solver.block_dim={solver.block_dim}")
    print(f"batch_count={batch_count}")

    @cuda.jit
    def kernel(a, tau):
        smem_a = cuda.shared.array(n_a, dtype=solver.value_type)
        smem_tau = cuda.shared.array(n_tau, dtype=solver.value_type)

        base_sample_idx = cuda.blockIdx.x * solver.batches_per_block

        load_to_shared_strided(a[base_sample_idx:], smem_a, solver.a_shape, solver.a_strides())
        cuda.syncthreads()

        solver.factorize(smem_a, smem_tau)
        cuda.syncthreads()

        store_from_shared_strided(smem_a, a[base_sample_idx:], solver.a_shape, solver.a_strides())
        store_from_shared_strided(smem_tau, tau[base_sample_idx:], solver.tau_shape, solver.tau_strides)

    a = prepare_random_matrix(
        (batch_count, solver.a_shape[1], solver.a_shape[2]),
        dtype=solver.precision,
        is_complex=(solver.data_type == "complex"),
    )

    a_d = cuda.to_device(a)
    tau_d = cuda.device_array((BLOCKS * solver.tau_shape[0], solver.tau_shape[1]), dtype=a.dtype)

    kernel[BLOCKS, solver.block_dim](a_d, tau_d)
    cuda.synchronize()

    a_result = a_d.copy_to_host()
    tau_result = tau_d.copy_to_host()

    error_sum = 0.0
    for sample_idx in range(0, batch_count):
        a_recreated = reconstruct_qr(solver, a_result[sample_idx], tau_result[sample_idx])
        error_sum += verify_relative_error(a_recreated, a[sample_idx], _PREC_TABLE[solver.precision], _ABS_ERR, solver)

    print(f"Successfully validated, with accumulated error: {error_sum}")


if __name__ == "__main__":
    main()
