# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import warnings

import numpy as np
from common import prepare_random_matrix
from common_numba import load_to_shared_batched, load_to_shared_strided, store_from_shared_batched, store_from_shared_strided
from numba import cuda

from nvmath.device import CholeskySolver, Solver

warnings.filterwarnings("ignore", module="numba")

M, N, K = (27, 27, 27)
PRECISION = np.float64
FILL_MODE = "lower"
DATA_TYPE = "complex"
BLOCK_DIM = (64, 1, 1)


def device_api_example(a, b):
    solver0 = Solver(
        function="potrf",
        precision=PRECISION,
        execution="Block",
        size=(M, N, K),
        fill_mode=FILL_MODE,
        data_type=DATA_TYPE,
        block_dim=BLOCK_DIM,
    )

    solver1 = Solver(
        function="potrs",
        precision=PRECISION,
        execution="Block",
        size=(M, N, K),
        fill_mode=FILL_MODE,
        data_type=DATA_TYPE,
        block_dim=BLOCK_DIM,
    )

    n_a = solver0.lda * solver0.n
    n_b = solver0.ldb * solver0.k

    @cuda.jit()
    def f(a, b, info):
        a_s = cuda.shared.array(n_a, dtype=solver0.value_type)
        b_s = cuda.shared.array(n_b, dtype=solver0.value_type)

        load_to_shared_batched(a, a_s, 0, (solver0.m, solver0.n), solver0.lda)
        cuda.syncthreads()

        solver0.execute(a_s, info)
        cuda.syncthreads()

        if info[0] != 0:
            return

        load_to_shared_batched(b, b_s, 0, (solver0.m, solver0.k), solver0.ldb)
        cuda.syncthreads()

        solver1.execute(a_s, b_s)
        cuda.syncthreads()

        store_from_shared_batched(a_s, a, 0, (solver0.m, solver0.n), solver0.lda)
        store_from_shared_batched(b_s, b, 0, (solver0.m, solver0.k), solver0.ldb)

    a_d = cuda.to_device(a)
    b_d = cuda.to_device(b)
    info_d = cuda.device_array(solver0.batches_per_block, dtype=solver0.info_type)

    f[1, solver0.block_dim](a_d, b_d, info_d)
    cuda.synchronize()

    info_result = info_d.copy_to_host()
    print(f"after device api run: info = {info_result[0]}")
    if info_result[0] != 0:
        raise RuntimeError(f"{info_result[0]}-th parameter is wrong")

    return b_d.copy_to_host()


def pythonic_api_example(a, b):
    solver = CholeskySolver(
        precision=PRECISION,
        execution="Block",
        size=(M, N, K),
        fill_mode=FILL_MODE,
        data_type=DATA_TYPE,
        block_dim=BLOCK_DIM,
    )

    n_a = solver.a_size()
    n_b = solver.b_size()

    @cuda.jit
    def f(a, b, info):
        a_s = cuda.shared.array(n_a, dtype=solver.value_type)
        b_s = cuda.shared.array(n_b, dtype=solver.value_type)

        load_to_shared_strided(a, a_s, solver.a_shape, solver.a_strides())
        cuda.syncthreads()

        solver.factorize(a_s, info)
        cuda.syncthreads()

        if info[0] != 0:
            return

        load_to_shared_strided(b, b_s, solver.b_shape, solver.b_strides())
        cuda.syncthreads()

        solver.solve(a_s, b_s)
        cuda.syncthreads()

        store_from_shared_strided(a_s, a, solver.a_shape, solver.a_strides())
        store_from_shared_strided(b_s, b, solver.b_shape, solver.b_strides())

    a_d = cuda.to_device(a)
    b_d = cuda.to_device(b)
    info_d = cuda.device_array(solver.info_shape, dtype=solver.info_type)

    f[1, solver.block_dim](a_d, b_d, info_d)
    cuda.synchronize()

    info_result = info_d.copy_to_host()
    print(f"after pythonic api run: info = {info_result[0]}")
    if info_result[0] != 0:
        raise RuntimeError(f"{-info_result[0]}-th parameter is wrong")

    return b_d.copy_to_host()


def main():
    a = prepare_random_matrix(
        (1, M, N),
        PRECISION,
        is_complex=True,
        is_hermitian=False,  # input A is not symmetric
        is_diag_dom=True,
    )
    b = prepare_random_matrix(
        (1, M, K),
        dtype=PRECISION,
        is_complex=True,
    )

    result_0 = device_api_example(a, b)
    result_1 = pythonic_api_example(a, b)

    diff = np.linalg.norm(result_0[0] - result_1[0])
    ref_norm = np.linalg.norm(result_1[0])

    if diff > 1e-6 * ref_norm + 1e-8:
        raise RuntimeError("Failed...")

    error = diff / ref_norm if ref_norm != 0.0 else 0.0
    print(f"Success with error: {error}")


if __name__ == "__main__":
    main()
