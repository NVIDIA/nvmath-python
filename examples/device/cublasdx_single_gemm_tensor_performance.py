# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from numba import cuda
from nvmath.device import matmul
from common import mm_perf_GFlops, random_real
from common_numba import time_numba
from nvmath.device.common import axpby, clear, copy, copy_fragment, copy_wait, make_tensor


def main():
    m, n, k = 64, 64, 64
    block_size = 128
    repeat = 4000
    alpha, beta = 1, 2
    ncycles = 1
    data_type = "real"
    precision = np.float16

    MM = matmul(
        size=(m, n, k),
        precision=(precision, precision, precision),
        data_type=data_type,
        arrangement=("row_major", "col_major", "col_major"),
        execution="Block",
        block_size=block_size,
        compiler="numba",
        execute_api="tensors",
        tensor_types=("suggested_smem_a", "suggested_smem_b", "suggested_rmem_c"),
    )
    grid_dim = 1

    a_size = MM.suggest_layout_smem_a().cosize
    b_size = MM.suggest_layout_smem_b().cosize
    c_size = MM.suggest_layout_rmem_c().cosize

    @cuda.jit(link=MM.files)
    def f(a, b, c, alpha, beta, output, repeat):
        # We have same precision for all tensors
        smem = cuda.shared.array(shape=(0,), dtype=precision, alignment=16)
        smem_a_buff, smem = smem[:a_size], smem[a_size:]
        smem_b_buff, smem = smem[:b_size], smem[b_size:]
        rmem_c_buff = cuda.local.array(shape=(c_size,), dtype=MM.c_value_type)
        rmem_c_out_buff = cuda.local.array(shape=(c_size,), dtype=MM.c_value_type)

        gmem_a = make_tensor(a, MM.get_layout_gmem_a())
        gmem_b = make_tensor(b, MM.get_layout_gmem_b())
        gmem_c = make_tensor(c, MM.get_layout_gmem_c())
        gmem_output = make_tensor(output, MM.get_layout_gmem_c())

        smem_a = make_tensor(smem_a_buff, MM.suggest_layout_smem_a())
        smem_b = make_tensor(smem_b_buff, MM.suggest_layout_smem_b())
        rmem_c = make_tensor(rmem_c_buff, MM.suggest_layout_rmem_c())
        rmem_c_out = make_tensor(rmem_c_out_buff, MM.suggest_layout_rmem_c())

        copy(gmem_a, smem_a)
        copy(gmem_b, smem_b)

        copy_wait()

        clear(rmem_c)
        copy_fragment(gmem_c, rmem_c_out)

        alpha = c.dtype.type(alpha)
        beta = c.dtype.type(beta)

        for r in range(repeat):
            MM.execute(smem_a, smem_b, rmem_c)
            axpby(alpha, rmem_c, beta, rmem_c_out)

        cuda.syncthreads()

        copy_fragment(rmem_c_out, gmem_output)

    a = random_real(MM.a_dim, precision, order="C")
    b = random_real(MM.b_dim, precision, order="F")
    c = random_real(MM.c_dim, precision, order="F")
    o = np.zeros_like(c)

    a_d = cuda.to_device(a)
    b_d = cuda.to_device(b)
    c_d = cuda.to_device(c)
    o_d = cuda.to_device(o)

    shared_memory_size = MM.get_shared_storage_size_ab(
        MM.suggest_layout_smem_a(),
        MM.suggest_layout_smem_b(),
    )

    time_ms = time_numba(f, grid_dim, MM.block_dim, shared_memory_size, ncycles, a_d, b_d, c_d, alpha, beta, o_d, repeat)
    time_2x_ms = time_numba(f, grid_dim, MM.block_dim, shared_memory_size, ncycles, a_d, b_d, c_d, alpha, beta, o_d, 2 * repeat)
    time_mm_ms = (time_2x_ms - time_ms) / repeat
    perf = mm_perf_GFlops((m, n, k), 1, time_mm_ms)

    # Correction call
    f[grid_dim, MM.block_dim, 0, shared_memory_size](a_d, b_d, c_d, alpha, beta, o_d, 1)
    cuda.synchronize()

    print(f"Time {time_mm_ms} ms\nPerf {perf} GFlop/s")

    print(f"m, n, k: {m}, {n}, {k}")
    print(f"Data type: {data_type}")
    print(f"Precision: {precision}")
    print(f"Block size: {block_size}")
    print(f"Leading dimensions: {MM.leading_dimension}")
    print(f"Shared memory: {shared_memory_size} bytes ({a_size + b_size} elements)")
    print(f"Avg time [ms]: {time_mm_ms}")
    print(f"Time (all) [ms]: {time_mm_ms * repeat}")
    print(f"Performance [GFLOPS]: {perf}")

    data_test = o_d.copy_to_host()
    data_ref = alpha * (a @ b) + beta * c
    error = np.linalg.norm(data_test - data_ref) / np.linalg.norm(data_ref)
    print(error)
    assert error < 1e-2


if __name__ == "__main__":
    main()
