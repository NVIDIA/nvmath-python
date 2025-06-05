# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

#
# Mirrors https://github.com/NVIDIA/CUDALibrarySamples/blob/master/MathDx/cuBLASDx/device_gemm_performance.cu
#

import cupy
import numpy as np
from numba import cuda
from common_cupy import time_cupy
from common_numba import time_numba
from nvmath.device import matmul
from nvmath.device.cublasdx import MAX_ALIGNMENT, SharedStorageCalc
from nvmath.linalg.advanced import Matmul as CublasltMatmul
from common import random_real
from nvmath.device.common import (
    axpby,
    clear,
    copy,
    copy_fragment,
    copy_wait,
    make_tensor,
)
from nvmath.device.common_cuda import Dim3


def main():
    m, n, k = 8192, 8192, 8192
    tile_m, tile_n, tile_k = 256, 128, 16
    block_size = 256
    alignment = MAX_ALIGNMENT
    alpha, beta = 1.1, 1.2
    ncycles = 100
    data_type = "real"
    precision = np.float32
    # Assume tile size is a factor of the problem size
    grid_dim = Dim3(m // tile_m, n // tile_n, 1)

    assert m % tile_m == 0
    assert n % tile_n == 0
    assert k % tile_k == 0

    MM = matmul(
        size=(tile_m, tile_n, tile_k),
        precision=(precision, precision, precision),
        data_type=data_type,
        arrangement=("col_major", "col_major", "col_major"),
        execution="Block",
        block_size=block_size,
        alignment=alignment,
        global_memory_alignment=alignment,
        static_block_dim=True,
        compiler="numba",
        execute_api="tensors",
        tensor_types=("suggested_smem_a", "suggested_smem_b", "suggested_rmem_c"),
    )

    a_size = MM.suggest_layout_smem_a().cosize
    b_size = MM.suggest_layout_smem_b().cosize
    c_size = MM.suggest_layout_rmem_c().cosize

    # Check that the alignment of the split shared memory is matching the
    # alignment of the matmul type.
    assert a_size * np.dtype(precision).itemsize % alignment.a == 0
    assert b_size * np.dtype(precision).itemsize % alignment.b == 0

    @cuda.jit(link=MM.files)
    def f(a, b, c, alpha, beta, output):
        block_m = cuda.blockIdx.x
        block_n = cuda.blockIdx.y

        # We have same precision for all tensors
        smem = cuda.shared.array(shape=(0,), dtype=precision, alignment=16)

        # 1. PREPARE GLOBAL MEMORY TENSORS

        a_slice = a[block_m * tile_m : (block_m + 1) * tile_m, :]
        b_slice = b[:, block_n * tile_n : (block_n + 1) * tile_n]
        c_tile = c[
            block_m * tile_m : (block_m + 1) * tile_m,
            block_n * tile_n : (block_n + 1) * tile_n,
        ]
        output_tile = output[
            block_m * tile_m : (block_m + 1) * tile_m,
            block_n * tile_n : (block_n + 1) * tile_n,
        ]
        gmem_c = make_tensor(c_tile, MM.get_layout_gmem_c(m))
        gmem_output = make_tensor(output_tile, MM.get_layout_gmem_c(m))

        # 2. PREPARE SHARED MEMORY TENSORS
        # We are slicing smem that is aligned to 16 bytes into pieces that are
        # multiple of 16 bytes. That guarantees that the alignment of the slices
        # is also 16 bytes. We must have alignment of the input match to the
        # alignment of the matmul type.
        # See check above.
        smem_a_buff, smem = smem[0:a_size], smem[a_size:]
        smem_b_buff, smem = smem[0:b_size], smem[b_size:]
        smem_a_n_buff, smem = smem[0:a_size], smem[a_size:]
        smem_b_n_buff, smem = smem[0:b_size], smem[b_size:]

        smem_a = make_tensor(smem_a_buff, MM.suggest_layout_smem_a())
        smem_b = make_tensor(smem_b_buff, MM.suggest_layout_smem_b())
        smem_a_n = make_tensor(smem_a_n_buff, MM.suggest_layout_smem_a())
        smem_b_n = make_tensor(smem_b_n_buff, MM.suggest_layout_smem_b())

        # 3. PREPARE 2-STAGE MEMORY PIPELINE

        stages = k // tile_k

        a_tile = a_slice[:, 0:tile_k]
        b_tile = b_slice[0:tile_k, :]

        gmem_a = make_tensor(a_tile, MM.get_layout_gmem_a(m))
        gmem_b = make_tensor(b_tile, MM.get_layout_gmem_b(k))

        copy(gmem_a, smem_a)
        copy(gmem_b, smem_b)

        # 4. EXECUTE GEMM WITH ACCUMULATION IN REGISTERS

        rmem_c_buff = cuda.local.array(shape=(c_size,), dtype=MM.c_value_type)
        rmem_c = make_tensor(rmem_c_buff, MM.suggest_layout_rmem_c())

        clear(rmem_c)

        for stage in range(1, stages):
            # Wait for previous stage
            copy_wait()

            # Copy tile for the next iteration
            a_tile = a_slice[:, stage * tile_k : (stage + 1) * tile_k]
            b_tile = b_slice[stage * tile_k : (stage + 1) * tile_k, :]

            gmem_a = make_tensor(a_tile, MM.get_layout_gmem_a(m))
            gmem_b = make_tensor(b_tile, MM.get_layout_gmem_b(k))

            copy(gmem_a, smem_a_n)
            copy(gmem_b, smem_b_n)

            # Accumulate results from this stage
            MM.execute(smem_a, smem_b, rmem_c)

            # Swap for the next iteration
            smem_a_n, smem_a = smem_a, smem_a_n
            smem_b_n, smem_b = smem_b, smem_b_n

        copy_wait()
        MM.execute(smem_a, smem_b, rmem_c)

        # 5. EPILOGUE

        rmem_c_out_buff = cuda.local.array(shape=(c_size,), dtype=MM.c_value_type)
        rmem_c_out = make_tensor(rmem_c_out_buff, MM.suggest_layout_rmem_c())

        copy_fragment(gmem_c, rmem_c_out)
        axpby(alpha, rmem_c, beta, rmem_c_out)
        copy_fragment(rmem_c_out, gmem_output)

    a = random_real((m, k), precision, order="F")
    b = random_real((k, n), precision, order="F")
    c = random_real((m, n), precision, order="F")
    o = np.zeros_like(c)

    a_d = cuda.to_device(a)
    b_d = cuda.to_device(b)
    c_d = cuda.to_device(c)
    o_d = cuda.to_device(o)

    smem_calc = SharedStorageCalc()
    smem_calc.add(MM.alignment.a, np.dtype(precision).itemsize, MM.suggest_layout_smem_a())
    smem_calc.add(MM.alignment.b, np.dtype(precision).itemsize, MM.suggest_layout_smem_b())
    smem_calc.add(MM.alignment.a, np.dtype(precision).itemsize, MM.suggest_layout_smem_a())
    smem_calc.add(MM.alignment.b, np.dtype(precision).itemsize, MM.suggest_layout_smem_b())
    shared_memory_size = smem_calc.get()

    time_mm_ms = time_numba(
        f,
        grid_dim,
        MM.block_dim,
        shared_memory_size,
        ncycles,
        a_d,
        b_d,
        c_d,
        alpha,
        beta,
        o_d,
    )

    print(f"tile_m, tile_n, tile_k: {tile_m}, {tile_n}, {tile_k}")
    print(f"Alignment: {MM.alignment}")
    print(f"Block size: {block_size}")
    print(f"m, n, k: {m}, {n}, {k}")
    print(f"Data type: {data_type}")
    print(f"Precision: {precision}")
    print(f"Leading dimensions: {MM.leading_dimension}")
    print(f"Shared memory: {shared_memory_size} bytes ({(MM.a_size + MM.b_size) * 2} elements)")
    print(f"Avg time [ms]: {time_mm_ms}")
    print(f"Time (all) [ms]: {time_mm_ms * ncycles}")

    data_test = o_d.copy_to_host()

    a_d2 = cupy.array(a)
    b_d2 = cupy.array(b)
    c_d2 = cupy.array(c)

    with CublasltMatmul(a_d2, b_d2, c=c_d2, alpha=alpha, beta=beta) as mm:
        mm.plan()

        time_cublaslt = time_cupy(lambda: mm.execute(), ncycles)
        data_ref = mm.execute()

    error = np.linalg.norm(data_test - cupy.asnumpy(data_ref)) / np.linalg.norm(data_ref)

    print(f"Avg time cublaslt [ms]: {time_cublaslt}")
    print(f"Time cublaslt (all) [ms]: {time_cublaslt * ncycles}")

    print(f"Numba cuBLASDx / cuBLASLt timinig = {time_mm_ms / time_cublaslt}")
    print(f"Error: {error}")
    assert error < 1e-5, f"error is higher than accepted value of 1e-5: {error}"


if __name__ == "__main__":
    main()
