# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

#
# Mirrors https://github.com/NVIDIA/CUDALibrarySamples/blob/master/MathDx/cuBLASDx/device_gemm_performance.cu
#

import cupy
import numpy as np
from common import device_shared_memory, random
from common_cupy import time_cupy
from common_numba import time_numba
from numba import cuda

from nvmath.bindings import mathdx
from nvmath.device import Matmul
from nvmath.device.common import (
    axpby,
    make_tensor,
)
from nvmath.device.common_cuda import Dim3, get_current_device_cc
from nvmath.device.cublasdx import MAX_ALIGNMENT, DevicePipeline
from nvmath.device.cublasdx_numba import pipeline_extensions
from nvmath.linalg.advanced import Matmul as CublasltMatmul


def main():
    m, n, k = 8192, 8192, 8192
    tile_m, tile_n, tile_k = 128, 128, 32
    block_size = 128
    alignment = MAX_ALIGNMENT
    alpha, beta = 1.1, 1.2
    ncycles = 100
    data_type = "real"
    # Assume tile size is a factor of the problem size
    grid_dim = Dim3(m // tile_m, n // tile_n, 1)

    assert m % tile_m == 0
    assert n % tile_n == 0
    assert k % tile_k == 0

    MM = Matmul(
        size=(tile_m, tile_n, tile_k),
        precision=(np.float16, np.float16, np.float32),
        data_type=data_type,
        arrangement=("row_major", "col_major", "row_major"),
        execution="Block",
        block_size=block_size,
        alignment=alignment,
        with_pipeline=True,
        enable_input_streaming=True,
        static_block_dim=True,
        # WAR for the TMA descriptor issue on SM 12.0
        sm=89 if get_current_device_cc().major >= 12 and mathdx.get_version_ex() < (0, 3, 2) else None,
    )

    @cuda.jit(extensions=pipeline_extensions, launch_bounds=[MM.block_size, 1])
    def matmul_kernel(alpha, beta, c, device_pipeline: DevicePipeline):
        smem = cuda.shared.array(shape=(0,), dtype=np.byte, alignment=device_pipeline.buffer_alignment)

        blockIdx = cuda.blockIdx
        c_tile = c[blockIdx.x * tile_m : (blockIdx.x + 1) * tile_m, blockIdx.y * tile_n : (blockIdx.y + 1) * tile_n]
        gmem_c = make_tensor(c_tile, MM.get_layout_gmem_c(m))

        tile_pipeline = device_pipeline.get_tile(smem, blockIdx.x, blockIdx.y)

        accumulator = MM.suggest_accumulator()
        tile_pipeline.execute(accumulator)

        tile_pipeline._del()

        # External epilogue
        if accumulator.is_thread_active():
            d_frag = accumulator.make_partition_and_copy(gmem_c)
            axpby(alpha, accumulator.get_results(), beta, d_frag)
            accumulator.partition_and_copy(d_frag, gmem_c)

    a = random((m, k), MM.a_value_type, arrangement=MM.arrangement.a)
    b = random((k, n), MM.b_value_type, arrangement=MM.arrangement.b)
    c = random((m, n), MM.c_value_type, arrangement=MM.arrangement.c)

    a_d = cuda.to_device(a)
    b_d = cuda.to_device(b)
    c_d = cuda.to_device(c)

    manual_pipeline_depth = 0
    override_pipeline_depth = manual_pipeline_depth != 0

    # TODO: find this value
    # sizeof(cublasdx::pipeline_stage_scratch_t)
    pipeline_stage_scratch_size = 16
    stage_shared_req = (
        tile_m * tile_k * np.dtype(MM.a_value_type).itemsize
        + tile_k * tile_n * np.dtype(MM.b_value_type).itemsize
        + pipeline_stage_scratch_size
    )

    available_shared_memory = device_shared_memory(MM.sm)
    print(f"Available shared memory: {available_shared_memory} bytes for SM {MM.sm}")
    maximal_pipeline_depth = min(16, (available_shared_memory - 32) // stage_shared_req)
    print(f"Maximal pipeline depth: {maximal_pipeline_depth}")
    pipeline_depth = manual_pipeline_depth if override_pipeline_depth else maximal_pipeline_depth

    print(f"Pipeline depth: {pipeline_depth}")

    k_stages = k // tile_k
    print(f"K stages: {k_stages}")
    assert k_stages >= pipeline_depth, (
        "PipelineDepth must be less or equal to GEMM k stages, please adjust manual_pipeline_depth"
    )

    device_pipeline = MM.suggest_device_pipeline(pipeline_depth, a_d, b_d)

    print(f"Grid dim: {grid_dim}")
    print(f"Pipeline block dim: {device_pipeline.block_dim}")
    print(f"Pipeline buffer size: {device_pipeline.buffer_size} bytes")

    data_test = None

    def get_results():
        nonlocal data_test
        data_test = c_d.copy_to_host()

    time_mm_ms = time_numba(
        matmul_kernel,
        grid_dim,
        MM.block_dim,
        device_pipeline.buffer_size,
        ncycles,
        alpha,
        beta,
        c_d,
        device_pipeline,
        get_results=get_results,
    )

    print(f"tile_m, tile_n, tile_k: {tile_m}, {tile_n}, {tile_k}")
    print(f"Alignment: {MM.alignment}")
    print(f"Block size: {block_size}")
    print(f"m, n, k: {m}, {n}, {k}")
    print(f"Data type: {data_type}")
    print(f"Precision: {MM.a_value_type}, {MM.b_value_type}, {MM.c_value_type}")
    print(f"Leading dimensions: {MM.leading_dimension}")
    print(f"Shared memory: {device_pipeline.buffer_size} bytes ({(MM.a_size + MM.b_size) * 2} elements)")
    print(f"Avg time [ms]: {time_mm_ms}")
    print(f"Time (all) [ms]: {time_mm_ms * ncycles}")

    cuda.synchronize()

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
    assert error < 1e-4, f"error is higher than accepted value of 1e-4: {error}"


if __name__ == "__main__":
    main()
