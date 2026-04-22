# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

#
# Mirrors https://github.com/NVIDIA/CUDALibrarySamples/blob/main/MathDx/cuBLASDx/01_gemm_introduction/introduction_pipeline.cu
#

import cupy
import numpy as np
from common import random_real
from common_numba import set_max_dynamic_shared_size_bytes
from numba import cuda

from nvmath.bindings import mathdx
from nvmath.device import Matmul, axpby
from nvmath.device.common import make_tensor
from nvmath.device.common_cuda import Dim3, get_current_device_cc
from nvmath.device.cublasdx import DevicePipeline
from nvmath.device.cublasdx_backend import MAX_ALIGNMENT
from nvmath.device.cublasdx_numba import pipeline_extensions


def main():
    m, n, k = 1024, 1024, 1024
    steps = 2
    tile_m, tile_n, tile_k = 128, 128, 32
    block_size = 128
    pipeline_depth = 2

    alpha, beta = 2.0, 3.0

    assert m % tile_m == 0
    assert n % tile_n == 0
    assert k % tile_k == 0

    grid_dim = Dim3(m // tile_m, n // tile_n, 1)

    MM = Matmul(
        size=(tile_m, tile_n, tile_k),
        precision=(np.float16, np.float16, np.float32),
        data_type="real",
        arrangement=("col_major", "row_major", "row_major"),
        alignment=MAX_ALIGNMENT,
        execution="Block",
        block_size=block_size,
        with_pipeline=True,
        enable_input_streaming=True,
        # WAR for the TMA descriptor issue on SM 12.0
        sm=89 if get_current_device_cc().major >= 12 and mathdx.get_version_ex() < (0, 3, 2) else None,
    )

    print(f"SM: {MM.sm}")

    @cuda.jit(extensions=pipeline_extensions, launch_bounds=[MM.block_size, 1])
    def matmul_kernel(alpha, beta, c, device_pipeline: DevicePipeline):
        smem = cuda.shared.array(shape=(0,), dtype=np.byte, alignment=device_pipeline.buffer_alignment)

        blockIdx = cuda.blockIdx
        c_tile = c[blockIdx.x * tile_m : (blockIdx.x + 1) * tile_m, blockIdx.y * tile_n : (blockIdx.y + 1) * tile_n]
        gmem_c = make_tensor(c_tile, MM.get_layout_gmem_c(n))

        tile_pipeline = device_pipeline.get_tile(smem, (blockIdx.x, np.int32(0)), (blockIdx.y, np.int32(0)))

        accumulator = MM.suggest_accumulator()
        tile_pipeline.execute(accumulator)

        for step in range(1, 4):
            i = np.int32(step % 2)
            j = np.int32(step // 2)
            device_pipeline.reset_tile(tile_pipeline, (blockIdx.x, i), (blockIdx.y, j))
            tile_pipeline.execute(accumulator)

        # External epilogue
        if accumulator.is_thread_active():
            d_frag = accumulator.make_partition_and_copy(gmem_c)
            axpby(alpha, accumulator.get_results(), beta, d_frag)
            accumulator.partition_and_copy(d_frag, gmem_c)

        tile_pipeline._del()

    a = random_real((m, k, steps), MM.a_value_type, order="C" if MM.arrangement.a == "row_major" else "F")
    b0 = random_real((steps, k, n), MM.b_value_type, order="C")
    b = np.moveaxis(b0, 0, -1)
    assert b[:, :, 0].shape == (k, n)
    assert b[:, :, 0].strides[-1] == np.dtype(b.dtype).itemsize, "b tile is not F-contiguous"

    c = random_real((m, n), MM.c_value_type, order="C" if MM.arrangement.c == "row_major" else "F")

    a_d = cuda.to_device(a)
    b_d = cuda.to_device(cupy.moveaxis(cupy.array(b0, order="K"), 0, -1))
    c_d = cuda.to_device(c)

    assert b.strides == b_d.strides

    # ================================
    # Pipeline configuration
    # ================================

    print(f"Pipeline depth: {pipeline_depth}")

    k_stages = k // tile_k
    print(f"K stages: {k_stages}")
    assert k_stages >= pipeline_depth, (
        "PipelineDepth must be less or equal to GEMM k stages, please adjust manual_pipeline_depth"
    )

    device_pipeline = MM.suggest_device_pipeline(pipeline_depth, a_d, b_d)

    print(f"Grid dim: {grid_dim}")
    print(f"Pipeline block dim: {device_pipeline.block_dim}")

    set_max_dynamic_shared_size_bytes(matmul_kernel, device_pipeline.buffer_size, alpha, beta, c_d, device_pipeline)
    matmul_kernel[grid_dim, device_pipeline.block_dim, 0, device_pipeline.buffer_size](alpha, beta, c_d, device_pipeline)
    cuda.synchronize()
    data_test = c_d.copy_to_host()

    mm_sum = sum(
        a[:, :, i].astype(MM.c_value_type) @ b[:, :, j].astype(MM.c_value_type) for i in range(steps) for j in range(steps)
    )
    data_ref = alpha * mm_sum + beta * c

    error = np.linalg.norm(data_test - data_ref) / np.linalg.norm(data_ref)
    print(f"Relative error: {error}")

    assert error < 1e-5, f"Error: {error} > 1e-5"


if __name__ == "__main__":
    main()
