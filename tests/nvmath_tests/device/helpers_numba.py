# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

from numba import cuda


def run_and_time(kernel, grid_dim, block_dim, shared_memory_size, ncycles, *args):
    ## Numba
    stream = cuda.stream()
    start, stop = cuda.event(), cuda.event()
    cuda.synchronize()

    # warmup + jit
    kernel[grid_dim, block_dim, stream, shared_memory_size](*args)
    stream.synchronize()

    # time
    start.record(stream)
    for _ in range(ncycles):
        kernel[grid_dim, block_dim, stream, shared_memory_size](*args)
    stop.record(stream)
    stream.synchronize()

    time_ms = cuda.event_elapsed_time(start, stop) / ncycles
    return time_ms


@cuda.jit(inline="always")
def shared_store_3d(smem, gmem, bid, M, N, BLOCK_SIZE):
    for index in range(cuda.threadIdx.x, M * N, BLOCK_SIZE):
        row = index % M
        col = index // M
        # smem is in fortran order
        # gmem is in fortran order
        gmem[bid, row, col] = smem[row + col * M]


@cuda.jit(inline="always")
def shared_load_3d(gmem, smem, bid, M, N, BLOCK_SIZE):
    for index in range(cuda.threadIdx.x, M * N, BLOCK_SIZE):
        row = index % M
        col = index // M
        # smem is in fortran order
        # gmem is in fortran order
        smem[row + col * M] = gmem[bid, row, col]
