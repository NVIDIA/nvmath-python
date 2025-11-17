# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from numba import cuda
from cuda.bindings import runtime as cudart
from nvmath.device import FFT
from common import fft_perf_GFlops, CHECK_CUDART
from common_numba import time_numba, get_active_blocks_per_multiprocessor


def run(fft_type, fft_size, direction=None):
    ncycles = 1
    repeat = 4000
    err, out = cudart.cudaGetDeviceProperties(0)
    sms = out.multiProcessorCount
    CHECK_CUDART(err)

    fft = FFT(
        fft_type=fft_type,
        size=fft_size,
        precision=np.float32,
        direction=direction,
        execution="Block",
        ffts_per_block="suggested",
    )

    complex_size = fft_size if fft_type == "c2c" else fft_size // 2 + 1

    @cuda.jit
    def f(data, repeat):
        thread_data = cuda.local.array(shape=(fft.storage_size,), dtype=fft.value_type)
        shared_mem = cuda.shared.array(shape=(0,), dtype=fft.value_type)

        local_fft_id = cuda.threadIdx.y
        fft_id = cuda.blockIdx.x * fft.ffts_per_block + local_fft_id

        index = cuda.threadIdx.x
        for i in range(fft.elements_per_thread):
            if index < complex_size:
                thread_data[i] = data[fft_id, index]
                index += fft.stride

        for r in range(repeat):
            fft.execute(thread_data, shared_mem)

        index = cuda.threadIdx.x
        for i in range(fft.elements_per_thread):
            if index < complex_size:
                data[fft_id, index] = thread_data[i]
                index += fft.stride

    dummy = cuda.to_device(np.ones((fft.ffts_per_block, complex_size), dtype=np.complex64))
    blocks_per_sm = get_active_blocks_per_multiprocessor(f, fft.block_dim, fft.shared_memory_size, dummy, repeat)
    batch_size = sms * blocks_per_sm * fft.ffts_per_block
    grid_dim = batch_size // fft.ffts_per_block
    assert batch_size % fft.ffts_per_block == 0

    data = np.ones((batch_size, complex_size), dtype=np.complex64)
    data_d = cuda.to_device(data)

    time_ms = time_numba(f, grid_dim, fft.block_dim, fft.shared_memory_size, ncycles, data_d, repeat)
    time_2x_ms = time_numba(f, grid_dim, fft.block_dim, fft.shared_memory_size, ncycles, data_d, 2 * repeat)
    time_fft_ms = (time_2x_ms - time_ms) / repeat
    perf = fft_perf_GFlops(fft_size, batch_size, time_fft_ms, coef=1.0 if fft_type == "c2c" else 0.5)

    print(f"{fft_type}, {fft_size}, {perf}, {time_fft_ms}")


def main():
    run("c2c", 512, direction="forward")
    run("c2c", 1024, direction="forward")
    run("c2c", 2048, direction="forward")
    run("c2c", 4096, direction="forward")
    run("c2c", 512, direction="inverse")
    run("c2c", 1024, direction="inverse")
    run("c2c", 2048, direction="inverse")
    run("c2c", 4096, direction="inverse")
    run("r2c", 512)
    run("r2c", 1024)
    run("r2c", 2048)
    run("r2c", 4096)
    run("c2r", 512)
    run("c2r", 1024)
    run("c2r", 2048)
    run("c2r", 4096)


if __name__ == "__main__":
    main()
