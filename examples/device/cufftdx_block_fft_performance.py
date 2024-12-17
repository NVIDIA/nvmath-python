# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from numba import cuda
from nvmath.device import fft
from common import fft_perf_GFlops, CHECK_CUDART
from common_numba import time_numba, get_active_blocks_per_multiprocessor
from cuda import cudart


def main():
    ncycles = 1
    repeat = 4000
    fft_size = 512
    ffts_per_block = 1
    err, out = cudart.cudaGetDeviceProperties(0)
    CHECK_CUDART(err)
    sms = out.multiProcessorCount
    elements_per_thread = 8

    FFT = fft(
        fft_type="c2c",
        size=fft_size,
        precision=np.float32,
        direction="forward",
        execution="Block",
        elements_per_thread=elements_per_thread,
        ffts_per_block=ffts_per_block,
        compiler="numba",
    )

    value_type = FFT.value_type
    storage_size = FFT.storage_size
    shared_memory_size = FFT.shared_memory_size
    stride = FFT.stride
    block_dim = FFT.block_dim

    @cuda.jit(link=FFT.files)
    def f(data, repeat):
        thread_data = cuda.local.array(shape=(storage_size,), dtype=value_type)
        shared_mem = cuda.shared.array(shape=(0,), dtype=value_type)

        local_fft_id = cuda.threadIdx.y
        fft_id = cuda.blockIdx.x * ffts_per_block + local_fft_id

        index = cuda.threadIdx.x
        for i in range(elements_per_thread):
            thread_data[i] = data[fft_id, index]
            index += stride

        for r in range(repeat):
            FFT(thread_data, shared_mem)

        index = cuda.threadIdx.x
        for i in range(elements_per_thread):
            data[fft_id, index] = thread_data[i]
            index += stride

    dummy = cuda.to_device(np.ones((ffts_per_block, fft_size), dtype=np.complex64))
    blocks_per_sm = get_active_blocks_per_multiprocessor(f, block_dim, shared_memory_size, dummy, repeat)
    batch_size = sms * blocks_per_sm * ffts_per_block

    grid_dim = batch_size // ffts_per_block
    assert batch_size % ffts_per_block == 0

    data = np.ones((batch_size, fft_size), dtype=np.complex64)
    data_d = cuda.to_device(data)

    time_ms = time_numba(f, grid_dim, block_dim, shared_memory_size, ncycles, data_d, repeat)
    time_2x_ms = time_numba(f, grid_dim, block_dim, shared_memory_size, ncycles, data_d, 2 * repeat)
    time_fft_ms = (time_2x_ms - time_ms) / repeat
    perf = fft_perf_GFlops(fft_size, batch_size, time_fft_ms)

    print(
        f"#SMs {sms}\nBlocks per SM {blocks_per_sm}\nFFts per block {ffts_per_block}\nBatch size {batch_size}"
        f"\nTime {time_fft_ms} ms\nPerf {perf} GFlop/s"
    )


if __name__ == "__main__":
    main()
