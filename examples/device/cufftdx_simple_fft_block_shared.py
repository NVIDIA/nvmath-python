# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

#
# Mirrors https://github.com/NVIDIA/CUDALibrarySamples/blob/master/MathDx/cuFFTDx/simple_fft_block_shared.cu
#

import numpy as np
from numba import cuda
from nvmath.device import fft


def main():
    FFT = fft(
        fft_type="c2c",
        size=128,
        precision=np.float32,
        direction="forward",
        elements_per_thread=8,
        ffts_per_block=2,
        execution="Block",
        compiler="numba",
        execute_api="shared_memory",
    )

    size = FFT.size
    value_type = FFT.value_type
    shared_memory_size = max(FFT.shared_memory_size, np.complex64(1.0).itemsize * size)
    stride = FFT.stride
    block_dim = FFT.block_dim
    ffts_per_block = FFT.ffts_per_block
    elements_per_thread = FFT.elements_per_thread

    @cuda.jit(link=FFT.files)
    def f(data):
        shared_mem = cuda.shared.array(shape=(0,), dtype=value_type)
        local_fft_id = cuda.threadIdx.y

        index = cuda.threadIdx.x
        for _ in range(elements_per_thread):
            shared_mem[local_fft_id * size + index] = data[local_fft_id, index]
            index += stride

        cuda.syncthreads()

        FFT(shared_mem)

        cuda.syncthreads()

        index = cuda.threadIdx.x
        for _ in range(elements_per_thread):
            data[local_fft_id, index] = shared_mem[local_fft_id * size + index]
            index += stride

    data = np.ones((ffts_per_block, size), dtype=np.complex64)
    data_d = cuda.to_device(data)

    print("input [1st FFT]:", data[0, :])

    f[1, block_dim, 0, shared_memory_size](data_d)
    cuda.synchronize()

    data_test = data_d.copy_to_host()

    print("output [1st FFT]:", data_test[0, :])

    data_ref = np.fft.fft(data, axis=-1)
    error = np.linalg.norm(data_test - data_ref) / np.linalg.norm(data_ref)
    assert error < 1e-5


if __name__ == "__main__":
    main()
