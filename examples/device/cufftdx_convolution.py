# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

#
# Mirrors https://github.com/NVIDIA/CUDALibrarySamples/blob/master/MathDx/cuFFTDx/convolution/convolution.cu
#

import numpy as np
from numba import cuda
from nvmath.device import fft
from common import random_complex
import functools


def main():
    size = 64
    ffts_per_block = 2
    elements_per_thread = 8

    FFT_base = functools.partial(
        fft,
        fft_type="c2c",
        size=size,
        precision=np.float32,
        ffts_per_block=ffts_per_block,
        elements_per_thread=ffts_per_block,
        execution="Block",
        compiler="numba",
    )
    FFT = FFT_base(direction="forward")
    IFFT = FFT_base(direction="inverse")

    size = FFT.size
    value_type = FFT.value_type
    storage_size = FFT.storage_size
    shared_memory_size = FFT.shared_memory_size
    stride = FFT.stride
    block_dim = FFT.block_dim

    @cuda.jit(link=FFT.files + IFFT.files)
    def f(data):
        thread_data = cuda.local.array(shape=(storage_size,), dtype=value_type)

        local_fft_id = cuda.threadIdx.y
        fft_id = cuda.blockIdx.x * ffts_per_block + local_fft_id

        index = cuda.threadIdx.x
        for i in range(elements_per_thread):
            thread_data[i] = data[fft_id, index]
            index += stride

        shared_mem = cuda.shared.array(shape=(0,), dtype=value_type)
        FFT(thread_data, shared_mem)

        for i in range(elements_per_thread):
            thread_data[i] = thread_data[i] / size

        IFFT(thread_data, shared_mem)

        index = cuda.threadIdx.x
        for i in range(elements_per_thread):
            data[fft_id, index] = thread_data[i]
            index += stride

    data = random_complex((ffts_per_block, size), real_dtype=np.float32)
    data_d = cuda.to_device(data)

    print("input [1st FFT]:", data[0, :])

    f[1, block_dim, 0, shared_memory_size](data_d)
    cuda.synchronize()

    data_test = data_d.copy_to_host()

    print("output [1st FFT]:", data_test[0, :])

    error = np.linalg.norm(data_test - data) / np.linalg.norm(data)
    assert error < 1e-5


if __name__ == "__main__":
    main()
