# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

#
# Mirrors https://github.com/NVIDIA/CUDALibrarySamples/blob/master/MathDx/cuFFTDx/convolution_r2c_c2r.cu
#

import numpy as np
from numba import cuda
from nvmath.device import FFT
from common import random_real


def main():
    ffts_per_block = 2
    fft_size = 128

    kwargs = {
        "size": fft_size,
        "precision": np.float32,
        "elements_per_thread": 2,
        "ffts_per_block": ffts_per_block,
        "execution": "Block",
    }
    fft = FFT(**kwargs, fft_type="r2c")
    ifft = FFT(**kwargs, fft_type="c2r")

    storage_size = max(fft.storage_size, ifft.storage_size)
    shared_memory_size = max(fft.shared_memory_size, ifft.shared_memory_size)
    assert ifft.stride == fft.stride
    assert ifft.block_dim == fft.block_dim
    assert ifft.elements_per_thread == fft.elements_per_thread

    @cuda.jit
    def f(data):
        thread_data = cuda.local.array(shape=(storage_size,), dtype=fft.value_type)
        shared_mem = cuda.shared.array(shape=(0,), dtype=fft.value_type)
        thread_data_real = thread_data.view(np.float32)

        local_fft_id = cuda.threadIdx.y
        fft_id = cuda.blockIdx.x * ffts_per_block + local_fft_id

        # Data being loaded is real, for we load fft_size real elements per batch
        index = cuda.threadIdx.x
        for i in range(fft.elements_per_thread):
            if index < fft_size:
                thread_data_real[i] = data[fft_id, index]
            index += fft.stride

        fft.execute(thread_data, shared_mem)

        # After the first transform, the data is complex, so we have fft_size//2+1 complex
        # elements per batch
        index = cuda.threadIdx.x
        for i in range(fft.elements_per_thread):
            if index < (fft_size // 2 + 1):
                thread_data[i] = thread_data[i] / fft_size
            index += fft.stride

        ifft.execute(thread_data, shared_mem)

        # After the second transform, the data is real again, so we store fft_size real
        # elements per batch
        index = cuda.threadIdx.x
        for i in range(fft.elements_per_thread):
            if index < fft_size:
                data[fft_id, index] = thread_data_real[i]
            index += fft.stride

    data = np.ones_like(random_real((ffts_per_block, fft_size), real_dtype=np.float32))
    data_d = cuda.to_device(data)

    print("input [1st FFT]:", data[0, :])

    f[1, fft.block_dim, 0, shared_memory_size](data_d)
    cuda.synchronize()

    data_test = data_d.copy_to_host()

    print("output [1st FFT]:", data_test[0, :])

    error = np.linalg.norm(data_test - data) / np.linalg.norm(data)
    assert error < 1e-5


if __name__ == "__main__":
    main()
