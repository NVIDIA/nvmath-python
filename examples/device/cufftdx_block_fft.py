# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

#
# Mirrors https://github.com/NVIDIA/CUDALibrarySamples/blob/master/MathDx/cuFFTDx/block_fft/block_fft.cu
#

import numpy as np
from numba import cuda
from nvmath.device import FFT


def main():
    fft = FFT(fft_type="c2c", size=64, precision=np.float32, direction="forward", execution="Block")

    @cuda.jit
    def f(data):
        thread_data = cuda.local.array(shape=(fft.storage_size,), dtype=fft.value_type)

        local_fft_id = cuda.threadIdx.y
        fft_id = cuda.blockIdx.x * fft.ffts_per_block + local_fft_id

        index = cuda.threadIdx.x
        for i in range(fft.elements_per_thread):
            thread_data[i] = data[fft_id, index]
            index += fft.stride

        shared_mem = cuda.shared.array(shape=(0,), dtype=fft.value_type)
        fft.execute(thread_data, shared_mem)

        index = cuda.threadIdx.x
        for i in range(fft.elements_per_thread):
            data[fft_id, index] = thread_data[i]
            index += fft.stride

    data = np.ones((fft.ffts_per_block, fft.size), dtype=np.complex64)
    data_d = cuda.to_device(data)

    print("input [1st FFT]:", data[0, :])

    f[1, fft.block_dim, 0, fft.shared_memory_size](data_d)
    cuda.synchronize()

    data_test = data_d.copy_to_host()

    print("output [1st FFT]:", data_test[0, :])

    data_ref = np.fft.fft(data, axis=-1)
    error = np.linalg.norm(data_test - data_ref) / np.linalg.norm(data_ref)
    assert error < 1e-5


if __name__ == "__main__":
    main()
