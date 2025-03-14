# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

#
# Mirrors https://github.com/NVIDIA/CUDALibrarySamples/blob/master/MathDx/cuFFTDx/simple_fft_thread.cu
#

import numpy as np
from numba import cuda
from nvmath.device import fft


def main():
    threads_count = 4

    FFT = fft(fft_type="c2c", size=8, precision=np.float64, direction="forward", execution="Thread", compiler="numba")

    size = FFT.size
    value_type = FFT.value_type
    storage_size = FFT.storage_size
    elements_per_thread = FFT.elements_per_thread

    @cuda.jit(link=FFT.files)
    def f(data):
        thread_data = cuda.local.array(shape=(storage_size,), dtype=value_type)

        local_fft_id = cuda.threadIdx.x

        for i in range(elements_per_thread):
            thread_data[i] = data[local_fft_id, i]

        FFT(thread_data)

        for i in range(elements_per_thread):
            data[local_fft_id, i] = thread_data[i]

    data = np.ones((threads_count, size), dtype=np.complex128)
    data_d = cuda.to_device(data)

    print("input [1st FFT]:", data[0, :])

    f[1, threads_count](data_d)
    cuda.synchronize()

    data_test = data_d.copy_to_host()

    print("output [1st FFT]:", data_test[0, :])

    data_ref = np.fft.fft(data, axis=-1)
    error = np.linalg.norm(data_test - data_ref) / np.linalg.norm(data_ref)
    assert error < 1e-5


if __name__ == "__main__":
    main()
