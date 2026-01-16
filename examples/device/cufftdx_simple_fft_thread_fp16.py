# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

#
# Mirrors https://github.com/NVIDIA/CUDALibrarySamples/blob/master/MathDx/cuFFTDx/simple_fft_thread_fp16.cu
#

import numpy as np
from numba import cuda
from nvmath.device import FFT, float16x4
from common import random_complex, fp16x2_to_complex64, complex64_to_fp16x2


def main():
    threads_count = 3

    fft = FFT(fft_type="c2c", size=8, precision=np.float16, direction="forward", execution="Thread")

    assert fft.implicit_type_batching == 2

    @cuda.jit
    def f(data):
        thread_data = cuda.local.array(shape=(fft.storage_size,), dtype=fft.value_type)

        local_fft_id = cuda.threadIdx.x

        for i in range(fft.elements_per_thread):
            r0 = data[2 * local_fft_id, 2 * i + 0]
            i0 = data[2 * local_fft_id, 2 * i + 1]
            r1 = data[2 * local_fft_id + 1, 2 * i + 0]
            i1 = data[2 * local_fft_id + 1, 2 * i + 1]
            thread_data[i] = float16x4(r0, r1, i0, i1)

        fft.execute(thread_data)

        for i in range(fft.elements_per_thread):
            rrii = thread_data[i]
            r0, r1, i0, i1 = rrii.x, rrii.y, rrii.z, rrii.w
            data[2 * local_fft_id, 2 * i + 0] = r0
            data[2 * local_fft_id, 2 * i + 1] = i0
            data[2 * local_fft_id + 1, 2 * i + 0] = r1
            data[2 * local_fft_id + 1, 2 * i + 1] = i1

    # Numpy has no FP16 complex, so we create a 2xlarger arrays of FP16 reals
    # Each consecutive pair of reals form one logical FP16 complex number
    data = random_complex((fft.implicit_type_batching * threads_count, fft.size), real_dtype=np.float32)
    data_fp16 = complex64_to_fp16x2(data)
    data_d = cuda.to_device(data_fp16)

    print("input [1st FFT]:", data[0, :])

    f[1, threads_count](data_d)
    cuda.synchronize()

    data_test = fp16x2_to_complex64(data_d.copy_to_host())

    print("output [1st FFT]:", data_test[0, :])

    data_ref = np.fft.fft(data, axis=-1)
    error = np.linalg.norm(data_test - data_ref) / np.linalg.norm(data_ref)
    assert error < 1e-3


if __name__ == "__main__":
    main()
