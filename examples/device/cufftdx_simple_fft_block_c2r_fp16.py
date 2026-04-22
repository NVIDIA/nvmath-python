# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

#
# Mirrors https://github.com/NVIDIA/CUDALibrarySamples/blob/master/MathDx/cuFFTDx/simple_fft_block_c2r_fp16.cu
#

import numpy as np
from common import complex64_to_fp16x2, random_real
from numba import cuda

from nvmath.device import FFT, float16x2_type, float16x4


def main():
    fft = FFT(
        fft_type="c2r",
        size=128,
        precision=np.float16,
        ffts_per_block=4,
        elements_per_thread=8,
        execution="Block",
    )

    assert fft.implicit_type_batching == 2
    assert fft.ffts_per_block % fft.implicit_type_batching == 0

    @cuda.jit
    def f(input, output):
        thread_data = cuda.local.array(shape=(fft.storage_size,), dtype=fft.value_type)
        thread_data_real = thread_data.view(float16x2_type)
        shared_mem = cuda.shared.array(shape=(0,), dtype=fft.value_type)

        local_fft_id = cuda.threadIdx.y

        index = cuda.threadIdx.x
        for i in range(fft.elements_per_thread):
            if index < fft.size // 2 + 1:
                r0 = input[2 * local_fft_id, 2 * index + 0]
                i0 = input[2 * local_fft_id, 2 * index + 1]
                r1 = input[2 * local_fft_id + 1, 2 * index + 0]
                i1 = input[2 * local_fft_id + 1, 2 * index + 1]
                thread_data[i] = float16x4(r0, r1, i0, i1)
            index += fft.stride

        fft.execute(thread_data, shared_mem)

        index = cuda.threadIdx.x
        for i in range(fft.elements_per_thread):
            if index < fft.size:
                rr = thread_data_real[i]
                output[2 * local_fft_id, index] = rr.x
                output[2 * local_fft_id + 1, index] = rr.y
            index += fft.stride

    # Numpy has no FP16 complex, so we create a 2xlarger arrays of FP16 reals
    # Each consecutive pair of reals form one logical FP16 complex number
    input = np.fft.rfft(random_real((fft.ffts_per_block, fft.size), real_dtype=np.float32))
    output_fp16 = np.zeros((fft.ffts_per_block, fft.size), dtype=np.float16)
    input_fp16 = complex64_to_fp16x2(input)
    input_d = cuda.to_device(input_fp16)
    output_d = cuda.to_device(output_fp16)

    print("input [1st FFT]:", input[0, :])

    f[1, fft.block_dim, 0, fft.shared_memory_size](input_d, output_d)
    cuda.synchronize()

    data_test = output_d.copy_to_host()

    print("output [1st FFT]:", data_test[0, :])

    data_ref = np.fft.irfft(input, axis=-1, n=fft.size, norm="forward")
    error = np.linalg.norm(data_test - data_ref) / np.linalg.norm(data_ref)
    assert error < 1e-2


if __name__ == "__main__":
    main()
