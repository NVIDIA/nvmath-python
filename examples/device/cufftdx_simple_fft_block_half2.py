# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

#
# Mirrors https://github.com/NVIDIA/CUDALibrarySamples/blob/master/MathDx/cuFFTDx/simple_fft_block_half2.cu
#

import numpy as np
from numba import cuda
from nvmath.device import fft, float16x4
from common import random_complex, fp16x2_to_complex64, complex64_to_fp16x2

def main():

    FFT = fft(fft_type='c2c', size=128, precision=np.float16, ffts_per_block=4, elements_per_thread=8, direction='forward', execution='Block', compiler='numba')

    size                   = FFT.size
    stride                 = FFT.stride
    value_type             = FFT.value_type
    storage_size           = FFT.storage_size
    elements_per_thread    = FFT.elements_per_thread
    implicit_type_batching = FFT.implicit_type_batching
    ffts_per_block         = FFT.ffts_per_block
    assert implicit_type_batching == 2
    assert ffts_per_block % implicit_type_batching == 0

    @cuda.jit(link=FFT.files)
    def f(data):

        thread_data = cuda.local.array(shape=(storage_size,), dtype=value_type)
        shared_mem = cuda.shared.array(shape=(0,), dtype=value_type)

        local_fft_id = cuda.threadIdx.y

        index = cuda.threadIdx.x
        for i in range(elements_per_thread):
            if index < size:
                r0 = data[2 * local_fft_id,     2 * index + 0]
                i0 = data[2 * local_fft_id,     2 * index + 1]
                r1 = data[2 * local_fft_id + 1, 2 * index + 0]
                i1 = data[2 * local_fft_id + 1, 2 * index + 1]
                thread_data[i] = float16x4(r0, r1, i0, i1)
            index += stride

        FFT(thread_data, shared_mem)

        index = cuda.threadIdx.x
        for i in range(elements_per_thread):
            if index < size:
                rrii = thread_data[i]
                r0, r1, i0, i1 = rrii.x, rrii.y, rrii.z, rrii.w
                data[2 * local_fft_id,     2 * index + 0] = r0
                data[2 * local_fft_id,     2 * index + 1] = i0
                data[2 * local_fft_id + 1, 2 * index + 0] = r1
                data[2 * local_fft_id + 1, 2 * index + 1] = i1
            index += stride

    # Numpy has no FP16 complex, so we create a 2xlarger arrays of FP16 reals
    # Each consecutive pair of reals form one logical FP16 complex number
    data = random_complex((ffts_per_block, size), real_dtype=np.float32)
    data_fp16 = complex64_to_fp16x2(data)
    data_d = cuda.to_device(data_fp16)

    print("input [1st FFT]:", data[0,:])

    f[1, FFT.block_dim, 0, FFT.shared_memory_size](data_d)
    cuda.synchronize()

    data_test = fp16x2_to_complex64(data_d.copy_to_host())

    print("output [1st FFT]:", data_test[0,:])

    data_ref = np.fft.fft(data, axis=-1)
    error = np.linalg.norm(data_test - data_ref) / np.linalg.norm(data_ref)
    assert error < 1e-2

if __name__ == "__main__":
    main()