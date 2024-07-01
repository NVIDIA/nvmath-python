# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

#
# Mirrors https://github.com/NVIDIA/CUDALibrarySamples/blob/master/MathDx/cuFFTDx/simple_fft_block_r2c.cu
#

import numpy as np
from numba import cuda
from nvmath.device import fft

def main():

    FFT = fft(fft_type='r2c', size=128, precision=np.float32, elements_per_thread=8, ffts_per_block=2, execution='Block', compiler='numba')

    fft_size            = FFT.size
    value_type          = FFT.value_type
    storage_size        = FFT.storage_size
    shared_memory_size  = FFT.shared_memory_size
    stride              = FFT.stride
    block_dim           = FFT.block_dim
    ffts_per_block      = FFT.ffts_per_block
    elements_per_thread = FFT.elements_per_thread


    @cuda.jit(link=FFT.files)
    def f(input, output):

        thread_data = cuda.local.array(shape=(storage_size,), dtype=value_type)
        thread_data_real = thread_data.view(np.float32)
        shared_mem = cuda.shared.array(shape=(0,), dtype=value_type)

        local_fft_id = cuda.threadIdx.y

        index = cuda.threadIdx.x
        for i in range(elements_per_thread):
            if index < fft_size:
                thread_data_real[i] = input[local_fft_id, index]
            index += stride

        FFT(thread_data, shared_mem)

        index = cuda.threadIdx.x
        for i in range(elements_per_thread):
            if index < fft_size // 2 + 1:
                output[local_fft_id, index] = thread_data[i]
            index += stride

    input = np.ones((ffts_per_block, fft_size), dtype=np.float32)
    output = np.zeros((ffts_per_block, fft_size // 2 + 1), dtype=np.complex64)
    input_d = cuda.to_device(input)
    output_d = cuda.to_device(output)

    print("input [1st FFT]:", input[0,:])

    f[1, block_dim, 0, shared_memory_size](input_d, output_d)
    cuda.synchronize()

    output_test = output_d.copy_to_host()

    print("output [1st FFT]:", output_test[0,:])

    data_ref = np.fft.rfft(input, axis=-1)
    error = np.linalg.norm(output_test - data_ref) / np.linalg.norm(data_ref)
    assert error < 1e-5

if __name__ == "__main__":
    main()