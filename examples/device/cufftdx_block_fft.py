# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

#
# Mirrors https://github.com/NVIDIA/CUDALibrarySamples/blob/master/MathDx/cuFFTDx/block_fft/block_fft.cu
#

import numpy as np
from numba import cuda
from nvmath.device import fft

def main():

    size = 64

    FFT = fft(fft_type='c2c', size=size, precision=np.float32, direction='forward', execution='Block', compiler='numba')

    size                = FFT.size
    value_type          = FFT.value_type
    storage_size        = FFT.storage_size
    shared_memory_size  = FFT.shared_memory_size
    stride              = FFT.stride
    block_dim           = FFT.block_dim
    ffts_per_block      = FFT.ffts_per_block
    elements_per_thread = FFT.elements_per_thread


    @cuda.jit(link=FFT.files)
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

        index = cuda.threadIdx.x
        for i in range(elements_per_thread):
            data[fft_id, index] = thread_data[i]
            index += stride

    data = np.ones((ffts_per_block, size), dtype=np.complex64)
    data_d = cuda.to_device(data)

    print("input [1st FFT]:", data[0,:])

    f[1, block_dim, 0, shared_memory_size](data_d)
    cuda.synchronize()

    data_test = data_d.copy_to_host()

    print("output [1st FFT]:", data_test[0,:])

    data_ref = np.fft.fft(data, axis=-1)
    error = np.linalg.norm(data_test - data_ref) / np.linalg.norm(data_ref)
    assert error < 1e-5

if __name__ == "__main__":
    main()