# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

#
# Mirrors https://github.com/NVIDIA/CUDALibrarySamples/blob/master/MathDx/cuFFTDx/convolution_r2c_c2r.cu
#

import numpy as np
from numba import cuda
from nvmath.device import fft
from common import random_real

def main():

    ffts_per_block = 2
    fft_size = 128

    kwargs = {'size':fft_size, 'precision':np.float32, 'elements_per_thread':2, 'ffts_per_block':ffts_per_block, 'execution':'Block', 'compiler':'numba'}
    FFT_fwd = fft(**kwargs, fft_type='r2c')
    FFT_inv = fft(**kwargs, fft_type='c2r')

    value_type          = FFT_fwd.value_type
    storage_size        = max(FFT_fwd.storage_size, FFT_inv.storage_size)
    shared_memory_size  = max(FFT_fwd.shared_memory_size, FFT_inv.shared_memory_size)
    stride              = FFT_fwd.stride
    block_dim           = FFT_fwd.block_dim
    elements_per_thread = FFT_fwd.elements_per_thread
    assert FFT_inv.stride == stride
    assert FFT_inv.block_dim == block_dim
    assert FFT_inv.elements_per_thread == elements_per_thread

    @cuda.jit(link=FFT_fwd.files + FFT_inv.files)
    def f(data):

        thread_data = cuda.local.array(shape=(storage_size,), dtype=value_type)
        shared_mem = cuda.shared.array(shape=(0,), dtype=value_type)
        thread_data_real = thread_data.view(np.float32)

        local_fft_id = cuda.threadIdx.y
        fft_id = cuda.blockIdx.x * ffts_per_block + local_fft_id

        # Data being loaded is real, for we load fft_size real elements per batch
        index = cuda.threadIdx.x
        for i in range(elements_per_thread):
            if index < fft_size:
                thread_data_real[i] = data[fft_id, index]
            index += stride

        FFT_fwd(thread_data, shared_mem)

        # After the first transform, the data is complex, so we have fft_size//2+1 complex elements per batch
        index = cuda.threadIdx.x
        for i in range(elements_per_thread):
            if index < (fft_size // 2 + 1):
                thread_data[i] = thread_data[i] / fft_size
            index += stride

        FFT_inv(thread_data, shared_mem)

        # After the second transform, the data is real again, so we store fft_size real elements per batch
        index = cuda.threadIdx.x
        for i in range(elements_per_thread):
            if index < fft_size:
                data[fft_id, index] = thread_data_real[i]
            index += stride

    data = np.ones_like(random_real((ffts_per_block, fft_size), real_dtype=np.float32))
    data_d = cuda.to_device(data)

    print("input [1st FFT]:", data[0,:])

    f[1, block_dim, 0, shared_memory_size](data_d)
    cuda.synchronize()

    data_test = data_d.copy_to_host()

    print("output [1st FFT]:", data_test[0,:])

    error = np.linalg.norm(data_test - data) / np.linalg.norm(data)
    assert error < 1e-5

if __name__ == "__main__":
    main()
