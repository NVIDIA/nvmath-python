# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from numba import cuda

from nvmath.device import fft, float32x2_type, float64x2_type
from .helpers import _TOLERANCE, l2error
import cupy
import time
import functools
from .helpers_numba import run_and_time


class FFTConvNumba:
    def __init__(self, size, precision, fft_type, ffts_per_block, elements_per_thread, use_vectorized_load_store):
        self._use_vectorized_load_store = use_vectorized_load_store
        assert precision in [np.float32, np.float64]
        assert fft_type == "c2c"

        make_fft = functools.partial(
            fft,
            fft_type="c2c",
            size=size,
            precision=precision,
            execution="Block",
            compiler="numba",
            ffts_per_block=ffts_per_block,
            elements_per_thread=elements_per_thread,
        )

        start = time.time()
        FWD = make_fft(direction="forward")
        INV = make_fft(direction="inverse")
        end = time.time()

        self.t_numba_jit_s = end - start

        complex_type = FWD.value_type
        storage_size = FWD.storage_size
        shared_memory_size = FWD.shared_memory_size
        ffts_per_block = FWD.ffts_per_block
        stride = FWD.stride
        elements_per_thread = FWD.elements_per_thread
        block_dim = FWD.block_dim

        assert FWD.value_type == INV.value_type
        assert FWD.storage_size == INV.storage_size
        assert FWD.shared_memory_size == INV.shared_memory_size
        assert FWD.ffts_per_block == INV.ffts_per_block
        assert FWD.stride == INV.stride
        assert FWD.size == INV.size
        assert FWD.elements_per_thread == INV.elements_per_thread
        assert FWD.block_dim == INV.block_dim
        assert FWD.elements_per_thread == INV.elements_per_thread
        assert FWD.ffts_per_block == INV.ffts_per_block

        assert FWD.size == size
        assert FWD.ffts_per_block == ffts_per_block
        assert FWD.elements_per_thread == elements_per_thread
        if precision == np.float32:
            assert complex_type == float32x2_type
        else:
            assert complex_type == float64x2_type
        assert all(code.endswith(".ltoir") for code in FWD.files + INV.files)

        @cuda.jit(link=FWD.files + INV.files)
        def f(input, output, filter):
            if use_vectorized_load_store:
                input_fp32x2 = input.view(complex_type)
                output_fp32x2 = output.view(complex_type)
                filter_fp32x2 = filter.view(complex_type)

            thread_data = cuda.local.array(shape=(storage_size,), dtype=complex_type)
            shared_mem = cuda.shared.array(shape=(0,), dtype=complex_type)

            local_fft_id = cuda.threadIdx.y
            global_fft_id = (cuda.blockIdx.x * ffts_per_block) + local_fft_id

            # Load data
            for i in range(elements_per_thread):
                idx = i * stride + cuda.threadIdx.x
                if idx < size:
                    if use_vectorized_load_store:
                        thread_data[i] = input_fp32x2[global_fft_id * size + idx]
                    else:
                        thread_data[i] = input[global_fft_id, idx]

            # Execute FFT
            FWD(thread_data, shared_mem)

            # Pointwise multiplication
            for i in range(elements_per_thread):
                idx = i * stride + cuda.threadIdx.x
                if idx < size:
                    if use_vectorized_load_store:
                        thread_data[i] = thread_data[i] * filter_fp32x2[global_fft_id * size + idx]
                    else:
                        thread_data[i] = thread_data[i] * filter[global_fft_id, idx]

            # Inverse FFT
            INV(thread_data, shared_mem)

            # Save results
            for i in range(elements_per_thread):
                idx = i * stride + cuda.threadIdx.x
                if idx < size:
                    if use_vectorized_load_store:
                        output_fp32x2[global_fft_id * size + idx] = thread_data[i]
                    else:
                        output[global_fft_id, idx] = thread_data[i]

        self._kernel = f
        self._ffts_per_block = ffts_per_block
        self._size = size
        self._block_dim = block_dim
        self._shared_memory_size = shared_memory_size

    def run(self, input, filter, reference, ncycles):
        (batch, ssize) = input.shape
        assert ssize == self._size
        assert batch % self._ffts_per_block == 0
        input_d = cuda.to_device(input)
        filter_d = cuda.to_device(filter)
        output_numba_d = cuda.to_device(cupy.zeros_like(input))

        grid_dim = (batch // self._ffts_per_block, 1, 1)

        if self._use_vectorized_load_store:
            args = (input_d.reshape((-1,)), output_numba_d.reshape((-1,)), filter_d.reshape((-1,)))
        else:
            args = (input_d, output_numba_d, filter_d)

        time_ms = run_and_time(self._kernel, grid_dim, self._block_dim, self._shared_memory_size, ncycles, *args)

        error = l2error(test=cupy.array(output_numba_d), ref=reference, module=cupy)

        print(f"FFTConvNumba size {self._size}, vectorized ? {'yes' if self._use_vectorized_load_store else 'no'}")
        print(f"FFTConvNumba Numba L2 error = {error}")
        print(f"FFTConvNumba Numba Time per kernel = {time_ms}")
        assert error < _TOLERANCE[np.float32]

        return {"time_ms": time_ms, "jit_ms": self.t_numba_jit_s}
