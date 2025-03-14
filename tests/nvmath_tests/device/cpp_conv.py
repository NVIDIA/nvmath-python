# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import cupy
from cuda import cuda
from .helpers import CHECK_CUDA, _TOLERANCE, l2error, free_array, convert_to_cuda_array, copy_to_cupy
from .helpers_cpp import run_and_time, compile_cpp_kernel


class FFTConvCpp:
    def __init__(self, size, precision, fft_type, sm, ffts_per_block, elements_per_thread):
        assert size >= 1
        assert precision in [np.float32, np.float64]
        assert fft_type == "c2c"
        assert sm[0] >= 7
        assert sm[1] >= 0
        assert ffts_per_block >= 1
        assert elements_per_thread >= 1

        #
        # Generate C++
        #

        cpp = f"""\

        #include <cufftdx.hpp>
        using namespace cufftdx;

        using Base = decltype(  Size<{ size }>()
                            + Precision<{ 'float' if precision == np.float32 else 'double' }>()
                            + Type<fft_type::{ fft_type }>()
                            + SM<{ sm[0] * 100 + sm[1] * 10 }>()
                            + FFTsPerBlock<{ ffts_per_block }>()
                            + ElementsPerThread<{ elements_per_thread }>()
                            + Block()
                            );

        using Fwd = decltype(  Base()
                            + Direction<fft_direction::forward>()
                            );

        using Inv = decltype(  Base()
                            + Direction<fft_direction::inverse>()
                            );

        static_assert(Fwd::shared_memory_size == Inv::shared_memory_size, "Shared memory size must match");

        __device__ const unsigned int shared_memory_size = Fwd::shared_memory_size;

        __global__ void kernel(void* input_void,
                               void* output_void,
                               void* filter_void) {{
            typename Fwd::value_type* input =  (typename Fwd::value_type*)input_void;
            typename Fwd::value_type* output = (typename Fwd::value_type*)output_void;
            typename Fwd::value_type* filter = (typename Fwd::value_type*)filter_void;
            using complex_type = typename Fwd::value_type;
            using scalar_type  = typename complex_type::value_type;

            complex_type thread_data[Fwd::storage_size];
            extern __shared__ complex_type shared_mem[];

            const unsigned global_fft_id = blockIdx.x * Fwd::ffts_per_block + threadIdx.y;
            const unsigned offset = global_fft_id * { size };

            // Load
            for(unsigned i = 0; i < Fwd::elements_per_thread; i++) {{
                const unsigned idx = threadIdx.x + i * Fwd::stride;
                if(idx < { size }) {{
                    thread_data[i] = input[offset + idx];
                }}
            }}

            // Execute FFT
            Fwd().execute(thread_data, shared_mem);

            // Scale values
            for(unsigned i = 0; i < Fwd::elements_per_thread; i++) {{
                const unsigned idx = threadIdx.x + i * Fwd::stride;
                if(idx < { size }) {{
                    thread_data[i] *= filter[offset + idx];
                }}
            }}

            // Execute inverse FFT
            Inv().execute(thread_data, shared_mem);

            // Save results
            for(unsigned i = 0; i < Fwd::elements_per_thread; i++) {{
                const unsigned idx = threadIdx.x + i * Fwd::stride;
                if(idx < { size }) {{
                    output[offset + idx] = thread_data[i];
                }}
            }}
        }}

        """

        # Get mangled name
        # This will change if the function arguments type or count change
        # To figure this out:
        # - dump cubin to file, say ${CUBIN} (see below)
        # - cuobjdump --dump-resource-usage ${CUBIN}
        mangled = "_Z6kernelPvS_S_"

        self._precision = precision
        self._module, self._kernel, self._shared_memory_size = compile_cpp_kernel(cpp, sm, mangled)
        self._size = size
        self._block_dim = (size // elements_per_thread, ffts_per_block, 1)
        self._ffts_per_block = ffts_per_block

        assert self._block_dim[0] * elements_per_thread == size
        assert self._block_dim[0] * self._block_dim[1] <= 1024

    def run(self, input, filter, reference, ncycles):
        (batch, ssize) = input.shape
        assert input.shape == filter.shape
        assert input.shape == reference.shape
        assert ssize == self._size
        assert batch >= 1
        assert batch % self._ffts_per_block == 0

        num_blocks = batch // self._ffts_per_block
        print(f"FFTConvCpp Batch {batch}, ffts_per_block {self._ffts_per_block}, num_blocks {num_blocks}, ncycles {ncycles}")
        assert num_blocks * self._ffts_per_block == batch

        # Create input
        output = cupy.zeros_like(input)
        dInput = convert_to_cuda_array(input)
        dOutput = convert_to_cuda_array(output)
        dFilter = convert_to_cuda_array(filter)

        # Run and time
        grid_dim = (num_blocks, 1, 1)
        time_ms = run_and_time(
            self._kernel, grid_dim, self._block_dim, self._shared_memory_size, ncycles, dInput, dOutput, dFilter
        )

        copy_to_cupy(dOutput, output)
        free_array(dInput)
        free_array(dOutput)
        free_array(dFilter)

        error = l2error(output, reference, module=cupy)
        print(f"FFTConvCpp CUDA C++ L2 error = {error}")
        print(f"FFTConvCpp CUDA C++ Time per kernel = {time_ms}")
        assert error < _TOLERANCE[self._precision]

        return {"time_ms": time_ms}

    def __del__(self):
        (err,) = cuda.cuModuleUnload(self._module)
        CHECK_CUDA(err)
