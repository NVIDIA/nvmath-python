# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

from cuda import cuda
from .helpers import CHECK_CUDA, _TOLERANCE, l2error, convert_to_cuda_array, free_array, copy_to_cupy
import numpy as np
from .helpers_cpp import run_and_time, compile_cpp_kernel
import cupy


class MatmulLoopCpp:
    def __init__(self, size, precision, data_type, sm, transpose_mode, block_size, repeat):
        m, n, k = size
        assert precision == np.float32
        assert data_type == "real"
        assert sm[0] >= 7
        assert sm[1] >= 0

        #
        # Generate C++
        #

        cpp = f"""\
        #include <cublasdx.hpp>
        using namespace cublasdx;

        using GEMM = decltype( Size< { m }, { n }, { k } >()
                               + Precision<float>()
                               + Type< type::{ data_type }>()
                               + Function<function::MM>()
                               + TransposeMode< transpose_mode::{ transpose_mode.a }, transpose_mode::{ transpose_mode.b }>()
                               + BlockDim<{ block_size }>()
                               + Block()
                               + SM<{ sm[0] * 100 + sm[1] * 10 }>()
                              );

        __device__ const unsigned int shared_memory_size = GEMM::shared_memory_size;

        __global__ void kernel(void* a_void,
                               void* b_void,
                               void* c_void) {{

            const GEMM::value_type* a = (const GEMM::value_type*) a_void;
            const GEMM::value_type* b = (const GEMM::value_type*) b_void;
            GEMM::value_type* c = (GEMM::value_type*) c_void;

            using value_type = GEMM::value_type;
            extern __shared__ __align__(16) char smem[];

            value_type* smem_a = reinterpret_cast<value_type*>(smem);
            value_type* smem_b = reinterpret_cast<value_type*>(smem) + GEMM::a_size;
            value_type* smem_c = reinterpret_cast<value_type*>(smem) + GEMM::a_size + GEMM::b_size;

            if ((threadIdx.x == 0) && (threadIdx.y == 0) && (threadIdx.z == 0)) {{
                for (unsigned int idx = 0; idx < GEMM::a_size; ++idx) {{
                    smem_a[idx] = a[idx];
                }}
                for (unsigned int idx = 0; idx < GEMM::b_size; ++idx) {{
                    smem_b[idx] = b[idx];
                }}
                for (unsigned int idx = 0; idx < GEMM::c_size; ++idx) {{
                    smem_c[idx] = 0.0;
                }}
            }}
            __syncthreads();

            for (unsigned int i = 0; i < { repeat }; i++) {{
                GEMM().execute(1.0, smem_a, smem_b, 0.0, smem_c);
            }}

            __syncthreads();
            if ((threadIdx.x == 0) && (threadIdx.y == 0) && (threadIdx.z == 0)) {{
                for (unsigned int idx = 0; idx < GEMM::c_size; ++idx) {{
                    c[idx] = smem_c[idx];
                }}
            }}
        }}
        """

        #
        # Compile kernel to SASS using NVRTC
        #

        # Get mangled name
        # This will change if the function arguments type or count change
        # To figure this out:
        # - dump cubin to file, say ${CUBIN} (see below)
        # - cuobjdump --dump-resource-usage ${CUBIN}
        mangled = "_Z6kernelPvS_S_"

        self._module, self._kernel, self._shared_memory_size = compile_cpp_kernel(cpp, sm, mangled)
        self._precision = precision
        self._size = size
        self._repeat = repeat
        self._block_dim = (block_size, 1, 1)

    def run(self, a, b, reference, ncycles):
        assert a.shape == (self._size[0], self._size[2])
        assert b.shape == (self._size[2], self._size[1])
        assert reference.shape == (self._size[0], self._size[1])
        print(f"MatmulLoopCpp ncycles {ncycles}")

        # Create input
        c = cupy.zeros(reference.shape, dtype=self._precision)
        dA = convert_to_cuda_array(a)
        dB = convert_to_cuda_array(b)
        dC = convert_to_cuda_array(c)

        time_ms = run_and_time(self._kernel, (1, 1, 1), self._block_dim, self._shared_memory_size, ncycles, dA, dB, dC)

        copy_to_cupy(dC, c)
        free_array(dA)
        free_array(dB)
        free_array(dC)

        error = l2error(c, reference, module=cupy)
        print(f"MatmulLoopCpp CUDA C++ L2 error = {error}")
        print(f"MatmulLoopCpp CUDA C++ Time per kernel = {time_ms}")
        assert error < _TOLERANCE[np.float32]

        return {"time_ms": time_ms}

    def __del__(self):
        (err,) = cuda.cuModuleUnload(self._module)
        CHECK_CUDA(err)
