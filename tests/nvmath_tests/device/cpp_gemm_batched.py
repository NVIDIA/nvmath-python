# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

from cuda import cuda
from .helpers import CHECK_CUDA, _TOLERANCE, l2error, convert_to_cuda_array, free_array, copy_to_cupy
import numpy as np
from .helpers_cpp import run_and_time, compile_cpp_kernel
import cupy


class MatmulBatchedCpp:
    def __init__(self, size, precision, data_type, sm, block_size, repeat):
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
                               + TransposeMode< transpose_mode::non_transposed, transpose_mode::non_transposed>()
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

            const unsigned bid = blockIdx.x;
            constexpr unsigned m = { m };
            constexpr unsigned n = { n };
            constexpr unsigned k = { k };
            constexpr unsigned lda = GEMM::lda;
            constexpr unsigned ldb = GEMM::ldb;
            constexpr unsigned ldc = GEMM::ldc;
            constexpr unsigned block_size = { block_size };

            // Global is row major
            // Shared is column major

            for(unsigned i = threadIdx.x ; i < m * k ; i += block_size) {{
                unsigned row = i / k;
                unsigned col = i % k;
                smem_a[row + col * lda] = a[bid * m * k + row * k + col];
            }}

            for(unsigned i = threadIdx.x ; i < k * n ; i += block_size) {{
                unsigned row = i / n;
                unsigned col = i % n;
                smem_b[row + col * ldb] = b[bid * k * n + row * n + col];
            }}

            for(unsigned i = threadIdx.x ; i < m * n ; i += block_size) {{
                unsigned row = i / n;
                unsigned col = i % n;
                smem_c[row + col * ldc] = 0;
            }}

            __syncthreads();

            for(int r = 0; r < { repeat }; r++) {{
                GEMM().execute(1.0, smem_a, smem_b, 0.0, smem_c);
            }}

            __syncthreads();

            for(unsigned i = threadIdx.x ; i < m * n ; i += block_size) {{
                unsigned row = i / n;
                unsigned col = i % n;
                c[bid * m * n + row * n + col] = smem_c[row + col * ldc];
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
        self._block_dim = (block_size, 1, 1)

    def run(self, a, b, reference, ncycles):
        batch = a.shape[0]
        m, n, k = self._size
        assert a.shape == (batch, m, k)
        assert b.shape == (batch, k, n)
        assert reference.shape == (batch, m, n)
        print(f"MatmulBatchedCpp ncycles {ncycles}")

        c = cupy.zeros((batch, m, n), dtype=self._precision)
        dA = convert_to_cuda_array(a)
        dB = convert_to_cuda_array(b)
        dC = convert_to_cuda_array(c)

        grid_dim = (batch, 1, 1)

        time_ms = run_and_time(self._kernel, grid_dim, self._block_dim, self._shared_memory_size, ncycles, dA, dB, dC)

        copy_to_cupy(dC, c)
        free_array(dA)
        free_array(dB)
        free_array(dC)

        error = l2error(c, reference, module=cupy)
        print(f"MatmulBatchedCpp CUDA C++ L2 error = {error}")
        print(f"MatmulBatchedCpp CUDA C++ Time per kernel = {time_ms}")
        assert error < _TOLERANCE[self._precision]

        return {"time_ms": time_ms}

    def __del__(self):
        (err,) = cuda.cuModuleUnload(self._module)
        CHECK_CUDA(err)
