# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

from .helpers import l2error, _TOLERANCE
import numpy as np
from nvmath.device import matmul
import cupy
from numba import cuda
from .helpers_numba import run_and_time


class NumbaGemmLoop:
    def __init__(self, size, precision, data_type, transpose_mode, block_size, repeat):
        assert precision == np.float32
        assert data_type == "real"

        MM = matmul(
            size=size,
            data_type="real",
            precision=np.float32,
            transpose_mode=transpose_mode,
            block_size=block_size,
            execution="Block",
            compiler="numba",
        )

        input_type = MM.input_type
        output_type = MM.output_type
        block_dim = MM.block_dim
        block_size = MM.block_size
        shared_memory_size = MM.shared_memory_size
        a_size = MM.a_size
        b_size = MM.b_size
        c_size = MM.c_size
        assert block_dim == (block_size, 1, 1)

        alpha = 1.0
        beta = 0.0
        m, n, k = size
        lda, ldb, ldc = MM.leading_dimension.a, MM.leading_dimension.b, MM.leading_dimension.c

        @cuda.jit(link=MM.files)
        def f(a_global, b_global, c_global):
            # Input/output
            a_smem = cuda.shared.array(shape=(a_size,), dtype=input_type)
            b_smem = cuda.shared.array(shape=(b_size,), dtype=input_type)
            c_smem = cuda.shared.array(shape=(c_size,), dtype=output_type)

            # Load global --> shared
            if cuda.threadIdx.x == 0 and cuda.threadIdx.y == 0 and cuda.threadIdx.z == 0:
                for i in range(m):
                    for j in range(k):
                        a_smem[j * lda + i] = a_global[i, j]  # m x k
                for i in range(k):
                    for j in range(n):
                        b_smem[j * ldb + i] = b_global[i, j]  # k x n
                for i in range(m):
                    for j in range(n):
                        c_smem[j * ldc + i] = 0  # m x n

            cuda.syncthreads()

            # Execute FFT
            for r in range(repeat):
                MM(alpha, a_smem, b_smem, beta, c_smem)

            cuda.syncthreads()

            # Store shared --> global
            if cuda.threadIdx.x == 0 and cuda.threadIdx.y == 0 and cuda.threadIdx.z == 0:
                for i in range(m):
                    for j in range(n):
                        c_global[i, j] = c_smem[j * ldc + i]

        self._precision = precision
        self._kernel = f
        self._size = size
        self._shared_memory_size = shared_memory_size
        self._repeat = repeat
        self._block_dim = block_dim

    def run(self, a, b, reference, ncycles):
        m, n, k = self._size
        assert a.shape == (m, k)
        assert b.shape == (k, n)
        assert reference.shape == (m, n)
        print(f"NumbaGemmLoop ncycles {ncycles}")

        c = cuda.to_device(cupy.zeros((m, n), dtype=self._precision))
        a_d = cuda.to_device(a)
        b_d = cuda.to_device(b)
        c_d = cuda.to_device(c)

        grid_dim = (1, 1, 1)

        time_ms = run_and_time(self._kernel, grid_dim, self._block_dim, 0, ncycles, a_d, b_d, c_d)

        output = cupy.array(c_d)
        error = l2error(test=output, ref=reference)

        print(f"NumbaGemmLoop numba error = {error}")
        print(f"NumbaGemmLoop numba time per kernel = {time_ms}")
        assert error < _TOLERANCE[self._precision]

        return {"time_ms": time_ms}
