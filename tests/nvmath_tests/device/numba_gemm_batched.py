# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

from .helpers import l2error, _TOLERANCE
import numpy as np
from nvmath.device import matmul, TransposeMode
import cupy
from numba import cuda
from .helpers_numba import run_and_time, shared_load_3d, shared_store_3d
import time

class NumbaGemmBatched:

    def __init__(self, size, precision, data_type, block_size, repeat):

        assert precision == np.float32
        assert data_type == 'real'

        start = time.time()
        MM = matmul(size=size, data_type='real', precision=np.float32, \
                    transpose_mode=TransposeMode('non_transposed', 'non_transposed'), \
                    block_size=block_size, execution='Block', compiler='numba')
        stop = time.time()
        t_numba_jit_s = stop - start

        (m, n, k) = size

        input_type  = MM.input_type
        output_type = MM.output_type
        block_dim   = MM.block_dim
        block_size  = MM.block_size
        shared_memory_size = MM.shared_memory_size
        a_size      = MM.a_size
        b_size      = MM.b_size
        c_size      = MM.c_size

        alpha = 1.0
        beta = 0.0

        @cuda.jit(link=MM.files)
        def f(a_global, b_global, c_global):

            bid = cuda.blockIdx.x

            # Input/output
            a_smem = cuda.shared.array(shape=(a_size,), dtype=input_type)
            b_smem = cuda.shared.array(shape=(b_size,), dtype=input_type)
            c_smem = cuda.shared.array(shape=(c_size,), dtype=output_type)

            # Load global --> shared
            shared_load_3d(a_global, a_smem, bid, m, k, block_size)
            shared_load_3d(b_global, b_smem, bid, k, n, block_size)
            cuda.syncthreads()

            # Execute FFT
            for r in range(repeat):
                MM(alpha, a_smem, b_smem, beta, c_smem)

            cuda.syncthreads()
            # Store shared --> global
            shared_store_3d(c_smem, c_global, bid, m, n, block_size)


        self._precision = precision
        self._kernel = f
        self._size = size
        self._shared_memory_size = shared_memory_size
        self._block_dim = block_dim

    def run(self, a, b, reference, ncycles):

        batch = a.shape[0]
        m, n, k = self._size
        assert a.shape         == (batch, m, k)
        assert b.shape         == (batch, k, n)
        assert reference.shape == (batch, m, n)
        print(f"NumbaGemmBatched ncycles {ncycles}")

        c = cuda.to_device(cupy.zeros_like(reference))
        a_d = cuda.to_device(a)
        b_d = cuda.to_device(b)
        c_d = cuda.to_device(c)

        grid_dim = (batch, 1, 1)

        time_ms = run_and_time(self._kernel, \
                               grid_dim, \
                               self._block_dim, \
                               0, \
                               ncycles, \
                               a_d, b_d, c_d)

        output = cupy.array(c_d)
        error = l2error(test=output, ref=reference)

        print(f"NumbaGemmLoop numba error = {error}")
        print(f"NumbaGemmLoop numba time per kernel = {time_ms}")
        assert error < _TOLERANCE[self._precision]

        return {'time_ms': time_ms}