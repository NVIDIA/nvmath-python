# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import cupy

from nvmath.device import TransposeMode
from .cpp_gemm_loop import MatmulLoopCpp
from .cpp_gemm_batched import MatmulBatchedCpp
from .numba_gemm_loop import NumbaGemmLoop
from .helpers import random_real, set_device, time_check_cupy
from ..helpers import print_aligned_table
from .numba_gemm_batched import NumbaGemmBatched

def test_batched_gemm_perf():

    TEST_CASES = [ \
        ((8,  8,  8),  65536, 1),
        ((8,  8,  8),  65536, 10),
        ((8,  8,  8),  65536, 100),
        ((8,  16, 8),  65536, 1),
        ((8,  16, 8),  65536, 10),
        ((8,  16, 8),  65536, 100),
        ((16, 16, 8),  65536, 1),
        ((16, 16, 8),  65536, 10),
        ((16, 16, 8),  65536, 100),
        ((32, 32, 64), 65536, 1),
        ((32, 32, 64), 65536, 10),
        ((32, 32, 64), 65536, 100),
    ]

    cols = ['m', 'n', 'k', 'batch', 'repeat', 'cupy [ms]', 'Numba [ms]', 'C++ [ms]']

    data = []

    for size, batch, repeat in TEST_CASES:

        print(f"Numba vs cupy host APIs (batched gemms), size {size}, batch {batch}, repeat {repeat}")

        m, n, k = size
        ncycles = 1
        SM = set_device()

        a = random_real((batch, m, k), np.float32, module=cupy)
        b = random_real((batch, k, n), np.float32, module=cupy)
        reference = cupy.einsum('bmk,bkn->bmn', a, b)

        def fun(a, b):
            for _ in range(repeat):
                out = cupy.einsum('bmk,bkn->bmn', a, b)
            return out

        t_cupy = time_check_cupy(fun, reference, ncycles, a, b)
        t_cpp = MatmulBatchedCpp(size=size, precision=np.float32, data_type='real', sm=SM, block_size=32, repeat=repeat).run(a=a, b=b, reference=reference, ncycles=ncycles)
        t_numba = NumbaGemmBatched(size=size, precision=np.float32, data_type='real', block_size=32, repeat=repeat).run(a=a, b=b, reference=reference, ncycles=ncycles)

        data.append({
            'm': m,
            'n': n,
            'k': k,
            'batch': batch,
            'repeat': repeat,
            'cupy [ms]': t_cupy['time_ms'],
            'Numba [ms]': t_numba['time_ms'],
            'C++ [ms]': t_cpp['time_ms'],
        })

    print_aligned_table(cols, data)

def test_gemm_loop_perf():

    TEST_CASES = [ \
        ((4, 4, 4), 100000),
        ((8, 8, 8), 100000),
        ((16, 16, 16), 100000),
        ((32, 32, 32), 10000)
    ]

    cols = ['m', 'n', 'k', 'repeat', 'Numba [ms]', 'C++ [ms]']

    data = []

    for size, repeat in TEST_CASES:

        print(f"Numba vs CUDA C++ (gemm loop), size = {size}, repeat = {repeat}")

        SM = set_device()
        m, n, k = size
        block_size = 128
        trans = TransposeMode('non_transposed', 'non_transposed')
        a = cupy.ones((m, k), dtype=np.float32)
        b = cupy.ones((k, n), dtype=np.float32)
        reference = a.dot(b)

        args = {
            'size':(m, n, k),
            'precision':np.float32,
            'data_type':'real',
            'transpose_mode':trans,
            'block_size':block_size,
        }

        run_args = {
            'a':a,
            'b':b,
            'reference':reference,
            'ncycles':10
        }

        mm_cpp = MatmulLoopCpp(**args, sm=SM, repeat=repeat).run(**run_args)
        mm_cpp_2 = MatmulLoopCpp(**args, sm=SM, repeat=2 * repeat).run(**run_args)

        print("------------------------------")

        mm_numba = NumbaGemmLoop(**args, repeat=repeat).run(**run_args)
        mm_numba_2 = NumbaGemmLoop(**args, repeat=2 * repeat).run(**run_args)

        print("------------------------------")

        time_cpp_ms = mm_cpp_2['time_ms'] - mm_cpp['time_ms']
        time_numba_ms = mm_numba_2['time_ms'] - mm_numba['time_ms']

        print("> CUDA C++ ", time_cpp_ms)
        print("> Numba ", time_numba_ms)

        data.append({
            'm': m,
            'n': n,
            'k': k,
            'repeat': repeat,
            'Numba [ms]': time_numba_ms,
            'C++ [ms]': time_cpp_ms,
        })

    print_aligned_table(cols, data)
