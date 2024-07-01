# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import cupy
import nvmath
import numpy as np

from nvmath_tests.helpers import time_cupy, print_aligned_table, matmul_perf_GFlops, matmul_flops

def run_test(data, precision, m, n, k, autotune = False, ncycles = 10):
    A = cupy.random.rand(m,k).astype(precision)
    B = cupy.random.rand(k,n).astype(precision)

    with nvmath.linalg.advanced.Matmul(A, B) as mm:

        mm.plan()
        if autotune:
            mm.autotune()

        time_nvmath = time_cupy(lambda: mm.execute(), ncycles)

    time_cp = time_cupy(lambda: cupy.matmul(A, B), ncycles)

    data.append({
        'precision': precision.__name__,
        'autotune': 'yes' if autotune else 'no',
        'm': m,
        'n': n,
        'k': k,
        'nvmath-python [ms]': time_nvmath['time_ms'],
        'cupy [ms]': time_cp['time_ms'],
        'dataset_size [MiB]': (m*k + k*n + m*n) * precision(1).itemsize / (2**20),
        'cupy [GFlop/s]': matmul_perf_GFlops(m, n, k, time_cp['time_ms'], precision),
        'nvmath-python [GFlop/s]': matmul_perf_GFlops(m, n, k, time_nvmath['time_ms'], precision),
        'speedup nvmath-python over cupy': time_cp['time_ms'] / time_nvmath['time_ms']
    })

    return data


def test_matmul_perf():

    data = []

    for precision in [np.float32, np.float16]:
        for m in [2 ** i for i in range(14)]:
            for k in [m, m//2, m//4]:
                n = m
                if matmul_flops(m, n, k, precision) < 1e8: continue # skip small cases

                run_test(data, precision, m, n, k)

    # Extra cases where autotuning was observed to result in a large speedup
    run_test(data, np.float32, 1085,  2219,  2438,  autotune = True) # observed on A100
    run_test(data, np.float32, 7672,  149,   9903,  autotune = True) # observed on A100
    run_test(data, np.float32, 1217,  2024,  4993,  autotune = True) # observed on A100
    run_test(data, np.float32, 413,   732,   61981, autotune = True) # observed on A100
    run_test(data, np.float32, 10777, 1352,  2407,  autotune = True) # observed on H100
    run_test(data, np.float32, 14320, 307,   7266,  autotune = True) # observed on H100
    run_test(data, np.float32, 1405,  399,   71663, autotune = True) # observed on A100
    run_test(data, np.float32, 9759,  1104,  5335,  autotune = True) # observed on H100
    run_test(data, np.float32, 11741, 543,   8863,  autotune = True) # observed on H100
    run_test(data, np.float32, 423,   2144,  55458, autotune = True) # observed on A100
    run_test(data, np.float32, 640,   2689,  49794, autotune = True) # observed on A100
    run_test(data, np.float32, 10277, 576,   20463, autotune = True) # observed on H100
    run_test(data, np.float32, 6710,  12849, 2262,  autotune = True) # observed on H100
    # run_test(data, np.float32, 6156,  8722,  8290,  autotune = True) # observed on A100

    print("\n")
    cols = ['precision', 'autotune', 'm', 'n', 'k', 'dataset_size [MiB]', 'cupy [ms]', 'nvmath-python [ms]', 'cupy [GFlop/s]', 'nvmath-python [GFlop/s]', 'speedup nvmath-python over cupy']
    print_aligned_table(cols, data)