# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import datetime

import cupy
import numpy as np
import math
import hypothesis


def nvmath_seed():
    """A decorator which sets a hypothesis seed to a hash of today's date.

    Setting the seed once per day allows random exploration of the parameter
    space, but having all runners in a pipeline use the same seed helps
    identify when failures are device/environment related.
    """
    return hypothesis.seed(hash(datetime.date.today()))


def numpy_type_to_str(np_dtype):
    if np_dtype == np.float16:
        return "float16"
    elif np_dtype == np.float32:
        return "float32"
    elif np_dtype == np.float64:
        return "float64"
    else:
        raise AssertionError()


def time_cupy(fun, ncycles, *args):
    args = [(cupy.array(arg) if isinstance(arg, np.ndarray | np.generic) else arg) for arg in args]
    start, stop = cupy.cuda.Event(), cupy.cuda.Event()
    out = fun(*args)

    start.record(None)
    for _ in range(ncycles):
        out = fun(*args)  # noqa: F841
    stop.record(None)
    stop.synchronize()

    t_cupy_ms = cupy.cuda.get_elapsed_time(start, stop) / ncycles

    return {"time_ms": t_cupy_ms}


def random_complex(shape, real_dtype, module=np):
    return module.random.randn(*shape).astype(real_dtype) + 1.0j * module.random.randn(*shape).astype(real_dtype)


# in: data_in = list(dict{k:v})
# out: dict{k: list(v)}
def transpose(data_in):
    headers = set()
    for d in data_in:
        for h in d:
            headers.add(h)
    headers = list(headers)

    data_out = {}
    for h in headers:
        data_out[h] = []

    for d in data_in:
        for h in headers:
            assert h in d
            data_out[h].append(d[h])

    return data_out


# headers = list(str)
# data = list(dict{ header : value })
def print_aligned_table(headers, data, print_headers=True):
    rows = transpose(data)

    assert len(headers) > 0
    nrows = len(rows[headers[0]])
    for h in headers:
        assert len(rows[h]) == nrows

    def convert(x):
        if isinstance(x, int):
            x = f"{x:>6d}"
        if isinstance(x, float):
            s = f"{x:>3.2e}"
        elif isinstance(x, str):
            s = x
        elif isinstance(x, type):
            s = numpy_type_to_str(x)
        else:
            print(x)
            raise AssertionError()
        return s

    for h in headers:
        rows[h] = [convert(x) for x in rows[h]]

    col_width = [max(len(str(h)), max(len(x) for x in rows[h])) + 1 for h in headers]

    if print_headers:
        headers_str = [f"{str(h):>{c}}" for h, c in zip(headers, col_width, strict=True)]
        print(",".join(headers_str))

    for row in range(nrows):
        row_str = [f"{rows[h][row]:>{c}}" for h, c in zip(headers, col_width, strict=True)]
        print(",".join(row_str))


def fft_conv_perf_GFlops(fft_size, batch, time_ms):
    fft_flops_per_batch = 5.0 * fft_size * math.log2(fft_size)
    return batch * (2.0 * fft_flops_per_batch + fft_size) / (1e-3 * time_ms) / 1e9


def fft_perf_GFlops(fft_size, batch, time_ms):
    fft_flops_per_batch = 5.0 * fft_size * math.log2(fft_size)
    return batch * fft_flops_per_batch / (1e-3 * time_ms) / 1e9


def matmul_flops(m, n, k, dtype):
    flopsCoef = 8 if np.issubdtype(dtype, np.complexfloating) else 2
    return flopsCoef * m * n * k


def matmul_perf_GFlops(m, n, k, time_ms, dtype=np.float64):
    flops = matmul_flops(m, n, k, dtype)
    return flops / (1e-3 * time_ms) / 1e9
