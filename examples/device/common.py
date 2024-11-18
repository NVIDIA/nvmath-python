# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

from cuda import cudart
import numpy as np
import math


def random_complex(shape, real_dtype):
    return np.random.randn(*shape).astype(real_dtype) + 1.0j * np.random.randn(*shape).astype(real_dtype)


def random_real(shape, real_dtype):
    return np.random.randn(*shape).astype(real_dtype)


def CHECK_CUDART(err):
    if err != cudart.cudaError_t.cudaSuccess:
        err2, str = cudart.cudaGetErrorString(cudart.cudaError_t.cudaSuccess)
        raise RuntimeError(f"CUDArt Error: {str}")


def fft_perf_GFlops(fft_size, batch, time_ms, coef=1.0):
    fft_flops_per_batch = coef * 5.0 * fft_size * math.log2(fft_size)
    return batch * fft_flops_per_batch / (1e-3 * time_ms) / 1e9


def mm_perf_GFlops(size, batch, time_ms, coef=1.0):
    return coef * 2.0 * batch * size[0] * size[1] * size[2] / (1e-3 * time_ms) / 1e9


def fp16x2_to_complex64(data):
    return data[..., ::2] + 1.0j * data[..., 1::2]


def complex64_to_fp16x2(data):
    shape = (*data.shape[:-1], data.shape[-1] * 2)
    output = np.zeros(shape=shape, dtype=np.float16)
    output[..., 0::2] = data.real
    output[..., 1::2] = data.imag
    return output
