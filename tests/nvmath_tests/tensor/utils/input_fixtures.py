# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import contextlib

import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = None
try:
    import torch
except ImportError:
    torch = None

import cuda.core.experimental as ccx


from .common_axes import MemBackend, Framework, DType
from .axes_utils import get_framework_dtype, is_complex


def get_random_input_data(
    framework: Framework,
    shape: int | tuple[int],
    dtype: DType,
    mem_backend: MemBackend,
    seed: int,
    lo: float = -0.5,
    hi: float = 0.5,
    device_id=None,
):
    assert lo < hi
    framework_dtype = get_framework_dtype(framework, dtype)
    if framework in [Framework.numpy, Framework.cupy]:

        def _create_array():
            if framework == Framework.numpy:
                assert mem_backend == MemBackend.cpu
                rng = np.random.default_rng(seed)
            else:
                assert mem_backend == MemBackend.cuda
                rng = cp.random.default_rng(seed)
            if not is_complex(dtype):
                a = rng.uniform(lo, hi, size=shape).astype(framework_dtype)
            else:
                real = rng.uniform(lo, hi, size=shape)
                imag = rng.uniform(lo, hi, size=shape)
                a = (real + 1j * imag).astype(framework_dtype)
            if len(shape) == 0:
                # real + 1j * imag will convert this to a scalar object,
                #   not a ndarray, here we convert it back to a ndarray
                if framework == Framework.numpy:
                    a = np.array(a)
                else:
                    a = cp.array(a)
            assert a.dtype == framework_dtype, f"{a.dtype} vs {framework_dtype}"
            assert a.shape == shape, f"{a.shape} vs {shape}"
            return a

        if mem_backend == MemBackend.cuda and device_id is not None:
            with get_framework_device_ctx(device_id, framework):
                return _create_array()
        else:
            return _create_array()

    elif framework == Framework.torch:
        if mem_backend == MemBackend.cpu:
            device = "cpu"
        elif device_id is not None:
            device = f"cuda:{device_id}"
        else:
            device = "cuda"
        g = torch.Generator(device=device)
        g = g.manual_seed(seed)
        t = torch.rand(size=shape, generator=g, device=device, dtype=framework_dtype)
        scale = torch.tensor(hi - lo, dtype=framework_dtype)
        if not is_complex(dtype):
            shift = torch.tensor(lo, dtype=framework_dtype)
        else:
            shift = torch.tensor(lo + 1j * lo, dtype=framework_dtype)
        t = t.mul_(scale).add_(shift)
        assert t.dtype == framework_dtype
        return t
    else:
        raise ValueError(f"Unknown framework {framework}")


def get_custom_stream(framework: Framework, device_id=None, is_numpy_stream_oriented=False):
    if framework == Framework.numpy:
        if is_numpy_stream_oriented:
            old_device = ccx.Device()
            device = ccx.Device(device_id)
            try:
                device.set_current()
                return device.create_stream()
            finally:
                old_device.set_current()
        else:
            return None
    elif framework == Framework.cupy:
        if device_id is None:
            return cp.cuda.Stream(non_blocking=True)
        else:
            with get_framework_device_ctx(device_id, framework):
                return cp.cuda.Stream(non_blocking=True)
    elif framework == Framework.torch:
        device = None if device_id is None else f"cuda:{device_id}"
        return torch.cuda.Stream(device=device)
    else:
        raise ValueError(f"Unknown GPU framework {framework}")


def get_framework_device_ctx(device_id: int, framework: Framework):
    if framework == Framework.numpy:
        return contextlib.nullcontext()
    elif framework == Framework.cupy:
        return cp.cuda.Device(device_id)
    elif framework == Framework.torch:
        return torch.cuda.device(device_id)
