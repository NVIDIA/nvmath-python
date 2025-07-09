# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0


import functools
import contextlib

from .common_axes import cp, Framework, torch


def multi_gpu_only(fn):
    import pytest

    @functools.wraps(fn)
    def inner(*args, **kwargs):
        if cp is None:
            pytest.skip("Test requires cupy")
        dev_count = cp.cuda.runtime.getDeviceCount()
        if dev_count < 2:
            pytest.skip(f"Test requires at least two gpus, got {dev_count}")
        else:
            return fn(*args, **kwargs)

    return inner


def get_custom_stream(framework: Framework, device_id=None):
    assert device_id is None or (isinstance(device_id, int) and device_id >= 0)
    match framework:
        case Framework.numpy | Framework.cupy | Framework.cupyx | Framework.scipy:
            if device_id is None:
                return cp.cuda.Stream(non_blocking=True)
            else:
                with cp.cuda.Device(device_id):
                    return cp.cuda.Stream(non_blocking=True)
        case Framework.torch:
            device = None if device_id is None else f"cuda:{device_id}"
            return torch.cuda.Stream(device=device)
        case _:
            raise ValueError(f"Unknown GPU framework {framework}")


def use_stream_or_dummy_ctx(framework: Framework, stream):
    if stream is None:
        return contextlib.nullcontext()
    match framework:
        case Framework.torch:
            return torch.cuda.stream(stream)
        case _:
            return stream
