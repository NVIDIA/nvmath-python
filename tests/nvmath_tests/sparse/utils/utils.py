# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0


import contextlib
import functools
import re
from typing import Literal

import numpy as np
import pytest

import nvmath.bindings.cusparse as cusparse
from nvmath._utils import get_nvrtc_version

try:
    from cuda.core import Device, StreamOptions, system
except ImportError:
    from cuda.core.experimental import Device, StreamOptions, system
from .common_axes import (
    Framework,
    copy_array,
    cp,
    device_id_from_array,
    framework_from_array,
    torch,
    value_data_from_array,
)

DEVICE_CC = int(Device().arch)


def get_framework_device_ctx(device_id: int | Literal["cpu"], framework: Framework):
    if device_id == "cpu":
        return contextlib.nullcontext()
    match framework:
        case Framework.numpy | Framework.scipy:
            return contextlib.nullcontext()
        case Framework.cupy | Framework.cupyx:
            if cp is None:
                raise RuntimeError("cupy is not installed")
            return cp.cuda.Device(device_id)
        case Framework.torch:
            if torch is None:
                raise RuntimeError("torch is not installed")
            return torch.cuda.device(device_id)
        case _:
            raise ValueError(f"Unknown framework {framework}")


def multi_gpu_only(fn):
    import pytest

    @functools.wraps(fn)
    def inner(*args, **kwargs):
        try:
            dev_count = system.get_num_devices()
        except AttributeError:
            dev_count = system.num_devices
        if dev_count < 2:
            pytest.skip(f"Test requires at least two gpus, got {dev_count}")
        else:
            return fn(*args, **kwargs)

    return inner


def get_custom_stream(framework: Framework, device_id=None):
    assert device_id is None or (isinstance(device_id, int) and device_id >= 0)
    match framework:
        case Framework.numpy | Framework.scipy:
            return Device(device_id).create_stream(options=StreamOptions(nonblocking=True))
        case Framework.cupy | Framework.cupyx:
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
        case Framework.numpy | Framework.scipy:
            return contextlib.nullcontext()
        case Framework.torch:
            return torch.cuda.stream(stream)
        case _:
            return stream


def to_dense_numpy(a):
    """
    Convert whatever is provided to a dense numpy array.
    """
    if isinstance(a, list):
        a = [to_dense_numpy(item) for item in a]
        assert all(item.shape == a[0].shape for item in a)
        if len(a) > 1:
            return np.stack(a)
        else:
            return a[0]

    match framework_from_array(a):
        case Framework.numpy:
            return np.asarray(a)
        case Framework.cupy:
            return cp.asnumpy(a)
        case Framework.cupyx | Framework.scipy:
            return to_dense_numpy(a.todense())
        case Framework.torch:
            assert a.dtype not in [torch.bfloat16, torch.complex32], (
                "numpy does not support bfloat16 and complex32. Please convert the array before calling this function."
            )

            if a.layout != torch.strided:
                return to_dense_numpy(a.to_dense())

            return a.cpu().numpy()
        case _:
            raise ValueError(f"Unsupported framework: {type(a)}")


def idfn(val):
    """
    Pytest does not pretty print (repr/str) parameters of custom types.
    """
    if hasattr(val, "pretty_name"):
        return val.pretty_name()
    # use default pytest pretty printing
    return None


def is_known_linker_error(message: str) -> bool:
    try:
        if get_nvrtc_version()[0] != 12:
            return False
    except Exception:
        pass

    return "multiply defined" in message and "_ZN6__halfC1E13__nv_bfloat16" in message


class allow_cusparse_unsupported:
    def __init__(self, *, enabled=True):
        self.regex = r"NOT_SUPPORTED \(10\)|ARCH_MISMATCH \(4\)"
        self.enabled = enabled

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, exc_traceback):
        # Unconditionally check for arch mismatch error, irrespective of the enabled flag.
        if re.search(r"ARCH_MISMATCH \(4\)", str(exc_value)):
            pytest.skip(f"cuSPARSE does not support this operation on this device: {str(exc_value)}")

        if not self.enabled:
            return False

        if (exc_type is cusparse.cuSPARSEError and re.search(self.regex, str(exc_value))) or (
            exc_type is RuntimeError and "BSR is not supported by the installed cuSPARSE library." in str(exc_value)
        ):
            pytest.skip(f"cuSPARSE does not support this operation: {str(exc_value)}")

        return False


def transform_sparse_array(a, transform_fn):
    with get_framework_device_ctx(device_id_from_array(a), framework_from_array(a)):
        a1 = copy_array(a)
        a_values = value_data_from_array(a1)
        transformed = transform_fn(a_values)
        if hasattr(a_values, "copy_"):
            a_values.copy_(transformed)
        else:
            a_values[...] = transformed

        return coalesce_array(a1)


def coalesce_array(a):
    fw = framework_from_array(a)

    with get_framework_device_ctx(device_id_from_array(a), fw):
        match fw:
            case Framework.torch:
                if a.layout == torch.sparse_coo:
                    return a.coalesce()
                return a
            case Framework.scipy | Framework.cupyx:
                if hasattr(a, "sum_duplicates"):
                    a.sum_duplicates()
                return a
            case _:
                raise TypeError(f"Unsupported sparse array framework: {fw}")


def shallow_state_snapshot(obj):
    """
    Get unique IDs of all attributes of the object,
    together with assert_snapshot_equal can be used to
    test if some operation overrides any attribute of the object.
    """
    return {k: id(v) for k, v in obj.__dict__.items()}


def ust_snapshot(ust):
    """
    Snapshot of direct attributes of the UST and the wrapped operand.
    """
    snap = shallow_state_snapshot(ust)
    if ust.wrapped_operand is not None:
        for key, value in shallow_state_snapshot(ust.wrapped_operand).items():
            snap[f"wrapped_operand__{key}"] = value
    return snap


def assert_snapshot_equal(current, ref):
    """
    Compare two ``shallow_state_snapshot`` for equality.
    """
    for key in current:
        if key not in ref:
            raise AssertionError(f"The object has new attribute {key} that was not present in the reference")
        if current[key] != ref[key]:
            raise AssertionError(f"The attribute {key} changed")
    for key in ref:
        if key not in current:
            raise AssertionError(f"The object does not have an attribute {key} that was present in the reference")
