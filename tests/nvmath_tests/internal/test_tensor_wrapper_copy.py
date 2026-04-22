# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
Self-contained tests for the copy_ function in nvmath.internal.tensor_wrapper.

Tests all admissible combinations of:
- Source memory spaces: CPU, GPU
- Destination memory spaces: CPU, GPU
- Frameworks: numpy, cupy, torch, ndbuffer
"""

from enum import Enum

import numpy as np
import pytest

from nvmath.internal import utils
from nvmath.internal.ndbuffer import ndbuffer
from nvmath.internal.tensor_ifc_ndbuffer import NDBufferTensor
from nvmath.internal.tensor_ifc_numpy import NumpyTensor
from nvmath.internal.tensor_wrapper import copy_, maybe_register_package, wrap_operand
from nvmath.internal.utils import get_or_create_stream
from nvmath_tests.helpers import (
    get_custom_stream,
    get_framework_device_ctx,
    get_random_input_data,
)

try:
    import cupy as cp
except ImportError:
    cp = None

try:
    import torch
except ImportError:
    torch = None

# When this test file was written, unfortunately, a lot of boilerplate
# code is needed to test the copy_ function because we need yet have
# a common place for several of these utilities inside the test suite.


class Framework(Enum):
    numpy = 1
    cupy = 2
    torch = 3
    ndbuffer = 4

    @classmethod
    def enabled(cls):
        """Yield only enabled frameworks based on availability."""
        yield cls.ndbuffer
        yield cls.numpy
        if cp is not None:
            yield cls.cupy
        if torch is not None:
            yield cls.torch


class MemBackend(Enum):
    cpu = 1
    cuda = 2


class DType(Enum):
    float32 = 1
    float64 = 2
    complex64 = 3
    complex128 = 4


# Helper functions
def get_framework_dtype(framework: Framework, dtype: DType):
    """Convert DType enum to framework-specific dtype."""
    dtype_map = {
        DType.float32: {
            "ndbuffer": np.float32,
            "numpy": np.float32,
            "cupy": np.float32,
            "torch": torch.float32 if torch else None,
        },
        DType.float64: {
            "ndbuffer": np.float64,
            "numpy": np.float64,
            "cupy": np.float64,
            "torch": torch.float64 if torch else None,
        },
        DType.complex64: {
            "ndbuffer": np.complex64,
            "numpy": np.complex64,
            "cupy": np.complex64,
            "torch": torch.complex64 if torch else None,
        },
        DType.complex128: {
            "ndbuffer": np.complex128,
            "numpy": np.complex128,
            "cupy": np.complex128,
            "torch": torch.complex128 if torch else None,
        },
    }
    return dtype_map[dtype][framework.name]


def to_numpy(arr):
    if isinstance(arr, np.ndarray):
        return arr

    if isinstance(arr, NDBufferTensor):
        if arr.device == "cpu":
            return NumpyTensor.create_host_from(arr, stream_holder=None).tensor
        device_id = arr.device_id
        stream_holder = get_or_create_stream(device_id, None, "cuda")
        return NumpyTensor.create_host_from(arr, stream_holder).tensor

    if cp is not None and isinstance(arr, cp.ndarray):
        return arr.get()

    if torch is not None and isinstance(arr, torch.Tensor):
        return arr.cpu().numpy()

    return arr


def arrays_equal(arr1, arr2):
    """Check if two arrays are equal, handling different frameworks."""
    return np.array_equal(to_numpy(arr1), to_numpy(arr2))


def create_empty_array(framework: Framework, mem_backend: MemBackend, shape: tuple, dtype: DType, device_id: int, stream=None):
    """Create an empty array for the given framework and memory backend."""
    native_dtype = get_framework_dtype(framework, dtype)

    match (framework, mem_backend):
        case (Framework.ndbuffer, MemBackend.cpu):
            dtype_instance = np.dtype(native_dtype)
            buf = ndbuffer.empty(
                shape, device_id=ndbuffer.CPU_DEVICE_ID, dtype_name=dtype_instance.name, itemsize=dtype_instance.itemsize
            )
            return NDBufferTensor(buf)
        case (Framework.ndbuffer, MemBackend.cuda):
            dtype_instance = np.dtype(native_dtype)
            with utils.device_ctx(device_id):
                buf = ndbuffer.empty(
                    shape, device_id=device_id, dtype_name=dtype_instance.name, itemsize=dtype_instance.itemsize, stream=stream
                )
                return NDBufferTensor(buf)
        case (Framework.numpy, MemBackend.cpu):
            return np.empty(shape, dtype=native_dtype)
        case (Framework.cupy, MemBackend.cuda):
            with get_framework_device_ctx(device_id, framework):
                return cp.empty(shape, dtype=native_dtype)
        case (Framework.torch, MemBackend.cpu):
            return torch.empty(shape, dtype=native_dtype)
        case (Framework.torch, MemBackend.cuda):
            return torch.empty(shape, dtype=native_dtype, device=f"cuda:{device_id}")
        case _:
            raise ValueError(f"Invalid framework: {framework} and memory backend: {mem_backend} combination.")


# Test fixtures
@pytest.fixture(scope="module")
def device_id():
    """Device ID to use for GPU tests."""
    return 0


def generate_copy_combinations():
    """
    Generate valid (src_framework, src_backend, dest_framework, dest_backend)
    combinations for cross-device (CPU <-> GPU) copy testing.
    """
    valid_combinations = {
        # NumPy
        (Framework.numpy, MemBackend.cpu, Framework.numpy, MemBackend.cpu),
        (Framework.numpy, MemBackend.cpu, Framework.ndbuffer, MemBackend.cpu),
        (Framework.numpy, MemBackend.cpu, Framework.ndbuffer, MemBackend.cuda),
        # CuPy
        (Framework.cupy, MemBackend.cuda, Framework.cupy, MemBackend.cuda),
        (Framework.cupy, MemBackend.cuda, Framework.ndbuffer, MemBackend.cpu),
        (Framework.cupy, MemBackend.cuda, Framework.ndbuffer, MemBackend.cuda),
        # Torch
        (Framework.torch, MemBackend.cpu, Framework.torch, MemBackend.cpu),
        (Framework.torch, MemBackend.cuda, Framework.torch, MemBackend.cuda),
        (Framework.torch, MemBackend.cuda, Framework.torch, MemBackend.cpu),
        (Framework.torch, MemBackend.cpu, Framework.torch, MemBackend.cuda),
    }

    for src_fw in Framework.enabled():
        for src_backend in MemBackend:
            for dest_fw in Framework.enabled():
                for dest_backend in MemBackend:
                    if (src_fw, src_backend, dest_fw, dest_backend) in valid_combinations:
                        yield (src_fw, src_backend, dest_fw, dest_backend)


@pytest.mark.parametrize(
    ("src_framework", "src_backend", "dest_framework", "dest_backend"),
    generate_copy_combinations(),
)
def test_copy_multiple_operands(src_framework, src_backend, dest_framework, dest_backend, device_id):
    """
    Test copying multiple operands for CuPy and PyTorch framework combinations.
    """
    # Skip if required framework is not available
    if (src_framework == Framework.cupy or dest_framework == Framework.cupy) and cp is None:
        pytest.skip("CuPy not available")

    if (src_framework == Framework.torch or dest_framework == Framework.torch) and torch is None:
        pytest.skip("PyTorch not available")

    # Assert we're not mixing CuPy and PyTorch
    cupy_involved = src_framework == Framework.cupy or dest_framework == Framework.cupy
    torch_involved = src_framework == Framework.torch or dest_framework == Framework.torch
    assert not (cupy_involved and torch_involved), (
        f"Invalid combination: {src_framework.name} -> {dest_framework.name}. Cannot mix CuPy and PyTorch frameworks."
    )

    # Create stream holder based on which framework is involved
    if cupy_involved:
        maybe_register_package("cupy")
        framework = Framework.cupy
        stream = get_custom_stream(framework, device_id)
        stream_holder = get_or_create_stream(device_id, stream, "cupy")
    elif torch_involved:
        maybe_register_package("torch")
        framework = Framework.torch
        stream = get_custom_stream(framework, device_id)
        stream_holder = get_or_create_stream(device_id, stream, "torch")
    else:
        framework = Framework.numpy
        stream = get_custom_stream(framework, device_id, is_numpy_stream_oriented=True)
        stream_holder = get_or_create_stream(device_id, stream, "cuda")

    # Create source arrays
    shapes = [(5, 6), (3, 4), (10,)]
    dtype = DType.float32
    src_arrays = [
        get_random_input_data(src_framework, shape, dtype, src_backend, get_framework_dtype, device_id=device_id)
        for shape in shapes
    ]

    # Create destination arrays
    dest_arrays = [
        create_empty_array(dest_framework, dest_backend, shape, dtype, device_id, stream=stream_holder) for shape in shapes
    ]
    src_wrapped = [wrap_operand(arr) for arr in src_arrays]
    dest_wrapped = [wrap_operand(arr) for arr in dest_arrays]
    copy_(src_wrapped, dest_wrapped, stream_holder)
    stream_holder.obj.sync()

    # Verify all copies
    for i, (src_arr, dest_arr) in enumerate(zip(src_arrays, dest_arrays, strict=False)):
        assert arrays_equal(src_arr, dest_arr), (
            f"Copy failed for operand {i}: {src_framework.name}/{src_backend.name} -> {dest_framework.name}/{dest_backend.name}"
        )
