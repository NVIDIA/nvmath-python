# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
import os
import contextlib
from typing import Union
from collections.abc import Callable
from enum import Enum

try:
    import cupy
except ImportError:
    cupy = None

try:
    import torch
except ImportError:
    torch = None

import numpy as np
import math
import hypothesis

try:
    from cuda.core import Device, Stream
except ImportError:
    from cuda.core.experimental import Device, Stream


def nvmath_seed():
    """Sets the hypothesis seed from the environment variable RANDOM_SEED."""
    if (seed := os.environ.get("RANDOM_SEED", default=None)) is not None:
        print(f"The nvmath-python hypothesis seed is '{seed}' from environment RANDOM_SEED")
        return hypothesis.seed(hash(seed))

    print("The nvmath-python hypothesis seed is unset.")

    def do_nothing(x):
        return x

    return do_nothing


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
    if cupy is None:
        raise RuntimeError("cupy is not installed")

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


# ============================================================================
# Common data generation utilities that are used by various tests
# ============================================================================


def get_framework_device_ctx(
    device_id: int, framework_enum: Enum
) -> Union[contextlib.nullcontext, "cupy.cuda.Device", "torch.cuda.device"]:
    """
    Get framework-specific device context.

    Args:
        device_id: Device ID to set
        framework_enum: Framework enum (numpy, cupy, or torch)

    Returns:
        Context manager for device setting
    """
    # Validate framework enum
    assert hasattr(framework_enum, "name"), f"framework_enum must be an Enum with a name attribute, got {type(framework_enum)}"
    assert framework_enum.name in ["numpy", "cupy", "torch"], (
        f"Unsupported framework '{framework_enum.name}'. Expected 'numpy', 'cupy', or 'torch'"
    )

    if framework_enum.name == "numpy":
        return contextlib.nullcontext()
    elif framework_enum.name == "cupy":
        if cupy is None:
            raise RuntimeError("cupy is not installed")
        return cupy.cuda.Device(device_id)
    elif framework_enum.name == "torch":
        if torch is None:
            raise RuntimeError("torch is not installed")
        return torch.cuda.device(device_id)
    else:
        raise ValueError(f"Unknown framework {framework_enum}")


def _create_numpy_array(
    shape: int | tuple[int, ...],
    native_dtype: type,
    lo: float,
    hi: float,
) -> np.ndarray:
    """
    Create a NumPy array with random data.
    Uses numpy.random.uniform with the global random state.

    Args:
        shape: Shape of the array to create
        native_dtype: Native NumPy dtype (e.g., np.float32, np.complex64)
        lo: Lower bound for uniform random distribution
        hi: Upper bound for uniform random distribution

    Returns:
        NumPy array with random values in [lo, hi) and the specified dtype
    """
    if np.dtype(native_dtype).kind == "c":
        real = np.random.uniform(lo, hi, size=shape)
        imag = np.random.uniform(lo, hi, size=shape)
        a = (real + 1j * imag).astype(native_dtype)
        # Handle scalar case: complex operation converts ndarray to scalar
        if len(shape) == 0:
            a = np.array(a)
    else:
        a = np.random.uniform(lo, hi, size=shape).astype(native_dtype)

    return a


def _create_cupy_array(
    shape: int | tuple[int, ...],
    native_dtype: type,
    lo: float,
    hi: float,
) -> "cupy.ndarray":
    """Create a CuPy array with random data on GPU.
    Uses cupy.random.uniform with the global random state.

    Args:
        shape: Shape of the array to create
        native_dtype: Native CuPy dtype (e.g., cupy.float32, cupy.complex64)
        lo: Lower bound for uniform random distribution
        hi: Upper bound for uniform random distribution

    Returns:
        CuPy array with random values in [lo, hi) and the specified dtype

    Raises:
        RuntimeError: If CuPy is not installed
    """
    if cupy is None:
        raise RuntimeError("cupy is not installed")

    if cupy.dtype(native_dtype).kind == "c":
        real = cupy.random.uniform(lo, hi, size=shape)
        imag = cupy.random.uniform(lo, hi, size=shape)
        a = (real + 1j * imag).astype(native_dtype)
        # Handle scalar case: complex operation converts ndarray to scalar
        if len(shape) == 0:
            a = cupy.array(a)
    else:
        a = cupy.random.uniform(lo, hi, size=shape).astype(native_dtype)

    return a


def _create_torch_tensor(
    shape: int | tuple[int, ...],
    native_dtype: type,
    mem_backend_name: str,
    lo: float,
    hi: float,
    device_id: int | None,
) -> "torch.Tensor":
    """
    Create a PyTorch tensor with random data.
    Uses torch.rand with a seeded generator for reproducibility. The generator
    is seeded using numpy's random state to ensure consistency across frameworks.

    Args:
        shape: Shape of the tensor to create
        native_dtype: Native PyTorch dtype (e.g., torch.float32, torch.complex64)
        mem_backend_name: Memory backend ('cpu' or 'cuda')
        lo: Lower bound for uniform random distribution
        hi: Upper bound for uniform random distribution
        device_id: CUDA device ID

    Returns:
        PyTorch tensor with random values in [lo, hi) and the specified dtype

    Raises:
        RuntimeError: If PyTorch is not installed
    """
    if torch is None:
        raise RuntimeError("torch is not installed")

    # Determine device
    if mem_backend_name == "cpu":
        device = "cpu"
    elif device_id is not None:
        device = f"cuda:{device_id}"
    else:
        device = "cuda"

    # For torch, we use a generator seeded with numpy's random state
    # to ensure reproducibility across frameworks
    torch_seed = np.random.randint(0, 2**31 - 1)
    g = torch.Generator(device=device)
    g.manual_seed(torch_seed)

    # Generate random tensor
    t = torch.rand(size=shape, generator=g, device=device, dtype=native_dtype)
    scale = torch.tensor(hi - lo, dtype=native_dtype)

    if native_dtype.is_complex:
        shift = torch.tensor(lo + 1j * lo, dtype=native_dtype)
    else:
        shift = torch.tensor(lo, dtype=native_dtype)

    t = t.mul_(scale).add_(shift)
    return t


def get_random_input_data(
    framework_enum: Enum,
    shape: int | tuple[int, ...],
    dtype_enum: Enum,
    mem_backend_enum: Enum,
    get_framework_dtype_func: Callable[[Enum, Enum], type],
    *,
    lo: float = -0.5,
    hi: float = 0.5,
    device_id: int | None = None,
) -> Union[np.ndarray, "cupy.ndarray", "torch.Tensor"]:
    """
    Generate random input data for testing.
    It uses the global random state.

    Args:
        framework_enum: Framework enum (numpy, cupy, or torch)
        shape: Shape of the array (int or tuple of ints)
        dtype_enum: DType enum
        mem_backend_enum: MemBackend enum (cuda or cpu)
        get_framework_dtype_func: Function to convert DType enum to framework dtype

    Keyword Args:
        lo: Lower bound for random values (default: -0.5)
        hi: Upper bound for random values (default: 0.5)
        device_id: Device ID (default: None)

    Returns:
        Random array with the specified properties
    """
    # Validate framework enum
    assert hasattr(framework_enum, "name"), f"framework_enum must be an Enum with a name attribute, got {type(framework_enum)}"
    assert framework_enum.name in ["numpy", "cupy", "torch"], (
        f"Unsupported framework '{framework_enum.name}'. Expected 'numpy', 'cupy', or 'torch'"
    )

    # Validate memory backend enum
    assert hasattr(mem_backend_enum, "name"), (
        f"mem_backend_enum must be an Enum with a name attribute, got {type(mem_backend_enum)}"
    )
    assert mem_backend_enum.name in ["cuda", "cpu"], (
        f"Unsupported memory backend '{mem_backend_enum.name}'. Expected 'cuda' or 'cpu'"
    )

    assert lo < hi, f"Lower bound ({lo}) must be less than upper bound ({hi})"

    # Get the native dtype for the framework
    native_dtype = get_framework_dtype_func(framework_enum, dtype_enum)

    if framework_enum.name == "numpy":
        assert mem_backend_enum.name == "cpu", f"NumPy framework requires CPU memory backend, got {mem_backend_enum.name}"
        a = _create_numpy_array(shape, native_dtype, lo, hi)

    elif framework_enum.name == "cupy":
        assert mem_backend_enum.name == "cuda", f"CuPy framework requires CUDA memory backend, got {mem_backend_enum.name}"
        # Use device context if device_id is specified
        if device_id is not None:
            with get_framework_device_ctx(device_id, framework_enum):
                a = _create_cupy_array(shape, native_dtype, lo, hi)
        else:
            a = _create_cupy_array(shape, native_dtype, lo, hi)

    elif framework_enum.name == "torch":
        a = _create_torch_tensor(shape, native_dtype, mem_backend_enum.name, lo, hi, device_id)

    else:
        raise ValueError(f"Unknown framework {framework_enum}")

    return a


def get_custom_stream(
    framework_enum: Enum,
    device_id: int | None = None,
    is_numpy_stream_oriented: bool = False,
) -> Union[Stream, "cupy.cuda.Stream", "torch.cuda.Stream"] | None:
    """
    Get a custom stream for the specified framework.

    Args:
        framework_enum: Framework enum (numpy, cupy, or torch)
        device_id: Device ID for multi-GPU setups
        is_numpy_stream_oriented: Whether to create CUDA stream for numpy

    Returns:
        Stream object or None
    """
    # Validate framework enum
    assert hasattr(framework_enum, "name"), f"framework_enum must be an Enum with a name attribute, got {type(framework_enum)}"
    assert framework_enum.name in ["numpy", "cupy", "torch"], (
        f"Unsupported framework '{framework_enum.name}'. Expected 'numpy', 'cupy', or 'torch'"
    )

    if framework_enum.name == "numpy":
        if is_numpy_stream_oriented:
            old_device = Device()
            device = Device(device_id)
            try:
                device.set_current()
                return device.create_stream()
            finally:
                old_device.set_current()
        else:
            return None
    elif framework_enum.name == "cupy":
        if cupy is None:
            raise RuntimeError("cupy is not installed")
        if device_id is None:
            return cupy.cuda.Stream(non_blocking=True)
        else:
            with get_framework_device_ctx(device_id, framework_enum):
                return cupy.cuda.Stream(non_blocking=True)
    elif framework_enum.name == "torch":
        if torch is None:
            raise RuntimeError("torch is not installed")
        device = None if device_id is None else f"cuda:{device_id}"
        return torch.cuda.Stream(device=device)
    else:
        raise ValueError(f"Unknown GPU framework {framework_enum}")


def use_stream(stream):
    if stream is None or isinstance(stream, Stream):
        return contextlib.nullcontext(stream)
    if cupy is not None and isinstance(stream, cupy.cuda.Stream):
        return stream
    elif torch is not None and isinstance(stream, torch.cuda.Stream):
        return torch.cuda.stream(stream)
    else:
        raise ValueError(f"Unknown stream type {type(stream)}")


def record_event(stream):
    if isinstance(stream, Stream):
        return stream.record()
    if cupy is not None and isinstance(stream, cupy.cuda.Stream):
        return stream.record()
    elif torch is not None and isinstance(stream, torch.cuda.Stream):
        return stream.record_event()
    else:
        raise ValueError(f"Unknown stream type {type(stream)}")


def wait_event(stream, event):
    if isinstance(stream, Stream):
        stream.wait(event)
    elif (
        cupy is not None and isinstance(stream, cupy.cuda.Stream) or torch is not None and isinstance(stream, torch.cuda.Stream)
    ):
        stream.wait_event(event)
    else:
        raise ValueError(f"Unknown stream type {type(stream)}")


def order_streams(
    stream0: Union[Stream, "cupy.cuda.Stream", "torch.cuda.Stream"],
    stream1: Union[Stream, "cupy.cuda.Stream", "torch.cuda.Stream"],
) -> None:
    """
    Order two streams such that stream1 waits for stream0 to complete.
    No operation is performed if either stream is None.

    Args:
        stream0: First stream
        stream1: Second stream
    """
    if stream0 is not None and stream1 is not None:
        event = record_event(stream0)
        wait_event(stream1, event)
