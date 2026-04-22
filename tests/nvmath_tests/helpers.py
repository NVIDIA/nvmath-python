# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
import contextlib
import gc
import os
import weakref
from collections.abc import Callable
from enum import Enum
from typing import Union

try:
    import cupy
except ImportError:
    cupy = None

try:
    import torch
except ImportError:
    torch = None

try:
    from nvmath.bindings import cusparseLt as _cusparseLt
    from nvmath.bindings._internal.utils import FunctionNotFoundError, NotSupportedError

    try:
        handle = _cusparseLt.init()
        _cusparseLt.get_version(handle)
        _cusparseLt.destroy(handle)
        _HAS_CUSPARSELT = True
    except (NotSupportedError, FunctionNotFoundError, RuntimeError):
        _HAS_CUSPARSELT = False
    except _cusparseLt.cuSPARSELtError as e:
        from nvmath.bindings import cusparse as _cusparse

        if e.status == _cusparse.Status.ARCH_MISMATCH:
            _HAS_CUSPARSELT = False
        else:
            raise
except ImportError:
    _HAS_CUSPARSELT = False

import math

import hypothesis
import numpy as np
import pytest

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


def requires_host_memory(min_memory: int | Callable[[...], int]):
    """
    Decorator that skips a test at runtime if host memory is insufficient.

    The memory check happens when the test runs, not during collection,
    since available memory may change between those times.

    Args:
        min_memory: Minimum required host memory in bytes.
                    Can be an int or a callable that takes test parameters
                    and returns the required bytes.

    Examples:
        ```python
        # Fixed memory requirement:
        @requires_host_memory(16 * 1024**3)  # 16 GB
        def test_large_operation(): ...


        # Dynamic memory based on parameters:
        @pytest.mark.parametrize("size", [1024, 2048, 4096])
        @requires_host_memory(lambda size: size * 1024 * 1024)
        def test_with_size(size): ...


        # Multiple parameters:
        @pytest.mark.parametrize("batch_size,dim", [(100, 1024), (1000, 2048)])
        @requires_host_memory(lambda batch_size, dim: batch_size * dim * 8)
        def test_with_params(batch_size, dim): ...
        ```
    """
    import functools
    import inspect

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Compute required memory at runtime
            if callable(min_memory):
                # Bind arguments to pass to the memory calculation function
                sig = inspect.signature(func)
                try:
                    bound = sig.bind(*args, **kwargs)
                    bound.apply_defaults()
                    required_memory = min_memory(**bound.arguments)
                except Exception as e:
                    # If binding fails, try with just kwargs
                    try:
                        required_memory = min_memory(**kwargs)
                    except Exception:
                        raise RuntimeError(
                            f"Could not compute memory requirement: {e}. "
                            f"Make sure the callable signature matches test parameters."
                        ) from e
            else:
                required_memory = min_memory

            skip_if_insufficient_host_memory(required_memory)

            return func(*args, **kwargs)

        return wrapper

    return decorator


requires_cusparselt = pytest.mark.skipif(not _HAS_CUSPARSELT, reason="cuSPARSELt is not available")


def skip_if_insufficient_host_memory(required_memory_bytes: int):
    """
    Skips the current test if available host memory is less than required.

    Call this function at the beginning of your test to check memory availability.
    This is an alternative to the @requires_host_memory decorator for cases
    where you want explicit control or need to compute memory requirements
    dynamically within the test.

    Args:
        required_memory_bytes: Minimum required host memory in bytes.

    Example:
        ```python
        def test_large_operation():
            skip_if_insufficient_host_memory(16 * 1024**3)  # Require 16 GB
            # ... rest of test


        @pytest.mark.parametrize("size", [1024, 2048, 4096])
        def test_with_size(size):
            skip_if_insufficient_host_memory(size * 1024 * 1024)  # size MB -> bytes
            # ... rest of test
        ```
    """

    import psutil

    available_memory = psutil.virtual_memory().available
    if available_memory < required_memory_bytes:
        pytest.skip(
            f"Test requires at least {required_memory_bytes:.3e} bytes host memory, "
            f"but only {available_memory:.3e} bytes available."
        )
    gc.collect()


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


class OutOfMemoryError(Exception):
    pass


@contextlib.contextmanager
def consistent_out_of_memory_error():
    """
    To simplify the test logic, check if the error looks like OOM
    and raise single exception type OutOfMemoryError.
    """
    try:
        yield
    except Exception as e:
        if (cupy is not None and isinstance(e, cupy.cuda.memory.OutOfMemoryError)) or ("CUDA_ERROR_OUT_OF_MEMORY" in str(e)):
            raise OutOfMemoryError() from e
        raise


@contextlib.contextmanager
def check_freed_after(obj, msg=""):
    """
    Assert that *obj* is destroyed once the caller drops its reference.

    Typical usage::

        with check_freed_after(result, "result should be solely owned"):
            del result

    The context manager creates a weak reference to *obj*, then drops
    its own strong reference.  Inside the ``with`` block the caller
    must ``del`` its variable.  On block exit the weak reference is
    checked — if the object is still alive, someone else is holding a
    reference (or it is trapped in a reference cycle, which is itself a
    bug that should be fixed rather than hidden by ``gc.collect()``).

    Note: this helper relies on CPython's deterministic reference-counting
    semantics, where an object is deallocated immediately when its
    reference count drops to zero.

    Args:
        obj: The object to check.
        msg: Optional message shown on assertion failure.

    Raises:
        TypeError: If *obj* does not support weak references.
        AssertionError: If the object is still alive after the block.
    """
    try:
        ref = weakref.ref(obj)
    except TypeError:
        raise TypeError(
            f"{type(obj).__name__!r} objects do not support weak references; check_freed_after cannot be used with this type"
        ) from None
    del obj
    yield
    assert ref() is None, msg or "Object was not destroyed; extra references likely exist"
