# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import random


try:
    import cupy as cp
except ImportError:
    cp = None
try:
    import torch
except ImportError:
    torch = None


import pytest

from nvmath_tests.helpers import (
    get_random_input_data as _get_random_input_data,
    get_custom_stream as _get_custom_stream,
    get_framework_device_ctx as _get_framework_device_ctx,
)

from .common_axes import MemBackend, Framework, DType, ShapeKind, OptFftType, OptFftLayout
from .axes_utils import get_framework_dtype, size_of, get_fft_dtype


def get_random_input_data(
    framework: Framework,
    shape: int | tuple[int],
    dtype: DType,
    mem_backend: MemBackend,
    *,
    lo: float = -0.5,
    hi: float = 0.5,
    device_id=None,
):
    """Generate random input data for FFT tests.

    This is a wrapper around the common get_random_input_data function
    that provides the module-specific utility functions.

    Note: Tests use per-test seeding for reproducibility
    via the conftest.py fixture, which sets the global random state.
    """
    return _get_random_input_data(
        framework,
        shape,
        dtype,
        mem_backend,
        get_framework_dtype,
        lo=lo,
        hi=hi,
        device_id=device_id,
    )


def get_1d_shape_cases(shape_kinds: list[ShapeKind], rng: random.Random, incl_1: bool = True):
    """
    Sample concrete shapes to test according to shape_kinds
    """
    assert len(shape_kinds), f"{shape_kinds}"

    def shape_gen():
        for shape_kind in shape_kinds:
            if shape_kind == ShapeKind.pow2:
                if incl_1:
                    yield ShapeKind.pow2, 1
                random_pow_2 = 2 ** rng.randint(1, 12)
                yield ShapeKind.pow2, random_pow_2
            elif shape_kind == ShapeKind.pow2357:
                a, b, c, d = rng.choices([0, 1, 2, 3], k=4)
                random_pow_2357 = (2**a) * (3**b) * (5**c) * (7**d)
                yield ShapeKind.pow2357, random_pow_2357
            elif shape_kind == ShapeKind.prime:
                random_prime = rng.choice([89, 103, 131, 397, 541, 853, 997, 12799])
                yield ShapeKind.prime, random_prime
            else:
                assert shape_kind == ShapeKind.random
                yield ShapeKind.random, rng.randint(1, 4096)

    return list(shape_gen())


def get_random_1d_shape(shape_kinds: list[ShapeKind], rng: random.Random, incl_1: bool = False):
    assert len(shape_kinds), f"{shape_kinds}"
    return rng.choice(get_1d_shape_cases(shape_kinds, rng=rng, incl_1=incl_1))


def get_custom_stream(framework: Framework, device_id=None, is_numpy_stream_oriented=False):
    """Get a custom stream for the specified framework.

    This is a wrapper around the common get_custom_stream function.
    """
    return _get_custom_stream(framework, device_id, is_numpy_stream_oriented)


def get_framework_device_ctx(device_id: int, framework: Framework):
    """Get framework-specific device context.

    This is a wrapper around the common get_framework_device_ctx function.
    """
    return _get_framework_device_ctx(device_id, framework)


def get_stream_pointer(stream) -> int:
    package = stream.__class__.__module__.split(".")[0]
    if package == "cupy":
        return stream.ptr
    elif package == "torch":
        return stream.cuda_stream
    else:
        raise ValueError(f"Unknown GPU framework {package}")


def init_assert_exec_backend_specified():
    import pytest
    import nvmath

    @pytest.fixture(autouse=True)
    def assert_exec_backend_specified(monkeypatch):
        """Make sure the tests pass the execution explicitly"""
        _actual_init = nvmath.fft.FFT.__init__

        def fft_init(self, *args, **kwargs):
            assert kwargs.get("execution") is not None, "The test must explicitly specify execution backend"
            _actual_init(self, *args, **kwargs)

        monkeypatch.setattr(nvmath.fft.FFT, "__init__", fft_init)

    return assert_exec_backend_specified


def get_primes_up_to(up_to):
    is_prime = [False, False] + [True] * (up_to - 1)
    for k in range(2, up_to + 1):
        if is_prime[k]:
            yield k
        c = k * k
        while c <= up_to:
            is_prime[c] = False
            c += k


@pytest.fixture
def fx_last_operand_layout(monkeypatch):
    import nvmath
    from .check_helpers import get_array_element_strides, get_raw_ptr

    _actual_init = nvmath.fft.FFT.__init__
    _actual_exec = nvmath.fft.FFT.execute
    layouts = {}
    ptrs = {}

    def wrapped_init(self, initial_operand, *args, **kwargs):
        nonlocal layouts, ptrs
        layouts["initial_operand"] = (tuple(initial_operand.shape), get_array_element_strides(initial_operand))
        ptrs["initial_operand"] = get_raw_ptr(initial_operand)
        ret = _actual_init(self, initial_operand, *args, **kwargs)
        layouts["operand"] = (self.operand.shape, self.operand.strides)
        ptrs["operand"] = self.operand.data_ptr
        assert self.operand_layout.shape == self.operand.shape
        assert self.operand_layout.strides == self.operand.strides
        if self.operand_backup is not None:
            layouts["operand_backup"] = (self.operand_backup.shape, self.operand_backup.strides)
            ptrs["operand_backup"] = self.operand_backup.data_ptr
        return ret

    monkeypatch.setattr(nvmath.fft.FFT, "__init__", wrapped_init)

    def wrapped_exec(self, *args, **kwargs):
        ret = _actual_exec(self, *args, **kwargs)
        layouts["result"] = (tuple(ret.shape), get_array_element_strides(ret))
        ptrs["result"] = get_raw_ptr(ret)
        return ret

    monkeypatch.setattr(nvmath.fft.FFT, "execute", wrapped_exec)

    def stride_order(shape, stride):
        return tuple(i for _, _, i in sorted(zip(stride, shape, range(len(shape)), strict=True)))

    def check_layouts(exec_backend, mem_backend, axes, result_layout, fft_type, is_dense, inplace):
        initial_shape, initial_strides = layouts["initial_operand"]
        if mem_backend == exec_backend.mem:
            assert "operand_backup" not in layouts
        else:
            assert ptrs["initial_operand"] == ptrs["operand_backup"]
            assert layouts["operand_backup"][0] == initial_shape
            assert layouts["operand_backup"][1] == initial_strides

        assert layouts["operand"][0] == initial_shape
        if fft_type != OptFftType.c2r and mem_backend == exec_backend.mem:
            assert ptrs["initial_operand"] == ptrs["operand"]
        else:
            assert ptrs["initial_operand"] != ptrs["operand"]
        if mem_backend == exec_backend.mem and (fft_type != OptFftType.c2r or is_dense):
            # nvmath should keep the strides for dense (possibly permuted) tensors
            assert layouts["operand"][1] == initial_strides

        if inplace:
            assert ptrs["result"] == ptrs["initial_operand"]
            assert layouts["result"] == layouts["initial_operand"]
        else:
            assert ptrs["result"] != ptrs["initial_operand"]
            # the frameworks gpu<->cpu copy does not necessarily keep the layout
            if mem_backend == exec_backend.mem:
                if result_layout == OptFftLayout.natural:
                    initial_order = stride_order(*layouts["initial_operand"])
                    res_order = stride_order(*layouts["result"])
                    assert initial_order == res_order
                else:
                    assert result_layout == OptFftLayout.optimized
                    res_layout = layouts["result"]
                    res_order = stride_order(*res_layout)
                    least_strided = res_order[: len(axes)]
                    assert sorted(axes) == sorted(least_strided), (
                        f"{sorted(axes)} vs {sorted(least_strided)}: result_layout={res_layout}"
                    )

    return check_layouts, layouts, ptrs


def align_up(num_bytes, alignment):
    return ((num_bytes + alignment - 1) // alignment) * alignment


def get_overaligned_view(alignment, framework, shape, dtype, mem_backend):
    from .check_helpers import get_raw_ptr, assert_array_type

    dtype_size = size_of(dtype)
    assert alignment % dtype_size == 0
    innermost_extent = shape[-1]
    offset_upperbound = alignment // dtype_size
    base_innermost = innermost_extent + offset_upperbound
    base_shape = list(shape)
    base_shape[-1] = base_innermost
    base_shape = tuple(base_shape)
    a = get_random_input_data(framework, base_shape, dtype, mem_backend)
    base_ptr = get_raw_ptr(a)
    complex_dtype = get_fft_dtype(dtype)
    required_alignment = size_of(complex_dtype)
    assert required_alignment % dtype_size == 0
    assert base_ptr % required_alignment == 0
    overaligned_offset = (-(base_ptr % -alignment)) // dtype_size
    assert 0 <= overaligned_offset < offset_upperbound
    slices = (slice(None),) * (len(shape) - 1) + (slice(overaligned_offset, overaligned_offset + innermost_extent),)
    aligned_view = a[slices]
    view_ptr = get_raw_ptr(aligned_view)
    assert view_ptr % alignment == 0
    assert_array_type(aligned_view, framework, mem_backend, dtype)
    return a, aligned_view


def free_cupy_pool():
    if cp is not None:
        cp.get_default_memory_pool().free_all_blocks()


def free_torch_pool():
    if torch is not None:
        torch.cuda.empty_cache()


def free_cuda_pool():
    from nvmath.internal.memory import free_reserved_memory

    free_reserved_memory()


def free_framework_pools(framework):
    if framework == Framework.numpy:
        free_cupy_pool()
        free_torch_pool()
    elif framework == Framework.cupy:
        free_cuda_pool()
        free_torch_pool()
    elif framework == Framework.torch:
        free_cuda_pool()
        free_cupy_pool()
