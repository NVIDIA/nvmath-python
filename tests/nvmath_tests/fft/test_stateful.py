# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import functools
import logging
import math
import os
import random
import sys
from ast import literal_eval

import pytest

try:
    import cupy as cp
except ImportError:
    cp = None
try:
    import torch
except ImportError:
    torch = None

import numpy as np

import nvmath
from nvmath.fft import ExecutionCPU, ExecutionCUDA
from nvmath.fft.fft import axis_order_in_memory, calculate_strides, get_fft_plan_traits
from nvmath.memory import _MEMORY_MANAGER

from ..helpers import check_freed_after
from .utils.axes_utils import (
    get_fft_dtype,
    get_ifft_dtype,
    is_complex,
    is_half,
    size_of,
)
from .utils.check_helpers import (
    add_in_place,
    assert_array_equal,
    assert_array_type,
    assert_eq,
    assert_norm_close,
    copy_array,
    get_array_device_id,
    get_array_element_strides,
    get_array_strides,
    get_device_ctx,
    get_fft_ref,
    get_ifft_ref,
    get_raw_ptr,
    get_scaled,
    intercept_default_allocations,
    is_decreasing,
    is_pow_2,
    record_event,
    should_skip_3d_unsupported,
    use_stream,
    wait_event,
)
from .utils.common_axes import (
    Direction,
    DType,
    ExecBackend,
    Framework,
    MemBackend,
    OptFftBlocking,
    OptFftLayout,
    OptFftType,
    ShapeKind,
)
from .utils.input_fixtures import (
    align_up,
    free_framework_pools,
    get_custom_stream,
    get_overaligned_view,
    get_random_input_data,
    init_assert_exec_backend_specified,
)
from .utils.support_matrix import (
    framework_exec_type_support,
    multi_gpu_only,
    opt_fft_type_direction_support,
    opt_fft_type_input_type_support,
    supported_backends,
    type_shape_support,
)

rng = random.Random(42)


# DO NOT REMOVE, this call creates a fixture that enforces
# specifying execution option to the FFT calls in tests
# defined in this file
assert_exec_backend_specified = init_assert_exec_backend_specified()


def get_default_num_threads():
    if os.name == "posix":
        num_threads = len(os.sched_getaffinity(0))
    else:
        num_threads = os.cpu_count() or 1
    return max(1, num_threads // 2)


@pytest.mark.parametrize(
    (
        "framework",
        "exec_backend",
        "mem_backend",
        "fft_dim",
        "batched",
        "dtype",
        "result_layout",
    ),
    [
        (
            framework,
            exec_backend,
            exec_backend.mem,
            fft_dim,
            batched,
            dtype,
            rng.choice(list(OptFftLayout)),
        )
        for framework in Framework.enabled()
        for exec_backend in supported_backends.exec
        if exec_backend.mem in supported_backends.framework_mem[framework]
        for fft_dim in [1, 2, 3]
        for batched in ["no", "left", "right"]
        for dtype in framework_exec_type_support[framework][exec_backend]
    ],
)
def test_stateful_nd_default_allocator(
    seeder,
    monkeypatch,
    framework,
    exec_backend,
    mem_backend,
    fft_dim,
    batched,
    dtype,
    result_layout,
):
    fft_dim_shape = {
        1: (1024,),
        2: (512, 512),
        3: (128, 128, 128),
    }
    shape = fft_dim_shape[fft_dim]

    if batched == "left":
        shape = (4,) + shape
        axes = tuple(i + 1 for i in range(fft_dim))
    elif batched == "right":
        shape = shape + (4,)
        axes = tuple(i for i in range(fft_dim))
    else:
        assert batched == "no"
        axes = None

    signal_0 = get_random_input_data(framework, shape, dtype, mem_backend)
    signal_1 = get_random_input_data(framework, shape, dtype, mem_backend)

    allocations = intercept_default_allocations(monkeypatch)
    expected_key = "torch" if framework == Framework.torch else "cupy"
    expected_allocations = 1 if exec_backend == ExecBackend.cufft else 0

    if dtype == DType.float16 and batched == "right":
        with pytest.raises(ValueError, match="is currently not supported for strided inputs"):
            nvmath.fft.FFT(
                signal_0,
                axes=axes,
                execution=exec_backend.nvname,
                options={"result_layout": result_layout.value},
            )
        return

    if should_skip_3d_unsupported(exec_backend, shape, axes):
        with pytest.raises(ValueError, match="The 3D batched FFT is not supported"):
            nvmath.fft.FFT(
                signal_0,
                axes=axes,
                execution=exec_backend.nvname,
                options={"result_layout": result_layout.value},
            )
        return

    with nvmath.fft.FFT(
        signal_0,
        axes=axes,
        execution=exec_backend.nvname,
        options={"result_layout": result_layout.value},
    ) as f:
        f.plan()
        fft_0 = f.execute()

        assert allocations[expected_key] == expected_allocations, f"{allocations}, {expected_key}"
        assert all(allocations[key] == 0 for key in allocations if key != expected_key), f"{allocations}, {expected_key}"

        f.reset_operand(signal_1)
        fft_1 = f.execute()

        assert allocations[expected_key] == expected_allocations, f"{allocations}, {expected_key}"
        assert all(allocations[key] == 0 for key in allocations if key != expected_key), f"{allocations}, {expected_key}"

        assert_array_type(fft_0, framework, mem_backend, get_fft_dtype(dtype))
        assert_array_type(fft_1, framework, mem_backend, get_fft_dtype(dtype))
        if result_layout == OptFftLayout.natural:
            fft_0_strides = get_array_strides(fft_0)
            assert is_decreasing(fft_0_strides), f"{fft_0_strides}"
            fft_1_strides = get_array_strides(fft_1)
            assert is_decreasing(fft_1_strides), f"{fft_1_strides}"

        assert_norm_close(fft_0, get_fft_ref(signal_0, axes=axes), exec_backend=exec_backend)
        assert_norm_close(fft_1, get_fft_ref(signal_1, axes=axes), exec_backend=exec_backend)


@pytest.mark.parametrize(
    (
        "framework",
        "exec_backend",
        "mem_backend",
        "fft_dim",
        "dtype",
    ),
    [
        (
            framework,
            exec_backend,
            exec_backend.mem,
            fft_dim,
            dtype,
        )
        for framework in Framework.enabled()
        for exec_backend in supported_backends.exec
        if exec_backend.mem in supported_backends.framework_mem[framework]
        for fft_dim in [1, 2, 3]
        for dtype in framework_exec_type_support[framework][exec_backend]
        if is_complex(dtype) and (not is_half(dtype) or fft_dim == 1)
    ],
)
def test_stateful_nd_custom_allocator(seeder, monkeypatch, framework, exec_backend, mem_backend, fft_dim, dtype):
    fft_dim_shape = {
        1: (512,),
        2: (256, 512),
        3: (64, 128, 32),
    }
    shape = fft_dim_shape[fft_dim]

    signal = get_random_input_data(framework, shape, dtype, mem_backend)

    allocations = intercept_default_allocations(monkeypatch)
    logger = logging.getLogger("dummy_logger")
    allocator = _MEMORY_MANAGER["_raw"](device_id=0, logger=logger)
    expected_allocations = 1 if exec_backend == ExecBackend.cufft else 0
    expected_key = "raw"

    with nvmath.fft.FFT(signal, execution=exec_backend.nvname, options={"allocator": allocator}) as f:
        f.plan()
        fft = f.execute(direction=Direction.forward.value)

        assert allocations[expected_key] == expected_allocations, f"{allocations}, {expected_key}"

        f.reset_operand(fft)
        ifft = f.execute(direction=Direction.inverse.value)

        assert allocations[expected_key] == expected_allocations, f"{allocations}, {expected_key}"
        assert all(allocations[key] == 0 for key in allocations if key != expected_key), f"{allocations}, {expected_key}"

        assert_array_type(fft, framework, mem_backend, dtype)
        assert_array_type(ifft, framework, mem_backend, dtype)

        assert_norm_close(fft, get_fft_ref(signal), exec_backend=exec_backend)
        assert_norm_close(ifft, get_scaled(signal, math.prod(shape)), exec_backend=exec_backend)


@pytest.mark.parametrize(
    (
        "framework",
        "exec_backend",
        "mem_backend",
        "release_workspace",
    ),
    [
        (
            framework,
            exec_backend,
            mem_backend,
            release_workspace,
        )
        for framework in Framework.enabled()
        for exec_backend in supported_backends.exec
        for mem_backend in supported_backends.framework_mem[framework]
        for release_workspace in [False, True]
    ],
)
def test_stateful_release_workspace(seeder, monkeypatch, framework, exec_backend, mem_backend, release_workspace):
    shape = (2048, 128)
    dtype = DType.float32

    signal_0 = get_random_input_data(framework, shape, dtype, mem_backend)
    signal_1 = get_random_input_data(framework, shape, dtype, mem_backend)

    allocations = intercept_default_allocations(monkeypatch)
    if framework == Framework.torch:
        expected_key = "torch"
    elif framework == Framework.cupy:
        expected_key = "cupy"
    elif framework == Framework.numpy:
        expected_key = "raw"
    else:
        raise ValueError(f"Unknown framework: {framework}")

    num_allocs_1, num_allocs_2 = (1, 2) if exec_backend == ExecBackend.cufft else (0, 0)

    with nvmath.fft.FFT(signal_0, execution=exec_backend.nvname) as f:
        f.plan()
        fft_0 = f.execute(direction=Direction.forward.value, release_workspace=release_workspace)

        assert allocations[expected_key] == num_allocs_1, f"{allocations}, {expected_key}"

        f.reset_operand(signal_1)
        fft_1 = f.execute(direction=Direction.forward.value, release_workspace=release_workspace)

        assert allocations[expected_key] == num_allocs_2 if release_workspace else 1, f"{allocations}, {expected_key}"
        assert all(allocations[key] == 0 for key in allocations if key != expected_key), f"{allocations}, {expected_key}"

        assert_array_type(fft_0, framework, mem_backend, get_fft_dtype(dtype))
        assert_array_type(fft_1, framework, mem_backend, get_fft_dtype(dtype))

        assert_norm_close(fft_0, get_fft_ref(signal_0), exec_backend=exec_backend)
        assert_norm_close(fft_1, get_fft_ref(signal_1), exec_backend=exec_backend)


@pytest.mark.parametrize(
    (
        "framework",
        "exec_backend",
        "mem_backend",
        "shape_kind",
        "shape",
        "axes",
        "dtype",
        "blocking",
    ),
    [
        (
            framework,
            exec_backend,
            mem_backend,
            shape_kind,
            repr(shape),
            repr(axes),
            dtype,
            blocking,
        )
        for framework in Framework.enabled()
        for exec_backend in [ExecBackend.cufft]
        if exec_backend in supported_backends.exec
        for mem_backend in supported_backends.framework_mem[framework]
        for shape_kind, shape, axes in [
            (ShapeKind.pow2, (641, 16, 16, 16), (1, 2, 3)),
            (ShapeKind.pow2, (4, 32, 32, 32), (1, 2, 3)),
            (ShapeKind.pow2, (32, 32, 32, 4), (0, 1, 2)),
            (ShapeKind.pow2, (4, 64, 64, 64), (1, 2, 3)),
            (ShapeKind.pow2, (64, 64, 64, 4), (0, 1, 2)),
            (ShapeKind.pow2, (1, 256, 256, 256), (1, 2, 3)),
            (ShapeKind.pow2, (128, 512, 17), (0, 1)),
            (ShapeKind.pow2, (17, 128, 512), (0, 1)),
            (ShapeKind.pow2, (64, 64, 641), (0, 1)),
            (ShapeKind.pow2, (4, 256), (1,)),
            (ShapeKind.pow2357, (2 * 3 * 5 * 7, 9, 7**2), (0, 1, 2)),
            (ShapeKind.pow2357, (4 * 49, 9 * 25, 13), (0, 1)),
            (ShapeKind.pow2357, (13, 4 * 49, 9 * 25), (1, 2)),
            (ShapeKind.prime, (31, 9973), (1,)),
            (ShapeKind.prime, (9973, 31), (0,)),
        ]
        for dtype in framework_exec_type_support[framework][exec_backend]
        if is_complex(dtype)
        and shape_kind in type_shape_support[exec_backend][dtype]
        and (not is_half(dtype) or math.prod(shape[a] for a in axes if a in axes) <= 1024)
        for blocking in OptFftBlocking
    ],
)
def test_custom_stream(seeder, framework, exec_backend, mem_backend, shape_kind, shape, axes, dtype, blocking):
    shape = literal_eval(shape)
    axes = literal_eval(axes)

    s0 = get_custom_stream(framework, is_numpy_stream_oriented=True)
    s1 = get_custom_stream(framework, is_numpy_stream_oriented=True)
    s2 = get_custom_stream(framework, is_numpy_stream_oriented=True)

    free_framework_pools(framework)

    with use_stream(s0):
        signal = get_random_input_data(framework, shape, dtype, mem_backend)
        scale = math.prod(shape[a] for a in axes)
        signal_scaled = get_scaled(signal, scale)

    e0 = record_event(s0)
    wait_event(s1, e0)

    if should_skip_3d_unsupported(exec_backend, shape, axes):
        with pytest.raises(ValueError, match="The 3D batched FFT is not supported"):
            nvmath.fft.FFT(
                signal,
                axes=axes,
                stream=s1,
                execution=exec_backend.nvname,
                options={"result_layout": "natural"},
            )
        return

    with nvmath.fft.FFT(
        signal,
        axes=axes,
        options={"blocking": blocking.value, "result_layout": "natural"},
        execution=exec_backend.nvname,
        stream=s1,
    ) as f:
        f.plan(stream=s1)
        fft = f.execute(direction=Direction.forward.value, stream=s1)
        f.reset_operand(fft, stream=s1)
        # While blocking = True makes the execute synchronous,
        # the cpu -> gpu in reset_operand is always async
        if mem_backend == MemBackend.cpu or blocking == OptFftBlocking.auto:
            e1 = record_event(s1)
            wait_event(s2, e1)
        ifft = f.execute(direction=Direction.inverse.value, stream=s2)

        with use_stream(s2):
            assert_array_type(ifft, framework, mem_backend, dtype)
            assert_norm_close(ifft, signal_scaled, shape_kind=shape_kind, exec_backend=exec_backend)


@pytest.mark.parametrize(
    (
        "framework",
        "exec_backend",
        "mem_backend",
        "shape_kind",
        "shape",
        "axes",
        "dtype",
        "blocking",
    ),
    [
        (
            framework,
            exec_backend,
            mem_backend,
            shape_kind,
            repr(shape),
            repr(axes),
            dtype,
            blocking,
        )
        for framework in Framework.enabled()
        for exec_backend in [ExecBackend.cufft]
        if exec_backend in supported_backends.exec
        for mem_backend in supported_backends.framework_mem[framework]
        for shape_kind, shape, axes in [
            (ShapeKind.pow2, (641, 16, 16, 16), (1, 2, 3)),
            (ShapeKind.pow2, (4, 32, 32, 32), (1, 2, 3)),
            (ShapeKind.pow2, (32, 32, 32, 4), (0, 1, 2)),
            (ShapeKind.pow2, (4, 64, 64, 64), (1, 2, 3)),
            (ShapeKind.pow2, (64, 64, 64, 4), (0, 1, 2)),
            (ShapeKind.pow2, (128, 512, 17), (0, 1)),
            (ShapeKind.pow2, (17, 128, 512), (0, 1)),
            (ShapeKind.pow2, (64, 64, 641), (0, 1)),
            (ShapeKind.pow2, (4, 256), (1,)),
            (ShapeKind.pow2357, (2 * 3 * 5 * 7, 9, 7**2), (0, 1, 2)),
            (ShapeKind.pow2357, (4 * 49, 9 * 25, 13), (0, 1)),
            (ShapeKind.pow2357, (13, 4 * 49, 9 * 25), (1, 2)),
            (ShapeKind.prime, (31, 9973), (1,)),
            (ShapeKind.prime, (9973, 31), (0,)),
        ]
        for dtype in framework_exec_type_support[framework][exec_backend]
        if is_complex(dtype)
        and shape_kind in type_shape_support[exec_backend][dtype]
        and (not is_half(dtype) or math.prod(shape[a] for a in axes if a in axes) <= 1024)
        for blocking in OptFftBlocking
    ],
)
def test_custom_stream_inplace(seeder, framework, exec_backend, mem_backend, shape_kind, shape, axes, dtype, blocking):
    shape = literal_eval(shape)
    axes = literal_eval(axes)
    s0 = get_custom_stream(framework, is_numpy_stream_oriented=True)
    s1 = get_custom_stream(framework, is_numpy_stream_oriented=True)

    free_framework_pools(framework)

    with use_stream(s0):
        signal = get_random_input_data(framework, shape, dtype, mem_backend)
        scale = math.prod(shape[a] for a in axes)
        signal_scaled = get_scaled(signal, scale * 2)

    if should_skip_3d_unsupported(exec_backend, shape, axes):
        with pytest.raises(ValueError, match="The 3D batched FFT is not supported"):
            nvmath.fft.FFT(
                signal,
                axes=axes,
                stream=s0,
                options={"blocking": blocking.value, "inplace": True},
                execution=exec_backend.nvname,
            )
        return

    with nvmath.fft.FFT(
        signal,
        axes=axes,
        options={"blocking": blocking.value, "inplace": True},
        stream=s0,
        execution=exec_backend.nvname,
    ) as f:
        f.plan(stream=s0)
        f.execute(direction=Direction.forward.value, stream=s0)
        if blocking == OptFftBlocking.auto:
            e = record_event(s0)
            wait_event(s1, e)
        with use_stream(s1):
            add_in_place(signal, signal)
        # Even though we're running in place, for CPU, the internal GPU
        # copy is not affected by the add_in_place
        if mem_backend == MemBackend.cpu:
            f.reset_operand(signal, stream=s1)
        f.execute(direction=Direction.inverse.value, stream=s1)

        with use_stream(s1):
            assert_array_type(signal, framework, mem_backend, dtype)
            assert_norm_close(signal, signal_scaled, shape_kind=shape_kind, exec_backend=exec_backend)


@pytest.mark.parametrize(
    (
        "framework",
        "exec_backend",
        "mem_backend",
        "shape_kind",
        "shape",
        "axes",
        "dtype",
        "blocking",
        "result_layout",
    ),
    [
        (
            framework,
            exec_backend,
            mem_backend,
            shape_kind,
            repr(shape),
            repr(axes),
            dtype,
            blocking,
            result_layout,
        )
        for framework in Framework.enabled()
        for exec_backend in [ExecBackend.cufft]
        if exec_backend in supported_backends.exec
        for mem_backend in supported_backends.framework_mem[framework]
        for shape_kind, shape, axes in [
            (ShapeKind.pow2, (641, 16, 16, 16), (1, 2, 3)),
            (ShapeKind.pow2, (4, 32, 32, 32), (1, 2, 3)),
            (ShapeKind.pow2, (32, 32, 32, 4), (0, 1, 2)),
            (ShapeKind.pow2, (4, 64, 64, 64), (1, 2, 3)),
            (ShapeKind.pow2, (64, 64, 64, 4), (0, 1, 2)),
            (ShapeKind.pow2, (128, 512, 17), (0, 1)),
            (ShapeKind.pow2, (17, 128, 512), (0, 1)),
            (ShapeKind.pow2, (64, 64, 641), (0, 1)),
            (ShapeKind.pow2, (4, 256), (1,)),
            (ShapeKind.pow2357, (2 * 3 * 5 * 7, 9, 7**2), (0, 1, 2)),
            (ShapeKind.pow2357, (4 * 49, 9 * 25, 13), (0, 1)),
            (ShapeKind.pow2357, (13, 4 * 49, 9 * 25), (1, 2)),
            (ShapeKind.prime, (31, 9973), (1,)),
            (ShapeKind.prime, (9973, 31), (0,)),
        ]
        for dtype in framework_exec_type_support[framework][exec_backend]
        if shape_kind in type_shape_support[exec_backend][dtype]
        and (not is_half(dtype) or math.prod(shape[a] for a in axes if a in axes) <= 1024)
        for blocking in OptFftBlocking
        for result_layout in OptFftLayout
    ],
)
def test_custom_stream_busy_input(
    seeder,
    framework,
    exec_backend,
    mem_backend,
    shape_kind,
    shape,
    axes,
    dtype,
    blocking,
    result_layout,
):
    shape = literal_eval(shape)
    axes = literal_eval(axes)
    s0 = get_custom_stream(framework)

    free_framework_pools(framework)

    with use_stream(s0):
        signal = get_random_input_data(framework, shape, dtype, mem_backend)
        ref = get_scaled(get_fft_ref(signal, axes=axes), 4)

    if should_skip_3d_unsupported(exec_backend, shape, axes):
        with pytest.raises(ValueError, match="The 3D batched FFT is not supported"):
            nvmath.fft.FFT(
                signal,
                axes=axes,
                stream=s0,
                execution=exec_backend.nvname,
                options={
                    "blocking": blocking.value,
                    "result_layout": result_layout.value,
                },
            )
        return

    with nvmath.fft.FFT(
        signal,
        axes=axes,
        options={"blocking": blocking.value, "result_layout": result_layout.value},
        stream=s0,
        execution=exec_backend.nvname,
    ) as f:
        f.plan(stream=s0)
        with use_stream(s0):
            add_in_place(signal, signal)
            add_in_place(signal, signal)
            if mem_backend == MemBackend.cpu:
                f.reset_operand(signal, stream=s0)

        fft = f.execute(direction=Direction.forward.value, stream=s0)

        with use_stream(s0):
            assert_array_type(fft, framework, mem_backend, get_fft_dtype(dtype))
            assert_norm_close(fft, ref, shape_kind=shape_kind, exec_backend=exec_backend)


@pytest.mark.parametrize(
    (
        "framework",
        "exec_backend",
        "mem_backend",
        "dtype",
    ),
    [
        (
            framework,
            exec_backend,
            mem_backend,
            dtype,
        )
        for framework in Framework.enabled()
        for exec_backend in [ExecBackend.cufft]
        if exec_backend in supported_backends.exec
        for mem_backend in [MemBackend.cuda]
        if mem_backend in supported_backends.framework_mem[framework]
        for dtype in framework_exec_type_support[framework][exec_backend]
    ],
)
@multi_gpu_only
def test_arrays_different_devices(seeder, framework, exec_backend, mem_backend, dtype):
    shape = (512, 512)
    signal_0 = get_random_input_data(framework, shape, dtype, mem_backend, device_id=0)
    signal_1 = get_random_input_data(framework, shape, dtype, mem_backend, device_id=1)

    fft_0, fft_1 = None, None
    try:
        fft_0 = nvmath.fft.FFT(
            signal_0,
            options={"blocking": "auto"},
            execution=exec_backend.nvname,
        )
        fft_1 = nvmath.fft.FFT(
            signal_1,
            options={"blocking": "auto"},
            execution=exec_backend.nvname,
        )
        fft_0.plan()
        fft_1.plan()
        fft_0_out = fft_0.execute(direction=Direction.forward.value)
        fft_1_out = fft_1.execute(direction=Direction.forward.value)

    finally:
        if fft_1 is not None:
            fft_1.free()
        if fft_0 is not None:
            fft_0.free()

    assert_eq(get_array_device_id(fft_0_out), 0)
    assert_eq(get_array_device_id(fft_1_out), 1)
    assert_array_type(fft_0_out, framework, mem_backend, get_fft_dtype(dtype))
    assert_array_type(fft_1_out, framework, mem_backend, get_fft_dtype(dtype))

    assert_norm_close(fft_0_out, get_fft_ref(signal_0), exec_backend=exec_backend)
    with get_device_ctx(1, framework):
        assert_norm_close(fft_1_out, get_fft_ref(signal_1), exec_backend=exec_backend)


@pytest.mark.parametrize(
    (
        "framework",
        "exec_backend",
        "mem_backend",
    ),
    [
        (
            framework,
            exec_backend,
            mem_backend,
        )
        for framework in Framework.enabled()
        for exec_backend in [ExecBackend.cufft]
        if exec_backend in supported_backends.exec
        for mem_backend in [MemBackend.cuda]
        if mem_backend in supported_backends.framework_mem[framework]
    ],
)
@multi_gpu_only
def test_multi_device_wrong_device(framework, exec_backend, mem_backend):
    shape = (32, 128)
    dtype = DType.complex64

    signal_0 = get_random_input_data(framework, shape, dtype, mem_backend, device_id=0)
    signal_1 = get_random_input_data(framework, shape, dtype, mem_backend, device_id=1)

    with nvmath.fft.FFT(
        signal_0,
        axes=[0, 1],
        options={"blocking": "auto", "result_layout": "natural"},
        execution=exec_backend.nvname,
    ) as f:
        f.plan()
        f.execute()
        with pytest.raises(ValueError, match="The new operand must be on the same device"):
            f.reset_operand(signal_1)


@pytest.mark.parametrize(
    (
        "framework",
        "exec_backend",
        "mem_backend",
    ),
    [
        (framework, exec_backend, mem_backend)
        for framework in Framework.enabled()
        for exec_backend in supported_backends.exec
        for mem_backend in supported_backends.framework_mem[framework]
    ],
)
def test_unsupported_shape_change(framework, exec_backend, mem_backend):
    shape_0 = (512, 128)
    shape_1 = (256, 1024)
    dtype = DType.float64

    signal_0 = get_random_input_data(framework, shape_0, dtype, mem_backend)
    signal_1 = get_random_input_data(framework, shape_1, dtype, mem_backend)

    with nvmath.fft.FFT(signal_0, execution=exec_backend.nvname) as f:
        f.plan()
        f.execute(direction=Direction.forward.value)
        with pytest.raises(ValueError, match="The new operand's traits"):
            f.reset_operand(signal_1)


@pytest.mark.parametrize(
    (
        "framework",
        "exec_backend",
        "mem_backend",
        "direction",
        "fft_kind",
        "dtype",
    ),
    [
        (
            framework,
            exec_backend,
            exec_backend.mem,
            direction,
            fft_kind,
            dtype,
        )
        for framework in Framework.enabled()
        for exec_backend in supported_backends.exec
        if exec_backend.mem in supported_backends.framework_mem[framework]
        for direction, fft_kind in [
            (Direction.forward, OptFftType.c2r),
            (Direction.inverse, OptFftType.r2c),
            (-1, OptFftType.c2r),
            (1, OptFftType.r2c),
        ]
        for dtype in opt_fft_type_input_type_support[fft_kind]
        if dtype in framework_exec_type_support[framework][exec_backend]
        if not is_half(dtype)
    ],
)
def test_incorrect_fft_kind_direction(framework, exec_backend, mem_backend, direction, fft_kind, dtype):
    shape = (16,)
    signal = get_random_input_data(framework, shape, dtype, mem_backend)

    with nvmath.fft.FFT(
        signal,
        options={"blocking": "auto", "fft_type": fft_kind.value},
        execution=exec_backend.nvname,
    ) as fft:
        fft.plan()
        if isinstance(direction, int):
            direction_name = "FORWARD" if direction == -1 else "INVERSE"
            direction_value = direction
        else:
            direction_name = direction_value = direction.value
        with pytest.raises(
            ValueError,
            match=f"{direction_name} is not compatible with the FFT type '{fft_kind.value}'",
        ):
            fft.execute(direction=direction_value)


@pytest.mark.parametrize(
    (
        "framework",
        "exec_backend",
        "mem_backend",
        "dtype",
        "inplace",
    ),
    [
        (
            framework,
            exec_backend,
            mem_backend,
            dtype,
            inplace,
        )
        for framework in Framework.enabled()
        for exec_backend in supported_backends.exec
        for mem_backend in supported_backends.framework_mem[framework]
        for dtype in framework_exec_type_support[framework][exec_backend]
        if is_complex(dtype) and not is_half(dtype)
        for inplace in [False, True]
    ],
)
def test_reset_operand_forward_inverse(seeder, framework, exec_backend, mem_backend, dtype, inplace):
    shape = (128, 256, 3)
    axes = (0, 1)
    signal = get_random_input_data(framework, shape, dtype, mem_backend)
    fft_ref = get_fft_ref(signal, axes=axes)
    scale = math.prod(shape[a] for a in axes)
    ifft_ref = get_scaled(signal, scale)

    with nvmath.fft.FFT(
        signal,
        axes=axes,
        options={"blocking": "auto", "result_layout": "natural", "inplace": inplace},
        execution=exec_backend.nvname,
    ) as fft:
        fft.plan()

        out = fft.execute(direction=-1)
        assert_array_type(out, framework, mem_backend, dtype)
        assert_norm_close(out, fft_ref, exec_backend=exec_backend)

        if not inplace:
            fft.reset_operand(out)

        out2 = fft.execute(direction="inverse")
        assert_array_type(out2, framework, mem_backend, dtype)
        assert_norm_close(out2, ifft_ref, exec_backend=exec_backend)


@pytest.mark.parametrize(
    (
        "framework",
        "exec_backend",
        "mem_backend",
        "direction",
        "dtype",
    ),
    [
        (
            framework,
            exec_backend,
            exec_backend.mem,
            direction,
            dtype,
        )
        for framework in Framework.enabled()
        for exec_backend in [ExecBackend.fftw]
        if exec_backend in supported_backends.exec and exec_backend.mem in supported_backends.framework_mem[framework]
        for direction in [Direction.forward, -1, Direction.inverse, 1]
        for dtype in framework_exec_type_support[framework][exec_backend]
        if is_complex(dtype) and not is_half(dtype)
    ],
)
def test_reset_operand_unsupported_direction(framework, exec_backend, mem_backend, direction, dtype):
    shape = (3, 16, 3)
    axes = (1, 2)
    signal = get_random_input_data(framework, shape, dtype, mem_backend)

    if isinstance(direction, int):
        direction_value = direction
        other_direction = -direction
    else:
        direction_value = direction.value
        other_direction = (Direction.forward if direction == Direction.inverse else Direction.inverse).value

    with nvmath.fft.FFT(
        signal,
        axes=axes,
        execution=exec_backend.nvname,
    ) as fft:
        fft.plan(direction=direction_value)
        fft.execute(direction=direction_value)

        with pytest.raises(nvmath.bindings.nvpl.fft.FFTWError, match="Invalid plan handle"):
            fft.execute(direction=other_direction)


@pytest.mark.parametrize(
    (
        "framework",
        "exec_backend",
        "mem_backend",
        "dtype",
        "case",
    ),
    [
        (
            framework,
            exec_backend,
            mem_backend,
            rng.choice([dt for dt in framework_exec_type_support[framework][exec_backend] if not is_half(dt)]),
            case,
        )
        for framework in Framework.enabled()
        for exec_backend in supported_backends.exec
        for mem_backend in supported_backends.framework_mem[framework]
        for case in ["implicit_dict", "explicit_dict", "cls", "cls_default"]
        # execution can be inferred from memory backend or is passed explicitly
        if exec_backend.mem == mem_backend or case in ("explicit_dict", "cls")
    ],
)
def test_execution_options(seeder, framework, exec_backend, mem_backend, dtype, case):
    shape = (3, 5, 7)
    axes = (0, 1, 2)
    signal = get_random_input_data(framework, shape, dtype, mem_backend)
    fft_ref = get_fft_ref(signal, axes=axes)

    if exec_backend not in supported_backends.exec:
        pytest.skip("Backend unsupported on this machine")

    if exec_backend == ExecBackend.cufft:
        options = {"device_id": 0}
        cls = ExecutionCUDA
    else:
        options = {"num_threads": 16}
        cls = ExecutionCPU

    if case == "explicit_dict":
        expected_options = dict(**options)
        options["name"] = exec_backend.nvname.upper()
        expected_options["name"] = exec_backend.nvname.lower()
        assert len(options) == 2
    elif case == "cls":
        expected_options = dict(name=exec_backend.nvname.lower(), **options)
        assert len(options) == 1
        options = cls(**options)
    elif case == "cls_default":
        expected_options = {"name": exec_backend.nvname.lower()}
        if exec_backend == ExecBackend.cufft:
            expected_options["device_id"] = 0
        else:
            expected_options["num_threads"] = get_default_num_threads()
        options = cls()
    else:
        assert case == "implicit_dict"
        expected_options = dict(name=exec_backend.nvname.lower(), **options)
        assert len(options) == 1

    with nvmath.fft.FFT(signal, axes=axes, execution=options, options={"result_layout": "natural"}) as fft:
        assert len(expected_options) == 2
        for k, v in expected_options.items():
            assert getattr(fft.execution_options, k) == v

        fft.plan()
        out = fft.execute()
        assert_array_type(out, framework, mem_backend, get_fft_dtype(dtype))
        assert_norm_close(out, fft_ref, exec_backend=exec_backend)


@pytest.mark.parametrize(
    (
        "framework",
        "exec_backend",
        "mem_backend",
        "dtype",
    ),
    [
        (
            framework,
            exec_backend,
            exec_backend.mem,
            dtype,
        )
        for framework in Framework.enabled()
        for exec_backend in [ExecBackend.fftw]
        if exec_backend in supported_backends.exec
        if exec_backend.mem in supported_backends.framework_mem[framework]
        for dtype in framework_exec_type_support[framework][exec_backend]
    ],
)
def test_num_threads_option(seeder, framework, exec_backend, mem_backend, dtype):
    if get_default_num_threads() < 8:
        pytest.skip("Not enough cores to run the test")

    shape = (127, 256, 128)
    axes = (1, 2)
    signal = get_random_input_data(framework, shape, dtype, mem_backend)
    ref = get_fft_ref(signal, axes)

    with nvmath.fft.FFT(signal, axes=axes, execution={"name": "cpu", "num_threads": 16}) as fft:
        fft.plan()
        fft_out_mt = fft.execute()

    with nvmath.fft.FFT(signal, axes=axes, execution={"name": "cpu", "num_threads": 1}) as fft:
        fft.plan()
        fft_out_1 = fft.execute()

    assert_array_type(fft_out_mt, framework, mem_backend, get_fft_dtype(dtype))
    assert_array_type(fft_out_1, framework, mem_backend, get_fft_dtype(dtype))
    assert_norm_close(fft_out_mt, ref, exec_backend=exec_backend)
    assert_norm_close(fft_out_1, ref, exec_backend=exec_backend)


@pytest.mark.parametrize(
    (
        "framework",
        "exec_backend",
        "mem_backend",
        "dtype",
        "inplace",
        "shape",
        "axes",
    ),
    [
        (
            framework,
            exec_backend,
            mem_backend,
            dtype,
            inplace,
            repr(shape),
            repr(axes),
        )
        for framework in Framework.enabled()
        for exec_backend in [ExecBackend.fftw]
        if exec_backend in supported_backends.exec
        for mem_backend in ({MemBackend.cuda: MemBackend.cpu, MemBackend.cpu: MemBackend.cuda}[exec_backend.mem],)
        if mem_backend in supported_backends.framework_mem[framework]
        for dtype in framework_exec_type_support[framework][exec_backend]
        for inplace in [False, True]
        if not inplace or is_complex(dtype)
        for shape, axes in [
            ((16,), (0,)),
            ((64, 256, 512), (1, 2)),
            ((65, 31, 127, 17), (0, 1, 2)),
        ]
    ],
)
def test_cpu_gpu_copy_sync(seeder, framework, exec_backend, mem_backend, dtype, inplace, shape, axes):
    if get_default_num_threads() < 8:
        pytest.skip("Not enough cores to run the test")
    free_framework_pools(framework)

    s_1 = get_custom_stream(framework)
    s_2 = get_custom_stream(framework)
    shape = literal_eval(shape)
    axes = literal_eval(axes)

    with use_stream(s_1):
        signal = get_random_input_data(framework, shape, dtype, mem_backend)
        noise = get_random_input_data(framework, shape, dtype, mem_backend)
        ref = get_fft_ref(get_scaled(signal, 4), axes)
        # There is a bug in cupy's fft cache - the same plan
        # will be used for different streams, without any synchronization.
        # https://github.com/cupy/cupy/issues/8079
        s_1.synchronize()
        add_in_place(signal, signal)

    with use_stream(s_2):
        signal_2 = get_random_input_data(framework, shape, dtype, mem_backend)
        ref_2 = get_fft_ref(get_scaled(signal_2, 4), axes)

    with nvmath.fft.FFT(
        signal,
        axes=axes,
        execution={"name": "cpu", "num_threads": 8},
        options={"inplace": inplace},
        stream=s_1,
    ) as fft:
        fft.plan(stream=s_1)
        with use_stream(s_1):
            # this should have no effect on the output result
            add_in_place(signal, noise)
        out = fft.execute(stream=s_1)
        with use_stream(s_1):
            add_in_place(out, out)

        with use_stream(s_2):
            add_in_place(signal_2, signal_2)
        fft.reset_operand(signal_2, stream=s_2)
        out_2 = fft.execute(stream=s_2)
        with use_stream(s_2):
            add_in_place(out_2, out_2)

    with use_stream(s_1):
        assert_array_type(out, framework, mem_backend, get_fft_dtype(dtype))
        assert_norm_close(out, ref, exec_backend=exec_backend)

    with use_stream(s_2):
        assert_array_type(out_2, framework, mem_backend, get_fft_dtype(dtype))
        assert_norm_close(out_2, ref_2, exec_backend=exec_backend)

    if inplace:
        assert out is signal
        assert out_2 is signal_2


@pytest.mark.parametrize(
    (
        "framework",
        "exec_backend",
        "mem_backend",
        "dtype",
    ),
    [
        (
            framework,
            exec_backend,
            exec_backend.mem,
            dtype,
        )
        for framework in Framework.enabled()
        for exec_backend in [ExecBackend.fftw]
        if exec_backend in supported_backends.exec
        if exec_backend.mem in supported_backends.framework_mem[framework]
        for dtype in framework_exec_type_support[framework][exec_backend]
    ],
)
def test_direction_planning_stateful_vs_statless(seeder, monkeypatch, framework, exec_backend, mem_backend, dtype):
    rets = []

    def wrap_call(fn):
        def wrapper(*args, **kwargs):
            ret = fn(*args, **kwargs)
            rets.append(ret.__reduce__()[1][:2])
            return ret

        return wrapper

    from nvmath.bindings.nvpl import fft as fftw

    monkeypatch.setattr(fftw, "plan_many", wrap_call(fftw.plan_many))

    if is_complex(dtype):
        fft_fn = nvmath.fft.fft
        ifft_fn = nvmath.fft.ifft
    else:
        fft_fn = nvmath.fft.rfft
        ifft_fn = nvmath.fft.irfft

    signal = get_random_input_data(framework, (15, 16), dtype, mem_backend)
    assert len(rets) == 0

    out = fft_fn(signal, execution=exec_backend.nvname)
    assert len(rets) == 1
    plan_forward, plan_backward = rets[-1]
    assert plan_forward != 0
    assert plan_backward == 0

    ifft_fn(out, execution=exec_backend.nvname)
    assert len(rets) == 2
    plan_forward, plan_backward = rets[-1]
    assert plan_forward == 0
    assert plan_backward != 0

    with nvmath.fft.FFT(signal, execution=exec_backend.nvname) as fft:
        fft.plan()
        assert len(rets) == 3
        plan_forward, plan_backward = rets[-1]
        if is_complex(dtype):
            assert plan_forward != 0
            assert plan_backward != 0
        else:
            assert plan_forward != 0
            assert plan_backward == 0

    if is_complex(dtype):
        with nvmath.fft.FFT(signal, options={"fft_type": "C2R"}, execution=exec_backend.nvname) as fft:
            fft.plan()
            assert len(rets) == 4
            plan_forward, plan_backward = rets[-1]
            assert plan_forward == 0
            assert plan_backward != 0


@pytest.mark.parametrize(
    (
        "framework",
        "exec_backend",
        "mem_backend",
        "fft_type",
        "direction",
        "dtype",
        "shape",
        "axes",
        "inplace",
        "layout",
        "required_alignment",
    ),
    [
        (
            framework,
            exec_backend,
            exec_backend.mem,
            fft_type,
            direction,
            dtype,
            repr(shape),
            repr(axes),
            inplace,
            OptFftLayout.natural if not inplace else rng.choice(list(OptFftLayout)),
            size_of(get_fft_dtype(dtype)),
        )
        for framework in Framework.enabled()
        for exec_backend in supported_backends.exec
        if exec_backend.mem in supported_backends.framework_mem[framework]
        for fft_type in OptFftType
        for direction in opt_fft_type_direction_support[fft_type]
        for dtype in opt_fft_type_input_type_support[fft_type]
        if dtype in framework_exec_type_support[framework][exec_backend]
        for shape, axes in [
            ((1,), (0,)),
            ((5413, 13), (0,)),
            ((17, 1024), (1,)),
            ((13, 99), (0, 1)),
            ((32, 32, 11), (0, 1)),
            ((11, 32, 32), (1, 2)),
            ((2999, 6, 7), (0, 1, 2)),
            ((8, 16, 8, 31), (0, 1, 2)),
            ((31, 16, 16, 8), (1, 2, 3)),
        ]
        for inplace in [False, True]
        if not inplace or fft_type == OptFftType.c2c
    ],
)
def test_reset_operand_decreasing_alignment(
    seeder,
    framework,
    exec_backend,
    mem_backend,
    fft_type,
    direction,
    dtype,
    inplace,
    shape,
    axes,
    required_alignment,
    layout,
):
    """
    The test checks if the plan does not depend on the initial data alignment,
    especially host execution libs, which take the pointers to data
    during the planning.
    """
    free_framework_pools(framework)
    shape = literal_eval(shape)
    axes = literal_eval(axes)
    fft_dim = len(shape)
    if fft_type != OptFftType.c2r:
        problem_shape = shape
    else:
        # assume last_axis_parity = odd
        problem_shape = tuple(e if axes[-1] != i else 2 * (e - 1) + 1 for i, e in enumerate(shape))

    if should_skip_3d_unsupported(exec_backend, problem_shape, axes):
        pytest.skip("Older CTK does not support 3D FFT")

    # we will create the plan with the tensor starting at
    # the address aligned to exactly 512 bytes and then reset_operand
    # to tensors starting at the 256, 128, 64...-byte alignments,
    # stopping at the minimal required alignment, i.e. size_of(complex_dtype)
    # to make sure we are aligned to 512 and no more, we start with
    # tensor aligned to 1024 and set the 512 offset
    overalignment_lg2 = 10
    overalignment_bytes = 2**overalignment_lg2
    dtype_size = size_of(dtype)
    overalignment_elements = overalignment_bytes // dtype_size
    complex_dtype = get_fft_dtype(dtype)
    assert required_alignment == size_of(complex_dtype)
    assert overalignment_bytes % required_alignment == 0

    max_offset = overalignment_elements - 1
    last_extent = shape[-1]
    last_extent_sample_stride = dtype_size * (last_extent + max_offset)
    last_extent_sample_stride = align_up(last_extent_sample_stride, overalignment_bytes) // dtype_size
    assert last_extent_sample_stride >= last_extent + max_offset
    assert (dtype_size * last_extent_sample_stride) % overalignment_bytes == 0

    alignment_cases = tuple(
        (
            i,
            alignment,
            # Move to a new sample - to preserve input chracterictis for inplace cases.
            # Move by overaligned offset to preserve the overlignment.
            # Then shift from overaligned position to get exactly pointer
            # aligned to the ``alignment``
            offset := i * last_extent_sample_stride + (alignment // dtype_size),
            offset + last_extent,
        )
        for i, a_lg2 in enumerate(range(overalignment_lg2 - 1, -1, -1))
        if (alignment := 2**a_lg2) >= required_alignment
    )
    assert alignment_cases[0][1] == overalignment_bytes // 2
    assert alignment_cases[-1][1] == required_alignment

    all_samples_last_extent = len(alignment_cases) * last_extent_sample_stride
    all_view_shape = list(shape)
    all_view_shape[-1] = all_samples_last_extent
    all_view_shape = tuple(all_view_shape)
    signal_base, signal_overaligned = get_overaligned_view(overalignment_bytes, framework, all_view_shape, dtype, mem_backend)
    base_ptr = get_raw_ptr(signal_base)
    overaligned_ptr = get_raw_ptr(signal_overaligned)
    assert overaligned_ptr % overalignment_bytes == 0, f"{base_ptr}, {overaligned_ptr}"
    assert signal_overaligned.shape == all_view_shape
    assert 0 <= overaligned_ptr - base_ptr <= max_offset * dtype_size
    assert_array_type(signal_overaligned, framework, mem_backend, dtype)
    overaligned_copy = signal_overaligned if not inplace else copy_array(signal_overaligned)
    assert_array_equal(signal_overaligned, overaligned_copy)

    def decreasingly_aligned_signals():
        prev_end = 0
        for i, alignment, offset_start, offset_end in alignment_cases:
            assert offset_start < offset_end
            assert prev_end <= offset_start
            prev_end = offset_end
            slices = (slice(None),) * (fft_dim - 1) + (slice(offset_start, offset_end),)
            sample = signal_overaligned[slices]
            sample_copy = overaligned_copy[slices]
            assert sample.shape == shape
            assert sample_copy.shape == shape

            if fft_type == OptFftType.c2r:
                real_dtype = get_ifft_dtype(dtype, fft_type=fft_type)
                real_sample = get_random_input_data(framework, problem_shape, real_dtype, mem_backend)
                complex_sample = get_fft_ref(real_sample, axes=axes)
                assert complex_sample.shape == shape
                sample[:] = complex_sample[:]

            sample_ptr = get_raw_ptr(sample)
            assert sample_ptr % alignment == 0, f"{i}, {alignment}, {sample_ptr}, {overaligned_ptr}, {base_ptr}"
            assert sample_ptr % (2 * alignment) == alignment, f"{i}, {alignment}, {sample_ptr}, {overaligned_ptr}, {base_ptr}"
            assert_array_type(sample, framework, mem_backend, dtype)
            try:
                assert_array_equal(sample, sample_copy)
            except AssertionError as e:
                raise AssertionError(f"The copied sample is not equal for {alignment} (i={i})") from e

            yield i, alignment, sample, sample_copy

    samples = decreasingly_aligned_signals()
    i, alignment, sample, sample_copy = next(samples)
    assert i == 0
    assert 2 * alignment == overalignment_bytes

    if direction == Direction.forward:
        ref_fn = functools.partial(get_fft_ref, axes=axes)
    else:
        ref_fn = functools.partial(
            get_ifft_ref,
            axes=axes,
            is_c2c=fft_type == OptFftType.c2c,
            last_axis_parity="odd",
        )

    try:
        with nvmath.fft.FFT(
            sample,
            execution=exec_backend.nvname,
            options={
                "fft_type": fft_type.value,
                "inplace": inplace,
                "result_layout": layout.value,
                "last_axis_parity": "odd",
            },
            axes=axes,
        ) as fft:
            fft.plan()
            res = fft.execute(direction=direction.value.lower())
            ref = ref_fn(sample_copy)
            assert_norm_close(res, ref, exec_backend=exec_backend, axes=axes)
            for i, alignment, sample, sample_copy in samples:
                assert i > 0 and 2 * alignment < overalignment_bytes
                fft.reset_operand(sample)
                res = fft.execute(direction=direction.value.lower())
                ref = ref_fn(sample_copy)
                try:
                    assert_norm_close(res, ref, exec_backend=exec_backend, axes=axes)
                except AssertionError as e:
                    raise AssertionError(
                        f"The output and reference are not close for tesnor aligned to {alignment} (i={i})"
                    ) from e
    except ValueError as e:
        if (
            is_half(dtype)
            and exec_backend == ExecBackend.cufft
            and max(axes) < len(shape) - 1
            and (
                ("The R2C FFT of half-precision tensor" in str(e) and not is_complex(dtype) and fft_type == OptFftType.r2c)
                or ("The C2R FFT of half-precision tensor" in str(e) and is_complex(dtype) and fft_type == OptFftType.c2r)
            )
        ) or (
            "incompatible with that of the original" in str(e)
            and fft_type == OptFftType.c2r
            and direction == Direction.inverse
            and len(shape) > 1
        ):
            # reset_operand will reject the operand as non-compatible
            # because the operand layout is non-contiguous
            # so the internal copy changes it
            pass
        else:
            raise
    except nvmath.bindings.cufft.cuFFTError as e:
        if (
            ("CUFFT_NOT_SUPPORTED" in str(e) or "CUFFT_SETUP_FAILED" in str(e))
            and is_half(dtype)
            and exec_backend == ExecBackend.cufft
            and any(not is_pow_2(problem_shape[a]) for a in axes)
        ):
            pass
        else:
            raise


@pytest.mark.skipif(cp is None, reason="CuPy is not available")
@pytest.mark.parametrize("use_plan_execute", [False, True], ids=["no_plan", "with_plan"])
def test_reference_count(use_plan_execute):
    """
    Test reference counts consistency before/after context manager.
    Only need a single scenario with CuPy to test scenario when tensors reside on GPU.
    """
    shape = 512, 512, 512
    axes = 0, 1
    a = cp.ones(shape, dtype=cp.complex64)

    initial_refcount_a = sys.getrefcount(a)

    # Create and optionally execute
    f = nvmath.fft.FFT(a, axes=axes, execution="cuda")
    with f:
        if use_plan_execute:
            f.plan()
            result = f.execute()
            cp.cuda.get_current_stream().synchronize()
        else:
            pass

    assert sys.getrefcount(a) == initial_refcount_a, "post op: a refcount changed"
    if use_plan_execute:
        with check_freed_after(result, "post op: result should have sole ownership"):
            del result


@pytest.mark.parametrize(
    (
        "framework",
        "exec_backend",
        "mem_backend",
        "shape",
        "ndim",
        "axes",
    ),
    [
        (
            framework,
            exec_backend,
            mem_backend,
            shape,
            ndim,
            axes,
        )
        for framework in Framework.enabled()
        for exec_backend in supported_backends.exec
        for mem_backend in supported_backends.framework_mem[framework]
        for shape, ndim, axes in [
            ((128,), 1, (0,)),
            ((128,), 1, (-1,)),
            ((64, 128), 2, (1,)),
            ((64, 128), 2, (0, 1)),
            ((64, 128), 2, (-1,)),
        ]
    ],
)
def test_key_from_init_matches_create_key(
    seeder,
    framework,
    exec_backend,
    mem_backend,
    shape,
    ndim,
    axes,
):
    """
    This test verifies that using the same data to create an FFT object and
    calling get_key() on the object produces the same key as calling
    the static create_key() method.
    """

    dtype = DType.complex64

    # Generate random operand
    signal = get_random_input_data(framework, shape, dtype, mem_backend)

    # Create FFT object and get key from instance method
    with nvmath.fft.FFT(signal, axes=axes, execution=exec_backend.nvname) as f:
        f.plan()
        key_from_instance = f.get_key()

    # Get key from static method
    key_from_static = nvmath.fft.FFT.create_key(signal, axes=axes, execution=exec_backend.nvname)

    # Verify they match
    assert key_from_instance == key_from_static, (
        f"Keys do not match for ndim={ndim}, axes={axes}\nInstance key: {key_from_instance}\nStatic key: {key_from_static}"
    )


@pytest.mark.parametrize("exec_backend", supported_backends.exec)
def test_key_cross_device_strided_operand(exec_backend):
    """
    create_key() and get_key() should return the same key for cross-device execution
    with a strided (non-contiguous) NumPy array as user-provided operand.
    """

    base_shape = 10, 30, 40
    a = np.arange(np.prod(base_shape), dtype=np.float32).reshape(base_shape)
    # Create a strided (non-contiguous) array via slicing
    a = a[4:, 3:, 1:]

    axes = (-2, -1)
    options = {"fft_type": "R2C"}

    # Get key from static method (computes from user's original operand)
    key_from_static = nvmath.fft.FFT.create_key(a, axes=axes, options=options, execution=exec_backend.nvname)

    # Create FFT object and get key from instance method
    with nvmath.fft.FFT(a, axes=axes, options=options, execution=exec_backend.nvname) as fft:
        fft.plan()
        key_from_instance = fft.get_key()

    # Verify they match
    assert key_from_instance == key_from_static, (
        f"Keys do not match for cross-device strided operand.\nInstance key: {key_from_instance}\nStatic key: {key_from_static}"
    )


@pytest.mark.parametrize(
    ("framework", "exec_backend"),
    [
        (framework, exec_backend)
        for framework in Framework.enabled()
        for exec_backend in supported_backends.exec
        if exec_backend.mem in supported_backends.framework_mem[framework]
    ],
)
def test_c2r_same_space_no_operand_ref(framework, exec_backend):
    """
    For C2R when execution and memory spaces are the same, after __init__
    completes, the FFT object does not hold a reference to the user-provided
    operand because it's copied into an internal buffer.
    The user operand's refcount should be the same as before __init__.
    This test verifies this assumption.
    """
    shape = (15, 16)
    dtype = DType.complex64
    mem_backend = exec_backend.mem

    a = get_random_input_data(framework, shape, dtype, mem_backend)
    initial_refcount = sys.getrefcount(a)

    fft = nvmath.fft.FFT(a, options={"fft_type": "C2R"}, execution=exec_backend.nvname)

    assert sys.getrefcount(a) == initial_refcount, (
        f"C2R same-space: FFT.__init__ should not hold a reference to the user operand. "
        f"Refcount after init: {sys.getrefcount(a)}, expected: {initial_refcount}"
    )

    fft.free()


@pytest.mark.parametrize(
    ("framework", "exec_backend", "mem_backend", "fft_type", "use_plan"),
    [
        (framework, exec_backend, mem_backend, fft_type, use_plan)
        for framework in Framework.enabled()
        for exec_backend in supported_backends.exec
        for mem_backend in supported_backends.framework_mem[framework]
        for fft_type in ["C2C", "C2R", "R2C"]
        for use_plan in [False, True]
    ],
)
def test_release_operand(framework, exec_backend, mem_backend, fft_type, use_plan):
    """
    Test that after release_operand(), the refcount of the user-provided
    operand returns to its initial value, both with and without planning/executing.
    """
    shape = (15, 16)
    dtype = DType.float32 if fft_type == "R2C" else DType.complex64
    a = get_random_input_data(framework, shape, dtype, mem_backend)
    initial_refcount = sys.getrefcount(a)

    fft = nvmath.fft.FFT(a, options={"fft_type": fft_type}, execution=exec_backend.nvname)
    if use_plan:
        fft.plan()
        result = fft.execute()
        with check_freed_after(result, "The caller should hold the only reference to the result buffer"):
            del result

    fft.release_operand()

    assert sys.getrefcount(a) == initial_refcount, (
        f"release_operand did not restore refcount for fft_type={fft_type}, use_plan={use_plan}. "
        f"Refcount after release: {sys.getrefcount(a)}, expected: {initial_refcount}"
    )

    fft.free()


@pytest.mark.parametrize(
    ("framework", "mem_backend", "fft_type"),
    [
        (framework, mem_backend, fft_type)
        for framework in Framework.enabled()
        for mem_backend in [MemBackend.cpu, MemBackend.cuda]
        if mem_backend in supported_backends.framework_mem[framework]
        for fft_type in ["C2C", "C2R", "R2C"]
    ],
)
def test_execute_after_release_operand_raises(framework, mem_backend, fft_type):
    """
    Test that calling execute() after release_operand() raises a RuntimeError.
    """
    if ExecBackend.cufft not in supported_backends.exec:
        pytest.skip("cufft not available")

    shape = (15, 16)
    dtype = DType.float32 if fft_type == "R2C" else DType.complex64
    a = get_random_input_data(framework, shape, dtype, mem_backend)

    fft = nvmath.fft.FFT(a, options={"fft_type": fft_type}, execution="cuda")
    fft.plan()
    fft.execute()

    fft.release_operand()
    with pytest.raises(RuntimeError, match="cannot be performed after the operand has been released"):
        fft.execute()

    # Verify the plan is still usable after release_operand.
    a_new = get_random_input_data(framework, shape, dtype, mem_backend)
    fft.reset_operand(a_new)
    result = fft.execute()
    if fft_type == "C2R":
        # last axis is even, so we need to use the even parity
        ref = get_ifft_ref(a_new, axes=list(range(len(shape))), is_c2c=False, last_axis_parity="even")
    else:
        ref = get_fft_ref(a_new)
    assert_norm_close(result, ref)

    fft.free()


@pytest.mark.parametrize(
    "case",
    [
        pytest.param("gpu_operand_gpu_exec", id="gpu_operand_gpu_exec"),
        pytest.param("gpu_operand_gpu_exec_c2r", id="gpu_operand_gpu_exec_c2r"),
        pytest.param("cpu_operand_gpu_exec", id="cpu_operand_gpu_exec"),
        pytest.param("gpu_operand_cpu_exec", id="gpu_operand_cpu_exec"),
    ],
)
def test_release_then_reset_unchecked(case):
    """
    Test that reset_operand_unchecked works after release_operand for all three cases.
    """
    cp = pytest.importorskip("cupy")
    shape = (64,)

    if case == "gpu_operand_gpu_exec":
        a = cp.random.rand(*shape).astype(cp.complex64)
        a_new = cp.random.rand(*shape).astype(cp.complex64)
        fft_type = "C2C"
        execution = "cuda"
    elif case == "gpu_operand_gpu_exec_c2r":
        c2r_shape = (shape[0] // 2 + 1,)
        a = cp.random.rand(*c2r_shape).astype(cp.complex64)
        a_new = cp.random.rand(*c2r_shape).astype(cp.complex64)
        fft_type = "C2R"
        execution = "cuda"
    elif case == "cpu_operand_gpu_exec":
        a = np.random.rand(*shape).astype(np.complex64)
        a_new = np.random.rand(*shape).astype(np.complex64)
        fft_type = "C2C"
        execution = "cuda"
    else:
        a = cp.random.rand(*shape).astype(cp.complex64)
        a_new = cp.random.rand(*shape).astype(cp.complex64)
        fft_type = "C2C"
        execution = "cpu"

    if fft_type == "C2R":
        ref = get_ifft_ref(a_new, axes=list(range(len(shape))), is_c2c=False, last_axis_parity="even")
    else:
        ref = get_fft_ref(a_new)

    try:
        fft = nvmath.fft.FFT(a, options={"fft_type": fft_type}, execution=execution)
    except RuntimeError as e:
        if "CPU execution is not available" in str(e):
            pytest.skip("FFTW not available")
        raise

    fft.plan()
    fft.execute()
    fft.release_operand()
    fft.reset_operand_unchecked(a_new)
    result = fft.execute()
    assert_norm_close(result, ref)

    fft.free()


@pytest.mark.parametrize(
    ("framework", "exec_backend"),
    [
        (framework, exec_backend)
        for framework in Framework.enabled()
        for exec_backend in supported_backends.exec
        if exec_backend.mem in supported_backends.framework_mem[framework]
    ],
)
def test_c2r_stale_operand_raises_on_second_execute(framework, exec_backend):
    """
    For C2R, calling execute() a second time without reset_operand() must raise
    a RuntimeError because cuFFT/FFTW destroys the internal operand copy during
    the first execute.
    """
    shape = (15, 16)
    dtype = DType.complex64
    mem_backend = exec_backend.mem

    a = get_random_input_data(framework, shape, dtype, mem_backend)

    with nvmath.fft.FFT(a, options={"fft_type": "C2R"}, execution=exec_backend.nvname) as f:
        f.plan()
        f.execute()

        with pytest.raises(RuntimeError, match="C2R FFTs.*execute.*cannot be called"):
            f.execute()


@pytest.mark.parametrize(
    ("framework", "exec_backend"),
    [
        (framework, exec_backend)
        for framework in Framework.enabled()
        for exec_backend in supported_backends.exec
        if exec_backend.mem in supported_backends.framework_mem[framework]
    ],
)
def test_c2r_reset_operand_between_executes(framework, exec_backend):
    """
    For C2R, calling reset_operand() or reset_operand_unchecked() between
    execute() calls must produce correct results each time.
    """
    shape = (64, 128)
    dtype = DType.complex64
    mem_backend = exec_backend.mem

    a = get_random_input_data(framework, shape, dtype, mem_backend)
    ref = get_ifft_ref(a, axes=[0, 1], is_c2c=False, last_axis_parity="even")

    with nvmath.fft.FFT(a, options={"fft_type": "C2R"}, execution=exec_backend.nvname) as f:
        f.plan()

        r1 = f.execute()
        assert_norm_close(r1, ref)

        f.reset_operand(a)
        r2 = f.execute()
        assert_norm_close(r2, ref)

        f.reset_operand_unchecked(a)
        r3 = f.execute()
        assert_norm_close(r3, ref)


@pytest.mark.parametrize(
    ("framework", "exec_backend", "mem_backend", "shape1", "shape2", "axes"),
    [
        (framework, exec_backend, mem_backend, shape1, shape2, axes)
        for framework in Framework.enabled()
        for exec_backend in supported_backends.exec
        for mem_backend in supported_backends.framework_mem[framework]
        for shape1, shape2, axes in [
            # 1D FFT with rearranged batch dims
            ((2, 3, 64), (3, 2, 64), (2,)),
            ((4, 2, 32), (2, 4, 32), (2,)),
            # 2D FFT with rearranged batch dims
            ((2, 3, 16, 32), (3, 2, 16, 32), (2, 3)),
            ((5, 2, 8, 16), (2, 5, 8, 16), (2, 3)),
        ]
    ],
)
def test_rearranged_batch_dims_produce_same_key(framework, exec_backend, mem_backend, shape1, shape2, axes):
    """
    Verify that rearranging batch dimensions produces the same FFT key
    for both 1D and 2D FFTs.
    This is the premise for test_reset_operand_with_rearranged_batch_dims.

    For the 1D case with shape1=(2,3,64), shape2=(3,2,64), axes=(2,):

                            Operand 1           Operand 2
                            ---------           ---------
    shape                   (2, 3, 64)          (3, 2, 64)          <- different
    strides                 (192, 64, 1)        (128, 64, 1)        <- different

    FFT axis (2):
      stride                1                   1                   <- same
      size                  64                  64                  <- same

    Batch axes (0, 1):
      batch_size            2 x 3 = 6           3 x 2 = 6           <- same
      sorted batch strides  (64, 192)           (64, 128)           <- different

    cuFFT plan parameters, the full key includes all of these:
      istride               1                   1                   <- same
      idistance             64 (min batch str)  64 (min batch str)  <- same
      ostride               1                   1                   <- same
      odistance             64                  64                  <- same
      fft_batch_size        6                   6                   <- same
      embedding_shape       (64,)               (64,)               <- same
      fft_in/out_shape      (64,)               (64,)               <- same
      data types            all identical                           <- same

    cuFFT treats batches as a flat sequence with a single distance
    parameter; it does not see the multi-dimensional batch layout.
    Since all key components match, the key is the same.
    """
    dtype = DType.complex64
    operand1 = get_random_input_data(framework, shape1, dtype, mem_backend)
    operand2 = get_random_input_data(framework, shape2, dtype, mem_backend)

    key1 = nvmath.fft.FFT.create_key(operand1, axes=axes, execution=exec_backend.nvname)
    key2 = nvmath.fft.FFT.create_key(operand2, axes=axes, execution=exec_backend.nvname)
    assert key1 == key2, (
        f"Expected matching keys for rearranged batch dims.\n"
        f"  shape1={shape1}, shape2={shape2}, axes={axes}\n"
        f"  key1={key1}\n  key2={key2}"
    )


@pytest.mark.parametrize(
    ("shape", "transpose_axes", "fft_axes"),
    [
        # 1D FFT: transpose batch dims of a (3,2,64) source to get (2,3,64)
        # with non-contiguous strides
        ((3, 2, 64), (1, 0, 2), (2,)),
        # 2D FFT: transpose batch dims
        ((3, 2, 16, 32), (1, 0, 2, 3), (2, 3)),
    ],
)
def test_transposed_strides_produce_same_key(shape, transpose_axes, fft_axes):
    """
    Verify that a contiguous operand and a transposed view with the same
    shape but different strides produce the same FFT key.
    This is the premise for test_reset_operand_with_transposed_strides.

    For the 1D case with shape=(3,2,64), transpose_axes=(1,0,2),
    fft_axes=(2,):

                            Operand 1           Operand 2
                            ---------           ---------
    shape                   (2, 3, 64)          (2, 3, 64)          <- same
    strides                 (192, 64, 1)        (64, 128, 1)        <- different

    FFT axis (2):
      stride                1                   1                   <- same
      size                  64                  64                  <- same

    Batch axes (0, 1):
      batch_size            2 x 3 = 6           2 x 3 = 6           <- same
      sorted batch strides  (64, 192)           (64, 128)           <- different

    cuFFT plan parameters, the full key includes all of these:
      istride               1                   1                   <- same
      idistance             64 (min batch str)  64 (min batch str)  <- same
      ostride               1                   1                   <- same
      odistance             64 (min result      64 (min result      <- same
                              batch stride)       batch stride)
      fft_batch_size        6                   6                   <- same
      embedding_shape       (64,)               (64,)               <- same
      fft_in/out_shape      (64,)               (64,)               <- same
      data types            all identical                           <- same

    cuFFT treats batches as a flat sequence with a single distance
    parameter; it does not see the multi-dimensional batch layout.
    Since all key components match, the key should be the same.
    """
    operand1_shape = tuple(shape[a] for a in transpose_axes)

    dtype = np.complex64
    operand1 = np.random.rand(*operand1_shape).astype(dtype)
    operand2 = np.random.rand(*shape).astype(dtype).transpose(transpose_axes)

    assert operand1.shape == operand2.shape
    assert operand1.strides != operand2.strides

    key1 = nvmath.fft.FFT.create_key(operand1, axes=fft_axes, execution="cuda")
    key2 = nvmath.fft.FFT.create_key(operand2, axes=fft_axes, execution="cuda")
    assert key1 == key2, (
        f"Expected matching keys for same shape with different strides.\n"
        f"  shape={operand1.shape}\n"
        f"  operand1 strides={operand1.strides}, operand2 strides={operand2.strides}\n"
        f"  key1={key1}\n  key2={key2}"
    )


def _reset_operand_different_shape_strides_impl(
    operand1, operand2, axes, exec_backend, reset_fn, result_layout="optimized", expect_optimized_layout=None
):
    """Shared logic for reset-operand tests: plan with operand1, execute,
    reset to operand2, verify class invariants and execution correctness.
    """
    ref1 = get_fft_ref(operand1, axes=list(axes))
    ref2 = get_fft_ref(operand2, axes=list(axes))

    options = nvmath.fft.FFTOptions(result_layout=result_layout)

    fft = nvmath.fft.FFT(operand1, axes=axes, execution=exec_backend.nvname, options=options)
    with fft:
        fft.plan()

        in_shape, in_strides = fft.get_input_layout()
        assert_eq(in_shape, operand1.shape)
        assert_eq(in_strides, get_array_element_strides(operand1))

        result1 = fft.execute()
        assert_norm_close(result1, ref1, exec_backend=exec_backend)

        if reset_fn == "checked":
            fft.reset_operand(operand2)
        else:
            fft.reset_operand_unchecked(operand2)

        # Class invariant: internal_operand_layout reflects the internal buffer.
        assert_eq(fft.internal_operand_layout.shape, tuple(fft.operand.shape))
        assert_eq(tuple(fft.internal_operand_layout.strides), tuple(fft.operand.strides))

        # get_input_layout() must be consistent with internal_operand_layout.
        in_shape2, in_strides2 = fft.get_input_layout()
        assert_eq(in_shape2, fft.internal_operand_layout.shape)
        assert_eq(in_strides2, fft.internal_operand_layout.strides)

        # After reset, plan_traits.result_shape/strides must match what a
        # fresh get_fft_plan_traits would produce for the internal buffer.
        fresh_traits = get_fft_plan_traits(
            fft.operand.shape,
            fft.operand.strides,
            fft.operand.dtype,
            fft.axes,
            fft.execution_options,
            fft_abstract_type=fft.fft_abstract_type,
            last_axis_parity=fft.options.last_axis_parity,
            result_layout=fft.options.result_layout,
        )
        assert_eq(tuple(fft.plan_traits.result_shape), tuple(fresh_traits.result_shape))
        assert_eq(tuple(fft.plan_traits.result_strides), tuple(fresh_traits.result_strides))

        # Verify the optimized branch was actually exercised:
        # for interleaved cases, the post-reset result_strides must differ from
        # what the natural formula (axis_order_in_memory) would produce.
        if expect_optimized_layout:
            natural_order = axis_order_in_memory(fft.operand.shape, fft.operand.strides)
            natural_strides = tuple(calculate_strides(list(fft.plan_traits.result_shape), natural_order))
            assert tuple(fft.plan_traits.result_strides) != natural_strides, (
                f"Expected optimized result_strides to differ from natural, but both are {natural_strides}"
            )

        result2 = fft.execute()
        assert_norm_close(result2, ref2, exec_backend=exec_backend)


@pytest.mark.parametrize(
    ("framework", "exec_backend", "mem_backend", "shape1", "shape2", "axes", "result_layout", "expect_optimized", "reset_fn"),
    [
        (framework, exec_backend, mem_backend, *case, reset_fn)
        for framework in Framework.enabled()
        for exec_backend in supported_backends.exec
        for mem_backend in supported_backends.framework_mem[framework]
        for case in [
            # Same shape — basic contract test (non-interleaved)
            ((64, 64), (64, 64), (0, 1), "optimized", False),
            # 1D FFT with rearranged batch dims (non-interleaved)
            ((2, 3, 64), (3, 2, 64), (2,), "optimized", False),
            ((4, 2, 32), (2, 4, 32), (2,), "optimized", False),
            # 2D FFT with rearranged batch dims (non-interleaved)
            ((2, 3, 16, 32), (3, 2, 16, 32), (2, 3), "optimized", False),
            # Interleaved samples (FFT on leading axis, batch on trailing axes).
            # These trigger optimized_result_layout=True with result_layout="optimized"
            # and natural layout with result_layout="natural".
            ((64, 2, 3), (64, 3, 2), (0,), "optimized", True),
            ((64, 2, 3), (64, 3, 2), (0,), "natural", False),
            ((16, 32, 2, 3), (16, 32, 3, 2), (0, 1), "optimized", True),
            ((16, 32, 2, 3), (16, 32, 3, 2), (0, 1), "natural", False),
        ]
        for reset_fn in ["checked", "unchecked"]
    ],
)
def test_reset_operand_with_rearranged_batch_dims(
    framework, exec_backend, mem_backend, shape1, shape2, axes, result_layout, expect_optimized, reset_fn
):
    """
    Reset the operand to one with rearranged batch dimensions but the
    same FFT key and verify that layout queries and execution
    produce correct results.
    """
    dtype = DType.complex64
    operand1 = get_random_input_data(framework, shape1, dtype, mem_backend)
    operand2 = get_random_input_data(framework, shape2, dtype, mem_backend)
    _reset_operand_different_shape_strides_impl(
        operand1, operand2, axes, exec_backend, reset_fn, result_layout, expect_optimized
    )


@pytest.mark.parametrize(
    (
        "framework",
        "exec_backend",
        "mem_backend",
        "source_shape",
        "transpose_axes",
        "fft_axes",
        "result_layout",
        "expect_optimized",
        "reset_fn",
    ),
    [
        (framework, exec_backend, mem_backend, *case, reset_fn)
        for framework in Framework.enabled()
        for exec_backend in supported_backends.exec
        for mem_backend in supported_backends.framework_mem[framework]
        for case in [
            # 1D FFT: transpose batch dims (non-interleaved)
            ((3, 2, 64), (1, 0, 2), (2,), "optimized", False),
            # 2D FFT: transpose batch dims (non-interleaved)
            ((3, 2, 16, 32), (1, 0, 2, 3), (2, 3), "optimized", False),
            # Interleaved samples: FFT on leading axis, transpose trailing
            # batch dims. Exercises optimized_result_layout branch.
            ((64, 3, 2), (0, 2, 1), (0,), "optimized", True),
            ((64, 3, 2), (0, 2, 1), (0,), "natural", False),
        ]
        for reset_fn in ["checked", "unchecked"]
    ],
)
def test_reset_operand_with_transposed_strides(
    framework,
    exec_backend,
    mem_backend,
    source_shape,
    transpose_axes,
    fft_axes,
    result_layout,
    expect_optimized,
    reset_fn,
):
    """
    Reset the operand to one with the same shape but different strides
    (transposed batch dimensions) but the same FFT key and verify that
    layout queries and execution produce correct results.
    See test_transposed_strides_produce_same_key for why the keys match.
    """
    dtype = DType.complex64
    target_shape = tuple(source_shape[a] for a in transpose_axes)

    operand1 = get_random_input_data(framework, target_shape, dtype, mem_backend)
    source = get_random_input_data(framework, source_shape, dtype, mem_backend)
    operand2 = source.permute(transpose_axes) if framework == Framework.torch else source.transpose(transpose_axes)

    assert operand1.shape == operand2.shape
    assert get_array_element_strides(operand1) != get_array_element_strides(operand2)

    _reset_operand_different_shape_strides_impl(
        operand1, operand2, fft_axes, exec_backend, reset_fn, result_layout, expect_optimized
    )
