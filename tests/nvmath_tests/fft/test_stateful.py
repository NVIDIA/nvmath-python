# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import os
import logging
import random
import math
import time
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

import nvmath
from nvmath.memory import _RawCUDAMemoryManager
from nvmath.fft import ExecutionCPU, ExecutionCUDA

from .utils.common_axes import (
    ExecBackend,
    MemBackend,
    Framework,
    DType,
    OptFftLayout,
    Direction,
    ShapeKind,
    OptFftBlocking,
    OptFftType,
    Direction,
)
from .utils.axes_utils import (
    is_complex,
    is_half,
    get_fft_dtype,
)
from .utils.support_matrix import (
    framework_exec_type_support,
    supported_backends,
    type_shape_support,
    multi_gpu_only,
    opt_fft_type_input_type_support,
)
from .utils.input_fixtures import (
    get_random_input_data,
    get_custom_stream,
    init_assert_exec_backend_specified,
)
from .utils.check_helpers import (
    get_fft_ref,
    get_scaled,
    record_event,
    use_stream,
    assert_norm_close,
    assert_array_type,
    assert_eq,
    get_array_device_id,
    get_array_strides,
    is_decreasing,
    intercept_default_allocations,
    add_in_place,
    free_cupy_pool,
    should_skip_3d_unsupported,
    copy_array,
)


rng = random.Random(42)


# DO NOT REMOVE, this call creates a fixture that enforces
# specifying execution option to the FFT calls in tests
# defined in this file
assert_exec_backend_specified = init_assert_exec_backend_specified()


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

    signal_0 = get_random_input_data(framework, shape, dtype, mem_backend, seed=55)
    signal_1 = get_random_input_data(framework, shape, dtype, mem_backend, seed=56)

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
        assert all(
            allocations[key] == 0 for key in allocations if key != expected_key
        ), f"{allocations}, {expected_key}"

        f.reset_operand(signal_1)
        fft_1 = f.execute()

        assert allocations[expected_key] == expected_allocations, f"{allocations}, {expected_key}"
        assert all(
            allocations[key] == 0 for key in allocations if key != expected_key
        ), f"{allocations}, {expected_key}"

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
def test_stateful_nd_custom_allocator(monkeypatch, framework, exec_backend, mem_backend, fft_dim, dtype):
    fft_dim_shape = {
        1: (512,),
        2: (256, 512),
        3: (64, 128, 32),
    }
    shape = fft_dim_shape[fft_dim]

    signal = get_random_input_data(framework, shape, dtype, mem_backend, seed=44)

    allocations = intercept_default_allocations(monkeypatch)
    logger = logging.getLogger("dummy_logger")
    allocator = _RawCUDAMemoryManager(device_id=0, logger=logger)
    expected_allocations = 1 if exec_backend == ExecBackend.cufft else 0
    expected_key = "raw"

    with nvmath.fft.FFT(signal, execution=exec_backend.nvname, options={"allocator": allocator}) as f:
        f.plan()
        fft = f.execute(direction=Direction.forward.value)

        assert allocations[expected_key] == expected_allocations, f"{allocations}, {expected_key}"

        f.reset_operand(fft)
        ifft = f.execute(direction=Direction.inverse.value)

        assert allocations[expected_key] == expected_allocations, f"{allocations}, {expected_key}"
        assert all(
            allocations[key] == 0 for key in allocations if key != expected_key
        ), f"{allocations}, {expected_key}"

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
def test_stateful_release_workspace(monkeypatch, framework, exec_backend, mem_backend, release_workspace):
    shape = (2048, 128)
    dtype = DType.float32

    signal_0 = get_random_input_data(framework, shape, dtype, mem_backend, seed=44)
    signal_1 = get_random_input_data(framework, shape, dtype, mem_backend, seed=45)

    allocations = intercept_default_allocations(monkeypatch)
    expected_key = "torch" if framework == Framework.torch else "cupy"

    num_allocs_1, num_allocs_2 = (1, 2) if exec_backend == ExecBackend.cufft else (0, 0)

    with nvmath.fft.FFT(signal_0, execution=exec_backend.nvname) as f:
        f.plan()
        fft_0 = f.execute(direction=Direction.forward.value, release_workspace=release_workspace)

        assert allocations[expected_key] == num_allocs_1, f"{allocations}, {expected_key}"

        f.reset_operand(signal_1)
        fft_1 = f.execute(direction=Direction.forward.value, release_workspace=release_workspace)

        assert allocations[expected_key] == num_allocs_2 if release_workspace else 1, f"{allocations}, {expected_key}"
        assert all(
            allocations[key] == 0 for key in allocations if key != expected_key
        ), f"{allocations}, {expected_key}"

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
def test_custom_stream(framework, exec_backend, mem_backend, shape_kind, shape, axes, dtype, blocking):
    shape = literal_eval(shape)
    axes = literal_eval(axes)

    s0 = get_custom_stream(framework)
    s1 = get_custom_stream(framework)
    s2 = get_custom_stream(framework)

    if framework != Framework.cupy:
        # for less memory footprint of the whole suite
        free_cupy_pool()

    with use_stream(s0):
        signal = get_random_input_data(framework, shape, dtype, mem_backend, seed=44)
        scale = math.prod(shape[a] for a in axes)
        signal_scaled = get_scaled(signal, scale)

    e0 = record_event(s0)
    s1.wait_event(e0)

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
            s2.wait_event(e1)
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
def test_custom_stream_inplace(framework, exec_backend, mem_backend, shape_kind, shape, axes, dtype, blocking):
    shape = literal_eval(shape)
    axes = literal_eval(axes)
    s0 = get_custom_stream(framework)
    s1 = get_custom_stream(framework)

    if framework != Framework.cupy:
        # for less memory footprint of the whole suite
        free_cupy_pool()

    with use_stream(s0):
        signal = get_random_input_data(framework, shape, dtype, mem_backend, seed=44)
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
            s1.wait_event(e)
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

    if framework != Framework.cupy:
        # for less memory footprint of the whole suite
        free_cupy_pool()

    with use_stream(s0):
        signal = get_random_input_data(framework, shape, dtype, mem_backend, seed=44)
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
def test_arrays_different_devices(framework, exec_backend, mem_backend, dtype):
    shape = (512, 512)
    signal_0 = get_random_input_data(framework, shape, dtype, mem_backend, seed=13, device_id=0)
    signal_1 = get_random_input_data(framework, shape, dtype, mem_backend, seed=12, device_id=1)

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
    with cp.cuda.Device(1):
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

    signal_0 = get_random_input_data(framework, shape, dtype, mem_backend, seed=44, device_id=0)
    signal_1 = get_random_input_data(framework, shape, dtype, mem_backend, seed=45, device_id=1)

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

    signal_0 = get_random_input_data(framework, shape_0, dtype, mem_backend, seed=44)
    signal_1 = get_random_input_data(framework, shape_1, dtype, mem_backend, seed=45)

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
    signal = get_random_input_data(framework, shape, dtype, mem_backend, seed=13)

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
def test_reset_operand_forward_inverse(framework, exec_backend, mem_backend, dtype, inplace):
    shape = (128, 256, 3)
    axes = (0, 1)
    signal = get_random_input_data(framework, shape, dtype, mem_backend, seed=13)
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
    signal = get_random_input_data(framework, shape, dtype, mem_backend, seed=13)

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
def test_execution_options(framework, exec_backend, mem_backend, dtype, case):
    shape = (3, 5, 7)
    axes = (0, 1, 2)
    signal = get_random_input_data(framework, shape, dtype, mem_backend, seed=13)
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
            expected_options["num_threads"] = len(os.sched_getaffinity(0))
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
def test_num_threads_option(framework, exec_backend, mem_backend, dtype):
    if len(os.sched_getaffinity(0)) < 16:
        pytest.skip("Not enough cores to run the test")

    def timed(cb):
        start = time.time_ns()
        ret = cb()
        end = time.time_ns()
        return end - start, ret

    shape = (127, 256, 128)
    axes = (1, 2)
    signal = get_random_input_data(framework, shape, dtype, mem_backend, seed=13)
    ref = get_fft_ref(signal, axes)

    with nvmath.fft.FFT(signal, axes=axes, execution={"name": "cpu", "num_threads": 16}) as fft:
        fft.plan()
        t_16, fft_out_16 = timed(lambda: fft.execute())

    with nvmath.fft.FFT(signal, axes=axes, execution={"name": "cpu", "num_threads": 1}) as fft:
        fft.plan()
        t_1, fft_out_1 = timed(lambda: fft.execute())

    assert t_1 > t_16, f"expected {t_1} > 2 * {t_16}"
    assert_array_type(fft_out_16, framework, mem_backend, get_fft_dtype(dtype))
    assert_array_type(fft_out_1, framework, mem_backend, get_fft_dtype(dtype))
    assert_norm_close(fft_out_16, ref, exec_backend=exec_backend)
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
def test_cpu_gpu_copy_sync(framework, exec_backend, mem_backend, dtype, inplace, shape, axes):
    if len(os.sched_getaffinity(0)) < 16:
        pytest.skip("Not enough cores to run the test")

    s_1 = get_custom_stream(framework)
    s_2 = get_custom_stream(framework)
    shape = literal_eval(shape)
    axes = literal_eval(axes)

    with use_stream(s_1):
        signal = get_random_input_data(framework, shape, dtype, mem_backend, seed=13)
        noise = get_random_input_data(framework, shape, dtype, mem_backend, seed=15)
        ref = get_fft_ref(get_scaled(signal, 4), axes)
        add_in_place(signal, signal)

    with use_stream(s_2):
        signal_2 = get_random_input_data(framework, shape, dtype, mem_backend, seed=15)
        ref_2 = get_fft_ref(get_scaled(signal_2, 4), axes)

    with nvmath.fft.FFT(
        signal,
        axes=axes,
        execution={"name": "cpu", "num_threads": 16},
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
def test_direction_planning_stateful_vs_statless(monkeypatch, framework, exec_backend, mem_backend, dtype):
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

    signal = get_random_input_data(framework, (15, 16), dtype, mem_backend, 445)
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
