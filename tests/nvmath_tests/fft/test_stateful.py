# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import logging
import random
import math
from ast import literal_eval

import pytest
import cupy as cp
try:
    import torch
except:
    torch = None

import nvmath
from nvmath.memory import _RawCUDAMemoryManager

from .utils.common_axes import (
    Backend,
    Framework,
    DType,
    OptFftLayout,
    Direction,
    ShapeKind,
    OptFftBlocking,
)
from .utils.axes_utils import (
    is_complex,
    is_half,
    get_fft_dtype,
)
from .utils.support_matrix import (
    framework_type_support,
    framework_backend_support,
    type_shape_support,
)
from .utils.input_fixtures import get_random_input_data, get_custom_stream
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
)


rng = random.Random(42)


@pytest.mark.parametrize(
    (
        "framework",
        "backend",
        "fft_dim",
        "batched",
        "dtype",
        "result_layout",
    ),
    [
        (
            framework,
            backend,
            fft_dim,
            batched,
            dtype,
            rng.choice(list(OptFftLayout)),
        )
        for framework in Framework.enabled()
        for backend in framework_backend_support[framework]
        for fft_dim in [1, 2, 3]
        for batched in ["no", "left", "right"]
        for dtype in framework_type_support[framework]
    ],
)
def test_stateful_nd_default_allocator(
    monkeypatch, framework, backend, fft_dim, batched, dtype, result_layout
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

    signal_0 = get_random_input_data(framework, shape, dtype, backend, seed=55)
    signal_1 = get_random_input_data(framework, shape, dtype, backend, seed=56)

    allocations = intercept_default_allocations(monkeypatch)
    expected_key = "torch" if framework == Framework.torch else "cupy"

    if dtype == DType.float16 and batched == "right":
        with pytest.raises(ValueError, match="is currently not supported for strided inputs"):
            nvmath.fft.FFT(signal_0, axes=axes, options={"result_layout": result_layout.value})
        return

    if should_skip_3d_unsupported(shape, axes):
        with pytest.raises(ValueError, match="The 3D batched FFT is not supported"):
            nvmath.fft.FFT(signal_0, axes=axes, options={"result_layout": result_layout.value})
        return

    with nvmath.fft.FFT(
        signal_0, axes=axes, options={"result_layout": result_layout.value}
    ) as f:

        f.plan()
        fft_0 = f.execute()

        assert allocations[expected_key] == 1, f"{allocations}, {expected_key}"

        f.reset_operand(signal_1)
        fft_1 = f.execute()

        assert allocations[expected_key] == 1, f"{allocations}, {expected_key}"
        assert all(
            allocations[key] == 0 for key in allocations if key != expected_key
        ), f"{allocations}, {expected_key}"

        assert_array_type(fft_0, framework, backend, get_fft_dtype(dtype))
        assert_array_type(fft_1, framework, backend, get_fft_dtype(dtype))
        if result_layout == OptFftLayout.natural:
            fft_0_strides = get_array_strides(fft_0)
            assert is_decreasing(fft_0_strides), f"{fft_0_strides}"
            fft_1_strides = get_array_strides(fft_1)
            assert is_decreasing(fft_1_strides), f"{fft_1_strides}"

        assert_norm_close(fft_0, get_fft_ref(signal_0, axes=axes))
        assert_norm_close(fft_1, get_fft_ref(signal_1, axes=axes))


@pytest.mark.parametrize(
    (
        "framework",
        "backend",
        "fft_dim",
        "dtype",
    ),
    [
        (
            framework,
            backend,
            fft_dim,
            dtype,
        )
        for framework in Framework.enabled()
        for backend in framework_backend_support[framework]
        for fft_dim in [1, 2, 3]
        for dtype in framework_type_support[framework]
        if is_complex(dtype) and (not is_half(dtype) or fft_dim == 1)
    ],
)
def test_stateful_nd_custom_allocator(monkeypatch, framework, backend, fft_dim, dtype):
    fft_dim_shape = {
        1: (512,),
        2: (256, 512),
        3: (64, 128, 32),
    }
    shape = fft_dim_shape[fft_dim]

    signal = get_random_input_data(framework, shape, dtype, backend, seed=44)

    allocations = intercept_default_allocations(monkeypatch)
    logger = logging.getLogger("dummy_logger")
    allocator = _RawCUDAMemoryManager(device_id=0, logger=logger)
    expected_key = "raw"

    with nvmath.fft.FFT(signal, options={"allocator": allocator}) as f:

        f.plan()
        fft = f.execute(direction=Direction.forward.value)

        assert allocations[expected_key] == 1, f"{allocations}, {expected_key}"

        f.reset_operand(fft)
        ifft = f.execute(direction=Direction.inverse.value)

        assert allocations[expected_key] == 1, f"{allocations}, {expected_key}"
        assert all(
            allocations[key] == 0 for key in allocations if key != expected_key
        ), f"{allocations}, {expected_key}"

        assert_array_type(fft, framework, backend, dtype)
        assert_array_type(ifft, framework, backend, dtype)

        assert_norm_close(fft, get_fft_ref(signal))
        assert_norm_close(ifft, get_scaled(signal, math.prod(shape)))


@pytest.mark.parametrize(
    (
        "framework",
        "backend",
        "release_workspace",
    ),
    [
        (
            framework,
            backend,
            release_workspace,
        )
        for framework in Framework.enabled()
        for backend in framework_backend_support[framework]
        for release_workspace in [False, True]
    ],
)
def test_stateful_release_workspace(monkeypatch, framework, backend, release_workspace):
    shape = (2048, 128)
    dtype = DType.float32

    signal_0 = get_random_input_data(framework, shape, dtype, backend, seed=44)
    signal_1 = get_random_input_data(framework, shape, dtype, backend, seed=45)

    allocations = intercept_default_allocations(monkeypatch)
    expected_key = "torch" if framework == Framework.torch else "cupy"

    with nvmath.fft.FFT(signal_0) as f:
        f.plan()
        fft_0 = f.execute(direction=Direction.forward.value, release_workspace=release_workspace)

        assert allocations[expected_key] == 1, f"{allocations}, {expected_key}"

        f.reset_operand(signal_1)
        fft_1 = f.execute(direction=Direction.forward.value, release_workspace=release_workspace)

        assert (
            allocations[expected_key] == 2 if release_workspace else 1
        ), f"{allocations}, {expected_key}"
        assert all(
            allocations[key] == 0 for key in allocations if key != expected_key
        ), f"{allocations}, {expected_key}"

        assert_array_type(fft_0, framework, backend, get_fft_dtype(dtype))
        assert_array_type(fft_1, framework, backend, get_fft_dtype(dtype))

        assert_norm_close(fft_0, get_fft_ref(signal_0))
        assert_norm_close(fft_1, get_fft_ref(signal_1))


@pytest.mark.parametrize(
    (
        "framework",
        "backend",
        "shape_kind",
        "shape",
        "axes",
        "dtype",
        "blocking",
    ),
    [
        (
            framework,
            backend,
            shape_kind,
            repr(shape),
            repr(axes),
            dtype,
            blocking,
        )
        for framework in Framework.enabled()
        for backend in Backend
        if backend in framework_backend_support[framework]
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
            (ShapeKind.prime, (31, 9973,), (1,)),
            (ShapeKind.prime, (9973, 31), (0,)),
        ]
        for dtype in framework_type_support[framework]
        if is_complex(dtype) and shape_kind in type_shape_support[dtype]
        and (not is_half(dtype) or math.prod(shape[a] for a in axes if a in axes) <= 1024)
        for blocking in OptFftBlocking
    ],
)
def test_custom_stream(framework, backend, shape_kind, shape, axes, dtype, blocking):
    shape = literal_eval(shape)
    axes = literal_eval(axes)

    s0 = get_custom_stream(framework)
    s1 = get_custom_stream(framework)
    s2 = get_custom_stream(framework)

    if framework != Framework.cupy:
        # for less memory footprint of the whole suite
        free_cupy_pool()

    with use_stream(s0):
        signal = get_random_input_data(framework, shape, dtype, backend, seed=44)
        scale = math.prod(shape[a] for a in axes)
        signal_scaled = get_scaled(signal, scale)

    e0 = record_event(s0)
    s1.wait_event(e0)

    if should_skip_3d_unsupported(shape, axes):
        with pytest.raises(ValueError, match="The 3D batched FFT is not supported"):
            nvmath.fft.FFT(signal, axes=axes, stream=s1, options={"result_layout": "natural"})
        return

    with nvmath.fft.FFT(
        signal,
        axes=axes,
        options={"blocking": blocking.value, "result_layout": "natural"},
        stream=s1,
    ) as f:
        f.plan(stream=s1)
        fft = f.execute(direction=Direction.forward.value, stream=s1)
        f.reset_operand(fft, stream=s1)
        # While blocking = True makes the execute synchronous,
        # the cpu -> gpu in reset_operand is always async
        if backend == Backend.cpu or blocking == OptFftBlocking.auto:
            e1 = record_event(s1)
            s2.wait_event(e1)
        ifft = f.execute(direction=Direction.inverse.value, stream=s2)

        with use_stream(s2):
            assert_array_type(ifft, framework, backend, dtype)
            assert_norm_close(ifft, signal_scaled, shape_kind=shape_kind)


@pytest.mark.parametrize(
    (
        "framework",
        "backend",
        "shape_kind",
        "shape",
        "axes",
        "dtype",
        "blocking",
    ),
    [
        (
            framework,
            backend,
            shape_kind,
            repr(shape),
            repr(axes),
            dtype,
            blocking,
        )
        for framework in Framework.enabled()
        for backend in Backend
        if backend in framework_backend_support[framework]
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
            (ShapeKind.prime, (31, 9973,), (1,)),
            (ShapeKind.prime, (9973, 31), (0,)),
        ]
        for dtype in framework_type_support[framework]
        if is_complex(dtype) and shape_kind in type_shape_support[dtype]
        and (not is_half(dtype) or math.prod(shape[a] for a in axes if a in axes) <= 1024)
        for blocking in OptFftBlocking
    ],
)
def test_custom_stream_inplace(framework, backend, shape_kind, shape, axes, dtype, blocking):
    shape = literal_eval(shape)
    axes = literal_eval(axes)
    s0 = get_custom_stream(framework)
    s1 = get_custom_stream(framework)

    if framework != Framework.cupy:
        # for less memory footprint of the whole suite
        free_cupy_pool()

    with use_stream(s0):
        signal = get_random_input_data(framework, shape, dtype, backend, seed=44)
        scale = math.prod(shape[a] for a in axes)
        signal_scaled = get_scaled(signal, scale * 2)

    if should_skip_3d_unsupported(shape, axes):
        with pytest.raises(ValueError, match="The 3D batched FFT is not supported"):
            nvmath.fft.FFT(signal, axes=axes, stream=s0, options={"blocking": blocking.value, "inplace": True})
        return

    with nvmath.fft.FFT(
        signal,
        axes=axes,
        options={"blocking": blocking.value, "inplace": True},
        stream=s0,
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
        if backend == Backend.cpu:
            f.reset_operand(signal, stream=s1)
        f.execute(direction=Direction.inverse.value, stream=s1)

        with use_stream(s1):
            assert_array_type(signal, framework, backend, dtype)
            assert_norm_close(signal, signal_scaled, shape_kind=shape_kind)


@pytest.mark.parametrize(
    (
        "framework",
        "backend",
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
            backend,
            shape_kind,
            repr(shape),
            repr(axes),
            dtype,
            blocking,
            result_layout,
        )
        for framework in Framework.enabled()
        for backend in Backend
        if backend in framework_backend_support[framework]
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
            (ShapeKind.prime, (31, 9973,), (1,)),
            (ShapeKind.prime, (9973, 31), (0,)),
        ]
        for dtype in framework_type_support[framework]
        if shape_kind in type_shape_support[dtype]
        and (not is_half(dtype) or math.prod(shape[a] for a in axes if a in axes) <= 1024)
        for blocking in OptFftBlocking
        for result_layout in OptFftLayout
    ],
)
def test_custom_stream_busy_input(framework, backend, shape_kind, shape, axes, dtype, blocking, result_layout):
    shape = literal_eval(shape)
    axes = literal_eval(axes)
    s0 = get_custom_stream(framework)

    if framework != Framework.cupy:
        # for less memory footprint of the whole suite
        free_cupy_pool()

    with use_stream(s0):
        signal = get_random_input_data(framework, shape, dtype, backend, seed=44)
        ref = get_scaled(get_fft_ref(signal, axes=axes), 4)

    if should_skip_3d_unsupported(shape, axes):
        with pytest.raises(ValueError, match="The 3D batched FFT is not supported"):
            nvmath.fft.FFT(
                signal,
                axes=axes,
                stream=s0,
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
    ) as f:
        f.plan(stream=s0)
        with use_stream(s0):
            add_in_place(signal, signal)
            add_in_place(signal, signal)
            if backend == Backend.cpu:
                f.reset_operand(signal, stream=s0)

        fft = f.execute(direction=Direction.forward.value, stream=s0)

        with use_stream(s0):
            assert_array_type(fft, framework, backend, get_fft_dtype(dtype))
            assert_norm_close(fft, ref, shape_kind=shape_kind)

@pytest.mark.parametrize(
    (
        "framework",
        "dtype",
    ),
    [
        (
            framework,
            dtype,
        )
        for framework in Framework.enabled()
        if Backend.gpu in framework_backend_support[framework]
        for dtype in framework_type_support[framework]
    ],
)
def test_arrays_different_devices(framework, dtype):
    dev_count = cp.cuda.runtime.getDeviceCount()
    if dev_count < 2:
        pytest.skip(f"Test requires at least two gpus, got {dev_count}")

    shape = (512, 512)
    signal_0 = get_random_input_data(
        framework, shape, dtype, Backend.gpu, seed=13, device_id=0
    )
    signal_1 = get_random_input_data(
        framework, shape, dtype, Backend.gpu, seed=12, device_id=1
    )

    from nvmath.bindings import cufft

    fft_0, fft_1 = None, None
    try:
        fft_0 = nvmath.fft.FFT(signal_0, options={"blocking": "auto"})
        fft_1 = nvmath.fft.FFT(signal_1, options={"blocking": "auto"})
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
    assert_array_type(fft_0_out, framework, Backend.gpu, get_fft_dtype(dtype))
    assert_array_type(fft_1_out, framework, Backend.gpu, get_fft_dtype(dtype))

    assert_norm_close(fft_0_out, get_fft_ref(signal_0))
    with cp.cuda.Device(1):
        assert_norm_close(fft_1_out, get_fft_ref(signal_1))


@pytest.mark.parametrize(
    ("framework",),
    [
        (framework,)
        for framework in Framework.enabled()
        if Backend.gpu in framework_backend_support[framework]
    ],
)
def test_multi_device_wrong_device(framework):
    dev_count = cp.cuda.runtime.getDeviceCount()
    backend = Backend.gpu
    if dev_count < 2:
        pytest.skip(f"Test requires at least two gpus, got {dev_count}")

    shape = (32, 128)
    dtype = DType.complex64

    signal_0 = get_random_input_data(
        framework, shape, dtype, backend, seed=44, device_id=0
    )
    signal_1 = get_random_input_data(
        framework, shape, dtype, backend, seed=45, device_id=1
    )

    with nvmath.fft.FFT(
        signal_0,
        axes=[0, 1],
        options={"blocking": "auto", "result_layout": "natural"},
    ) as f:
        f.plan()
        f.execute()
        with pytest.raises(
            ValueError, match="The new operand must be on the same device"
        ):
            f.reset_operand(signal_1)


@pytest.mark.parametrize(
    (
        "framework",
        "backend",
    ),
    [
        (framework, backend)
        for framework in Framework.enabled()
        for backend in framework_backend_support[framework]
    ],
)
def test_unsupported_shape_change(framework, backend):
    shape_0 = (512, 128)
    shape_1 = (256, 1024)
    dtype = DType.float64

    signal_0 = get_random_input_data(framework, shape_0, dtype, backend, seed=44)
    signal_1 = get_random_input_data(framework, shape_1, dtype, backend, seed=45)

    with nvmath.fft.FFT(signal_0) as f:
        f.plan()
        f.execute(direction=Direction.forward.value)
        with pytest.raises(ValueError, match="The new operand's traits"):
            f.reset_operand(signal_1)
