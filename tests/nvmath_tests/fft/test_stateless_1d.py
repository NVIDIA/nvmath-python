# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import random
import math
from ast import literal_eval

import pytest
import numpy as np
import cupy as cp
try:
    import torch
except:
    torch = None

import nvmath

from .utils.common_axes import (
    Backend,
    Framework,
    DType,
    ShapeKind,
    OptFftBlocking,
    OptFftType,
    OptFftLayout,
    Direction,
)
from .utils.axes_utils import (
    get_framework_dtype,
    is_complex,
    is_half,
    get_fft_dtype,
    get_ifft_dtype,
    framework_dtype,
    r2c_dtype,
    enum_val_str,
)
from .utils.support_matrix import (
    framework_type_support,
    framework_backend_support,
    type_shape_support,
    opt_fft_type_direction_support,
    opt_fft_type_input_type_support,
    inplace_opt_ftt_type_support,
)
from .utils.input_fixtures import (
    get_1d_shape_cases,
    get_random_1d_shape,
    get_random_input_data,
    get_custom_stream,
)
from .utils.check_helpers import (
    assert_eq,
    copy_array,
    use_stream,
    get_array_device_id,
    intercept_device_id,
    get_fft_ref,
    get_scaled,
    get_ifft_c2r_options,
    get_cufft_version,
    unfold,
    assert_norm_close,
    assert_array_type,
)

rng = random.Random(101)


@pytest.mark.parametrize(
    ("framework", "backend", "dtype", "shape_kind", "shape"),
    [
        (framework, backend, dtype, shape_kind, shape)
        for framework in Framework.enabled()
        for backend in framework_backend_support[framework]
        for dtype in framework_type_support[framework]
        for shape_kind, shape in get_1d_shape_cases(type_shape_support[dtype], rng)
    ],
)
def test_fft_ifft(framework, backend, dtype, shape_kind, shape):
    signal = get_random_input_data(framework, (shape,), dtype, backend, seed=42)

    if shape == 1 and is_half(dtype) and get_cufft_version() < 10702:
        with pytest.raises(ValueError, match="sample size 1 and half-precision type"):
            nvmath.fft.fft(signal) if is_complex(dtype) else nvmath.fft.rfft(signal)
        return

    sample_fft = nvmath.fft.fft(signal) if is_complex(dtype) else nvmath.fft.rfft(signal)
    assert_array_type(sample_fft, framework, backend, get_fft_dtype(dtype))
    assert_norm_close(sample_fft, get_fft_ref(signal), shape_kind=shape_kind)

    ifft_fn = nvmath.fft.ifft if is_complex(dtype) else nvmath.fft.irfft
    sample_ifft = ifft_fn(
        sample_fft, options=get_ifft_c2r_options(dtype, shape)
    )
    assert_array_type(sample_ifft, framework, backend, dtype)
    assert_norm_close(sample_ifft, get_scaled(signal, shape), shape_kind=shape_kind)


@pytest.mark.parametrize(
    ("framework", "fft_type", "dtype", "backend", "shape_kind", "shape"),
    [
        (
            framework,
            fft_type,
            dtype,
            backend,
        )
        + get_random_1d_shape(type_shape_support[dtype], rng)
        for framework in Framework.enabled()
        for fft_type in [OptFftType.r2c, OptFftType.c2c]
        for dtype in opt_fft_type_input_type_support[fft_type]
        if dtype in framework_type_support[framework]
        for backend in framework_backend_support[framework]
    ],
)
def test_fft_explicit_fft_type(framework, fft_type, dtype, backend, shape_kind, shape):
    sample = get_random_input_data(framework, (shape,), dtype, backend, seed=17)

    fn = nvmath.fft.fft if is_complex(dtype) else nvmath.fft.rfft
    sample_fft = fn(sample, options={"fft_type": fft_type.value})
    assert_array_type(sample_fft, framework, backend, get_fft_dtype(dtype))
    assert_norm_close(sample_fft, get_fft_ref(sample), shape_kind=shape_kind)


@pytest.mark.parametrize(
    ("framework", "fft_type", "dtype", "backend", "shape_kind", "shape"),
    [
        (
            framework,
            fft_type,
            dtype,
            backend,
        )
        + get_random_1d_shape(type_shape_support[dtype], rng)
        for framework in Framework.enabled()
        for fft_type in [OptFftType.c2r, OptFftType.c2c]
        for dtype in opt_fft_type_input_type_support[fft_type]
        if dtype in framework_type_support[framework]
        for backend in framework_backend_support[framework]
    ],
)
def test_ifft_explicit_fft_type(framework, fft_type, dtype, backend, shape_kind, shape):
    signal_dtype = get_ifft_dtype(dtype, fft_type)
    signal = get_random_input_data(framework, (shape,), signal_dtype, backend, seed=21)
    sample_fft = get_fft_ref(signal)
    assert_array_type(sample_fft, framework, backend, dtype)

    sample_ifft = nvmath.fft.ifft(
        sample_fft,
        options={
            "fft_type": fft_type.value,
            "last_axis_size": "even" if shape % 2 == 0 else "odd",
        },
    )
    assert_array_type(sample_ifft, framework, backend, signal_dtype)
    assert_norm_close(sample_ifft, get_scaled(signal, shape), shape_kind=shape_kind)


@pytest.mark.parametrize(
    ("framework", "dtype", "backend", "shape_kind", "shape"),
    [
        (framework, dtype, backend, shape_kind, shape)
        for framework in Framework.enabled()
        for fft_type in inplace_opt_ftt_type_support[True]
        for dtype in opt_fft_type_input_type_support[fft_type]
        if dtype in framework_type_support[framework]
        for backend in framework_backend_support[framework]
        for shape_kind, shape in get_1d_shape_cases(type_shape_support[dtype], rng)
    ],
)
def test_fft_inplace(framework, dtype, backend, shape_kind, shape):
    sample = get_random_input_data(framework, (shape,), dtype, backend, seed=71)
    ref_fft = get_fft_ref(sample)

    if shape == 1 and is_half(dtype) and get_cufft_version() < 10702:
        with pytest.raises(ValueError, match="sample size 1 and half-precision type"):
            nvmath.fft.fft(sample, options={"inplace": True})
        return

    nvmath.fft.fft(sample, options={"inplace": True})
    assert_array_type(sample, framework, backend, dtype)
    assert_norm_close(sample, ref_fft, shape_kind=shape_kind)


@pytest.mark.parametrize(
    ("framework", "dtype", "backend", "shape_kind", "shape"),
    [
        (framework, dtype, backend, shape_kind, shape)
        for framework in Framework.enabled()
        for fft_type in inplace_opt_ftt_type_support[True]
        for dtype in opt_fft_type_input_type_support[fft_type]
        if dtype in framework_type_support[framework]
        for backend in framework_backend_support[framework]
        for shape_kind, shape in get_1d_shape_cases(type_shape_support[dtype], rng)
    ],
)
def test_ifft_inplace(framework, dtype, backend, shape_kind, shape):
    signal = get_random_input_data(framework, (shape,), dtype, backend, seed=32)
    sample = get_fft_ref(signal)

    if shape == 1 and is_half(dtype) and get_cufft_version() < 10702:
        with pytest.raises(ValueError, match="sample size 1 and half-precision type"):
            nvmath.fft.ifft(sample, options={"inplace": True})
        return

    nvmath.fft.ifft(sample, options={"inplace": True})
    assert_array_type(sample, framework, backend, dtype)
    assert_norm_close(sample, get_scaled(signal, shape), shape_kind=shape_kind)


@pytest.mark.parametrize(
    (
        "framework",
        "backend",
        "dtype",
        "window_size",
        "shape_kind",
        "batch_size",
        "step_size",
        "result_layout",
    ),
    [
        (
            framework,
            backend,
            dtype,
            window_size,
            shape_kind,
            batch_size,
            step_size,
            result_layout,
        )
        for framework in Framework.enabled()
        for backend in framework_backend_support[framework]
        for dtype in framework_type_support[framework]
        for window_size, shape_kind in (
            (1, ShapeKind.pow2),
            (2, ShapeKind.pow2),
            (4, ShapeKind.pow2),
            (256, ShapeKind.pow2),
            (2 * 3 * 5 * 7, ShapeKind.pow2357),
            (199, ShapeKind.prime),
        )
        if shape_kind in type_shape_support[dtype]
        for batch_size in (1, 3, 16, 99)
        for step_size in sorted(
            step
            for step in set([1, window_size - 1, window_size, window_size + 1])
            if step > 0
        )
        for result_layout in OptFftLayout
    ],
)
def test_fft_ifft_overlap(framework, backend, dtype, window_size, shape_kind, batch_size, step_size, result_layout):
    signal_size = batch_size * step_size + window_size - 1
    signal = get_random_input_data(framework, (signal_size,), dtype, backend, seed=42)
    batch = unfold(signal, 0, window_size, step_size)
    assert_eq(batch.shape[0], batch_size)
    fft_fn = nvmath.fft.fft if is_complex(dtype) else nvmath.fft.rfft
    ifft_fn = nvmath.fft.ifft if is_complex(dtype) else nvmath.fft.irfft

    try:
        fft = fft_fn(batch, axes=(1,), options={"result_layout": result_layout.value})
        assert_array_type(fft, framework, backend, get_fft_dtype(dtype))
        assert_norm_close(fft, get_fft_ref(batch, axes=(1,)), shape_kind=shape_kind)
    except ValueError as e:
        assert is_half(dtype)
        if "sample size 1 and half-precision type" in str(e):
            assert window_size == 1 and get_cufft_version() < 10702
        else:
            assert not is_complex(dtype) and backend == Backend.cpu
            assert "is currently not supported for strided inputs" in str(e)
        return

    try:
        ifft = ifft_fn(
            fft,
            axes=(1,),
            options={
                "result_layout": result_layout.value,
                "last_axis_size": "odd" if window_size % 2 else "even",
            },
        )
        assert_array_type(ifft, framework, backend, dtype)
        assert_norm_close(ifft, get_scaled(batch, window_size), shape_kind=shape_kind)
    except ValueError as e:
        assert (
            not is_complex(dtype)
            and is_half(dtype)
            and (backend == Backend.cpu or result_layout == OptFftLayout.natural)
        )
        assert "is currently not supported for strided outputs" in str(e)
        return

@pytest.mark.parametrize(
    ("framework", "backend", "dtype", "blocking", "shape_kind", "shape"),
    [
        (framework, backend, dtype, blocking)
        + get_random_1d_shape(type_shape_support[dtype], rng)
        for framework in Framework.enabled()
        for backend in framework_backend_support[framework]
        for dtype in framework_type_support[framework]
        for blocking in OptFftBlocking
    ],
)
def test_ifft_fft_blocking(
    monkeypatch, framework, backend, dtype, blocking, shape_kind, shape
):
    synchronization_num = 0
    _actual_sync = cp.cuda.Event.synchronize

    def _synchronize(self):
        nonlocal synchronization_num
        synchronization_num += 1
        _actual_sync(self)

    monkeypatch.setattr(cp.cuda.Event, "synchronize", _synchronize)

    sample = get_random_input_data(framework, (shape,), dtype, backend, seed=33)
    sample_fft_ref = get_fft_ref(sample)
    sample_scaled = get_scaled(sample, shape)

    if is_complex(dtype):
        sample_fft = nvmath.fft.fft(sample, options={"blocking": blocking.value})
    else:
        sample_fft = nvmath.fft.rfft(sample, options={"blocking": blocking.value})
    assert_array_type(sample_fft, framework, backend, get_fft_dtype(dtype))
    assert_norm_close(sample_fft, sample_fft_ref, shape_kind=shape_kind)

    if is_complex(dtype):
        sample_ifft = nvmath.fft.ifft(
            sample_fft,
            options={"blocking": blocking.value, **get_ifft_c2r_options(dtype, shape)},
        )
        assert_array_type(sample_ifft, framework, backend, dtype)
        assert_norm_close(sample_ifft, sample_scaled, shape_kind=shape_kind)

    if backend == Backend.cpu or blocking == OptFftBlocking.true:
        expected_syncs = (1 + is_complex(dtype)) * 2  # 2x for plan creation and fft execution
    else:
        expected_syncs = 1 + is_complex(dtype)  # 2x for plan creation only
    assert_eq(synchronization_num, expected_syncs)


@pytest.mark.parametrize(
    ("framework", "backend", "dtype", "shape_kind", "shape"),
    [
        (framework, backend, dtype)
        + get_random_1d_shape(type_shape_support[dtype], rng)
        for framework in Framework.enabled()
        for backend in framework_backend_support[framework]
        for fft_type in inplace_opt_ftt_type_support[True]
        for dtype in framework_type_support[framework]
        if dtype in opt_fft_type_input_type_support[fft_type]
    ],
)
def test_fft_ifft_inplace_blocking_auto(framework, backend, dtype, shape_kind, shape):
    # only C2C supports inplace FFT
    signal = get_random_input_data(framework, (shape,), dtype, backend, seed=63)
    signal_scaled = get_scaled(signal, shape)
    signal_copy = copy_array(signal)

    nvmath.fft.fft(signal_copy, options={"inplace": True, "blocking": "auto"})
    nvmath.fft.ifft(signal_copy, options={"inplace": True, "blocking": "auto"})

    assert_array_type(signal_copy, framework, backend, dtype)
    assert_norm_close(signal_copy, signal_scaled, shape_kind=shape_kind)


@pytest.mark.parametrize(
    ("framework", "backend", "dtype"),
    [
        (framework, backend, dtype)
        for framework in Framework.enabled()
        for backend in framework_backend_support[framework]
        for dtype in framework_type_support[framework]
        for fft_type in inplace_opt_ftt_type_support[True]
        if dtype in opt_fft_type_input_type_support[fft_type]
    ],
)
def test_fft_explicit_device_id(monkeypatch, framework, backend, dtype):

    dev_count = cp.cuda.runtime.getDeviceCount()
    if dev_count < 2:
        pytest.skip(f"Test requires at least two gpus, got {dev_count}")

    from nvmath.bindings import cufft

    device_ids = intercept_device_id(
        monkeypatch, (cufft, "xt_make_plan_many"), (cufft, "xt_exec")
    )

    shape = 4096
    signal = get_random_input_data(framework, (shape,), dtype, backend, seed=318)
    signal_copy = copy_array(signal)

    array_device = 0
    if is_complex(dtype):
        nvmath.fft.fft(
            signal_copy, options={"inplace": True, "blocking": "auto", "device_id": 0}
        )
    else:
        nvmath.fft.rfft(
            signal_copy, options={"inplace": True, "blocking": "auto", "device_id": 0}
        )
    expected_device = 0 if backend == Backend.cpu else array_device
    assert_eq(device_ids["xt_make_plan_many"], expected_device)
    assert_eq(device_ids["xt_exec"], expected_device)

    if is_complex(dtype):
        expected_device = 1 if backend == Backend.cpu else array_device
        nvmath.fft.ifft(
            signal_copy, options={"inplace": True, "blocking": "auto", "device_id": 1}
        )
        assert_eq(device_ids["xt_make_plan_many"], expected_device)
        assert_eq(device_ids["xt_exec"], expected_device)

        if backend == Backend.gpu:
            assert_eq(get_array_device_id(signal_copy), array_device)
        assert_array_type(signal_copy, framework, backend, dtype)
        assert_norm_close(signal_copy, get_scaled(signal, shape))


@pytest.mark.parametrize(
    ("framework", "blocking", "dtype"),
    [
        (framework, blocking, dtype)
        for framework in Framework.enabled()
        if Backend.gpu in framework_backend_support[framework]
        for blocking in OptFftBlocking
        for dtype in framework_type_support[framework]
    ],
)
def test_fft_array_device_id(monkeypatch, framework, blocking, dtype):
    dev_count = cp.cuda.runtime.getDeviceCount()
    if dev_count < 2:
        pytest.skip(f"Test requires at least two gpus, got {dev_count}")

    shape = 2048
    signal_1 = get_random_input_data(
        framework, (shape,), dtype, Backend.gpu, seed=415, device_id=1
    )
    signal_0 = get_random_input_data(
        framework, (shape,), dtype, Backend.gpu, seed=416, device_id=0
    )

    from nvmath.bindings import cufft

    device_ids = intercept_device_id(
        monkeypatch, (cufft, "xt_make_plan_many"), (cufft, "xt_exec")
    )

    fft_fn = nvmath.fft.fft if is_complex(dtype) else nvmath.fft.rfft

    fft_1 = fft_fn(signal_1, options={"blocking": blocking.value})
    assert_eq(device_ids["xt_make_plan_many"], 1)
    assert_eq(device_ids["xt_exec"], 1)
    fft_0 = fft_fn(signal_0, options={"blocking": blocking.value})
    assert_eq(device_ids["xt_make_plan_many"], 0)
    assert_eq(device_ids["xt_exec"], 0)

    for dev_id, fft in enumerate((fft_0, fft_1)):
        assert_eq(get_array_device_id(fft), dev_id)
        assert_array_type(fft, framework, Backend.gpu, get_fft_dtype(dtype))

    assert_norm_close(fft_0, get_fft_ref(signal_0))
    with cp.cuda.Device(1):
        assert_norm_close(fft_1, get_fft_ref(signal_1))

    if is_complex(dtype):
        ifft_1 = nvmath.fft.ifft(
            fft_1,
            options={"blocking": blocking.value, **get_ifft_c2r_options(dtype, shape)},
        )
        assert_eq(device_ids["xt_make_plan_many"], 1)
        assert_eq(device_ids["xt_exec"], 1)
        ifft_0 = nvmath.fft.ifft(
            fft_0,
            options={"blocking": blocking.value, **get_ifft_c2r_options(dtype, shape)},
        )
        assert_eq(device_ids["xt_make_plan_many"], 0)
        assert_eq(device_ids["xt_exec"], 0)

        for dev_id, ifft in enumerate((ifft_0, ifft_1)):
            assert_eq(get_array_device_id(ifft), dev_id)
            assert_array_type(ifft, framework, Backend.gpu, dtype)

        assert_norm_close(ifft_0, get_scaled(signal_0, shape))
        with cp.cuda.Device(1):
            assert_norm_close(ifft_1, get_scaled(signal_1, shape))


@pytest.mark.parametrize(
    ("framework", "dtype"),
    [
        (framework, dtype)
        for framework in Framework.enabled()
        if Backend.gpu in framework_backend_support[framework]
        for dtype in framework_type_support[framework]
        if not is_half(dtype)  # for the fft size, the halfs lack precision
    ],
)
def test_fft_custom_stream(framework, dtype):
    stream = get_custom_stream(framework)
    shape = 1024 * 1024

    with use_stream(stream):
        signal = get_random_input_data(
            framework, (shape,), dtype, Backend.gpu, seed=234
        )
        fft_ref = get_fft_ref(signal)
        fft_ref = fft_ref * 42

    fft_fn = nvmath.fft.fft if is_complex(dtype) else nvmath.fft.rfft
    fft = fft_fn(
        signal,
        options={
            "blocking": "auto",
        },
        stream=stream,
    )

    with use_stream(stream):
        # The stateless API synchronizes on plan creation,
        # so to see proper stream handling, we do some postprocessing
        # on the stream
        fft = fft * 42
        assert_norm_close(fft, fft_ref)


@pytest.mark.parametrize(
    ("framework", "backend", "dtype"),
    [
        (framework, backend, dtype)
        for framework in Framework.enabled()
        for dtype in DType
        if dtype not in framework_type_support[framework]
        for backend in framework_backend_support[framework]
    ],
)
def test_fft_unsupported_type(framework, dtype, backend):
    if dtype not in framework_dtype[framework]:
        pytest.skip(
            f"Type {enum_val_str(dtype)} not representable in {enum_val_str(framework)}"
        )
    if framework == Framework.numpy:
        sample = np.arange(128).astype(get_framework_dtype(framework, dtype))
    elif framework == Framework.cupy:
        sample = cp.arange(128).astype(get_framework_dtype(framework, dtype))
    elif framework == Framework.torch:
        sample = torch.arange(128).type(get_framework_dtype(framework, dtype))
        if backend == Backend.gpu:
            sample = sample.cuda()
    else:
        raise ValueError(f"Unknown framework {framework}")

    if is_complex(dtype):
        match = f"Unsupported dtype '{enum_val_str(dtype)}' for FFT"
    else:
        match = f"This function expects complex operand, found {enum_val_str(dtype)}"
    with pytest.raises(
        ValueError, match=match
    ):
        nvmath.fft.fft(sample)


@pytest.mark.parametrize(
    ("framework", "backend", "dtype"),
    [
        (framework, backend, dtype)
        for framework in Framework.enabled()
        for dtype in DType
        if dtype not in framework_type_support[framework]
        for backend in framework_backend_support[framework]
    ],
)
def test_ifft_unsupported_type(framework, dtype, backend):
    if dtype not in framework_dtype[framework]:
        pytest.skip(
            f"Type {enum_val_str(dtype)} not representable in {enum_val_str(framework)}"
        )
    if framework == Framework.numpy:
        sample = np.arange(64).astype(get_framework_dtype(framework, dtype))
    elif framework == Framework.cupy:
        sample = cp.arange(64).astype(get_framework_dtype(framework, dtype))
    elif framework == Framework.torch:
        sample = torch.arange(64).type(get_framework_dtype(framework, dtype))
        if backend == Backend.gpu:
            sample = sample.cuda()
    else:
        raise ValueError(f"Unknown framework {framework}")

    if is_complex(dtype):
        match = f"Unsupported dtype '{enum_val_str(dtype)}' for FFT"
    else:
        match = f"This function expects complex operand, found {enum_val_str(dtype)}"
    with pytest.raises(
        ValueError, match=match
    ):
        nvmath.fft.ifft(sample)


@pytest.mark.parametrize(
    ("framework", "backend", "dtype"),
    [
        (framework, backend, dtype)
        for framework in Framework.enabled()
        for dtype in DType
        # real type representable in the framework
        # but not the corresponding complex dtype
        if dtype in framework_dtype[framework]
        and dtype in r2c_dtype
        and r2c_dtype[dtype] not in framework_dtype[framework]
        for backend in framework_backend_support[framework]
    ],
)
def test_rfft_unsupported_result_dtype(framework, dtype, backend):
    signal = get_random_input_data(framework, (32,), dtype, backend, seed=234)
    with pytest.raises(
        TypeError,
        match=(
            f"The result data type {enum_val_str(r2c_dtype[dtype])} is not "
            f"supported by the operand package '{enum_val_str(framework)}'."
        ),
    ):
        nvmath.fft.rfft(signal)


@pytest.mark.parametrize(
    ("framework", "backend", "dtype", "shape_kind"),
    [
        (framework, backend, dtype, shape_kind)
        for framework in Framework.enabled()
        for backend in framework_backend_support[framework]
        for dtype in framework_type_support[framework]
        for shape_kind in ShapeKind
        if shape_kind not in type_shape_support[dtype]
        and shape_kind != ShapeKind.random
    ],
)
def test_fft_unsupported_shape(framework, dtype, backend, shape_kind):
    # Note, there is an undocumented partial support for not pow of 2 shapes above 128
    # This test selects shapes that are actually failing only to check the handling
    # of unsupported input.
    shapes = {
        ShapeKind.pow2357: 210,
        ShapeKind.prime: 127,
    }
    sample = get_random_input_data(
        framework, (shapes[shape_kind],), dtype, backend, seed=101
    )

    fft_fn = nvmath.fft.fft if is_complex(dtype) else nvmath.fft.rfft

    with pytest.raises(
        nvmath.bindings.cufft.cuFFTError,
        match="(CUFFT_NOT_SUPPORTED|CUFFT_SETUP_FAILED)",
    ):
        fft_fn(sample)


@pytest.mark.parametrize(
    ("framework", "backend", "dtype", "shape_kind"),
    [
        (framework, backend, dtype, shape_kind)
        for framework in Framework.enabled()
        for backend in framework_backend_support[framework]
        for dtype in framework_type_support[framework]
        if is_complex(dtype)
        for shape_kind in ShapeKind
        if shape_kind not in type_shape_support[dtype]
        and shape_kind != ShapeKind.random
    ],
)
def test_ifft_unsupported_shape(framework, dtype, backend, shape_kind):
    # Note, there is an undocumented partial support for prime shapes above 128
    # This test selects shapes that are actually failing only to check the handling
    # of unsupported input.
    shapes = {
        ShapeKind.pow2357: 630,
        ShapeKind.prime: 101,
    }
    sample = get_random_input_data(
        framework, (shapes[shape_kind],), dtype, backend, seed=101
    )
    with pytest.raises(
        nvmath.bindings.cufft.cuFFTError,
        match="(CUFFT_NOT_SUPPORTED|CUFFT_SETUP_FAILED)",
    ):
        nvmath.fft.ifft(sample)


@pytest.mark.parametrize(
    ("framework", "backend", "dtype", "shape", "axes"),
    [
        (framework, backend, dtype, repr(shape), repr(axes))
        for framework in Framework.enabled()
        for backend in framework_backend_support[framework]
        for dtype in framework_type_support[framework]
        if is_complex(dtype)
        for shape, axes in [
            ((1,), None),
            ((2, 2, 1), (1, 2)),
        ]
    ],
)
def test_irfft_unsupported_empty_output(framework, dtype, backend, shape, axes):
    shape = literal_eval(shape)
    axes = literal_eval(axes)
    signal = get_random_input_data(framework, shape, dtype, backend, seed=101)
    sample_vol = math.prod(shape) if axes is None else math.prod(shape[a] for a in axes)

    if sample_vol == 1 and is_half(dtype) and get_cufft_version() < 10702:
        with pytest.raises(ValueError, match="sample size 1 and half-precision type"):
            nvmath.fft.ifft(signal, options={"inplace": True})
    else:
        nvmath.fft.irfft(signal, axes=axes, options={"last_axis_size": "odd"})

    with pytest.raises(
        ValueError,
        match="The size of the last FFT axis in the result for FFT type 'C2R' is 0",
    ):
        nvmath.fft.irfft(signal, axes=axes)


@pytest.mark.parametrize(
    ("framework", "fft_type", "direction", "backend"),
    [
        (framework, fft_type, direction, backend)
        for framework in Framework.enabled()
        for fft_type in OptFftType
        for direction in Direction
        if (direction not in opt_fft_type_direction_support[fft_type])
        for backend in framework_backend_support[framework]
    ],
)
def test_incompatible_fft_type_direction(framework, fft_type, direction, backend):
    if fft_type == OptFftType.r2c:
        if direction == Direction.inverse:
            pytest.skip("No fft function API for real inverse FFT")
        dtype = DType.float32
        fn = nvmath.fft.rfft
    else:
        dtype = DType.complex64
        fn = nvmath.fft.fft if direction == Direction.forward else nvmath.fft.ifft
    sample = get_random_input_data(framework, (128,), dtype, backend, seed=15)
    with pytest.raises(
        ValueError,
        match=(
            f"The specified direction {direction.value} "
            f"is not compatible with the FFT type '{fft_type.value}'"
        ),
    ):
        fn(sample, options={"fft_type": fft_type.value})


@pytest.mark.parametrize(
    ("framework", "fft_type", "direction", "dtype", "backend"),
    [
        (framework, fft_type, direction, dtype, backend)
        for framework in Framework.enabled()
        for fft_type in OptFftType
        for direction in opt_fft_type_direction_support[fft_type]
        for dtype in framework_type_support[framework]
        if dtype not in opt_fft_type_input_type_support[fft_type]
        for backend in framework_backend_support[framework]
    ],
)
def test_incompatible_fft_type_dtype(framework, fft_type, direction, dtype, backend):
    sample = get_random_input_data(framework, (512,), dtype, backend, seed=17)
    if is_complex(dtype):
        fn = nvmath.fft.fft if direction == Direction.forward else nvmath.fft.ifft
    else:
        if direction == Direction.inverse:
            pytest.skip("No fft function API for inverse real FFT")
        else:
            fn = nvmath.fft.rfft
    # TODO Should those errors be unified?
    if fft_type == OptFftType.c2c:
        with pytest.raises(
            nvmath.bindings.cufft.cuFFTError,
            match=f"CUFFT_INVALID_TYPE",
        ):
            fn(sample, options={"fft_type": fft_type.value})
    else:
        with pytest.raises(
            AssertionError,
            match=f"Internal Error \\(name='{enum_val_str(dtype)}'\\)",
        ):
            fn(sample, options={"fft_type": fft_type.value})


@pytest.mark.parametrize(
    ("framework", "fft_type", "direction", "backend", "dtype"),
    [
        (
            framework,
            fft_type,
            direction,
            backend,
            opt_fft_type_input_type_support[fft_type][-1],
        )
        for framework in Framework.enabled()
        for fft_type in OptFftType
        if fft_type not in inplace_opt_ftt_type_support[True]
        for direction in opt_fft_type_direction_support[fft_type]
        for backend in framework_backend_support[framework]
    ],
)
def test_inplace_unsupported_fft_type(framework, fft_type, direction, backend, dtype):
    sample = get_random_input_data(framework, (256,), dtype, backend, seed=19)
    if is_complex(dtype):
        fn = nvmath.fft.fft if direction == Direction.forward else nvmath.fft.ifft
    else:
        if direction == Direction.inverse:
            pytest.skip("No fft function API for inverse real FFT")
        else:
            fn = nvmath.fft.rfft
    with pytest.raises(
        ValueError,
        match=(
            f"The in-place option \\(FFTOptions\\.inplace=True\\) is only supported for "
            f"complex-to-complex FFT\\. The FFT type is '{fft_type.value}'"
        ),
    ):
        fn(sample, options={"fft_type": fft_type.value, "inplace": True})


@pytest.mark.parametrize(
    ("framework", "backend", "dtype"),
    [
        (framework, backend, dtype)
        for framework in Framework.enabled()
        for backend in framework_backend_support[framework]
        for dtype in opt_fft_type_input_type_support[OptFftType.r2c]
        if dtype in framework_type_support[framework]
    ],
)
def test_inplace_unsupported_implicit_r2c(framework, backend, dtype):
    sample = get_random_input_data(framework, (256,), dtype, backend, seed=19)
    with pytest.raises(
        ValueError,
        match=(
            f"The in-place option \\(FFTOptions\\.inplace=True\\) is only supported for "
            f"complex-to-complex FFT\\. The FFT type is '{OptFftType.r2c.value}'"
        ),
    ):
        nvmath.fft.rfft(sample, options={"inplace": True})


@pytest.mark.parametrize(
    ("framework", "dtype"),
    [
        (framework, dtype)
        for framework in Framework.enabled()
        if Backend.gpu in framework_backend_support[framework]
        for dtype in framework_type_support[framework][-1:]
    ],
)
def test_fft_wrong_device_stream(framework, dtype):
    dev_count = cp.cuda.runtime.getDeviceCount()
    if dev_count < 2:
        pytest.skip(f"Test requires at least two gpus, got {dev_count}")

    with cp.cuda.Device(0):
        stream = get_custom_stream(framework)

    shape = 256
    signal = get_random_input_data(
        framework,
        (shape,),
        dtype,
        Backend.gpu,
        seed=12345,
        device_id=1,
    )

    import cupy_backends

    with pytest.raises(
        cupy_backends.cuda.api.runtime.CUDARuntimeError,
        match="cudaErrorInvalidResourceHandle: invalid resource handle",
    ):
        nvmath.fft.fft(signal, stream=stream)
