# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import random
import math
from ast import literal_eval

import pytest
import numpy as np
import cuda.core.experimental as ccx

from nvmath.memory import BaseCUDAMemoryManager, MemoryPointer

try:
    import cupy as cp
except ImportError:
    cp = None
try:
    import torch
except ImportError:
    torch = None

import nvmath

from .utils.common_axes import (
    ExecBackend,
    MemBackend,
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
)
from .utils.support_matrix import (
    type_shape_support,
    opt_fft_type_direction_support,
    opt_fft_type_input_type_support,
    inplace_opt_ftt_type_support,
    framework_exec_type_support,
    supported_backends,
    multi_gpu_only,
)
from .utils.input_fixtures import (
    get_1d_shape_cases,
    get_random_1d_shape,
    get_random_input_data,
    get_custom_stream,
    get_stream_pointer,
    init_assert_exec_backend_specified,
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

# DO NOT REMOVE, this call creates a fixture that enforces
# specifying execution option to the FFT calls in tests
# defined in this file
assert_exec_backend_specified = init_assert_exec_backend_specified()


@pytest.mark.parametrize(
    ("framework", "exec_backend", "mem_backend", "dtype", "shape_kind", "shape"),
    [
        (framework, exec_backend, mem_backend, dtype, shape_kind, shape)
        for framework in Framework.enabled()
        for exec_backend in supported_backends.exec
        for mem_backend in supported_backends.framework_mem[framework]
        for dtype in framework_exec_type_support[framework][exec_backend]
        for shape_kind, shape in get_1d_shape_cases(type_shape_support[exec_backend][dtype], rng)
    ],
)
def test_fft_ifft(framework, exec_backend, mem_backend, dtype, shape_kind, shape):
    signal = get_random_input_data(framework, (shape,), dtype, mem_backend, seed=42)

    if exec_backend == ExecBackend.cufft and shape == 1 and is_half(dtype) and get_cufft_version() < 10702:
        with pytest.raises(ValueError, match="sample size 1 and half-precision type"):
            nvmath.fft.fft(signal) if is_complex(dtype) else nvmath.fft.rfft(signal)
        return

    norm_ctx = {
        "shape_kind": shape_kind,
        "exec_backend": exec_backend,
    }

    fn = nvmath.fft.fft if is_complex(dtype) else nvmath.fft.rfft
    sample_fft = fn(signal, execution=exec_backend.nvname)
    assert_array_type(sample_fft, framework, mem_backend, get_fft_dtype(dtype))
    assert_norm_close(sample_fft, get_fft_ref(signal), **norm_ctx)

    ifft_fn = nvmath.fft.ifft if is_complex(dtype) else nvmath.fft.irfft
    sample_ifft = ifft_fn(
        sample_fft,
        execution=exec_backend.nvname,
        options=get_ifft_c2r_options(dtype, shape),
    )
    assert_array_type(sample_ifft, framework, mem_backend, dtype)
    assert_norm_close(sample_ifft, get_scaled(signal, shape), **norm_ctx)


@pytest.mark.parametrize(
    (
        "framework",
        "fft_type",
        "exec_backend",
        "mem_backend",
        "dtype",
        "shape_kind",
        "shape",
    ),
    [
        (
            framework,
            fft_type,
            exec_backend,
            mem_backend,
            dtype,
        )
        + get_random_1d_shape(type_shape_support[exec_backend][dtype], rng)
        for framework in Framework.enabled()
        for fft_type in [OptFftType.r2c, OptFftType.c2c]
        for exec_backend in supported_backends.exec
        for mem_backend in supported_backends.framework_mem[framework]
        for dtype in framework_exec_type_support[framework][exec_backend]
        if dtype in opt_fft_type_input_type_support[fft_type]
    ],
)
def test_fft_explicit_fft_type(framework, fft_type, exec_backend, mem_backend, dtype, shape_kind, shape):
    sample = get_random_input_data(framework, (shape,), dtype, mem_backend, seed=17)

    fn = nvmath.fft.fft if is_complex(dtype) else nvmath.fft.rfft
    sample_fft = fn(sample, options={"fft_type": fft_type.value}, execution=exec_backend.nvname)
    assert_array_type(sample_fft, framework, mem_backend, get_fft_dtype(dtype))
    assert_norm_close(
        sample_fft,
        get_fft_ref(sample),
        shape_kind=shape_kind,
        exec_backend=exec_backend,
    )


@pytest.mark.parametrize(
    (
        "framework",
        "fft_type",
        "exec_backend",
        "mem_backend",
        "dtype",
        "shape_kind",
        "shape",
    ),
    [
        (
            framework,
            fft_type,
            exec_backend,
            mem_backend,
            dtype,
        )
        + get_random_1d_shape(type_shape_support[exec_backend][dtype], rng)
        for framework in Framework.enabled()
        for fft_type in [OptFftType.c2r, OptFftType.c2c]
        for exec_backend in supported_backends.exec
        for mem_backend in supported_backends.framework_mem[framework]
        for dtype in framework_exec_type_support[framework][exec_backend]
        if dtype in opt_fft_type_input_type_support[fft_type]
    ],
)
def test_ifft_explicit_fft_type(framework, fft_type, exec_backend, mem_backend, dtype, shape_kind, shape):
    signal_dtype = get_ifft_dtype(dtype, fft_type)
    signal = get_random_input_data(framework, (shape,), signal_dtype, mem_backend, seed=21)
    sample_fft = get_fft_ref(signal)
    assert_array_type(sample_fft, framework, mem_backend, dtype)

    fn = nvmath.fft.ifft if is_complex(signal_dtype) else nvmath.fft.irfft
    sample_ifft = fn(
        sample_fft,
        options={
            "fft_type": fft_type.value,
            "last_axis_parity": "even" if shape % 2 == 0 else "odd",
        },
        execution=exec_backend.nvname,
    )
    assert_array_type(sample_ifft, framework, mem_backend, signal_dtype)
    assert_norm_close(
        sample_ifft,
        get_scaled(signal, shape),
        shape_kind=shape_kind,
        exec_backend=exec_backend,
    )


@pytest.mark.parametrize(
    ("framework", "exec_backend", "mem_backend", "dtype", "shape_kind", "shape"),
    [
        (framework, exec_backend, mem_backend, dtype, shape_kind, shape)
        for framework in Framework.enabled()
        for fft_type in inplace_opt_ftt_type_support[True]
        for exec_backend in supported_backends.exec
        for mem_backend in supported_backends.framework_mem[framework]
        for dtype in framework_exec_type_support[framework][exec_backend]
        if dtype in opt_fft_type_input_type_support[fft_type]
        for shape_kind, shape in get_1d_shape_cases(type_shape_support[exec_backend][dtype], rng)
    ],
)
def test_fft_inplace(framework, exec_backend, mem_backend, dtype, shape_kind, shape):
    sample = get_random_input_data(framework, (shape,), dtype, mem_backend, seed=71)
    ref_fft = get_fft_ref(sample)

    if exec_backend == ExecBackend.cufft and shape == 1 and is_half(dtype) and get_cufft_version() < 10702:
        with pytest.raises(ValueError, match="sample size 1 and half-precision type"):
            nvmath.fft.fft(sample, options={"inplace": True})
        return

    nvmath.fft.fft(sample, options={"inplace": True}, execution=exec_backend.nvname)
    assert_array_type(sample, framework, mem_backend, dtype)
    assert_norm_close(
        sample,
        ref_fft,
        shape_kind=shape_kind,
        exec_backend=exec_backend,
    )


@pytest.mark.parametrize(
    ("framework", "exec_backend", "mem_backend", "dtype", "shape_kind", "shape"),
    [
        (framework, exec_backend, mem_backend, dtype, shape_kind, shape)
        for framework in Framework.enabled()
        for fft_type in inplace_opt_ftt_type_support[True]
        for exec_backend in supported_backends.exec
        for mem_backend in supported_backends.framework_mem[framework]
        for dtype in framework_exec_type_support[framework][exec_backend]
        if dtype in opt_fft_type_input_type_support[fft_type]
        for shape_kind, shape in get_1d_shape_cases(type_shape_support[exec_backend][dtype], rng)
    ],
)
def test_ifft_inplace(framework, exec_backend, mem_backend, dtype, shape_kind, shape):
    signal = get_random_input_data(framework, (shape,), dtype, mem_backend, seed=32)
    sample = get_fft_ref(signal)

    if exec_backend == ExecBackend.cufft and shape == 1 and is_half(dtype) and get_cufft_version() < 10702:
        with pytest.raises(ValueError, match="sample size 1 and half-precision type"):
            nvmath.fft.ifft(sample, options={"inplace": True})
        return

    nvmath.fft.ifft(sample, options={"inplace": True}, execution=exec_backend.nvname)
    assert_array_type(sample, framework, mem_backend, dtype)
    assert_norm_close(
        sample,
        get_scaled(signal, shape),
        shape_kind=shape_kind,
        exec_backend=exec_backend,
    )


@pytest.mark.parametrize(
    (
        "framework",
        "exec_backend",
        "mem_backend",
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
            exec_backend,
            mem_backend,
            dtype,
            window_size,
            shape_kind,
            batch_size,
            step_size,
            result_layout,
        )
        for framework in Framework.enabled()
        for exec_backend in supported_backends.exec
        for mem_backend in supported_backends.framework_mem[framework]
        for dtype in framework_exec_type_support[framework][exec_backend]
        for window_size, shape_kind in (
            (1, ShapeKind.pow2),
            (2, ShapeKind.pow2),
            (4, ShapeKind.pow2),
            (256, ShapeKind.pow2),
            (2 * 3 * 5 * 7, ShapeKind.pow2357),
            (199, ShapeKind.prime),
        )
        if shape_kind in type_shape_support[exec_backend][dtype]
        for batch_size in (1, 3, 16, 99)
        for step_size in sorted(step for step in {1, window_size - 1, window_size, window_size + 1} if step > 0)
        for result_layout in OptFftLayout
    ],
)
def test_fft_ifft_overlap(
    framework,
    exec_backend,
    mem_backend,
    dtype,
    window_size,
    shape_kind,
    batch_size,
    step_size,
    result_layout,
):
    signal_size = batch_size * step_size + window_size - 1
    signal = get_random_input_data(framework, (signal_size,), dtype, mem_backend, seed=42)
    batch = unfold(signal, 0, window_size, step_size)
    assert_eq(batch.shape[0], batch_size)
    fft_fn = nvmath.fft.fft if is_complex(dtype) else nvmath.fft.rfft
    ifft_fn = nvmath.fft.ifft if is_complex(dtype) else nvmath.fft.irfft

    try:
        fft = fft_fn(
            batch,
            axes=(1,),
            options={"result_layout": result_layout.value},
            execution=exec_backend.nvname,
        )
        assert_array_type(fft, framework, mem_backend, get_fft_dtype(dtype))
        assert_norm_close(
            fft,
            get_fft_ref(batch, axes=(1,)),
            shape_kind=shape_kind,
            exec_backend=exec_backend,
        )
    except ValueError as e:
        assert is_half(dtype)
        if "sample size 1 and half-precision type" in str(e):
            assert exec_backend == ExecBackend.cufft and window_size == 1 and get_cufft_version() < 10702
        else:
            assert not is_complex(dtype) and mem_backend == mem_backend.cpu
            assert "is currently not supported for strided inputs" in str(e)
        return

    try:
        ifft = ifft_fn(
            fft,
            axes=(1,),
            options={
                "result_layout": result_layout.value,
                "last_axis_parity": "odd" if window_size % 2 else "even",
            },
            execution=exec_backend.nvname,
        )
        assert_array_type(ifft, framework, mem_backend, dtype)
        assert_norm_close(
            ifft,
            get_scaled(batch, window_size),
            shape_kind=shape_kind,
            exec_backend=exec_backend,
        )
    except ValueError as e:
        assert (
            exec_backend == ExecBackend.cufft
            and not is_complex(dtype)
            and is_half(dtype)
            and (mem_backend != exec_backend or result_layout == OptFftLayout.natural)
        )
        assert "is currently not supported for strided outputs" in str(e)
        return


@pytest.mark.parametrize(
    ("framework", "exec_backend", "mem_backend", "dtype", "blocking", "shape_kind", "shape"),
    [
        (framework, exec_backend, mem_backend, dtype, blocking)
        + get_random_1d_shape(type_shape_support[exec_backend][dtype], rng)
        for framework in Framework.enabled()
        for exec_backend in [ExecBackend.cufft]
        if exec_backend in supported_backends.exec
        for mem_backend in supported_backends.framework_mem[framework]
        for dtype in framework_exec_type_support[framework][exec_backend]
        for blocking in OptFftBlocking
    ],
)
def test_ifft_fft_blocking(monkeypatch, framework, exec_backend, mem_backend, dtype, blocking, shape_kind, shape):
    synchronization_num = 0
    _actual_sync = ccx.Event.sync

    def _synchronize(self):
        nonlocal synchronization_num
        synchronization_num += 1
        _actual_sync(self)

    monkeypatch.setattr(ccx.Event, "sync", _synchronize)

    sample = get_random_input_data(framework, (shape,), dtype, mem_backend, seed=33)
    sample_fft_ref = get_fft_ref(sample)
    sample_scaled = get_scaled(sample, shape)

    if is_complex(dtype):
        sample_fft = nvmath.fft.fft(
            sample,
            options={"blocking": blocking.value},
            execution=exec_backend.nvname,
        )
    else:
        sample_fft = nvmath.fft.rfft(
            sample,
            options={"blocking": blocking.value},
            execution=exec_backend.nvname,
        )
    assert_array_type(sample_fft, framework, mem_backend, get_fft_dtype(dtype))
    assert_norm_close(
        sample_fft,
        sample_fft_ref,
        shape_kind=shape_kind,
        exec_backend=exec_backend,
    )

    if is_complex(dtype):
        sample_ifft = nvmath.fft.ifft(
            sample_fft,
            execution=exec_backend.nvname,
            options={"blocking": blocking.value, **get_ifft_c2r_options(dtype, shape)},
        )
        assert_array_type(sample_ifft, framework, mem_backend, dtype)
        assert_norm_close(
            sample_ifft,
            sample_scaled,
            shape_kind=shape_kind,
            exec_backend=exec_backend,
        )

    if mem_backend == MemBackend.cpu or blocking == OptFftBlocking.true:
        expected_syncs = (1 + is_complex(dtype)) * 2  # 2x for plan creation and fft execution
    else:
        expected_syncs = 1 + is_complex(dtype)  # 2x for plan creation only
    assert_eq(synchronization_num, expected_syncs)


@pytest.mark.parametrize(
    (
        "framework",
        "exec_backend",
        "mem_backend",
        "dtype",
        "fft_type",
        "shape_kind",
        "shape",
    ),
    [
        (framework, exec_backend, mem_backend, dtype, fft_type)
        + get_random_1d_shape(type_shape_support[exec_backend][dtype], rng)
        for framework in Framework.enabled()
        for exec_backend in supported_backends.exec
        for mem_backend in supported_backends.framework_mem[framework]
        for dtype in framework_exec_type_support[framework][exec_backend]
        for fft_type in inplace_opt_ftt_type_support[True]
        if dtype in opt_fft_type_input_type_support[fft_type]
    ],
)
def test_fft_ifft_inplace_blocking_auto(framework, exec_backend, mem_backend, dtype, fft_type, shape_kind, shape):
    assert fft_type == OptFftType.c2c  # only C2C supports inplace FFT
    signal = get_random_input_data(framework, (shape,), dtype, mem_backend, seed=63)
    signal_scaled = get_scaled(signal, shape)
    signal_copy = copy_array(signal)

    nvmath.fft.fft(
        signal_copy,
        execution=exec_backend.nvname,
        options={"inplace": True, "blocking": "auto"},
    )
    nvmath.fft.ifft(
        signal_copy,
        execution=exec_backend.nvname,
        options={"inplace": True, "blocking": "auto"},
    )

    assert_array_type(signal_copy, framework, mem_backend, dtype)
    assert_norm_close(
        signal_copy,
        signal_scaled,
        shape_kind=shape_kind,
        exec_backend=exec_backend,
    )


@pytest.mark.parametrize(
    ("framework", "exec_backend", "mem_backend", "dtype"),
    [
        (framework, exec_backend, mem_backend, dtype)
        for framework in Framework.enabled()
        for exec_backend in [ExecBackend.cufft]
        if exec_backend in supported_backends.exec
        for mem_backend in supported_backends.framework_mem[framework]
        for dtype in framework_exec_type_support[framework][exec_backend]
        for fft_type in inplace_opt_ftt_type_support[True]
        if dtype in opt_fft_type_input_type_support[fft_type]
    ],
)
@multi_gpu_only
def test_fft_explicit_device_id(monkeypatch, framework, exec_backend, mem_backend, dtype):
    from nvmath.bindings import cufft  # type: ignore

    device_ids = intercept_device_id(monkeypatch, (cufft, "xt_make_plan_many"), (cufft, "xt_exec"))

    shape = 4096
    signal = get_random_input_data(framework, (shape,), dtype, mem_backend, seed=318)
    signal_copy = copy_array(signal)

    array_device = 0
    fn = nvmath.fft.fft if is_complex(dtype) else nvmath.fft.rfft
    fn(
        signal_copy,
        execution={"name": "cuda", "device_id": 0},
        options={
            "inplace": True,
            "blocking": "auto",
        },
    )
    expected_device = 0 if mem_backend == MemBackend.cpu else array_device
    assert_eq(device_ids["xt_make_plan_many"], expected_device)
    assert_eq(device_ids["xt_exec"], expected_device)

    if is_complex(dtype):
        expected_device = 1 if mem_backend == MemBackend.cpu else array_device
        nvmath.fft.ifft(
            signal_copy,
            execution={"name": "cuda", "device_id": 1},
            options={"inplace": True, "blocking": "auto"},
        )
        assert_eq(device_ids["xt_make_plan_many"], expected_device)
        assert_eq(device_ids["xt_exec"], expected_device)

        if mem_backend == MemBackend.cuda:
            assert_eq(get_array_device_id(signal_copy), array_device)
        assert_array_type(signal_copy, framework, mem_backend, dtype)
        assert_norm_close(
            signal_copy,
            get_scaled(signal, shape),
            exec_backend=exec_backend,
        )


@pytest.mark.parametrize(
    ("framework", "exec_backend", "mem_backend", "blocking", "dtype"),
    [
        (framework, exec_backend, mem_backend, blocking, dtype)
        for framework in Framework.enabled()
        for exec_backend in [ExecBackend.cufft]
        if exec_backend in supported_backends.exec
        for mem_backend in [MemBackend.cuda]
        if mem_backend in supported_backends.framework_mem[framework]
        for blocking in OptFftBlocking
        for dtype in framework_exec_type_support[framework][exec_backend]
    ],
)
@multi_gpu_only
def test_fft_array_device_id(monkeypatch, framework, exec_backend, mem_backend, blocking, dtype):
    shape = 2048
    signal_1 = get_random_input_data(framework, (shape,), dtype, mem_backend, seed=415, device_id=1)
    signal_0 = get_random_input_data(framework, (shape,), dtype, mem_backend, seed=416, device_id=0)

    from nvmath.bindings import cufft  # type: ignore

    device_ids = intercept_device_id(monkeypatch, (cufft, "xt_make_plan_many"), (cufft, "xt_exec"))

    fft_fn = nvmath.fft.fft if is_complex(dtype) else nvmath.fft.rfft

    fft_1 = fft_fn(
        signal_1,
        options={"blocking": blocking.value},
        execution=exec_backend.nvname,
    )
    assert_eq(device_ids["xt_make_plan_many"], 1)
    assert_eq(device_ids["xt_exec"], 1)
    fft_0 = fft_fn(
        signal_0,
        options={"blocking": blocking.value},
        execution=exec_backend.nvname,
    )
    assert_eq(device_ids["xt_make_plan_many"], 0)
    assert_eq(device_ids["xt_exec"], 0)

    for dev_id, fft in enumerate((fft_0, fft_1)):
        assert_eq(get_array_device_id(fft), dev_id)
        assert_array_type(fft, framework, mem_backend, get_fft_dtype(dtype))

    assert_norm_close(
        fft_0,
        get_fft_ref(signal_0),
        exec_backend=exec_backend,
    )
    with cp.cuda.Device(1):
        assert_norm_close(
            fft_1,
            get_fft_ref(signal_1),
            exec_backend=exec_backend,
        )

    if is_complex(dtype):
        ifft_1 = nvmath.fft.ifft(
            fft_1,
            execution=exec_backend.nvname,
            options={"blocking": blocking.value, **get_ifft_c2r_options(dtype, shape)},
        )
        assert_eq(device_ids["xt_make_plan_many"], 1)
        assert_eq(device_ids["xt_exec"], 1)
        ifft_0 = nvmath.fft.ifft(
            fft_0,
            execution=exec_backend.nvname,
            options={"blocking": blocking.value, **get_ifft_c2r_options(dtype, shape)},
        )
        assert_eq(device_ids["xt_make_plan_many"], 0)
        assert_eq(device_ids["xt_exec"], 0)

        for dev_id, ifft in enumerate((ifft_0, ifft_1)):
            assert_eq(get_array_device_id(ifft), dev_id)
            assert_array_type(ifft, framework, mem_backend, dtype)

        assert_norm_close(
            ifft_0,
            get_scaled(signal_0, shape),
            exec_backend=exec_backend,
        )
        with cp.cuda.Device(1):
            assert_norm_close(
                ifft_1,
                get_scaled(signal_1, shape),
                exec_backend=exec_backend,
            )


@pytest.mark.parametrize(
    ("framework", "exec_backend", "mem_backend", "dtype", "use_stream_ptr"),
    [
        (framework, exec_backend, mem_backend, dtype, use_stream_ptr)
        for framework in Framework.enabled()
        for exec_backend in [ExecBackend.cufft]
        if exec_backend in supported_backends.exec
        for mem_backend in [MemBackend.cuda]
        if mem_backend in supported_backends.framework_mem[framework]
        for dtype in framework_exec_type_support[framework][exec_backend]
        if not is_half(dtype)  # for the fft size, the halfs lack precision
        for use_stream_ptr in (True, False)
    ],
)
def test_fft_custom_stream(framework, exec_backend, mem_backend, dtype, use_stream_ptr):
    stream = get_custom_stream(framework)
    shape = 1024 * 1024

    with use_stream(stream):
        signal = get_random_input_data(framework, (shape,), dtype, mem_backend, seed=234)
        fft_ref = get_fft_ref(signal)
        fft_ref = fft_ref * 42

    fft_fn = nvmath.fft.fft if is_complex(dtype) else nvmath.fft.rfft
    try:
        fft = fft_fn(
            signal,
            execution=exec_backend.nvname,
            options={
                "blocking": "auto",
            },
            stream=get_stream_pointer(stream) if use_stream_ptr else stream,
        )
    except TypeError as e:
        assert "A stream object must be provided for PyTorch operands" in str(e) and framework == Framework.torch
        return

    with use_stream(stream):
        # The stateless API synchronizes on plan creation,
        # so to see proper stream handling, we do some postprocessing
        # on the stream
        fft = fft * 42
        assert_norm_close(
            fft,
            fft_ref,
            exec_backend=exec_backend,
        )


@pytest.mark.parametrize(
    ("framework", "dtype", "exec_backend", "mem_backend"),
    [
        (framework, dtype, exec_backend, mem_backend)
        for framework in Framework.enabled()
        for dtype in DType
        for exec_backend in supported_backends.exec
        for mem_backend in supported_backends.framework_mem[framework]
        if dtype not in framework_exec_type_support[framework][exec_backend]
    ],
)
def test_fft_ifft_unsupported_type(framework, dtype, exec_backend, mem_backend):
    if dtype not in framework_dtype[framework]:
        pytest.skip(f"Type {dtype.name} not representable in {framework.name}")
    if framework == Framework.numpy:
        sample = np.arange(128).astype(get_framework_dtype(framework, dtype))
    elif framework == Framework.cupy:
        sample = cp.arange(128).astype(get_framework_dtype(framework, dtype))
    elif framework == Framework.torch:
        sample = torch.arange(128).type(get_framework_dtype(framework, dtype))
        if mem_backend == MemBackend.cuda:
            sample = sample.cuda()
    else:
        raise ValueError(f"Unknown framework {framework}")

    if is_complex(dtype):
        if exec_backend == ExecBackend.fftw and is_half(dtype):
            fft_match = "FFT supports following input types"
        else:
            fft_match = f"Unsupported dtype '{dtype.name}' for FFT"
    else:
        fft_match = f"This function expects complex operand, found {dtype.name}"

    with pytest.raises(ValueError, match=fft_match):
        nvmath.fft.fft(sample, execution=exec_backend.nvname)

    with pytest.raises(ValueError, match=fft_match):
        nvmath.fft.ifft(sample, execution=exec_backend.nvname)

    with pytest.raises(ValueError, match=fft_match):
        nvmath.fft.irfft(sample, execution=exec_backend.nvname)

    if dtype in r2c_dtype and r2c_dtype[dtype] not in framework_dtype[framework]:
        err_cls = TypeError
        rfft_match = f"The result data type {r2c_dtype[dtype].name} is not supported by the operand package '{framework.name}'."
    elif is_complex(dtype):
        err_cls = RuntimeError
        rfft_match = f"expects a real input, but got {dtype.name}"
    elif is_half(dtype) and dtype != DType.bfloat16:
        err_cls = ValueError
        rfft_match = "FFT supports following input types"
    else:
        err_cls = ValueError
        rfft_match = f"Unsupported dtype '{dtype.name}' for FFT"

    with pytest.raises(err_cls, match=rfft_match):
        nvmath.fft.rfft(sample, execution=exec_backend.nvname)


@pytest.mark.parametrize(
    ("framework", "exec_backend", "mem_backend", "dtype", "shape_kind"),
    [
        (framework, exec_backend, mem_backend, dtype, shape_kind)
        for framework in Framework.enabled()
        for exec_backend in supported_backends.exec
        for mem_backend in supported_backends.framework_mem[framework]
        for dtype in framework_exec_type_support[framework][exec_backend]
        for shape_kind in ShapeKind
        if shape_kind not in type_shape_support[exec_backend][dtype] and shape_kind != ShapeKind.random
    ],
)
def test_fft_unsupported_shape(framework, exec_backend, mem_backend, dtype, shape_kind):
    assert exec_backend == ExecBackend.cufft, f"Didn't expect difference in shape kind support for {exec_backend}"
    # Note, cufft has undocumented partial support for not pow of 2 shapes above 128
    # This test selects shapes that are actually failing only to check the handling
    # of unsupported input.
    shapes = {
        ShapeKind.pow2357: 210,
        ShapeKind.prime: 127,
    }
    sample = get_random_input_data(framework, (shapes[shape_kind],), dtype, mem_backend, seed=101)

    fft_fn = nvmath.fft.fft if is_complex(dtype) else nvmath.fft.rfft

    with pytest.raises(
        nvmath.bindings.cufft.cuFFTError,
        match="(CUFFT_NOT_SUPPORTED|CUFFT_SETUP_FAILED)",
    ):
        fft_fn(sample, execution=exec_backend.nvname)


@pytest.mark.parametrize(
    ("framework", "exec_backend", "mem_backend", "dtype", "shape_kind"),
    [
        (framework, exec_backend, mem_backend, dtype, shape_kind)
        for framework in Framework.enabled()
        for exec_backend in supported_backends.exec
        for mem_backend in supported_backends.framework_mem[framework]
        for dtype in framework_exec_type_support[framework][exec_backend]
        if is_complex(dtype)
        for shape_kind in ShapeKind
        if shape_kind not in type_shape_support[exec_backend][dtype] and shape_kind != ShapeKind.random
    ],
)
def test_ifft_unsupported_shape(framework, exec_backend, mem_backend, dtype, shape_kind):
    assert exec_backend == ExecBackend.cufft, f"Didn't expect difference in shape kind support for {exec_backend}"
    # Note, cufft has undocumented partial support for prime shapes above 128
    # This test selects shapes that are actually failing only to check the handling
    # of unsupported input.
    shapes = {
        ShapeKind.pow2357: 630,
        ShapeKind.prime: 101,
    }
    sample = get_random_input_data(framework, (shapes[shape_kind],), dtype, mem_backend, seed=101)
    with pytest.raises(
        nvmath.bindings.cufft.cuFFTError,
        match="(CUFFT_NOT_SUPPORTED|CUFFT_SETUP_FAILED)",
    ):
        nvmath.fft.ifft(sample, execution=exec_backend.nvname)


@pytest.mark.parametrize(
    ("framework", "exec_backend", "mem_backend", "dtype", "shape", "axes"),
    [
        (framework, exec_backend, mem_backend, dtype, repr(shape), repr(axes))
        for framework in Framework.enabled()
        for exec_backend in supported_backends.exec
        for mem_backend in supported_backends.framework_mem[framework]
        for dtype in framework_exec_type_support[framework][exec_backend]
        if is_complex(dtype)
        for shape, axes in [
            ((1,), None),
            ((2, 2, 1), (1, 2)),
        ]
    ],
)
def test_irfft_unsupported_empty_output(framework, exec_backend, dtype, mem_backend, shape, axes):
    shape = literal_eval(shape)
    axes = literal_eval(axes)
    signal = get_random_input_data(framework, shape, dtype, mem_backend, seed=101)
    sample_vol = math.prod(shape) if axes is None else math.prod(shape[a] for a in axes)

    if exec_backend == ExecBackend.cufft and sample_vol == 1 and is_half(dtype) and get_cufft_version() < 10702:
        with pytest.raises(ValueError, match="sample size 1 and half-precision type"):
            nvmath.fft.ifft(signal, execution=exec_backend.nvname, options={"inplace": True})
    else:
        nvmath.fft.irfft(
            signal,
            axes=axes,
            execution=exec_backend.nvname,
            options={"last_axis_parity": "odd"},
        )

    with pytest.raises(
        ValueError,
        match="The size of the last FFT axis in the result for FFT type 'C2R' is 0",
    ):
        nvmath.fft.irfft(signal, execution=exec_backend.nvname, axes=axes)


@pytest.mark.parametrize(
    ("framework", "fft_type", "direction", "exec_backend", "mem_backend"),
    [
        (framework, fft_type, direction, exec_backend, mem_backend)
        for framework in Framework.enabled()
        for fft_type in OptFftType
        for direction in Direction
        if (direction not in opt_fft_type_direction_support[fft_type])
        for exec_backend in supported_backends.exec
        for mem_backend in supported_backends.framework_mem[framework]
    ],
)
def test_incompatible_fft_type_direction(framework, fft_type, direction, exec_backend, mem_backend):
    if fft_type == OptFftType.r2c:
        if direction == Direction.inverse:
            pytest.skip("No fft function API for real inverse FFT")
        dtype = DType.float32
        fn = nvmath.fft.rfft
    else:
        dtype = DType.complex64
        fn = nvmath.fft.fft if direction == Direction.forward else nvmath.fft.ifft
    sample = get_random_input_data(framework, (128,), dtype, mem_backend, seed=15)
    with pytest.raises(
        ValueError,
        match=(f"The specified direction {direction.value} is not compatible with the FFT type '{fft_type.value}'"),
    ):
        fn(sample, execution=exec_backend.nvname, options={"fft_type": fft_type.value})


@pytest.mark.parametrize(
    ("framework", "fft_type", "direction", "exec_backend", "mem_backend", "dtype"),
    [
        (framework, fft_type, direction, exec_backend, mem_backend, dtype)
        for framework in Framework.enabled()
        for fft_type in OptFftType
        for direction in opt_fft_type_direction_support[fft_type]
        for exec_backend in supported_backends.exec
        for mem_backend in supported_backends.framework_mem[framework]
        for dtype in framework_exec_type_support[framework][exec_backend]
        if dtype not in opt_fft_type_input_type_support[fft_type]
    ],
)
def test_incompatible_fft_type_dtype(framework, fft_type, direction, exec_backend, mem_backend, dtype):
    sample = get_random_input_data(framework, (512,), dtype, mem_backend, seed=17)
    if is_complex(dtype):
        fn = nvmath.fft.fft if direction == Direction.forward else nvmath.fft.ifft
    else:
        if direction == Direction.inverse:
            pytest.skip("No fft function API for inverse real FFT")
        else:
            fn = nvmath.fft.rfft
    if fft_type == OptFftType.c2c:
        if exec_backend == ExecBackend.fftw:
            err_cls = ValueError
            match = f"Got unsupported input data type {dtype.name} for the C2C transform"
        else:
            err_cls = nvmath.bindings.cufft.cuFFTError
            match = "CUFFT_INVALID_TYPE"
        with pytest.raises(
            err_cls,
            match=match,
        ):
            fn(
                sample,
                execution=exec_backend.nvname,
                options={"fft_type": fft_type.value},
            )
    else:
        with pytest.raises(
            AssertionError,
            match=f"Internal Error \\(name='{dtype.name}'\\)",
        ):
            fn(
                sample,
                execution=exec_backend.nvname,
                options={"fft_type": fft_type.value},
            )


@pytest.mark.parametrize(
    ("framework", "fft_type", "direction", "exec_backend", "mem_backend", "dtype"),
    [
        (
            framework,
            fft_type,
            direction,
            exec_backend,
            mem_backend,
            dtype,
        )
        for framework in Framework.enabled()
        for fft_type in OptFftType
        if fft_type not in inplace_opt_ftt_type_support[True]
        for direction in opt_fft_type_direction_support[fft_type]
        for exec_backend in supported_backends.exec
        for mem_backend in supported_backends.framework_mem[framework]
        for dtype in framework_exec_type_support[framework][exec_backend]
        if dtype in opt_fft_type_input_type_support[fft_type]
    ],
)
def test_inplace_unsupported_fft_type(framework, fft_type, direction, exec_backend, mem_backend, dtype):
    sample = get_random_input_data(framework, (256,), dtype, mem_backend, seed=19)
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
        fn(
            sample,
            execution=exec_backend.nvname,
            options={"fft_type": fft_type.value, "inplace": True},
        )


@pytest.mark.parametrize(
    ("framework", "exec_backend", "mem_backend", "dtype"),
    [
        (
            Framework.numpy,
            exec_backend,
            MemBackend.cpu,
            dtype,
        )
        for exec_backend in ExecBackend
        if exec_backend not in supported_backends.exec
        for dtype in [DType.float32, DType.complex64]
    ],
)
def test_unsupported_execution_backend(framework, exec_backend, mem_backend, dtype):
    sample = get_random_input_data(framework, (16,), dtype, mem_backend, seed=119)
    fn = nvmath.fft.fft if is_complex(dtype) else nvmath.fft.rfft

    with pytest.raises(
        RuntimeError,
        match=f"FFT {exec_backend.nvname.upper()} execution",
    ):
        fn(sample, execution=exec_backend.nvname)


@pytest.mark.parametrize(
    ("framework", "mem_backend", "dtype"),
    [
        (
            Framework.numpy,
            MemBackend.cpu,
            dtype,
        )
        for dtype in [DType.float32, DType.complex64]
    ],
)
def test_wrong_execution_backend(framework, mem_backend, dtype):
    sample = get_random_input_data(framework, (16,), dtype, mem_backend, seed=119)
    fn = nvmath.fft.fft if is_complex(dtype) else nvmath.fft.rfft

    with pytest.raises(
        ValueError,
        match="The 'execution' options must be",
    ):
        fn(sample, execution="tpu", options={"device_id": 1})

    with pytest.raises(
        ValueError,
        match="The 'execution' options must be",
    ):
        fn(sample, execution="TPU")

    with pytest.raises(
        ValueError,
        match="The 'execution' options must be",
    ):
        fn(
            sample,
            execution={"name": "tty", "num_threads": 7},
            options={"device_id": 1},
        )

    with pytest.raises(
        ValueError,
        match="The 'execution' options must be",
    ):
        fn(
            sample,
            execution={"name": "tty", "device_id": 7},
            options={"result_layout": "odd"},
        )

    with pytest.raises(
        ValueError,
        match="The 'execution' options must be",
    ):
        fn(sample, execution=7, options={"device_id": 1})

    with pytest.raises(
        ValueError,
        match="The 'execution' options must be",
    ):
        fn(sample, execution=lambda x: x)

    with pytest.raises(
        ValueError,
        match="The 'execution' options must be",
    ):
        fn(sample, execution={"name": lambda x: x}, options={"device_id": 1})

    with pytest.raises(
        TypeError,
        match="got an unexpected keyword argument 'namex'",
    ):
        fn(sample, execution={"namex": "cpu"})


@pytest.mark.parametrize(
    ("framework", "exec_backend", "mem_backend", "dtype"),
    [
        (
            Framework.numpy,
            exec_backend,
            MemBackend.cpu,
            dtype,
        )
        for exec_backend in supported_backends.exec
        if exec_backend != ExecBackend.cufft
        for dtype in [DType.float32, DType.complex64]
    ],
)
def test_cpu_execution_wrong_options(framework, exec_backend, mem_backend, dtype):
    sample = get_random_input_data(framework, (16,), dtype, mem_backend, seed=119)
    fn = nvmath.fft.fft if is_complex(dtype) else nvmath.fft.rfft
    assert exec_backend.nvname == "cpu"

    with pytest.raises(
        ValueError,
        match="The 'device_id' is not a valid option when 'execution' is specified to be 'cpu'",
    ):
        fn(sample, execution="CPU", options={"device_id": 1})

    with pytest.raises(
        ValueError,
        match="The 'device_id' is not a valid option when 'execution' is specified to be 'cpu'",
    ):
        fn(sample, execution="cpu", options={"last_axis_parity": "odd", "device_id": 1})

    with pytest.raises(
        TypeError,
        match="unexpected keyword argument 'device_id'",
    ):
        fn(sample, execution={"name": "CPU", "num_threads": 3, "device_id": 1})

    with pytest.raises(
        TypeError,
        match="unexpected keyword argument 'non_existing'",
    ):
        fn(
            sample,
            execution={"name": "cpu", "non_existing": 1},
            options={"result_layout": "natural"},
        )

    with pytest.raises(
        ValueError,
        match="The 'num_threads' must be a positive integer",
    ):
        fn(
            sample,
            execution={"name": "CPU", "num_threads": "a_lot_please"},
            options={"result_layout": "natural"},
        )

    with pytest.raises(
        ValueError,
        match="The 'num_threads' must be a positive integer",
    ):
        fn(
            sample,
            execution={"name": "cpu", "num_threads": -7},
            options={"result_layout": "natural"},
        )


@pytest.mark.parametrize(
    ("framework", "exec_backend", "mem_backend", "dtype"),
    [
        (
            Framework.cupy,
            exec_backend,
            MemBackend.cuda,
            dtype,
        )
        for exec_backend in supported_backends.exec
        if exec_backend == ExecBackend.cufft
        for dtype in [DType.float32, DType.complex64]
    ],
)
def test_gpu_execution_wrong_options(framework, exec_backend, mem_backend, dtype):
    sample = get_random_input_data(framework, (16,), dtype, mem_backend, seed=119)
    fn = nvmath.fft.fft if is_complex(dtype) else nvmath.fft.rfft
    assert exec_backend.nvname == "cuda"

    with pytest.raises(
        TypeError,
        match="unexpected keyword argument 'num_threads'",
    ):
        fn(sample, execution="CUDA", options={"num_threads": 1})

    with pytest.raises(
        TypeError,
        match="unexpected keyword argument 'num_threads'",
    ):
        fn(sample, execution={"name": "cuda", "num_threads": 3, "device_id": 1})


@pytest.mark.parametrize(
    ("framework", "exec_backend", "mem_backend", "dtype"),
    [
        (
            Framework.cupy,
            exec_backend,
            MemBackend.cuda,
            dtype,
        )
        for exec_backend in supported_backends.exec
        if exec_backend == ExecBackend.cufft
        for dtype in [DType.float32, DType.complex64]
    ],
)
def test_conflicting_device_id_option(framework, exec_backend, mem_backend, dtype):
    sample = get_random_input_data(framework, (16,), dtype, mem_backend, seed=119)
    fn = nvmath.fft.fft if is_complex(dtype) else nvmath.fft.rfft
    assert exec_backend.nvname == "cuda"

    fn(
        sample,
        execution={
            "name": "CUDA",
            "device_id": 1,
        },
        options={"device_id": 1},
    )

    with pytest.raises(
        ValueError,
        match="Got conflicting 'device_id' passed in 'execution' \\(2\\) and 'options' \\(1\\)",
    ):
        fn(
            sample,
            execution={
                "name": "CUDA",
                "device_id": 2,
            },
            options={"device_id": 1},
        )


@pytest.mark.parametrize(
    ("framework", "exec_backend", "mem_backend", "dtype"),
    [
        (
            Framework.numpy,
            exec_backend,
            MemBackend.cpu,
            dtype,
        )
        for exec_backend in supported_backends.exec
        if exec_backend != ExecBackend.cufft
        for dtype in [DType.float32, DType.complex64]
    ],
)
def test_stride_overflow_error(framework, exec_backend, mem_backend, dtype):
    sample = get_random_input_data(framework, (2**31,), dtype, mem_backend, seed=19)
    fn = nvmath.fft.fft if is_complex(dtype) else nvmath.fft.rfft
    with pytest.raises(
        ValueError,
        match="shape extents or stride larger than `2147483647` are not currently supported",
    ):
        fn(sample, execution=exec_backend.nvname)

    sample = sample.reshape((-1, 1))
    fn = nvmath.fft.fft if is_complex(dtype) else nvmath.fft.rfft
    with pytest.raises(
        ValueError,
        match="shape extents or stride larger than `2147483647` are not currently supported",
    ):
        fn(sample, axes=(1,), execution=exec_backend.nvname)


@pytest.mark.parametrize(
    ("framework", "exec_backend", "mem_backend", "dtype"),
    [
        (framework, exec_backend, mem_backend, dtype)
        for framework in Framework.enabled()
        for exec_backend in supported_backends.exec
        for mem_backend in supported_backends.framework_mem[framework]
        for dtype in framework_exec_type_support[framework][exec_backend]
    ],
)
def test_inplace_unsupported_implicit_r2c_c2r(framework, exec_backend, mem_backend, dtype):
    sample = get_random_input_data(framework, (256,), dtype, mem_backend, seed=19)
    fn = nvmath.fft.irfft if is_complex(dtype) else nvmath.fft.rfft
    implicit_kind = OptFftType.c2r if is_complex(dtype) else OptFftType.r2c
    with pytest.raises(
        ValueError,
        match=(
            f"The in-place option \\(FFTOptions\\.inplace=True\\) is only supported for "
            f"complex-to-complex FFT\\. The FFT type is '{implicit_kind.value}'"
        ),
    ):
        fn(sample, execution=exec_backend.nvname, options={"inplace": True})


@pytest.mark.parametrize(
    ("framework", "exec_backend", "mem_backend", "dtype"),
    [
        (framework, exec_backend, mem_backend, dtype)
        for framework in Framework.enabled()
        for exec_backend in [ExecBackend.cufft]
        if exec_backend in supported_backends.exec
        for mem_backend in [MemBackend.cuda]
        if mem_backend in supported_backends.framework_mem[framework]
        for dtype in framework_exec_type_support[framework][exec_backend][-1:]
    ],
)
@multi_gpu_only
def test_fft_wrong_device_stream(framework, exec_backend, mem_backend, dtype):
    with cp.cuda.Device(0):
        stream = get_custom_stream(framework)

    shape = 256
    signal = get_random_input_data(
        framework,
        (shape,),
        dtype,
        mem_backend,
        seed=12345,
        device_id=1,
    )

    with pytest.raises(Exception):
        nvmath.fft.fft(signal, stream=stream, execution=exec_backend.nvname)


class CustomMemoryManager(BaseCUDAMemoryManager):
    """
    This test class implements a mock memory manager that simulates memory allocation
    and deallocation without actually performing them.

    The class tracks whether its finalizer was called to verify proper cleanup behavior.
    It uses a fake pointer value since no real memory is allocated.
    """

    def __init__(self, use_finalizer):
        self._finalizer_called = False
        self._use_finalizer = use_finalizer

    def memalloc(self, size):
        def create_finalizer():
            def finalizer():
                # Ensure the finalizer would be called only once
                assert self._finalizer_called is False
                self._finalizer_called = True

            return finalizer

        fake_ptr = 123  # We don't actually allocate memory, so just use a fake pointer for simplicity
        return MemoryPointer(fake_ptr, size, finalizer=create_finalizer() if self._use_finalizer else None)


@pytest.mark.parametrize("use_finalizer", [True, False])
def test_memory_manager_explicit_free(use_finalizer):
    custom_mgr = CustomMemoryManager(use_finalizer)
    assert custom_mgr._finalizer_called is False
    ptr = custom_mgr.memalloc(1000)
    ptr.free()  # Explicitly release the resource
    assert custom_mgr._finalizer_called == use_finalizer  # Verify the finalizer was called or not

    # Test double-free
    if use_finalizer:
        with pytest.raises(RuntimeError, match="The buffer has already been freed."):
            ptr.free()  # double-free should raise an error
    else:
        ptr.free()  # Nothing bad happened if the finalizer is not provided

    ptr = None  # This should NOT trigger another finalizer even when GC'ed,
    # verified by the finalizer being called only once above
