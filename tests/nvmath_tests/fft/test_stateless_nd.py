# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

from ast import literal_eval
import re
import random
import math

import pytest

import nvmath

from .utils.common_axes import (
    Framework,
    ExecBackend,
    MemBackend,
    DType,
    ShapeKind,
    OptFftLayout,
    OptFftType,
    Direction,
)
from .utils.axes_utils import (
    is_complex,
    is_half,
    get_fft_dtype,
    get_ifft_dtype,
    size_of,
)
from .utils.support_matrix import (
    type_shape_support,
    opt_fft_type_input_type_support,
    opt_fft_type_direction_support,
    inplace_opt_ftt_type_support,
    framework_exec_type_support,
    supported_backends,
)
from .utils.input_fixtures import (
    get_random_input_data,
    init_assert_exec_backend_specified,
    fx_last_operand_layout,
)
from .utils.check_helpers import (
    is_decreasing,
    copy_array,
    get_fft_ref,
    get_ifft_ref,
    get_scaled,
    check_layout_fallback,
    get_permuted_copy,
    get_rev_perm,
    get_transposed,
    get_cufft_version,
    unfold,
    as_strided,
    get_ifft_c2r_options,
    get_array_strides,
    get_array_element_strides,
    assert_norm_close,
    assert_array_type,
    assert_eq,
    should_skip_3d_unsupported,
    assert_array_equal,
    get_raw_ptr,
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
        "array_dim",
        "axis",
        "dtype",
        "shape_kind",
        "result_layout",
    ),
    [
        (
            framework,
            exec_backend,
            mem_backend,
            array_dim,
            axis,
            dtype,
            rng.choice(type_shape_support[exec_backend][dtype]),
            rng.choice(list(OptFftLayout)),
        )
        for framework in Framework.enabled()
        for exec_backend in supported_backends.exec
        for mem_backend in supported_backends.framework_mem[framework]
        for array_dim in [2, 3]
        for axis in [0, array_dim - 1]
        for dtype in [DType.float16, DType.float32, DType.complex64]
        if dtype in framework_exec_type_support[framework][exec_backend]
    ],
)
def test_fft_ifft_1d(
    fx_last_operand_layout,  # noqa: F811
    framework,
    exec_backend,
    mem_backend,
    array_dim,
    axis,
    dtype,
    shape_kind,
    result_layout,
):
    shapes = {
        ShapeKind.pow2: 128,
        ShapeKind.pow2357: 2 * 3 * 5 * 7,
        ShapeKind.prime: 127,
        ShapeKind.random: 207,
    }
    shape = (shapes[shape_kind],) * array_dim
    signal = get_random_input_data(framework, shape, dtype, mem_backend, seed=55)
    check_layouts, *_ = fx_last_operand_layout

    fft_fn = nvmath.fft.fft if is_complex(dtype) else nvmath.fft.rfft

    if dtype == DType.float16 and axis != array_dim - 1:
        with pytest.raises(ValueError, match="is currently not supported for strided inputs"):
            fft_fn(
                signal,
                axes=[axis],
                execution=exec_backend.nvname,
                options={"result_layout": result_layout.value},
            )
        return

    fft = fft_fn(
        signal,
        axes=[axis],
        execution=exec_backend.nvname,
        options={"result_layout": result_layout.value},
    )
    check_layouts(
        exec_backend,
        mem_backend,
        (axis,),
        result_layout,
        OptFftType.c2c if is_complex(dtype) else OptFftType.r2c,
        is_dense=True,
        inplace=False,
    )
    assert_array_type(fft, framework, mem_backend, get_fft_dtype(dtype))
    assert_norm_close(
        fft,
        get_fft_ref(signal, axes=[axis]),
        axes=[axis],
        shape_kind=shape_kind,
        exec_backend=exec_backend,
    )

    if is_complex(dtype):
        options = {
            "result_layout": result_layout.value,
            **get_ifft_c2r_options(dtype, shape[axis]),
        }
        ifft = nvmath.fft.ifft(
            fft,
            options=options,
            execution=exec_backend.nvname,
            axes=[axis],
        )
        check_layouts(
            exec_backend,
            mem_backend,
            (axis,),
            result_layout,
            OptFftType.c2c,
            is_dense=True,
            inplace=False,
        )
        assert_array_type(ifft, framework, mem_backend, dtype)
        assert_norm_close(
            ifft,
            get_scaled(signal, shape[axis]),
            shape_kind=shape_kind,
            exec_backend=exec_backend,
        )


@pytest.mark.parametrize(
    (
        "framework",
        "exec_backend",
        "mem_backend",
        "array_dim",
        "first_axis",
        "dtype",
        "shape_kind",
        "result_layout",
    ),
    [
        (
            framework,
            exec_backend,
            mem_backend,
            array_dim,
            first_axis,
            dtype,
            rng.choice(type_shape_support[exec_backend][dtype]),
            rng.choice(list(OptFftLayout)),
        )
        for framework in Framework.enabled()
        for exec_backend in supported_backends.exec
        for mem_backend in supported_backends.framework_mem[framework]
        for array_dim in [2, 3]
        for first_axis in range(array_dim - 2 + 1)
        for dtype in framework_exec_type_support[framework][exec_backend]
    ],
)
def test_fft_ifft_2d(
    fx_last_operand_layout,  # noqa: F811
    framework,
    exec_backend,
    mem_backend,
    array_dim,
    first_axis,
    dtype,
    shape_kind,
    result_layout,
):
    shapes = {
        ShapeKind.pow2: (64, 256, 128),
        ShapeKind.pow2357: (9 * 49, 4 * 25, 2 * 3 * 5 * 7),
        ShapeKind.prime: (127, 233, 277),
        ShapeKind.random: (209, 178, 361),
    }
    shape = shapes[shape_kind][:array_dim]
    axes = list(range(first_axis, first_axis + 2))
    signal = get_random_input_data(framework, shape, dtype, mem_backend, seed=55)
    check_layouts, *_ = fx_last_operand_layout

    fft_fn = nvmath.fft.fft if is_complex(dtype) else nvmath.fft.rfft

    if dtype == DType.float16 and first_axis != array_dim - 2:
        with pytest.raises(ValueError, match="is currently not supported for strided inputs"):
            fft_fn(
                signal,
                axes=axes,
                execution=exec_backend.nvname,
                options={"result_layout": result_layout.value},
            )
        return

    fft = fft_fn(
        signal,
        axes=axes,
        execution=exec_backend.nvname,
        options={"result_layout": result_layout.value},
    )
    check_layouts(
        exec_backend,
        mem_backend,
        axes,
        result_layout,
        OptFftType.c2c if is_complex(dtype) else OptFftType.r2c,
        is_dense=True,
        inplace=False,
    )
    assert_array_type(fft, framework, mem_backend, get_fft_dtype(dtype))
    assert_norm_close(
        fft,
        get_fft_ref(signal, axes=axes),
        axes=axes,
        shape_kind=shape_kind,
        exec_backend=exec_backend,
    )
    if is_complex(dtype):
        options = {
            "result_layout": result_layout.value,
            **get_ifft_c2r_options(dtype, shape[axes[-1]]),
        }
        ifft = nvmath.fft.ifft(fft, execution=exec_backend.nvname, options=options, axes=axes)
        check_layouts(
            exec_backend,
            mem_backend,
            axes,
            result_layout,
            OptFftType.c2c,
            is_dense=True,
            inplace=False,
        )
        assert_array_type(ifft, framework, mem_backend, dtype)
        volume = math.prod(shape[axis] for axis in axes)
        assert_norm_close(
            ifft,
            get_scaled(signal, volume),
            shape_kind=shape_kind,
            exec_backend=exec_backend,
        )


@pytest.mark.parametrize(
    (
        "framework",
        "exec_backend",
        "mem_backend",
        "array_dim",
        "first_axis",
        "dtype",
        "shape_kind",
        "result_layout",
    ),
    [
        (
            framework,
            exec_backend,
            mem_backend,
            array_dim,
            first_axis,
            dtype,
            rng.choice(type_shape_support[exec_backend][dtype]),
            rng.choice(list(OptFftLayout)),
        )
        for framework in Framework.enabled()
        for exec_backend in supported_backends.exec
        for mem_backend in supported_backends.framework_mem[framework]
        for array_dim in [3, 4]
        for first_axis in range(array_dim - 3 + 1)
        for dtype in framework_exec_type_support[framework][exec_backend]
        # TODO(ktokarski) Prepare smaller cases for 3D halfs instead
        if not is_half(dtype)  # the values overflow with this many elements
    ],
)
def test_fft_ifft_3d(
    fx_last_operand_layout,  # noqa: F811
    framework,
    exec_backend,
    mem_backend,
    array_dim,
    first_axis,
    dtype,
    shape_kind,
    result_layout,
):
    shapes = {
        ShapeKind.pow2: (128, 64, 32, 16),
        ShapeKind.pow2357: (6, 441, 210, 30),
        ShapeKind.prime: (17, 127, 47, 13),
        ShapeKind.random: (22, 178, 361, 26),
    }
    shape = shapes[shape_kind][:array_dim]
    axes = list(range(first_axis, first_axis + 3))
    check_layouts, *_ = fx_last_operand_layout

    if should_skip_3d_unsupported(exec_backend, shape, axes):
        pytest.skip("Pre 11.4.2 CTK does not support 3D batched FFT")

    signal = get_random_input_data(framework, shape, dtype, mem_backend, seed=55)

    fft_fn = nvmath.fft.fft if is_complex(dtype) else nvmath.fft.rfft
    fft = fft_fn(
        signal,
        axes=axes,
        execution=exec_backend.nvname,
        options={"result_layout": result_layout.value},
    )
    check_layouts(
        exec_backend,
        mem_backend,
        axes,
        result_layout,
        OptFftType.c2c if is_complex(dtype) else OptFftType.r2c,
        is_dense=True,
        inplace=False,
    )
    assert_array_type(fft, framework, mem_backend, get_fft_dtype(dtype))
    assert_norm_close(
        fft,
        get_fft_ref(signal, axes=axes),
        axes=axes,
        shape_kind=shape_kind,
        exec_backend=exec_backend,
    )

    if is_complex(dtype):
        options = {
            "result_layout": result_layout.value,
            **get_ifft_c2r_options(dtype, shape[axes[-1]]),
        }
        ifft = nvmath.fft.ifft(fft, execution=exec_backend.nvname, options=options, axes=axes)
        check_layouts(
            exec_backend,
            mem_backend,
            axes,
            result_layout,
            OptFftType.c2c,
            is_dense=True,
            inplace=False,
        )
        assert_array_type(ifft, framework, mem_backend, dtype)
        volume = math.prod(shape[axis] for axis in axes)
        assert_norm_close(
            ifft,
            get_scaled(signal, volume),
            shape_kind=shape_kind,
            exec_backend=exec_backend,
        )


@pytest.mark.parametrize(
    (
        "framework",
        "exec_backend",
        "mem_backend",
        "shape",
        "axes",
        "dtype",
        "result_layout",
    ),
    [
        (
            framework,
            exec_backend,
            mem_backend,
            repr(shape),
            repr(axes),
            dtype,
            result_layout,
        )
        for framework in Framework.enabled()
        for exec_backend in supported_backends.exec
        for mem_backend in supported_backends.framework_mem[framework]
        for shape, axes in [
            ((1024,), (0,)),
            ((128, 3), (0,)),
            ((7, 128), (1,)),
            ((64, 32), (0, 1)),
            ((64, 32, 7), (0, 1)),
            ((127, 17, 31), (1, 2)),
            ((16, 8, 32), (0, 1, 2)),
            ((16, 8, 32, 3), (0, 1, 2)),
            ((3, 16, 8, 32), (1, 2, 3)),
        ]
        for dtype in framework_exec_type_support[framework][exec_backend]
        if not is_complex(dtype) and not is_half(dtype)
        for result_layout in OptFftLayout
    ],
)
def test_irfft_preserves_input(
    fx_last_operand_layout,  # noqa: F811
    framework,
    exec_backend,
    mem_backend,
    shape,
    axes,
    dtype,
    result_layout,
):
    shape = literal_eval(shape)
    axes = literal_eval(axes)
    check_layouts, *_ = fx_last_operand_layout

    if should_skip_3d_unsupported(exec_backend, shape, axes):
        pytest.skip("Pre 11.4.2 CTK does not support 3D batched FFT")

    signal = get_random_input_data(framework, shape, dtype, mem_backend, seed=105)

    exec_options = {"name": exec_backend.nvname}
    if exec_backend == ExecBackend.fftw:
        exec_options["num_threads"] = 16

    fft = nvmath.fft.rfft(
        signal,
        axes=axes,
        execution=exec_options,
        options={"result_layout": result_layout.value},
    )
    check_layouts(
        exec_backend,
        mem_backend,
        axes,
        result_layout,
        OptFftType.r2c,
        is_dense=True,
        inplace=False,
    )

    assert_array_type(fft, framework, mem_backend, get_fft_dtype(dtype))
    assert_norm_close(
        fft,
        get_fft_ref(signal, axes=tuple(sorted(axes))),
        axes=axes,
        exec_backend=exec_backend,
    )
    fft_copy = copy_array(fft)
    ifft = nvmath.fft.irfft(
        fft,
        axes=axes,
        execution=exec_options,
        options={
            "last_axis_parity": "odd" if shape[axes[-1]] % 2 else "even",
            "result_layout": result_layout.value,
        },
    )
    check_layouts(
        exec_backend,
        mem_backend,
        axes,
        result_layout,
        OptFftType.c2r,
        is_dense=True,
        inplace=False,
    )

    assert_array_type(ifft, framework, mem_backend, dtype)
    volume = math.prod(shape[a] for a in axes)
    assert_norm_close(
        ifft,
        get_scaled(signal, volume),
        axes=axes,
        exec_backend=exec_backend,
    )
    assert_array_equal(fft, fft_copy)


@pytest.mark.parametrize(
    (
        "framework",
        "exec_backend",
        "mem_backend",
        "fft_dim",
        "batched",
        "fft_type",
        "dtype",
        "result_layout",
        "shape_kind",
    ),
    [
        (
            framework,
            exec_backend,
            mem_backend,
            fft_dim,
            batched,
            fft_type,
            dtype,
            result_layout,
            rng.choice(type_shape_support[exec_backend][dtype]),
        )
        for framework in Framework.enabled()
        for exec_backend in supported_backends.exec
        for mem_backend in supported_backends.framework_mem[framework]
        for fft_dim in [1, 2, 3]
        for batched in ["no", "left", "right"]
        for fft_type in inplace_opt_ftt_type_support[True]
        for dtype in opt_fft_type_input_type_support[fft_type]
        if dtype in framework_exec_type_support[framework][exec_backend]
        for result_layout in OptFftLayout
    ],
)
def test_inplace(
    framework,
    exec_backend,
    mem_backend,
    fft_dim,
    batched,
    fft_type,
    dtype,
    result_layout,
    shape_kind,
):
    extent = {
        ShapeKind.pow2: [1024, 256, 64],
        ShapeKind.pow2357: [720, 210, 48],
        ShapeKind.prime: [997, 151, 43],
        ShapeKind.random: [361, 178, 22],
    }
    shape = (extent[shape_kind][fft_dim - 1],) * fft_dim

    if batched == "left":
        shape = (3,) + shape
        axes = tuple(range(1, fft_dim + 1))
    elif batched == "right":
        shape = shape + (7,)
        axes = tuple(-i for i in range(2, fft_dim + 2))
    else:
        assert batched == "no"
        axes = None

    if should_skip_3d_unsupported(exec_backend, shape, axes):
        pytest.skip("Pre 11.4.2 CTK does not support 3D batched FFT")

    signal = get_random_input_data(framework, shape, dtype, mem_backend, seed=105)
    signal_copy = copy_array(signal)

    nvmath.fft.fft(
        signal,
        axes=axes,
        execution=exec_backend.nvname,
        options={
            "inplace": True,
            "result_layout": result_layout.value,
            "fft_type": fft_type.value,
        },
    )
    assert_array_type(signal, framework, mem_backend, dtype)
    if result_layout == OptFftLayout.natural:
        fft_strides = get_array_strides(signal)
        assert is_decreasing(fft_strides), f"{fft_strides}"
    assert_norm_close(
        signal,
        get_fft_ref(signal_copy, axes=axes),
        axes=axes,
        shape_kind=shape_kind,
        exec_backend=exec_backend,
    )

    if fft_dim == 1 or not is_half(dtype):  # the half types overflow for bigger sizes
        nvmath.fft.ifft(
            signal,
            axes=axes,
            execution=exec_backend.nvname,
            options={
                "inplace": True,
                "result_layout": result_layout.value,
                "fft_type": fft_type.value,
            },
        )

        assert_array_type(signal, framework, mem_backend, dtype)

        if result_layout == OptFftLayout.natural:
            ifft_strides = get_array_strides(signal)
            assert is_decreasing(ifft_strides), f"{ifft_strides}"

        if axes is not None:
            volume = math.prod(shape[axis] for axis in axes)
        else:
            volume = math.prod(shape)
        assert_norm_close(
            signal,
            get_scaled(signal_copy, volume),
            shape_kind=shape_kind,
            exec_backend=exec_backend,
        )


@pytest.mark.parametrize(
    (
        "framework",
        "exec_backend",
        "mem_backend",
        "axes",
        "batched",
        "dtype",
        "fft_shape",
    ),
    [
        (
            framework,
            exec_backend,
            mem_backend,
            repr(axes),
            batched,
            dtype := rng.choice([dt for dt in framework_exec_type_support[framework][exec_backend] if is_complex(dt)]),
            repr(
                rng.sample(
                    ([17, 31, 101] if ShapeKind.prime in type_shape_support[exec_backend][dtype] else [16, 32, 64]),
                    k=3,
                )[: len(axes)]
            ),
        )
        for framework in Framework.enabled()
        for exec_backend in supported_backends.exec
        for mem_backend in supported_backends.framework_mem[framework]
        for axes in [[0, 1], [1, 0], [0, 2, 1], [2, 1, 0], [1, 2, 0]]
        for batched in ["no", "left", "right"]
    ],
)
def test_permuted_axes_c2c(framework, exec_backend, mem_backend, axes, batched, dtype, fft_shape):
    axes = literal_eval(axes)
    fft_shape = tuple(literal_eval(fft_shape))
    if batched == "left":
        shape = (13,) + fft_shape
        axes = [axis + 1 for axis in axes]
    elif batched == "right":
        shape = fft_shape + (11,)
    else:
        assert batched == "no"
        shape = fft_shape

    if should_skip_3d_unsupported(exec_backend, shape, axes):
        pytest.skip("Pre 11.4.2 CTK does not support 3D batched FFT")

    signal = get_random_input_data(framework, shape, dtype, mem_backend, seed=105)
    fft = nvmath.fft.fft(signal, execution=exec_backend.nvname, axes=axes)
    assert_array_type(fft, framework, mem_backend, get_fft_dtype(dtype))
    assert_norm_close(
        fft,
        get_fft_ref(signal, axes=tuple(sorted(axes))),
        axes=axes,
        exec_backend=exec_backend,
    )
    ifft = nvmath.fft.ifft(fft, execution=exec_backend.nvname, axes=axes)
    assert_array_type(ifft, framework, mem_backend, dtype)
    volume = math.prod(fft_shape)
    assert_norm_close(
        ifft,
        get_scaled(signal, volume),
        axes=axes,
        exec_backend=exec_backend,
    )


@pytest.mark.parametrize(
    (
        "framework",
        "exec_backend",
        "mem_backend",
        "axes",
        "shape",
        "dtype",
        "result_layout",
    ),
    [
        (
            framework,
            exec_backend,
            mem_backend,
            repr(axes),
            repr(shape),
            rng.choice([dt for dt in framework_exec_type_support[framework][exec_backend] if is_complex(dt)]),
            result_layout,
        )
        for framework in Framework.enabled()
        for exec_backend in supported_backends.exec
        for mem_backend in supported_backends.framework_mem[framework]
        for shape, axes in [
            ((64, 1), (0, 1)),
            ((32, 1, 4, 1), (0, 1)),
            ((32, 1, 4, 1), (1, 0)),
            ((4, 1, 32, 1), (2, 3)),
            ((4, 1, 32, 1), (3, 2)),
            ((32, 8, 1), (0, 1, 2)),
            ((32, 8, 1), (0, 2, 1)),
            ((8, 1, 32), (0, 1, 2)),
            ((8, 1, 32), (2, 1, 0)),
            ((8, 1, 32, 4, 1), (0, 1, 2)),
            ((4, 1, 8, 1, 32), (4, 3, 2)),
        ]
        for result_layout in OptFftLayout
    ],
)
def test_permuted_axes_c2c_repeated_strides(framework, exec_backend, mem_backend, axes, shape, dtype, result_layout):
    axes = literal_eval(axes)
    shape = literal_eval(shape)

    if should_skip_3d_unsupported(exec_backend, shape, axes):
        pytest.skip("Pre 11.4.2 CTK does not support 3D batched FFT")

    signal = get_random_input_data(framework, shape, dtype, mem_backend, seed=105)
    fft = nvmath.fft.fft(
        signal,
        axes=axes,
        execution=exec_backend.nvname,
        options={"result_layout": result_layout.value},
    )
    assert_array_type(fft, framework, mem_backend, get_fft_dtype(dtype))
    assert_norm_close(
        fft,
        get_fft_ref(signal, axes=tuple(sorted(axes))),
        axes=axes,
        exec_backend=exec_backend,
    )
    ifft = nvmath.fft.ifft(
        fft,
        axes=axes,
        execution=exec_backend.nvname,
        options={"result_layout": result_layout.value},
    )
    assert_array_type(ifft, framework, mem_backend, dtype)
    volume = math.prod(shape[a] for a in axes)
    assert_norm_close(
        ifft,
        get_scaled(signal, volume),
        axes=axes,
        exec_backend=exec_backend,
    )


@pytest.mark.parametrize(
    (
        "framework",
        "exec_backend",
        "mem_backend",
        "axes",
        "shape",
        "dtype",
        "result_layout",
    ),
    [
        (
            framework,
            exec_backend,
            mem_backend,
            repr(axes),
            repr(shape),
            rng.choice([dt for dt in framework_exec_type_support[framework][exec_backend] if is_complex(dt)]),
            result_layout,
        )
        for framework in Framework.enabled()
        for exec_backend in supported_backends.exec
        for mem_backend in supported_backends.framework_mem[framework]
        for shape, axes in [
            ((64, 1), (0, 1)),
            ((32, 1, 4, 1), (0, 1)),
            ((32, 1, 4, 1), (1, 0)),
            ((4, 1, 32, 1), (2, 3)),
            ((4, 1, 32, 1), (3, 2)),
            ((32, 8, 1), (0, 1, 2)),
            ((32, 8, 1), (0, 2, 1)),
            ((8, 1, 32), (0, 1, 2)),
            ((8, 1, 32), (2, 1, 0)),
            ((8, 1, 32, 4, 1), (0, 1, 2)),
            ((4, 1, 8, 1, 32), (4, 3, 2)),
        ]
        for result_layout in OptFftLayout
    ],
)
def test_permuted_axes_c2c_repeated_strides_inplace(framework, exec_backend, mem_backend, axes, shape, dtype, result_layout):
    axes = literal_eval(axes)
    shape = literal_eval(shape)

    if should_skip_3d_unsupported(exec_backend, shape, axes):
        pytest.skip("Pre 11.4.2 CTK does not support 3D batched FFT")

    data = get_random_input_data(framework, shape, dtype, mem_backend, seed=105)
    signal_copy = copy_array(data)
    nvmath.fft.fft(
        data,
        axes=axes,
        execution=exec_backend.nvname,
        options={"inplace": True, "result_layout": result_layout.value},
    )
    assert_array_type(data, framework, mem_backend, dtype)
    assert_norm_close(
        data,
        get_fft_ref(signal_copy, axes=tuple(sorted(axes))),
        axes=axes,
        exec_backend=exec_backend,
    )
    nvmath.fft.ifft(
        data,
        axes=axes,
        execution=exec_backend.nvname,
        options={"inplace": True, "result_layout": result_layout.value},
    )
    assert_array_type(data, framework, mem_backend, dtype)
    volume = math.prod(shape[a] for a in axes)
    assert_norm_close(
        data,
        get_scaled(signal_copy, volume),
        axes=axes,
        exec_backend=exec_backend,
    )


@pytest.mark.parametrize(
    (
        "framework",
        "exec_backend",
        "mem_backend",
        "axes",
        "batched",
        "dtype",
        "fft_shape",
        "result_layout",
    ),
    [
        (
            framework,
            exec_backend,
            mem_backend,
            repr(axes),
            batched,
            dtype := rng.choice([dt for dt in framework_exec_type_support[framework][exec_backend] if not is_complex(dt)]),
            repr(
                rng.sample(
                    ([17, 31, 101] if ShapeKind.prime in type_shape_support[exec_backend][dtype] else [16, 32, 64]),
                    k=3,
                )
            ),
            result_layout,
        )
        for framework in Framework.enabled()
        for exec_backend in supported_backends.exec
        for mem_backend in supported_backends.framework_mem[framework]
        for axes in [[1, 0, 2]]
        for batched in ["no", "left", "right"]
        for result_layout in OptFftLayout
    ],
)
def test_permuted_axes_r2c_c2r(framework, exec_backend, mem_backend, axes, batched, dtype, fft_shape, result_layout):
    axes = literal_eval(axes)
    fft_shape = tuple(literal_eval(fft_shape))
    if batched == "left":
        shape = (7,) + fft_shape
        axes = [axis + 1 for axis in axes]
    elif batched == "right":
        shape = fft_shape + (9,)
    else:
        assert batched == "no"
        shape = fft_shape

    if should_skip_3d_unsupported(exec_backend, shape, axes):
        pytest.skip("Pre 11.4.2 CTK does not support 3D batched FFT")

    signal = get_random_input_data(framework, shape, dtype, mem_backend, seed=105)

    if is_half(dtype) and batched == "right":
        with pytest.raises(ValueError, match="is currently not supported for strided inputs"):
            fft = nvmath.fft.rfft(
                signal,
                axes=axes,
                execution=exec_backend.nvname,
                options={"result_layout": result_layout.value},
            )
        return

    fft = nvmath.fft.rfft(
        signal,
        axes=axes,
        execution=exec_backend.nvname,
        options={"result_layout": result_layout.value},
    )
    assert_array_type(fft, framework, mem_backend, get_fft_dtype(dtype))
    assert_norm_close(
        fft,
        get_fft_ref(signal, axes=tuple(sorted(axes))),
        axes=axes,
        exec_backend=exec_backend,
    )
    ifft = nvmath.fft.irfft(
        fft,
        axes=axes,
        execution=exec_backend.nvname,
        options={
            "last_axis_parity": "odd" if shape[axes[-1]] % 2 else "even",
            "result_layout": result_layout.value,
        },
    )
    assert_array_type(ifft, framework, mem_backend, dtype)
    volume = math.prod(fft_shape)
    assert_norm_close(
        ifft,
        get_scaled(signal, volume),
        axes=axes,
        exec_backend=exec_backend,
    )


@pytest.mark.parametrize(
    (
        "framework",
        "exec_backend",
        "mem_backend",
        "axes",
        "shape",
        "dtype",
        "result_layout",
    ),
    [
        (
            framework,
            exec_backend,
            mem_backend,
            repr(axes),
            repr(shape),
            rng.choice([dt for dt in framework_exec_type_support[framework][exec_backend] if not is_complex(dt)]),
            result_layout,
        )
        for framework in Framework.enabled()
        for exec_backend in supported_backends.exec
        for mem_backend in supported_backends.framework_mem[framework]
        for shape, axes in [
            ((64, 1), (0, 1)),
            ((4, 1, 32, 1, 8), (3, 2, 4)),
            ((32, 1, 8, 4, 1), (1, 0, 2)),
        ]
        for result_layout in OptFftLayout
    ],
)
def test_permuted_axes_r2c_c2r_repeated_strides(framework, exec_backend, mem_backend, axes, shape, dtype, result_layout):
    axes = literal_eval(axes)
    shape = literal_eval(shape)

    if should_skip_3d_unsupported(exec_backend, shape, axes):
        pytest.skip("Pre 11.4.2 CTK does not support 3D batched FFT")

    signal = get_random_input_data(framework, shape, dtype, mem_backend, seed=105)

    if is_half(dtype) and max(axes) < len(shape) - 1:
        with pytest.raises(ValueError, match="is currently not supported for strided inputs"):
            nvmath.fft.rfft(
                signal,
                axes=axes,
                execution=exec_backend.nvname,
                options={"result_layout": result_layout.value},
            )
        return

    fft = nvmath.fft.rfft(
        signal,
        axes=axes,
        execution=exec_backend.nvname,
        options={"result_layout": result_layout.value},
    )
    assert_array_type(fft, framework, mem_backend, get_fft_dtype(dtype))
    assert_norm_close(
        fft,
        get_fft_ref(signal, axes=axes),
        axes=axes,
        exec_backend=exec_backend,
    )
    ifft = nvmath.fft.irfft(
        fft,
        axes=axes,
        execution=exec_backend.nvname,
        options={
            "last_axis_parity": "odd" if shape[axes[-1]] % 2 else "even",
            "result_layout": result_layout.value,
        },
    )
    assert_array_type(ifft, framework, mem_backend, dtype)
    volume = math.prod(shape[a] for a in axes)
    assert_norm_close(
        ifft,
        get_scaled(signal, volume),
        axes=axes,
        exec_backend=exec_backend,
    )


@pytest.mark.parametrize(
    (
        "framework",
        "exec_backend",
        "mem_backend",
        "axes",
        "shape",
        "dtype",
        "result_layout",
    ),
    [
        (
            framework,
            exec_backend,
            mem_backend,
            repr(axes),
            repr(shape),
            rng.choice([dt for dt in framework_exec_type_support[framework][exec_backend] if not is_complex(dt)]),
            result_layout,
        )
        for framework in Framework.enabled()
        for exec_backend in supported_backends.exec
        for mem_backend in supported_backends.framework_mem[framework]
        for shape, axes in [
            ((64, 1), (1, 0)),
            ((32, 1, 1), (1, 0)),
            ((4, 1, 32, 1), (3, 2)),
            ((4, 1, 16, 32, 1), (3, 4, 2)),
            ((8, 8, 1, 8, 1), (2, 0, 1)),
        ]
        for result_layout in OptFftLayout
    ],
)
def test_permuted_axes_r2c_c2r_repeated_strides_fallback(
    framework, exec_backend, mem_backend, axes, shape, dtype, result_layout
):
    axes = literal_eval(axes)
    shape = literal_eval(shape)

    if should_skip_3d_unsupported(exec_backend, shape, axes):
        pytest.skip("Pre 11.4.2 CTK does not support 3D batched FFT")

    signal = get_random_input_data(framework, shape, dtype, mem_backend, seed=105)

    fft = check_layout_fallback(
        signal,
        axes,
        lambda signal, axes: nvmath.fft.rfft(
            signal,
            execution=exec_backend.nvname,
            axes=axes,
            options={"result_layout": result_layout.value},
        ),
    )
    assert_array_type(fft, framework, mem_backend, get_fft_dtype(dtype))
    assert_norm_close(
        fft,
        get_fft_ref(signal, axes=axes),
        axes=axes,
        exec_backend=exec_backend,
    )
    last_axis_parity = "odd" if shape[axes[-1]] % 2 else "even"
    ifft = check_layout_fallback(
        fft,
        axes,
        lambda fft, axes: nvmath.fft.irfft(
            fft,
            execution=exec_backend.nvname,
            axes=axes,
            options={
                "last_axis_parity": last_axis_parity,
                "result_layout": result_layout.value,
            },
        ),
    )
    assert_array_type(ifft, framework, mem_backend, dtype)
    volume = math.prod(shape[a] for a in axes)
    assert_norm_close(
        ifft,
        get_scaled(signal, volume),
        axes=axes,
        exec_backend=exec_backend,
    )


@pytest.mark.parametrize(
    (
        "result_layout",
        "framework",
        "exec_backend",
        "mem_backend",
        "fft_shape",
        "batch_shape",
        "batched",
        "dtype",
    ),
    [
        (
            result_layout,
            framework,
            exec_backend,
            mem_backend,
            repr(fft_shape),
            repr(batch_shape),
            batched,
            dtype,
        )
        for result_layout in OptFftLayout
        for framework in Framework.enabled()
        for exec_backend in supported_backends.exec
        for mem_backend in supported_backends.framework_mem[framework]
        for fft_shape in [
            (128,),
            (32, 16),
            (32, 1),
            (1, 32),
            (1, 32, 32),
            (32, 1, 32),
            (32, 32, 1),
        ]
        for batch_shape in [(1,), (1, 1), (2, 1), (1, 64)]
        for batched in ["left", "right"]
        for dtype in framework_exec_type_support[framework][exec_backend]
    ],
)
def test_fft_repeated_strides(
    result_layout,
    framework,
    exec_backend,
    mem_backend,
    fft_shape,
    batch_shape,
    batched,
    dtype,
):
    fft_shape = literal_eval(fft_shape)
    batch_shape = literal_eval(batch_shape)
    if batched == "left":
        shape = batch_shape + fft_shape
        axes = tuple(range(offset := len(batch_shape), offset + len(fft_shape)))
    else:
        assert batched == "right"
        shape = fft_shape + batch_shape
        axes = tuple(range(len(fft_shape)))

    if should_skip_3d_unsupported(exec_backend, shape, axes):
        pytest.skip("Pre 11.4.2 CTK does not support 3D batched FFT")

    signal = get_random_input_data(framework, shape, dtype, mem_backend, seed=105)
    fft_fn = nvmath.fft.fft if is_complex(dtype) else nvmath.fft.rfft

    if dtype == DType.float16 and batched == "right" and any(d != 1 for d in batch_shape):
        with pytest.raises(ValueError, match="is currently not supported for strided inputs"):
            fft = fft_fn(
                signal,
                axes=axes,
                execution=exec_backend.nvname,
                options={"result_layout": result_layout.value},
            )
    else:
        try:
            fft = fft_fn(
                signal,
                axes=axes,
                execution=exec_backend.nvname,
                options={"result_layout": result_layout.value},
            )
            assert_array_type(fft, framework, mem_backend, get_fft_dtype(dtype))
            assert_norm_close(
                fft,
                get_fft_ref(signal, axes=axes),
                axes=axes,
                exec_backend=exec_backend,
            )
        except nvmath.bindings.cufft.cuFFTError as e:
            assert exec_backend == ExecBackend.cufft, f"{exec_backend}"
            assert "CUFFT_SETUP_FAILED" in str(e)
            assert dtype == DType.float16 and fft_shape[-1] == 1
            assert get_cufft_version() < 10702  # 10702 is shipped with CTK 11.7


@pytest.mark.parametrize(
    (
        "result_layout",
        "framework",
        "exec_backend",
        "mem_backend",
        "fft_shape",
        "batch_shape",
        "batched",
        "dtype",
    ),
    [
        (
            result_layout,
            framework,
            exec_backend,
            mem_backend,
            repr(fft_shape),
            repr(batch_shape),
            batched,
            dtype,
        )
        for result_layout in OptFftLayout
        for framework in Framework.enabled()
        for exec_backend in supported_backends.exec
        for mem_backend in supported_backends.framework_mem[framework]
        for fft_shape in [
            (128,),
            (32, 16),
            (32, 1),
            (1, 32),
            (1, 32, 32),
            (32, 1, 32),
            (32, 32, 1),
        ]
        for batch_shape in [(1,), (1, 1), (2, 1), (1, 64)]
        for batched in ["left", "right"]
        for dtype in framework_exec_type_support[framework][exec_backend]
        # r2c halfs do not support strided inputs
        if dtype != DType.float16 or batched == "left" or all(d == 1 for d in batch_shape)
    ],
)
def test_ifft_repeated_strides(
    result_layout,
    framework,
    exec_backend,
    mem_backend,
    fft_shape,
    batch_shape,
    batched,
    dtype,
):
    fft_shape = literal_eval(fft_shape)
    batch_shape = literal_eval(batch_shape)
    if batched == "left":
        shape = batch_shape + fft_shape
        axes = tuple(range(offset := len(batch_shape), offset + len(fft_shape)))
    else:
        assert batched == "right"
        shape = fft_shape + batch_shape
        axes = tuple(range(len(fft_shape)))

    if should_skip_3d_unsupported(exec_backend, shape, axes):
        pytest.skip("Pre 11.4.2 CTK does not support 3D batched FFT")

    try:
        signal = get_random_input_data(framework, shape, dtype, mem_backend, seed=105)
        fft_fn = nvmath.fft.fft if is_complex(dtype) else nvmath.fft.rfft
        fft = fft_fn(
            signal,
            axes=axes,
            execution=exec_backend.nvname,
            options={"result_layout": "natural"},
        )
        assert_array_type(fft, framework, mem_backend, get_fft_dtype(dtype))
        assert_norm_close(
            fft,
            get_fft_ref(signal, axes=axes),
            axes=axes,
            exec_backend=exec_backend,
        )
        assert is_decreasing(get_array_strides(fft))

        ifft_fn = nvmath.fft.ifft if is_complex(dtype) else nvmath.fft.irfft
        ifft = ifft_fn(
            fft,
            axes=axes,
            execution=exec_backend.nvname,
            options={
                "last_axis_parity": "even" if shape[axes[-1]] % 2 == 0 else "odd",
                "result_layout": result_layout.value,
            },
        )
        assert_array_type(ifft, framework, mem_backend, dtype)
        volume = math.prod(fft_shape)
        assert_norm_close(
            ifft,
            get_scaled(signal, volume),
            axes=axes,
            exec_backend=exec_backend,
        )
    except nvmath.bindings.cufft.cuFFTError as e:
        assert exec_backend == ExecBackend.cufft, f"{exec_backend}"
        assert "CUFFT_SETUP_FAILED" in str(e)
        assert dtype == DType.float16 and fft_shape[-1] == 1
        assert get_cufft_version() < 10702  # 10702 is shipped with CTK 11.7


@pytest.mark.parametrize(
    (
        "result_layout",
        "framework",
        "exec_backend",
        "mem_backend",
        "shape",
        "axes",
    ),
    [
        (
            result_layout,
            framework,
            exec_backend,
            mem_backend,
            repr(shape),
            repr(axes),
        )
        for result_layout in OptFftLayout
        for framework in Framework.enabled()
        for exec_backend in supported_backends.exec
        if DType.complex32 in framework_exec_type_support[framework][exec_backend]
        for mem_backend in supported_backends.framework_mem[framework]
        for shape, axes in [
            ((513, 4), (0,)),
            ((128, 33, 2), (0, 1)),
        ]
    ],
)
def test_irfft_half_strided_output(result_layout, framework, exec_backend, mem_backend, shape, axes):
    shape = literal_eval(shape)
    axes = literal_eval(axes)

    if should_skip_3d_unsupported(exec_backend, shape, axes):
        pytest.skip("Pre 11.4.2 CTK does not support 3D batched FFT")

    real_shape = list(shape)
    real_shape[axes[-1]] = (shape[axes[-1]] - 1) * 2
    signal = get_random_input_data(framework, real_shape, DType.float16, mem_backend, seed=105)
    # normalize the fft so we don't overflow in half-precision case
    fft = copy_array(get_fft_ref(signal, axes=axes, norm="forward"))
    assert_array_type(fft, framework, mem_backend, DType.complex32)
    assert_eq(fft.shape, shape)
    if result_layout == OptFftLayout.natural:
        with pytest.raises(ValueError, match="is currently not supported for strided outputs"):
            nvmath.fft.irfft(
                fft,
                axes=axes,
                execution=exec_backend.nvname,
                options={"result_layout": result_layout.value},
            )
    else:
        ifft = nvmath.fft.irfft(
            fft,
            axes=axes,
            execution=exec_backend.nvname,
            options={"result_layout": result_layout.value},
        )
        assert_array_type(ifft, framework, mem_backend, DType.float16)
        fft2 = nvmath.fft.rfft(ifft, axes=axes, execution=exec_backend.nvname)
        assert_array_type(fft2, framework, mem_backend, DType.complex32)
        volume = math.prod(ifft.shape[a] for a in axes)
        assert_norm_close(
            fft2,
            get_scaled(fft, volume),
            axes=axes,
            exec_backend=exec_backend,
        )


@pytest.mark.parametrize(
    (
        "framework",
        "exec_backend",
        "mem_backend",
        "fft_type",
        "direction",
        "dtype",
        "base_shape",
        "view_shape_kind",
        "view_shape",
        "slices",
        "axes",
        "fft_dim",
        "inplace",
        "layout",
    ),
    [
        (
            framework,
            exec_backend,
            mem_backend,
            fft_type,
            direction,
            dtype,
            repr(base_shape),
            view_shape_kind,
            repr(view_shape),
            repr(slices),
            repr(axes),
            f"FftDim.{len(axes)}",
            inplace,
            layout,
        )
        for framework in Framework.enabled()
        for exec_backend in supported_backends.exec
        for mem_backend in supported_backends.framework_mem[framework]
        for fft_type in OptFftType
        for direction in opt_fft_type_direction_support[fft_type]
        for dtype in opt_fft_type_input_type_support[fft_type]
        if dtype in framework_exec_type_support[framework][exec_backend]
        for base_shape, view_shape_kind, view_shape, slices, axes in [
            ((256,), ShapeKind.pow2, (16,), ((16,),), (0,)),
            ((39, 2), ShapeKind.pow2, (16, 2), ((0, 16), None), (0,)),
            ((39, 2), ShapeKind.pow2, (16, 2), ((16, 32), None), (0,)),
            ((39, 7), ShapeKind.pow2, (16, 5), ((0, 16), (0, 5)), (0,)),
            ((39, 2), ShapeKind.pow2, (8, 2), ((16, 32, 2), None), (0,)),
            ((2, 39), ShapeKind.pow2, (2, 8), (None, (16, 32, 2)), (1,)),
            ((5, 2048), ShapeKind.pow2, (5, 16), (None, (1, 17)), (1,)),
            ((5, 2048), ShapeKind.pow2, (5, 16), (None, (2, 18)), (1,)),
            ((5, 2048), ShapeKind.random, (5, 177), (None, (2, 179)), (1,)),
            ((5, 2048), ShapeKind.pow2, (1, 16), ((None, None, 5), (1, 17)), (1,)),
            ((5, 2048), ShapeKind.pow2, (1, 16), ((0, 1), (2, 18)), (1,)),
            ((2048, 5), ShapeKind.pow2, (16, 1), ((1, 17), (0, 1)), (0,)),
            ((2048, 5), ShapeKind.pow2, (16, 1), ((2, 18), (None, None, 5)), (0,)),
            # using big base shape to catch errors around insufficient allocation
            ((128, 1024), ShapeKind.pow2, (8, 16), ((-10, -2), (-32, -16)), (0, 1)),
            ((1024, 4096, 7), ShapeKind.pow2, (256, 16, 7), ((2, 258), (6, 22), None), (0, 1)),
            ((7, 1024, 4096), ShapeKind.pow2, (7, 32, 16), (None, (2, 34), (6, 22)), (1, 2)),
            ((3, 17, 13), ShapeKind.pow2, (2, 8, 8), ((1, 3), (8,), (1, 9)), (1, 2)),
            ((55, 55, 55), ShapeKind.random, (55, 19, 28), (None, (None, None, 3), (None, None, 2)), (0, 1, 2)),
            ((13, 17, 13), ShapeKind.pow2, (8, 8, 8), ((1, 9), (8,), (1, 9)), (0, 1, 2)),
            ((13, 17, 13, 11), ShapeKind.pow2, (8, 8, 8, 11), ((1, 9), (8,), (1, 9), None), (0, 1, 2)),
            ((11, 13, 17, 13), ShapeKind.pow2, (11, 8, 8, 8), (None, (1, 9), (8,), (1, 9)), (1, 2, 3)),
        ]
        for inplace in [False, True]
        if not inplace or fft_type == OptFftType.c2c
        for layout in OptFftLayout
        if not inplace or layout == OptFftLayout.natural
    ],
)
def test_sliced_tensor(
    fx_last_operand_layout,  # noqa: F811
    framework,
    exec_backend,
    mem_backend,
    fft_type,
    direction,
    dtype,
    base_shape,
    view_shape_kind,
    view_shape,
    slices,
    axes,
    fft_dim,
    inplace,
    layout,
):
    base_shape = literal_eval(base_shape)
    view_shape = literal_eval(view_shape)
    slices = literal_eval(slices)
    axes = literal_eval(axes)
    assert len(slices) == len(base_shape) == len(view_shape)
    check_layouts, *_ = fx_last_operand_layout

    if fft_type == OptFftType.c2c:
        if direction == Direction.forward:
            fft_fn = nvmath.fft.fft
        else:
            assert direction == Direction.inverse
            fft_fn = nvmath.fft.ifft
        complex_dtype = dtype
    elif fft_type == OptFftType.r2c:
        fft_fn = nvmath.fft.rfft
        complex_dtype = get_fft_dtype(dtype)
    else:
        assert fft_type == OptFftType.c2r
        fft_fn = nvmath.fft.irfft
        complex_dtype = dtype

    assert is_complex(complex_dtype)

    signal_base = get_random_input_data(framework, base_shape, dtype, mem_backend, seed=105)
    slices = tuple(slice(*s) if s is not None else slice(s) for s in slices)
    signal = signal_base[slices]
    assert_array_type(signal, framework, mem_backend, dtype)
    assert signal.shape == view_shape
    last_axis_parity = "odd"

    if fft_type != OptFftType.c2r:
        # problem size as defined by cufft, i.e. shape of the input for r2c, c2c
        # and the shape of the output for c2r
        instance_shape = view_shape
    else:
        real_dtype = get_ifft_dtype(dtype, fft_type=fft_type)
        # assuming last_axis_parit == "odd"
        instance_shape = tuple(e if i != axes[-1] else 2 * (e - 1) + 1 for i, e in enumerate(view_shape))
        real_sample = get_random_input_data(framework, instance_shape, real_dtype, mem_backend, seed=106)
        complex_sample = get_fft_ref(real_sample, axes=axes)
        assert_array_type(complex_sample, framework, mem_backend, dtype)
        assert complex_sample.shape == view_shape
        signal[:] = complex_sample[:]
        assert_array_type(signal, framework, mem_backend, dtype)
        assert signal.shape == view_shape

    signal_copy = signal if not inplace else copy_array(signal)

    if should_skip_3d_unsupported(exec_backend, view_shape, axes):
        pytest.skip("Skipping 3D for older cufft")

    alignment_excpt_clss = (nvmath.bindings.cufft.cuFFTError,)
    if nvmath.bindings.nvpl is not None:
        alignment_excpt_clss += (nvmath.bindings.nvpl.fft.FFTWUnaligned,)

    try:
        try:
            out = fft_fn(
                signal,
                axes=axes,
                options={
                    "last_axis_parity": last_axis_parity,
                    "inplace": inplace,
                    "result_layout": layout.value,
                },
                execution=exec_backend.nvname,
            )
            check_layouts(
                exec_backend,
                mem_backend,
                axes,
                layout,
                fft_type,
                is_dense=False,
                inplace=inplace,
            )
        except nvmath.fft.UnsupportedLayoutError as e:
            # with slices, we don't really require permutation,
            # just the `step` may make the embedding not possible
            assert e.permutation == tuple(range(len(view_shape)))
            assert exec_backend == mem_backend.exec
            cont_signal = copy_array(signal)
            out = fft_fn(
                cont_signal,
                axes=axes,
                options={
                    "last_axis_parity": last_axis_parity,
                    "inplace": inplace,
                    "result_layout": layout.value,
                },
                execution=exec_backend.nvname,
            )
    except alignment_excpt_clss as e:
        str_e = str(e)
        if (
            exec_backend == ExecBackend.cufft
            and is_half(dtype)
            and ("CUFFT_NOT_SUPPORTED" in str_e or "CUFFT_SETUP_FAILED" in str_e)
        ):
            # only pow2 problem sizes are supported for halfs
            assert any(math.gcd(instance_shape[a], 2**30) != instance_shape[a] for a in axes)
        else:
            alignment = size_of(complex_dtype)
            if exec_backend == ExecBackend.cufft:
                assert "CUFFT_INVALID_VALUE" in str_e, str_e
            else:
                assert "input tensor's underlying memory" in str_e, str_e
                assert f"pointer must be aligned to at least {alignment} bytes" in str_e, str_e
            assert fft_type in (OptFftType.c2r, OptFftType.r2c), f"{fft_type}"
            assert get_raw_ptr(signal) % alignment != 0
            strides = get_array_element_strides(signal)
            start_offset = sum(stride * (extent_slice.start or 0) for stride, extent_slice in zip(strides, slices, strict=True))
            assert start_offset % 2 == 1, f"{strides} {slices} {start_offset}"
    except ValueError as e:
        str_e = str(e)
        if "The R2C FFT of half-precision tensor" in str_e:
            assert exec_backend == ExecBackend.cufft and is_half(dtype) and fft_type == OptFftType.r2c
        elif "The C2R FFT of half-precision tensor" in str_e:
            assert exec_backend == ExecBackend.cufft and is_half(dtype) and fft_type == OptFftType.c2r
            assert layout == OptFftLayout.natural
        else:
            raise
    else:
        if fft_type == OptFftType.c2r:
            assert out.shape == instance_shape

        if direction == Direction.forward:
            ref = get_fft_ref(signal_copy, axes=axes)
            out_dtype = get_fft_dtype(dtype)
        else:
            ref = get_ifft_ref(
                signal_copy,
                axes=axes,
                last_axis_parity="odd",
                is_c2c=fft_type == OptFftType.c2c,
            )
            out_dtype = get_ifft_dtype(dtype, fft_type)

        assert_array_type(out, framework, mem_backend, out_dtype)
        assert_norm_close(out, ref, axes=axes, exec_backend=exec_backend, shape_kind=view_shape_kind)


@pytest.mark.parametrize(
    (
        "framework",
        "exec_backend",
        "shape",
        "slice_descs",
        "dtype",
    ),
    [
        (framework, exec_backend, repr(shape), repr(slice_descs), dtype)
        for framework in Framework.enabled()
        for exec_backend in supported_backends.exec
        if exec_backend.mem in supported_backends.framework_mem[framework]
        for shape, slice_descs in [
            ((8,) * fft_dim, ((None, None),) * (fft_dim - 1) + ((start, end),))
            for fft_dim in [1, 2, 3]
            for start in [1, 2, 3, 4]
            for end in [None]
        ]
        for dtype in framework_exec_type_support[framework][exec_backend]
    ],
)
def test_sliced_tensor_unaligned(framework, exec_backend, shape, slice_descs, dtype):
    mem_backend = exec_backend.mem
    shape = literal_eval(shape)
    slice_descs = literal_eval(slice_descs)
    assert len(shape) == len(slice_descs)

    if should_skip_3d_unsupported(exec_backend, shape):
        pytest.skip("Pre 11.4.2 CTK does not support 3D batched FFT")

    base_shape = tuple(d + (start or 0) - (end or 0) for d, (start, end) in zip(shape, slice_descs, strict=True))
    signal = get_random_input_data(framework, base_shape, dtype, mem_backend, seed=105)
    signal_sliced = signal[tuple(slice(*slice_desc) for slice_desc in slice_descs)]
    assert signal_sliced.shape == shape
    assert signal_sliced.dtype == signal.dtype

    excpt_clss = (nvmath.bindings.cufft.cuFFTError,)
    if nvmath.bindings.nvpl is not None:
        excpt_clss += (nvmath.bindings.nvpl.fft.FFTWUnaligned,)
    try:
        fft_fn = nvmath.fft.fft if is_complex(dtype) else nvmath.fft.rfft
        fft = fft_fn(
            signal_sliced,
            execution=exec_backend.nvname,
        )
    except excpt_clss as e:
        assert not is_complex(dtype)

        str_e = str(e)
        if exec_backend == ExecBackend.cufft:
            assert "CUFFT_INVALID_VALUE" in str_e, str_e
        else:
            alig = 16 if dtype == DType.float64 else 8
            assert "input tensor's underlying memory" in str_e, str_e
            assert f"pointer must be aligned to at least {alig} bytes" in str_e, str_e

        inner_offset = slice_descs[-1][0]
        # The R2C float input must be aligned to a pair of elements
        # not just a single element
        assert inner_offset is not None and inner_offset % 2 == 1
    else:
        assert_array_type(fft, framework, mem_backend, get_fft_dtype(dtype))
        assert_norm_close(
            fft,
            get_fft_ref(signal_sliced),
            exec_backend=exec_backend,
        )


@pytest.mark.parametrize(
    (
        "framework",
        "exec_backend",
        "mem_backend",
        "batched",
        "dtype",
        "shape_kind",
    ),
    [
        (
            framework,
            exec_backend,
            mem_backend,
            batched,
            dtype,
            rng.choice(type_shape_support[exec_backend][dtype]),
        )
        for framework in Framework.enabled()
        for exec_backend in supported_backends.exec
        for mem_backend in supported_backends.framework_mem[framework]
        for batched in ["no", "left", "right"]
        for dtype in framework_exec_type_support[framework][exec_backend]
        if is_complex(dtype) and not is_half(dtype)
    ],
)
def test_permuted_tensor(framework, exec_backend, mem_backend, batched, dtype, shape_kind):
    extent = {
        ShapeKind.pow2: (128, 64),
        ShapeKind.pow2357: (48, 48),
        ShapeKind.prime: (43, 43),
        ShapeKind.random: (22, 22),
    }
    shape = extent[shape_kind]

    if batched == "left":
        shape = (8,) + shape
        transpose_axes = [2, 1]
        axes = [1, 2]
    elif batched == "right":
        shape = shape + (4,)
        transpose_axes = [1, 0]
        axes = [0, 1]
    else:
        assert batched == "no"
        transpose_axes = [0, 1]
        axes = None

    signal = get_random_input_data(framework, shape, dtype, mem_backend, seed=105)
    signal_transposed = get_transposed(signal, *transpose_axes)
    assert not is_decreasing(get_array_strides(signal_transposed))
    assert signal.dtype == signal_transposed.dtype

    fft = nvmath.fft.fft(
        signal_transposed,
        axes=axes,
        execution=exec_backend.nvname,
    )
    assert_array_type(fft, framework, mem_backend, get_fft_dtype(dtype))
    assert_norm_close(
        fft,
        get_fft_ref(signal_transposed, axes=axes),
        axes=axes,
        exec_backend=exec_backend,
    )


@pytest.mark.parametrize(
    (
        "framework",
        "exec_backend",
        "mem_backend",
        "fft_dim",
        "batched",
        "batch_size",
        "dtype",
        "result_layout",
    ),
    [
        (
            framework,
            exec_backend,
            mem_backend,
            fft_dim,
            batched,
            batch_size,
            dtype,
            result_layout,
        )
        for framework in Framework.enabled()
        for exec_backend in supported_backends.exec
        for mem_backend in supported_backends.framework_mem[framework]
        for fft_dim in [1, 2, 3]
        for batched, batch_size in [
            ("no", None),
        ]
        + [(side, 2**batch_size_log) for side in ("left", "right") for batch_size_log in (0, 1, 2, 3, 10)]
        for dtype in framework_exec_type_support[framework][exec_backend]
        for result_layout in OptFftLayout
    ],
)
def test_single_element(
    framework,
    exec_backend,
    mem_backend,
    fft_dim,
    batched,
    batch_size,
    dtype,
    result_layout,
):
    sample_shape = (1,) * fft_dim
    if batched == "left":
        shape = (batch_size,) + sample_shape
        axes = tuple(range(1, 1 + fft_dim))
    elif batched == "right":
        shape = sample_shape + (batch_size,)
        axes = tuple(range(fft_dim))
    else:
        assert batched == "no"
        shape = sample_shape
        axes = None

    if should_skip_3d_unsupported(exec_backend, shape, axes):
        pytest.skip("Pre 11.4.2 CTK does not support 3D batched FFT")

    signal = get_random_input_data(framework, shape, dtype, mem_backend, seed=105)
    fft_fn = nvmath.fft.fft if is_complex(dtype) else nvmath.fft.rfft

    if not is_complex(dtype) and is_half(dtype) and batched == "right" and batch_size > 1:
        with pytest.raises(ValueError, match="R2C FFT of half-precision tensor"):
            fft_fn(
                signal,
                axes=axes,
                execution=exec_backend.nvname,
                options={"result_layout": result_layout.value},
            )
        return

    if exec_backend == ExecBackend.cufft and is_half(dtype) and get_cufft_version() < 10702:
        with pytest.raises(ValueError, match="sample size 1 and half-precision type"):
            fft_fn(
                signal,
                axes=axes,
                execution=exec_backend.nvname,
                options={"result_layout": result_layout.value},
            )
        return

    fft = fft_fn(
        signal,
        axes=axes,
        execution=exec_backend.nvname,
        options={"result_layout": result_layout.value},
    )
    assert_array_type(fft, framework, mem_backend, get_fft_dtype(dtype))
    assert_norm_close(
        fft,
        get_fft_ref(signal, axes=axes),
        axes=axes,
        exec_backend=exec_backend,
    )

    ifft_fn = nvmath.fft.ifft if is_complex(dtype) else nvmath.fft.irfft
    ifft = ifft_fn(
        fft,
        axes=axes,
        execution=exec_backend.nvname,
        options={"last_axis_parity": "odd", "result_layout": result_layout.value},
    )
    assert_norm_close(
        ifft,
        signal,
        exec_backend=exec_backend,
    )


@pytest.mark.parametrize(
    (
        "framework",
        "exec_backend",
        "mem_backend",
        "sample_shape",
        "batched",
        "batch_size",
        "batch_step",
        "dtype",
        "result_layout",
    ),
    [
        (
            framework,
            exec_backend,
            mem_backend,
            repr(sample_shape),
            batched,
            batch_size,
            batch_step,
            dtype,
            result_layout,
        )
        for framework in Framework.enabled()
        for exec_backend in supported_backends.exec
        for mem_backend in supported_backends.framework_mem[framework]
        for sample_shape in [(64,), (16, 16), (4, 4, 4)]
        for batched, batch_size in [
            ("no", None),
        ]
        + [(side, extent) for side in ("left", "right") for extent in (4, 1024)]
        for batch_step in (None, 2)
        if batch_step is None or (batch_size is not None and batch_size > batch_step)
        for dtype in framework_exec_type_support[framework][exec_backend]
        for result_layout in OptFftLayout
    ],
)
def test_single_element_view(
    framework,
    exec_backend,
    mem_backend,
    sample_shape,
    batched,
    batch_size,
    batch_step,
    dtype,
    result_layout,
):
    sample_shape = literal_eval(sample_shape)
    if batched == "left":
        shape = (batch_size,) + sample_shape
        axes = tuple(range(1, 1 + len(sample_shape)))
    elif batched == "right":
        shape = sample_shape + (batch_size,)
        axes = tuple(range(len(sample_shape)))
    else:
        assert batched == "no"
        shape = sample_shape
        axes = None

    if should_skip_3d_unsupported(exec_backend, shape, axes):
        pytest.skip("Pre 11.4.2 CTK does not support 3D batched FFT")

    signal = get_random_input_data(framework, shape, dtype, mem_backend, seed=105)
    signal_view = signal[
        tuple((slice(None, None, batch_step) if axes and axis not in axes else slice(1)) for axis in range(len(shape)))
    ]
    actual_batch_size = (batch_size or 1) // (batch_step or 1)
    assert math.prod(signal_view.shape) == actual_batch_size
    fft_fn = nvmath.fft.fft if is_complex(dtype) else nvmath.fft.rfft

    if not is_complex(dtype) and is_half(dtype) and batched == "right" and actual_batch_size > 1:
        with pytest.raises(ValueError, match="R2C FFT of half-precision tensor"):
            fft_fn(
                signal_view,
                axes=axes,
                execution=exec_backend.nvname,
                options={"result_layout": result_layout.value},
            )
        return

    if exec_backend == ExecBackend.cufft and is_half(dtype) and get_cufft_version() < 10702:
        with pytest.raises(ValueError, match="sample size 1 and half-precision type"):
            fft_fn(
                signal_view,
                axes=axes,
                execution=exec_backend.nvname,
                options={"result_layout": result_layout.value},
            )
        return

    fft = fft_fn(
        signal_view,
        axes=axes,
        execution=exec_backend.nvname,
        options={"result_layout": result_layout.value},
    )
    assert_array_type(fft, framework, mem_backend, get_fft_dtype(dtype))
    assert_norm_close(
        fft,
        get_fft_ref(signal_view, axes=axes),
        axes=axes,
        exec_backend=exec_backend,
    )

    ifft_fn = nvmath.fft.ifft if is_complex(dtype) else nvmath.fft.irfft
    ifft = ifft_fn(
        fft,
        axes=axes,
        execution=exec_backend.nvname,
        options={"last_axis_parity": "odd", "result_layout": result_layout.value},
    )
    assert_norm_close(
        ifft,
        signal_view,
        exec_backend=exec_backend,
    )


@pytest.mark.parametrize(
    (
        "framework",
        "exec_backend",
        "mem_backend",
        "base_shape",
        "axes",
        "steps",
        "dtype",
    ),
    [
        (
            framework,
            exec_backend,
            mem_backend,
            repr(base_shape),
            repr(axes),
            repr(step),
            rng.choice(
                [
                    dtype
                    for dtype in framework_exec_type_support[framework][exec_backend]
                    if is_complex(dtype) and not is_half(dtype)
                ]
            ),
        )
        for framework in Framework.enabled()
        for exec_backend in supported_backends.exec
        for mem_backend in supported_backends.framework_mem[framework]
        for base_shape, axes, step in [
            ((2, 55), (0,), (None, 2)),
            ((2, 55), (1,), (None, 2)),
            ((100, 101, 5), (0, 1), (None, 2, 3)),
            ((100, 101, 5), (1, 2), (None, 2, 3)),
            ((100, 101, 5), (0, 1, 2), (None, 2, 3)),
        ]
    ],
)
def test_inplace_sliced_non_overlapping(
    framework,
    exec_backend,
    mem_backend,
    base_shape,
    axes,
    steps,
    dtype,
):
    base_shape = literal_eval(base_shape)
    axes = literal_eval(axes)
    steps = literal_eval(steps)
    assert len(base_shape) == len(steps)
    signal_base = get_random_input_data(framework, base_shape, dtype, mem_backend, seed=105)
    signal = signal_base[tuple(slice(None, None, step) for step in steps)]
    signal_copy = copy_array(signal)
    shape = tuple(
        (e + impl_step - 1) // impl_step
        for e, step in zip(base_shape, steps, strict=True)
        for impl_step in [1 if step is None else step]
    )
    assert signal.shape == shape
    if should_skip_3d_unsupported(exec_backend, shape, axes):
        pytest.skip("Pre 11.4.2 CTK does not support 3D batched FFT")

    try:
        fft = nvmath.fft.fft(
            signal,
            axes=axes,
            execution=exec_backend.nvname,
            options={"inplace": True},
        )
    except nvmath.fft.UnsupportedLayoutError:
        # The embedding check does usually block the step-sliced
        # operands. For mem_backend != exec_backend though
        # the copy is made so we have contiguous layout
        # and we can make sure that the overlapping check is not
        # too strict
        assert mem_backend.exec == exec_backend
        return
    assert fft is signal
    assert_norm_close(
        fft,
        get_fft_ref(signal_copy, axes=axes),
        axes=axes,
        exec_backend=exec_backend,
    )
    ifft = nvmath.fft.ifft(
        fft,
        axes=axes,
        execution=exec_backend.nvname,
        options={
            "inplace": True,
            "last_axis_parity": "odd" if shape[axes[-1]] else "even",
        },
    )
    assert ifft is signal
    assert_norm_close(
        ifft,
        get_scaled(signal_copy, math.prod(shape[a] for a in axes)),
        axes=axes,
        exec_backend=exec_backend,
    )


@pytest.mark.parametrize(
    (
        "framework",
        "exec_backend",
        "mem_backend",
        "shape",
        "axes",
        "unfold_args",
        "dtype",
        "result_layout",
    ),
    [
        (
            framework,
            exec_backend,
            mem_backend,
            repr(shape),
            repr(axes),
            repr(unfold_args),
            dtype,
            result_layout,
        )
        for framework in Framework.enabled()
        for exec_backend in supported_backends.exec
        for mem_backend in supported_backends.framework_mem[framework]
        for shape, axes, unfold_args in [
            ((128,), (0,), (0, 8, 1)),
            ((128,), (0,), (0, 8, 7)),
            ((128,), (0,), (0, 8, 8)),
            ((128,), (0,), (0, 8, 16)),
            ((128,), (1,), (0, 8, 1)),
            ((128,), (1,), (0, 8, 7)),
            ((128,), (1,), (0, 8, 8)),
            ((128,), (1,), (0, 8, 16)),
            ((128,), (0, 1), (0, 8, 1)),
            ((128,), (0, 1), (0, 8, 7)),
            ((128,), (0, 1), (0, 8, 8)),
            ((128,), (0, 1), (0, 8, 16)),
            ((32, 32), (0, 1, 2), (0, 4, 2)),
            ((32, 32), (0, 1, 2), (0, 4, 4)),
            ((32, 32), (0, 1, 2), (0, 4, 8)),
            ((32, 32), (0, 1, 2), (1, 4, 2)),
            ((32, 32), (0, 1, 2), (1, 4, 4)),
            ((32, 32), (0, 1, 2), (1, 4, 8)),
        ]
        for dtype in framework_exec_type_support[framework][exec_backend]
        if is_complex(dtype)
        for result_layout in OptFftLayout
    ],
)
def test_inplace_overlapping(
    framework,
    exec_backend,
    mem_backend,
    shape,
    axes,
    unfold_args,
    dtype,
    result_layout,
):
    shape = literal_eval(shape)
    axes = literal_eval(axes)
    unfold_args = literal_eval(unfold_args)
    unfold_dim, unfold_window_size, unfold_step = unfold_args
    signal = get_random_input_data(framework, shape, dtype, mem_backend, seed=105)
    signal_view = unfold(signal, unfold_dim, unfold_window_size, unfold_step)
    signal_view_copy = copy_array(signal_view)

    if should_skip_3d_unsupported(exec_backend, shape, axes):
        pytest.skip("Pre 11.4.2 CTK does not support 3D batched FFT")

    if unfold_window_size > unfold_step:
        with pytest.raises(ValueError, match="overlaps in memory"):
            nvmath.fft.fft(
                signal_view,
                axes=axes,
                execution=exec_backend.nvname,
                options={"inplace": True, "result_layout": result_layout.value},
            )

        with pytest.raises(ValueError, match="overlaps in memory"):
            nvmath.fft.ifft(
                signal_view,
                axes=axes,
                execution=exec_backend.nvname,
                options={"inplace": True, "result_layout": result_layout.value},
            )
    else:
        nvmath.fft.fft(
            signal_view,
            axes=axes,
            execution=exec_backend.nvname,
            options={"inplace": True, "result_layout": result_layout.value},
        )
        assert_norm_close(
            signal_view,
            get_fft_ref(signal_view_copy, axes=axes),
            axes=axes,
            exec_backend=exec_backend,
        )


@pytest.mark.parametrize(
    (
        "framework",
        "exec_backend",
        "mem_backend",
        "shape",
        "axes",
        "stride",
        "dtype",
        "result_layout",
        "inplace",
    ),
    [
        (
            framework,
            exec_backend,
            mem_backend,
            repr(shape),
            repr(axes),
            stride,
            dtype,
            result_layout,
            inplace,
        )
        for framework in Framework.enabled()
        for exec_backend in supported_backends.exec
        for mem_backend in supported_backends.framework_mem[framework]
        for shape, axes, stride in [
            ((128, 1, 1), (1,), 1),
            ((128, 1, 1), (1,), 2),
            ((128, 1, 1), (1,), 4),
            ((1, 1, 128), (1,), 1),
            ((1, 1, 128), (1,), 2),
            ((1, 1, 128), (1,), 4),
            ((128, 1, 1), (1, 2), 1),
            ((128, 1, 1), (1, 2), 2),
            ((128, 1, 1), (1, 2), 4),
            ((1, 1, 1), (1, 2), 2),
            ((1, 1, 1), (0, 1), 2),
            ((128, 1, 1, 1), (1, 2), 1),
            ((128, 1, 1, 1), (1, 2), 2),
            ((128, 1, 1, 1), (1, 2), 4),
            ((128, 1, 1, 1), (1, 2, 3), 1),
            ((128, 1, 1, 1), (1, 2, 3), 2),
            ((128, 1, 1, 1), (1, 2, 3), 4),
        ]
        for dtype in framework_exec_type_support[framework][exec_backend]
        if not is_half(dtype)
        for result_layout in OptFftLayout
        for inplace in [True, False]
        if not inplace or is_complex(dtype)
    ],
)
def test_repeated_strides_strided(
    framework,
    exec_backend,
    mem_backend,
    shape,
    axes,
    stride,
    dtype,
    result_layout,
    inplace,
):
    shape = literal_eval(shape)
    axes = literal_eval(axes)

    if should_skip_3d_unsupported(exec_backend, shape, axes):
        pytest.skip("Pre 11.4.2 CTK does not support 3D batched FFT")

    vol = math.prod(shape) * stride
    signal = get_random_input_data(framework, (vol,), dtype, mem_backend, seed=105)
    signal = signal[::stride].reshape(shape)
    signal_copy = copy_array(signal)
    fft_fn = nvmath.fft.fft if is_complex(dtype) else nvmath.fft.rfft
    fft = fft_fn(
        signal,
        axes=axes,
        execution=exec_backend.nvname,
        options={"result_layout": result_layout.value, "inplace": inplace},
    )
    assert_array_type(fft, framework, mem_backend, get_fft_dtype(dtype))
    assert_norm_close(
        fft,
        get_fft_ref(signal_copy, axes=axes),
        axes=axes,
        exec_backend=exec_backend,
    )


@pytest.mark.parametrize(
    (
        "framework",
        "exec_backend",
        "mem_backend",
        "shape",
        "strides",
        "axes",
        "dtype",
        "result_layout",
        "inplace",
    ),
    [
        (
            framework,
            exec_backend,
            mem_backend,
            repr(shape),
            repr(strides),
            repr(axes),
            dtype,
            result_layout,
            inplace,
        )
        for framework in Framework.enabled()
        for exec_backend in supported_backends.exec
        for mem_backend in supported_backends.framework_mem[framework]
        for shape, strides, axes in [
            # non-tilable batch
            ((4, 4, 4, 4), (1, 1, 4, 4), (3, 1)),
            ((8, 8, 8, 8, 8), (1, 8, 8, 64, 64), (2, 3, 0)),
            # non-tialbe batch, correct output embedding
            ((4, 1, 4, 4), (1, 4, 4, 4), (0, 2)),
            # wrong output embedding
            ((8, 4, 2, 4), (8, 2, 1, 8), (0, 1, 2)),
            ((5, 4, 8, 16), (8, 1, 4, 32), (1, 2, 3)),
            ((5, 4, 8, 16), (8, 2, 8, 64), (1, 2, 3)),
            ((5, 4, 8, 16), (16, 2, 16, 128), (1, 2, 3)),
            ((4, 8, 5), (1, 8, 1), (0, 1)),
            ((4, 8, 5), (4, 8, 4), (0, 1)),
            ((4, 8, 5), (4, 32, 4), (0, 1)),
        ]
        for dtype in framework_exec_type_support[framework][exec_backend]
        for result_layout in OptFftLayout
        for inplace in [True, False]
        if not inplace or is_complex(dtype)
    ],
)
def test_overlapping_non_tilable_output(
    framework,
    exec_backend,
    mem_backend,
    shape,
    strides,
    axes,
    dtype,
    result_layout,
    inplace,
):
    shape = literal_eval(shape)
    strides = literal_eval(strides)
    axes = literal_eval(axes)
    assert len(shape) == len(strides)

    if should_skip_3d_unsupported(exec_backend, shape, axes):
        pytest.skip("Pre 11.4.2 CTK does not support 3D batched FFT")

    max_offset = sum((extent - 1) * stride for extent, stride in zip(shape, strides, strict=True))
    signal = get_random_input_data(framework, (max_offset + 1,), dtype, mem_backend, seed=105)
    signal_view = as_strided(signal, shape, strides)
    signal_view_copy = copy_array(signal_view)
    fft_fn = nvmath.fft.fft if is_complex(dtype) else nvmath.fft.rfft
    try:
        fft = fft_fn(
            signal_view,
            axes=axes,
            execution=exec_backend.nvname,
            options={"result_layout": result_layout.value, "inplace": inplace},
        )
    except nvmath.fft.UnsupportedLayoutError as e:
        data_transposed = get_permuted_copy(signal_view, e.permutation)
        assert is_decreasing(get_array_strides(data_transposed))
        res_transposed = fft_fn(
            data_transposed,
            axes=e.axes,
            execution=exec_backend.nvname,
            options={"result_layout": result_layout.value, "inplace": inplace},
        )
        fft = get_permuted_copy(res_transposed, get_rev_perm(e.permutation))
    except ValueError as e:
        assert "overlaps in memory" in str(e)
        assert inplace
        return

    assert_array_type(fft, framework, mem_backend, get_fft_dtype(dtype))
    assert_norm_close(
        fft,
        get_fft_ref(signal_view_copy, axes=axes),
        axes=axes,
        exec_backend=exec_backend,
    )


@pytest.mark.parametrize(
    ("framework", "exec_backend", "mem_backend", "array_dim", "fft_dim", "dtype"),
    [
        (
            framework,
            exec_backend,
            mem_backend,
            array_dim,
            fft_dim,
            rng.choice(framework_exec_type_support[framework][exec_backend]),
        )
        for framework in Framework.enabled()
        for exec_backend in supported_backends.exec
        for mem_backend in supported_backends.framework_mem[framework]
        for array_dim in [10, 20, 32]
        for fft_dim in [1, 2, 3]
    ],
)
def test_high_dim_array(framework, exec_backend, mem_backend, array_dim, fft_dim, dtype):
    if array_dim <= 20:
        shape = (2,) * array_dim
    else:
        shape = tuple(2 if i >= array_dim - fft_dim else 1 for i in range(array_dim))
    signal = get_random_input_data(framework, shape, dtype, mem_backend, seed=444)
    axes = tuple(range(array_dim - fft_dim, array_dim))

    if should_skip_3d_unsupported(exec_backend, shape, axes):
        pytest.skip("Pre 11.4.2 CTK does not support 3D batched FFT")

    if is_complex(dtype):
        sample_fft = nvmath.fft.fft(
            signal,
            axes=axes,
            execution=exec_backend.nvname,
        )
    else:
        sample_fft = nvmath.fft.rfft(
            signal,
            axes=axes,
            execution=exec_backend.nvname,
        )
    fft_ref = get_fft_ref(signal, axes=axes)
    assert_norm_close(
        sample_fft,
        fft_ref,
        axes=axes,
        exec_backend=exec_backend,
    )


@pytest.mark.parametrize(
    ("framework", "exec_backend", "mem_backend", "fft_dim", "dtype"),
    [
        (
            framework,
            exec_backend,
            mem_backend,
            fft_dim,
            rng.choice(framework_exec_type_support[framework][exec_backend]),
        )
        for framework in Framework.enabled()
        for exec_backend in supported_backends.exec
        for mem_backend in supported_backends.framework_mem[framework]
        for fft_dim in [2, 3]
    ],
)
def test_unsupported_axes_gaps(framework, exec_backend, mem_backend, fft_dim, dtype):
    shape = (32,) * (fft_dim + 1)
    signal = get_random_input_data(framework, shape, dtype, mem_backend, seed=444)
    axes = (0, fft_dim)
    example_transpose = tuple(range(1, fft_dim)) + (0, fft_dim)
    fft_fn = nvmath.fft.fft if is_complex(dtype) else nvmath.fft.rfft
    with pytest.raises(
        nvmath.fft.UnsupportedLayoutError,
        match=f"create a transposed view using transpose{re.escape(str(example_transpose))}",
    ):
        fft_fn(
            signal,
            axes=axes,
            execution=exec_backend.nvname,
        )


@pytest.mark.parametrize(
    ("framework", "exec_backend", "mem_backend", "fft_dim", "negative", "dtype"),
    [
        (
            framework,
            exec_backend,
            mem_backend,
            fft_dim,
            negative,
            rng.choice(framework_exec_type_support[framework][exec_backend]),
        )
        for framework in Framework.enabled()
        for exec_backend in supported_backends.exec
        for mem_backend in supported_backends.framework_mem[framework]
        for fft_dim in [1, 2, 3]
        for negative in [True, False]
    ],
)
def test_unsupported_axes_out_of_range(framework, exec_backend, mem_backend, fft_dim, negative, dtype):
    shape = (32,) * (fft_dim - 1)
    signal = get_random_input_data(framework, shape, dtype, mem_backend, seed=444)
    if not negative:
        axes = tuple(range(fft_dim))
    else:
        axes = (-fft_dim - 1, -fft_dim)
    fft_fn = nvmath.fft.fft if is_complex(dtype) else nvmath.fft.rfft
    if len(shape) == 0:
        with pytest.raises(ValueError, match="FFT does not support scalars."):
            fft_fn(
                signal,
                axes=axes,
                execution=exec_backend.nvname,
            )
    else:
        with pytest.raises(
            ValueError,
            match=f"{re.escape(str(axes))} are out of bounds for a {fft_dim - 1}-D tensor",
        ):
            fft_fn(
                signal,
                axes=axes,
                execution=exec_backend.nvname,
            )


@pytest.mark.parametrize(
    ("framework", "exec_backend", "mem_backend", "fft_dim", "use_axes", "dtype"),
    [
        (
            framework,
            exec_backend,
            mem_backend,
            fft_dim,
            use_axes,
            rng.choice(framework_exec_type_support[framework][exec_backend]),
        )
        for framework in Framework.enabled()
        for exec_backend in supported_backends.exec
        for mem_backend in supported_backends.framework_mem[framework]
        for fft_dim in [4, 5]
        for use_axes in [False, True]
    ],
)
def test_unsupported_fft_dim(framework, exec_backend, mem_backend, fft_dim, use_axes, dtype):
    shape = (2,) * fft_dim
    signal = get_random_input_data(framework, shape, dtype, mem_backend, seed=444)

    fft_fn = nvmath.fft.fft if is_complex(dtype) else nvmath.fft.rfft
    if not use_axes:
        with pytest.raises(
            ValueError,
            match="FFTs in number of dimensions > 3 is not supported",
        ):
            fft_fn(
                signal,
                execution=exec_backend.nvname,
            )
    else:
        with pytest.raises(
            ValueError,
            match="Only up to 3D FFTs are currently supported.",
        ):
            fft_fn(
                signal,
                axes=tuple(range(fft_dim)),
                execution=exec_backend.nvname,
            )
