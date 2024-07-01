# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import itertools
from ast import literal_eval
import re
import random
import math

import pytest

import nvmath

from .utils.common_axes import (
    Framework,
    Backend,
    DType,
    ShapeKind,
    OptFftLayout,
    OptFftType,
)
from .utils.axes_utils import (
    is_complex,
    is_half,
    get_fft_dtype,
    get_ifft_dtype,
)
from .utils.support_matrix import (
    framework_type_support,
    framework_backend_support,
    type_shape_support,
    opt_fft_type_input_type_support,
    inplace_opt_ftt_type_support,
)
from .utils.input_fixtures import (
    get_random_input_data,
)
from .utils.check_helpers import (
    assert_eq,
    is_decreasing,
    copy_array,
    get_fft_ref,
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
    assert_norm_close,
    assert_array_type,
    assert_eq,
    should_skip_3d_unsupported,
)


rng = random.Random(42)


@pytest.mark.parametrize(
    (
        "framework",
        "backend",
        "array_dim",
        "axis",
        "dtype",
        "shape_kind",
        "result_layout",
    ),
    [
        (
            framework,
            backend,
            array_dim,
            axis,
            dtype,
            rng.choice(type_shape_support[dtype]),
            rng.choice(list(OptFftLayout)),
        )
        for framework in Framework.enabled()
        for backend in framework_backend_support[framework]
        for array_dim in [2, 3]
        for axis in [0, array_dim - 1]
        for dtype in [DType.float16, DType.float32, DType.complex64]
        if dtype in framework_type_support[framework]
    ],
)
def test_fft_ifft_1d(
    framework, backend, array_dim, axis, dtype, shape_kind, result_layout
):
    shapes = {
        ShapeKind.pow2: 128,
        ShapeKind.pow2357: 2 * 3 * 5 * 7,
        ShapeKind.prime: 127,
        ShapeKind.random: 414,
    }
    shape = (shapes[shape_kind],) * array_dim
    signal = get_random_input_data(framework, shape, dtype, backend, seed=55)

    fft_fn = nvmath.fft.fft if is_complex(dtype) else nvmath.fft.rfft

    if dtype == DType.float16 and axis != array_dim - 1:
        with pytest.raises(ValueError, match="is currently not supported for strided inputs"):
            fft_fn(signal, axes=[axis], options={"result_layout": result_layout.value})
        return
    
    fft = fft_fn(
        signal, axes=[axis], options={"result_layout": result_layout.value}
    )
    assert_array_type(fft, framework, backend, get_fft_dtype(dtype))
    if result_layout == OptFftLayout.natural:
        fft_strides = get_array_strides(fft)
        assert is_decreasing(fft_strides), f"{fft_strides}"
    assert_norm_close(fft, get_fft_ref(signal, axes=[axis]), axes=[axis])

    if is_complex(dtype):
        options = {
            "result_layout": result_layout.value,
            **get_ifft_c2r_options(dtype, shape[axis]),
        }
        ifft = nvmath.fft.ifft(fft, options=options, axes=[axis])
        assert_array_type(ifft, framework, backend, dtype)
        if result_layout == OptFftLayout.natural:
            ifft_strides = get_array_strides(ifft)
            assert is_decreasing(ifft_strides), f"{ifft_strides}"
        assert_norm_close(ifft, get_scaled(signal, shape[axis]))


@pytest.mark.parametrize(
    (
        "framework",
        "backend",
        "array_dim",
        "first_axis",
        "dtype",
        "shape_kind",
        "result_layout",
    ),
    [
        (
            framework,
            backend,
            array_dim,
            first_axis,
            dtype,
            rng.choice(type_shape_support[dtype]),
            rng.choice(list(OptFftLayout)),
        )
        for framework in Framework.enabled()
        for backend in framework_backend_support[framework]
        for array_dim in [2, 3]
        for first_axis in range(array_dim - 2 + 1)
        for dtype in framework_type_support[framework]
    ],
)
def test_fft_ifft_2d(
    framework, backend, array_dim, first_axis, dtype, shape_kind, result_layout
):
    shapes = {
        ShapeKind.pow2: (64, 256, 128),
        ShapeKind.pow2357: (9 * 49, 4 * 25, 2 * 3 * 5 * 7),
        ShapeKind.prime: (127, 233, 277),
        ShapeKind.random: (209, 178, 361),
    }
    shape = shapes[shape_kind][:array_dim]
    axes = list(range(first_axis, first_axis + 2))
    signal = get_random_input_data(framework, shape, dtype, backend, seed=55)

    fft_fn = nvmath.fft.fft if is_complex(dtype) else nvmath.fft.rfft

    if dtype == DType.float16 and first_axis != array_dim - 2:
        with pytest.raises(ValueError, match="is currently not supported for strided inputs"):
            fft_fn(signal, axes=axes, options={"result_layout": result_layout.value})
        return

    fft = fft_fn(
        signal, axes=axes, options={"result_layout": result_layout.value}
    )
    assert_array_type(fft, framework, backend, get_fft_dtype(dtype))
    if result_layout == OptFftLayout.natural:
        fft_strides = get_array_strides(fft)
        assert is_decreasing(fft_strides), f"{fft_strides}"
    assert_norm_close(fft, get_fft_ref(signal, axes=axes), axes=axes, shape_kind=shape_kind)
    if is_complex(dtype):
        options = {
            "result_layout": result_layout.value,
            **get_ifft_c2r_options(dtype, shape[axes[-1]]),
        }
        ifft = nvmath.fft.ifft(fft, options=options, axes=axes)
        assert_array_type(ifft, framework, backend, dtype)
        if result_layout == OptFftLayout.natural:
            ifft_strides = get_array_strides(ifft)
            assert is_decreasing(ifft_strides), f"{ifft_strides}"
        volume = math.prod(shape[axis] for axis in axes)
        assert_norm_close(ifft, get_scaled(signal, volume), shape_kind=shape_kind)


@pytest.mark.parametrize(
    (
        "framework",
        "backend",
        "array_dim",
        "first_axis",
        "dtype",
        "shape_kind",
        "result_layout",
    ),
    [
        (
            framework,
            backend,
            array_dim,
            first_axis,
            dtype,
            rng.choice(type_shape_support[dtype]),
            rng.choice(list(OptFftLayout)),
        )
        for framework in Framework.enabled()
        for backend in framework_backend_support[framework]
        for array_dim in [3, 4]
        for first_axis in range(array_dim - 3 + 1)
        for dtype in framework_type_support[framework]
        if not is_half(dtype)  # the values overflow with this many elements
    ],
)
def test_fft_ifft_3d(
    framework, backend, array_dim, first_axis, dtype, shape_kind, result_layout
):
    shapes = {
        ShapeKind.pow2: (128, 64, 32, 16),
        ShapeKind.pow2357: (6, 441, 210, 30),
        ShapeKind.prime: (17, 127, 47, 13),
        ShapeKind.random: (22, 178, 361, 26),
    }
    shape = shapes[shape_kind][:array_dim]
    axes = list(range(first_axis, first_axis + 3))

    if should_skip_3d_unsupported(shape, axes):
        pytest.skip("Pre 11.4.2 CTK does not support 3D batched FFT")

    signal = get_random_input_data(framework, shape, dtype, backend, seed=55)

    fft_fn = nvmath.fft.fft if is_complex(dtype) else nvmath.fft.rfft
    fft = fft_fn(
        signal, axes=axes, options={"result_layout": result_layout.value}
    )
    assert_array_type(fft, framework, backend, get_fft_dtype(dtype))
    if result_layout == OptFftLayout.natural:
        fft_strides = get_array_strides(fft)
        assert is_decreasing(fft_strides), f"{fft_strides}"
    assert_norm_close(fft, get_fft_ref(signal, axes=axes), axes=axes, shape_kind=shape_kind)

    if is_complex(dtype):
        options = {
            "result_layout": result_layout.value,
            **get_ifft_c2r_options(dtype, shape[axes[-1]]),
        }
        ifft = nvmath.fft.ifft(fft, options=options, axes=axes)
        assert_array_type(ifft, framework, backend, dtype)
        if result_layout == OptFftLayout.natural:
            ifft_strides = get_array_strides(ifft)
            assert is_decreasing(ifft_strides), f"{ifft_strides}"
        volume = math.prod(shape[axis] for axis in axes)
        assert_norm_close(ifft, get_scaled(signal, volume), shape_kind=shape_kind)


@pytest.mark.parametrize(
    (
        "framework",
        "backend",
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
            backend,
            fft_dim,
            batched,
            fft_type,
            dtype,
            result_layout,
            rng.choice(type_shape_support[dtype]),
        )
        for framework in Framework.enabled()
        for backend in framework_backend_support[framework]
        for fft_dim in [1, 2, 3]
        for batched in ["no", "left", "right"]
        for fft_type in inplace_opt_ftt_type_support[True]
        for dtype in opt_fft_type_input_type_support[fft_type]
        if dtype in framework_type_support[framework]
        for result_layout in OptFftLayout
    ],
)
def test_inplace(
    framework, backend, fft_dim, batched, fft_type, dtype, result_layout, shape_kind
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

    if should_skip_3d_unsupported(shape, axes):
        pytest.skip("Pre 11.4.2 CTK does not support 3D batched FFT")

    signal = get_random_input_data(framework, shape, dtype, backend, seed=105)
    signal_copy = copy_array(signal)

    nvmath.fft.fft(
        signal,
        axes=axes,
        options={
            "inplace": True,
            "result_layout": result_layout.value,
            "fft_type": fft_type.value,
        },
    )
    assert_array_type(signal, framework, backend, dtype)
    if result_layout == OptFftLayout.natural:
        fft_strides = get_array_strides(signal)
        assert is_decreasing(fft_strides), f"{fft_strides}"
    assert_norm_close(signal, get_fft_ref(signal_copy, axes=axes), axes=axes, shape_kind=shape_kind)

    if fft_dim == 1 or not is_half(dtype):  # the half types overflow for bigger sizes
        nvmath.fft.ifft(
            signal,
            axes=axes,
            options={
                "inplace": True,
                "result_layout": result_layout.value,
                "fft_type": fft_type.value,
            },
        )

        assert_array_type(signal, framework, backend, dtype)

        if result_layout == OptFftLayout.natural:
            ifft_strides = get_array_strides(signal)
            assert is_decreasing(ifft_strides), f"{ifft_strides}"

        if axes is not None:
            volume = math.prod(shape[axis] for axis in axes)
        else:
            volume = math.prod(shape)
        assert_norm_close(signal, get_scaled(signal_copy, volume), shape_kind=shape_kind)


@pytest.mark.parametrize(
    (
        "framework",
        "backend",
        "axes",
        "batched",
        "dtype",
        "fft_shape",
    ),
    [
        (
            framework,
            backend,
            repr(axes),
            batched,
            dtype := rng.choice(
                [dt for dt in framework_type_support[framework] if is_complex(dt)]
            ),
            repr(
                rng.sample(
                    (
                        [17, 31, 101]
                        if ShapeKind.prime in type_shape_support[dtype]
                        else [16, 32, 64]
                    ),
                    k=3,
                )[: len(axes)]
            ),
        )
        for framework in Framework.enabled()
        for backend in framework_backend_support[framework]
        for axes in [[0, 1], [1, 0], [0, 2, 1], [2, 1, 0], [1, 2, 0]]
        for batched in ["no", "left", "right"]
    ],
)
def test_permuted_axes_c2c(framework, backend, axes, batched, dtype, fft_shape):
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
    
    if should_skip_3d_unsupported(shape, axes):
        pytest.skip("Pre 11.4.2 CTK does not support 3D batched FFT")

    signal = get_random_input_data(framework, shape, dtype, backend, seed=105)
    fft = nvmath.fft.fft(signal, axes=axes)
    assert_array_type(fft, framework, backend, get_fft_dtype(dtype))
    assert_norm_close(fft, get_fft_ref(signal, axes=tuple(sorted(axes))), axes=axes)
    ifft = nvmath.fft.ifft(fft, axes=axes)
    assert_array_type(ifft, framework, backend, dtype)
    volume = math.prod(fft_shape)
    assert_norm_close(ifft, get_scaled(signal, volume), axes=axes)


@pytest.mark.parametrize(
    (
        "framework",
        "backend",
        "axes",
        "shape",
        "dtype",
        "result_layout",
    ),
    [
        (
            framework,
            backend,
            repr(axes),
            repr(shape),
            rng.choice([dt for dt in framework_type_support[framework] if is_complex(dt)]),
            result_layout
        )
        for framework in Framework.enabled()
        for backend in framework_backend_support[framework]
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
def test_permuted_axes_c2c_repeated_strides(
    framework, backend, axes, shape, dtype, result_layout
):
    axes = literal_eval(axes)
    shape = literal_eval(shape)

    if should_skip_3d_unsupported(shape, axes):
        pytest.skip("Pre 11.4.2 CTK does not support 3D batched FFT")

    signal = get_random_input_data(framework, shape, dtype, backend, seed=105)
    fft = nvmath.fft.fft(signal, axes=axes, options={"result_layout": result_layout.value})
    assert_array_type(fft, framework, backend, get_fft_dtype(dtype))
    assert_norm_close(fft, get_fft_ref(signal, axes=tuple(sorted(axes))), axes=axes)
    ifft = nvmath.fft.ifft(fft, axes=axes, options={"result_layout": result_layout.value})
    assert_array_type(ifft, framework, backend, dtype)
    volume = math.prod(shape[a] for a in axes)
    assert_norm_close(ifft, get_scaled(signal, volume), axes=axes)


@pytest.mark.parametrize(
    (
        "framework",
        "backend",
        "axes",
        "shape",
        "dtype",
        "result_layout",
    ),
    [
        (
            framework,
            backend,
            repr(axes),
            repr(shape),
            rng.choice([dt for dt in framework_type_support[framework] if is_complex(dt)]),
            result_layout
        )
        for framework in Framework.enabled()
        for backend in framework_backend_support[framework]
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
def test_permuted_axes_c2c_repeated_strides_inplace(
    framework, backend, axes, shape, dtype, result_layout
):
    axes = literal_eval(axes)
    shape = literal_eval(shape)

    if should_skip_3d_unsupported(shape, axes):
        pytest.skip("Pre 11.4.2 CTK does not support 3D batched FFT")

    data = get_random_input_data(framework, shape, dtype, backend, seed=105)
    signal_copy = copy_array(data)
    nvmath.fft.fft(data, axes=axes, options={"inplace": True, "result_layout": result_layout.value})
    assert_array_type(data, framework, backend, dtype)
    assert_norm_close(data, get_fft_ref(signal_copy, axes=tuple(sorted(axes))), axes=axes)
    nvmath.fft.ifft(data, axes=axes, options={"inplace": True, "result_layout": result_layout.value})
    assert_array_type(data, framework, backend, dtype)
    volume = math.prod(shape[a] for a in axes)
    assert_norm_close(data, get_scaled(signal_copy, volume), axes=axes)


@pytest.mark.parametrize(
    (
        "framework",
        "backend",
        "axes",
        "batched",
        "dtype",
        "fft_shape",
        "result_layout",
    ),
    [
        (
            framework,
            backend,
            repr(axes),
            batched,
            dtype := rng.choice(
                [dt for dt in framework_type_support[framework] if not is_complex(dt)]
            ),
            repr(
                rng.sample(
                    (
                        [17, 31, 101]
                        if ShapeKind.prime in type_shape_support[dtype]
                        else [16, 32, 64]
                    ),
                    k=3,
                )
            ),
            result_layout,
        )
        for framework in Framework.enabled()
        for backend in framework_backend_support[framework]
        for axes in [[1, 0, 2]]
        for batched in ["no", "left", "right"]
        for result_layout in OptFftLayout
    ],
)
def test_permuted_axes_r2c_c2r(framework, backend, axes, batched, dtype, fft_shape, result_layout):
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
    
    if should_skip_3d_unsupported(shape, axes):
        pytest.skip("Pre 11.4.2 CTK does not support 3D batched FFT")

    signal = get_random_input_data(framework, shape, dtype, backend, seed=105)

    if is_half(dtype) and batched == "right":
        with pytest.raises(ValueError, match="is currently not supported for strided inputs"):
            fft = nvmath.fft.rfft(
                signal,
                axes=axes,
                options={"result_layout": result_layout.value}
            )
        return

    fft = nvmath.fft.rfft(signal, axes=axes, options={"result_layout": result_layout.value})
    assert_array_type(fft, framework, backend, get_fft_dtype(dtype))
    assert_norm_close(fft, get_fft_ref(signal, axes=tuple(sorted(axes))), axes=axes)
    ifft = nvmath.fft.irfft(
        fft,
        axes=axes,
        options={
            "last_axis_size": "odd" if shape[axes[-1]] % 2 else "even",
            "result_layout": result_layout.value
        },
    )
    assert_array_type(ifft, framework, backend, dtype)
    volume = math.prod(fft_shape)
    assert_norm_close(ifft, get_scaled(signal, volume), axes=axes)


@pytest.mark.parametrize(
    (
        "framework",
        "backend",
        "axes",
        "shape",
        "dtype",
        "result_layout",
    ),
    [
        (
            framework,
            backend,
            repr(axes),
            repr(shape),
            rng.choice([dt for dt in framework_type_support[framework] if not is_complex(dt)]),
            result_layout
        )
        for framework in Framework.enabled()
        for backend in framework_backend_support[framework]
        for shape, axes in [
            ((64, 1), (0, 1)),
            ((4, 1, 32, 1, 8), (3, 2, 4)),
            ((32, 1, 8, 4, 1), (1, 0, 2)),
        ]
        for result_layout in OptFftLayout
    ],
)
def test_permuted_axes_r2c_c2r_repeated_strides(
    framework, backend, axes, shape, dtype, result_layout
):
    axes = literal_eval(axes)
    shape = literal_eval(shape)

    if should_skip_3d_unsupported(shape, axes):
        pytest.skip("Pre 11.4.2 CTK does not support 3D batched FFT")

    signal = get_random_input_data(framework, shape, dtype, backend, seed=105)

    if is_half(dtype) and max(axes) < len(shape) - 1:
        with pytest.raises(ValueError, match="is currently not supported for strided inputs"):
            nvmath.fft.rfft(
                signal, axes=axes, options={"result_layout": result_layout.value}
            )
        return

    fft = nvmath.fft.rfft(
        signal, axes=axes, options={"result_layout": result_layout.value}
    )
    assert_array_type(fft, framework, backend, get_fft_dtype(dtype))
    assert_norm_close(fft, get_fft_ref(signal, axes=axes), axes=axes)
    ifft = nvmath.fft.irfft(
        fft,
        axes=axes,
        options={
            "last_axis_size": "odd" if shape[axes[-1]] % 2 else "even",
            "result_layout": result_layout.value,
        },
    )
    assert_array_type(ifft, framework, backend, dtype)
    volume = math.prod(shape[a] for a in axes)
    assert_norm_close(ifft, get_scaled(signal, volume), axes=axes)


@pytest.mark.parametrize(
    (
        "framework",
        "backend",
        "axes",
        "shape",
        "dtype",
        "result_layout",
    ),
    [
        (
            framework,
            backend,
            repr(axes),
            repr(shape),
            rng.choice([dt for dt in framework_type_support[framework] if not is_complex(dt)]),
            result_layout,
        )
        for framework in Framework.enabled()
        for backend in framework_backend_support[framework]
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
    framework, backend, axes, shape, dtype, result_layout
):
    axes = literal_eval(axes)
    shape = literal_eval(shape)

    if should_skip_3d_unsupported(shape, axes):
        pytest.skip("Pre 11.4.2 CTK does not support 3D batched FFT")

    signal = get_random_input_data(framework, shape, dtype, backend, seed=105)

    fft = check_layout_fallback(
        signal,
        axes,
        lambda signal, axes: nvmath.fft.rfft(
            signal, axes=axes, options={"result_layout": result_layout.value}
        ),
    )
    assert_array_type(fft, framework, backend, get_fft_dtype(dtype))
    assert_norm_close(fft, get_fft_ref(signal, axes=axes), axes=axes)
    last_axis_size =  "odd" if shape[axes[-1]] % 2 else "even"
    ifft = check_layout_fallback(
        fft,
        axes,
        lambda fft, axes: nvmath.fft.irfft(
                fft,
                axes=axes,
                options={
                    "last_axis_size": last_axis_size,
                    "result_layout": result_layout.value,
                },
            ),
    )
    assert_array_type(ifft, framework, backend, dtype)
    volume = math.prod(shape[a] for a in axes)
    assert_norm_close(ifft, get_scaled(signal, volume), axes=axes)


@pytest.mark.parametrize(
    ("result_layout", "framework", "backend", "fft_shape", "batch_shape", "batched", "dtype"),
    [
        (
            result_layout,
            framework,
            backend,
            repr(fft_shape),
            repr(batch_shape),
            batched,
            dtype,
        )
        for result_layout in OptFftLayout
        for framework in Framework.enabled()
        for backend in framework_backend_support[framework]
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
        for dtype in framework_type_support[framework]
    ],
)
def test_fft_repeated_strides(result_layout, framework, backend, fft_shape, batch_shape, batched, dtype):
    fft_shape = literal_eval(fft_shape)
    batch_shape = literal_eval(batch_shape)
    if batched == "left":
        shape = batch_shape + fft_shape
        axes = tuple(range(offset := len(batch_shape), offset + len(fft_shape)))
    else:
        assert batched == "right"
        shape = fft_shape + batch_shape
        axes = tuple(range(len(fft_shape)))
    
    if should_skip_3d_unsupported(shape, axes):
        pytest.skip("Pre 11.4.2 CTK does not support 3D batched FFT")

    signal = get_random_input_data(framework, shape, dtype, backend, seed=105)
    fft_fn = nvmath.fft.fft if is_complex(dtype) else nvmath.fft.rfft

    if dtype == DType.float16 and batched == "right" and any(d != 1 for d in batch_shape):
        with pytest.raises(ValueError, match="is currently not supported for strided inputs"):
            fft = fft_fn(signal, axes=axes, options={"result_layout": result_layout.value})        
    else:
        try:
            fft = fft_fn(signal, axes=axes, options={"result_layout": result_layout.value})
            assert_array_type(fft, framework, backend, get_fft_dtype(dtype))
            assert_norm_close(fft, get_fft_ref(signal, axes=axes), axes=axes)
        except nvmath.bindings.cufft.cuFFTError as e:
            assert "CUFFT_SETUP_FAILED" in str(e)
            assert dtype == DType.float16 and fft_shape[-1] == 1
            assert get_cufft_version() < 10702  # 10702 is shipped with CTK 11.7


@pytest.mark.parametrize(
    (
        "result_layout",
        "framework",
        "backend",
        "fft_shape",
        "batch_shape",
        "batched",
        "dtype",
    ),
    [
        (
            result_layout,
            framework,
            backend,
            repr(fft_shape),
            repr(batch_shape),
            batched,
            dtype,
        )
        for result_layout in OptFftLayout
        for framework in Framework.enabled()
        for backend in framework_backend_support[framework]
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
        for dtype in framework_type_support[framework]
        # r2c halfs do not support strided inputs
        if dtype != DType.float16
        or batched == "left"
        or all(d == 1 for d in batch_shape)
    ],
)
def test_ifft_repeated_strides(
    result_layout, framework, backend, fft_shape, batch_shape, batched, dtype
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
    
    if should_skip_3d_unsupported(shape, axes):
        pytest.skip("Pre 11.4.2 CTK does not support 3D batched FFT")

    try:
        signal = get_random_input_data(framework, shape, dtype, backend, seed=105)
        fft_fn = nvmath.fft.fft if is_complex(dtype) else nvmath.fft.rfft
        fft = fft_fn(signal, axes=axes, options={"result_layout": "natural"})
        assert_array_type(fft, framework, backend, get_fft_dtype(dtype))
        assert_norm_close(fft, get_fft_ref(signal, axes=axes), axes=axes)
        assert is_decreasing(get_array_strides(fft))

        ifft_fn = nvmath.fft.ifft if is_complex(dtype) else nvmath.fft.irfft
        ifft = ifft_fn(
            fft,
            axes=axes,
            options={
                "last_axis_size": "even" if shape[axes[-1]] % 2 == 0 else "odd",
                "result_layout": result_layout.value,
            },
        )
        assert_array_type(ifft, framework, backend, dtype)
        volume = math.prod(fft_shape)
        assert_norm_close(ifft, get_scaled(signal, volume), axes=axes)
    except nvmath.bindings.cufft.cuFFTError as e:
        assert "CUFFT_SETUP_FAILED" in str(e)
        assert dtype == DType.float16 and fft_shape[-1] == 1
        assert get_cufft_version() < 10702  # 10702 is shipped with CTK 11.7


@pytest.mark.parametrize(
    (
        "result_layout",
        "framework",
        "backend",
        "shape",
        "axes",
    ),
    [
        (
            result_layout,
            framework,
            backend,
            repr(shape),
            repr(axes),
        )
        for result_layout in OptFftLayout
        for framework in Framework.enabled()
        if DType.complex32 in framework_type_support[framework]
        for backend in framework_backend_support[framework]
        for shape, axes in [
            ((513, 4), (0,)),
            ((128, 33, 2), (0, 1,)),
        ]
    ],
)
def test_irfft_half_strided_output(
    result_layout, framework, backend, shape, axes
):
    shape = literal_eval(shape)
    axes = literal_eval(axes)

    if should_skip_3d_unsupported(shape, axes):
        pytest.skip("Pre 11.4.2 CTK does not support 3D batched FFT")

    real_shape = list(shape)
    real_shape[axes[-1]] = (shape[axes[-1]] - 1) * 2
    signal = get_random_input_data(framework, real_shape, DType.float16, backend, seed=105)
    # normalize the fft so we don't overflow in half-precision case
    fft = copy_array(get_fft_ref(signal, axes=axes, norm="forward"))
    assert_array_type(fft, framework, backend, DType.complex32)
    assert_eq(fft.shape, shape)
    if result_layout == OptFftLayout.natural:
        with pytest.raises(ValueError, match="is currently not supported for strided outputs"):
            nvmath.fft.irfft(fft, axes=axes, options={"result_layout": result_layout.value})
    else:
        ifft = nvmath.fft.irfft(fft, axes=axes, options={"result_layout": result_layout.value})
        assert_array_type(ifft, framework, backend, DType.float16)
        fft2 = nvmath.fft.rfft(ifft, axes=axes)
        assert_array_type(fft2, framework, backend, DType.complex32)
        volume = math.prod(ifft.shape[a] for a in axes)
        assert_norm_close(fft2, get_scaled(fft, volume), axes=axes)


@pytest.mark.parametrize(
    (
        "framework",
        "backend",
        "fft_dim",
        "batched",
        "dtype",
        "shape_kind",
    ),
    [
        (
            framework,
            backend,
            fft_dim,
            batched,
            dtype,
            rng.choice(type_shape_support[dtype]),
        )
        for framework in Framework.enabled()
        for backend in framework_backend_support[framework]
        for fft_dim in [1, 2, 3]
        for batched in ["no", "left", "right"]
        for dtype in framework_type_support[framework]
        if is_complex(dtype) and not is_half(dtype)
    ],
)
def test_sliced_tensor_complex(framework, backend, fft_dim, batched, dtype, shape_kind):
    extent = {
        ShapeKind.pow2: (64 + 5, 64 + 5, 64 + 5),
        ShapeKind.pow2357: (48 + 5, 48 + 5, 48 + 5),
        ShapeKind.prime: (43 + 5, 43 + 5, 43 + 5),
        ShapeKind.random: (22 + 5, 22 + 5, 22 + 5),
    }
    shape = extent[shape_kind][:fft_dim]

    slices = tuple(slice(3, -2) for _ in range(fft_dim))
    if batched == "left":
        shape = (8,) + shape
        axes = tuple(range(1, fft_dim + 1))
        slices = (slice(None),) + slices
    elif batched == "right":
        shape = shape + (16,)
        axes = tuple(-i for i in range(2, fft_dim + 2))
        slices = slices + (slice(None),)
    else:
        assert batched == "no"
        axes = None
    
    if should_skip_3d_unsupported(shape, axes):
        pytest.skip("Pre 11.4.2 CTK does not support 3D batched FFT")

    signal = get_random_input_data(framework, shape, dtype, backend, seed=105)
    signal_sliced = signal[slices]
    assert get_array_strides(signal) == get_array_strides(signal_sliced)
    assert signal.dtype == signal_sliced.dtype

    if is_complex(dtype):
        fft = nvmath.fft.fft(signal_sliced, axes=axes)
    else:
        fft = nvmath.fft.rfft(signal_sliced, axes=axes)
    assert_array_type(fft, framework, backend, get_fft_dtype(dtype))
    assert_norm_close(fft, get_fft_ref(signal_sliced, axes=axes), axes=axes)


@pytest.mark.parametrize(
    (
        "framework",
        "fft_dim",
        "dtype",
    ),
    [
        (framework, fft_dim, dtype)
        for framework in Framework.enabled()
        if Backend.gpu in framework_backend_support[framework]
        for fft_dim in [1, 2, 3]
        for dtype in framework_type_support[framework]
        if not is_complex(dtype) and not is_half(dtype)
    ],
)
def test_sliced_tensor_unsupported(framework, fft_dim, dtype):
    backend = Backend.gpu
    shape = (64,) * fft_dim  # note, the sliced shape will not be power of 2 (59)

    if should_skip_3d_unsupported(shape):
        pytest.skip("Pre 11.4.2 CTK does not support 3D batched FFT")

    signal = get_random_input_data(framework, shape, dtype, backend, seed=105)
    signal_sliced = signal[tuple(slice(3, -2) for _ in shape)]
    assert get_array_strides(signal) == get_array_strides(signal_sliced)
    assert signal.dtype == signal_sliced.dtype

    # TODO Is this documented?
    with pytest.raises(nvmath.bindings.cufft.cuFFTError, match="CUFFT_INVALID_VALUE"):
        nvmath.fft.rfft(signal_sliced)


@pytest.mark.parametrize(
    (
        "framework",
        "fft_dim",
        "dtype",
    ),
    [
        (framework, fft_dim, dtype)
        for framework in Framework.enabled()
        if Backend.gpu in framework_backend_support[framework]
        for fft_dim in [2, 3]
        for dtype in framework_type_support[framework]
        if not is_complex(dtype)
    ],
)
def test_sliced_tensor_unsupported_reversed_axes(framework, fft_dim, dtype):
    backend = Backend.gpu
    shape = (64 + 5,) * fft_dim

    signal = get_random_input_data(framework, shape, dtype, backend, seed=105)
    if fft_dim == 2:
        signal_sliced = signal[2:-3, 3:-2]
        axes = [1, 0]
    else:
        assert fft_dim == 3
        signal_sliced = signal[3:-2, 3:-2, 2:-3]
        axes = [2, 1, 0]

    if should_skip_3d_unsupported(shape, axes):
        pytest.skip("Pre 11.4.2 CTK does not support 3D batched FFT")

    assert get_array_strides(signal) == get_array_strides(signal_sliced)
    assert signal.dtype == signal_sliced.dtype

    # TODO is this error expected or misleading?
    with pytest.raises(
        nvmath.fft.UnsupportedLayoutError, match="To convert to a supported layout"
    ):
        nvmath.fft.rfft(signal_sliced, axes=axes)


@pytest.mark.parametrize(
    (
        "framework",
        "backend",
        "batched",
        "dtype",
        "shape_kind",
    ),
    [
        (
            framework,
            backend,
            batched,
            dtype,
            rng.choice(type_shape_support[dtype]),
        )
        for framework in Framework.enabled()
        for backend in framework_backend_support[framework]
        for batched in ["no", "left", "right"]
        for dtype in framework_type_support[framework]
        if is_complex(dtype) and not is_half(dtype)
    ],
)
def test_permuted_tensor(framework, backend, batched, dtype, shape_kind):
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

    signal = get_random_input_data(framework, shape, dtype, backend, seed=105)
    signal_transposed = get_transposed(signal, *transpose_axes)
    assert not is_decreasing(get_array_strides(signal_transposed))
    assert signal.dtype == signal_transposed.dtype

    fft = nvmath.fft.fft(signal_transposed, axes=axes)
    assert_array_type(fft, framework, backend, get_fft_dtype(dtype))
    assert_norm_close(fft, get_fft_ref(signal_transposed, axes=axes), axes=axes)


@pytest.mark.parametrize(
    (
        "framework",
        "backend",
        "fft_dim",
        "batched",
        "batch_size",
        "dtype",
        "result_layout"
    ),
    [
        (
            framework,
            backend,
            fft_dim,
            batched,
            batch_size,
            dtype,
            result_layout,
        )
        for framework in Framework.enabled()
        for backend in framework_backend_support[framework]
        for fft_dim in [1, 2, 3]
        for batched, batch_size in [
            ("no", None),
        ]
        + [
            (side, 2**batch_size_log)
            for side in ("left", "right")
            for batch_size_log in (0, 1, 2, 3, 10)
        ]
        for dtype in framework_type_support[framework]
        for result_layout in OptFftLayout
    ],
)
def test_single_element(
    framework, backend, fft_dim, batched, batch_size, dtype, result_layout
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

    if should_skip_3d_unsupported(shape, axes):
        pytest.skip("Pre 11.4.2 CTK does not support 3D batched FFT")

    signal = get_random_input_data(framework, shape, dtype, backend, seed=105)
    fft_fn = nvmath.fft.fft if is_complex(dtype) else nvmath.fft.rfft

    if (
        not is_complex(dtype)
        and is_half(dtype)
        and batched == "right"
        and batch_size > 1
    ):
        with pytest.raises(ValueError, match="R2C FFT of half-precision tensor"):
            fft_fn(signal, axes=axes, options={"result_layout": result_layout.value})
        return

    if is_half(dtype) and get_cufft_version() < 10702:
        with pytest.raises(ValueError, match="sample size 1 and half-precision type"):
            fft_fn(signal, axes=axes, options={"result_layout": result_layout.value})
        return

    fft = fft_fn(signal, axes=axes, options={"result_layout": result_layout.value})
    assert_array_type(fft, framework, backend, get_fft_dtype(dtype))
    assert_norm_close(fft, get_fft_ref(signal, axes=axes), axes=axes)

    ifft_fn = nvmath.fft.ifft if is_complex(dtype) else nvmath.fft.irfft
    ifft = ifft_fn(
        fft,
        axes=axes,
        options={"last_axis_size": "odd", "result_layout": result_layout.value},
    )
    assert_norm_close(ifft, signal)


@pytest.mark.parametrize(
    (
        "framework",
        "backend",
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
            backend,
            repr(sample_shape),
            batched,
            batch_size,
            batch_step,
            dtype,
            result_layout,
        )
        for framework in Framework.enabled()
        for backend in framework_backend_support[framework]
        for sample_shape in [(64,), (16, 16), (4, 4, 4)]
        for batched, batch_size in [
            ("no", None),
        ]
        + [
            (side, extent)
            for side in ("left", "right")
            for extent in (4, 1024)
        ]
        for batch_step in (None, 2)
        if batch_step is None or (batch_size is not None and batch_size > batch_step)
        for dtype in framework_type_support[framework]
        for result_layout in OptFftLayout
    ],
)
def test_single_element_view(
    framework,
    backend,
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
    
    if should_skip_3d_unsupported(shape, axes):
        pytest.skip("Pre 11.4.2 CTK does not support 3D batched FFT")

    signal = get_random_input_data(framework, shape, dtype, backend, seed=105)
    signal_view = signal[
        tuple(
            (slice(None, None, batch_step) if axes and axis not in axes else slice(1))
            for axis in range(len(shape))
        )
    ]
    actual_batch_size = (batch_size or 1) // (batch_step or 1)
    assert math.prod(signal_view.shape) == actual_batch_size
    fft_fn = nvmath.fft.fft if is_complex(dtype) else nvmath.fft.rfft

    if (
        not is_complex(dtype)
        and is_half(dtype)
        and batched == "right"
        and actual_batch_size > 1
    ):
        with pytest.raises(ValueError, match="R2C FFT of half-precision tensor"):
            fft_fn(signal_view, axes=axes, options={"result_layout": result_layout.value})
        return

    if is_half(dtype) and get_cufft_version() < 10702:
        with pytest.raises(ValueError, match="sample size 1 and half-precision type"):
            fft_fn(signal_view, axes=axes, options={"result_layout": result_layout.value})
        return

    fft = fft_fn(signal_view, axes=axes, options={"result_layout": result_layout.value})
    assert_array_type(fft, framework, backend, get_fft_dtype(dtype))
    assert_norm_close(fft, get_fft_ref(signal_view, axes=axes), axes=axes)

    ifft_fn = nvmath.fft.ifft if is_complex(dtype) else nvmath.fft.irfft
    ifft = ifft_fn(
        fft,
        axes=axes,
        options={"last_axis_size": "odd", "result_layout": result_layout.value},
    )
    assert_norm_close(ifft, signal_view)


@pytest.mark.parametrize(
    (
        "framework",
        "backend",
        "shape",
        "axes",
        "slices",
        "dtype",
        "result_layout",
    ),
    [
        (
            framework,
            backend,
            repr(shape),
            repr(axes),
            repr(slices),
            dtype,
            result_layout,
        )
        for framework in Framework.enabled()
        for backend in framework_backend_support[framework]
        for shape, axes, slices in 
        [
            ((39, 2,), (0,), ((16,), (None,))),
            ((39, 7,), (0,), ((16,), (5,),)),
            ((39, 2,), (0,), ((16, 32), (None,))),
            ((39, 2,), (0,), ((16, 32, 2), (None,))),
            ((39, 7,), (0,), ((16, 32, 2), (5,),)),
            ((17, 17,), (1,), ((1, 17, 2), (16,),)),
            ((17, 13, 3), (0, 1), ((8,), (1, 9), (1, 3))),
            ((3, 17, 13), (1, 2), ((1, 3), (8,), (1, 9))),
            ((13, 17, 13), (0, 1, 2), ((1, 9), (8,), (1, 9)))
        ]
        for dtype in framework_type_support[framework]
        if is_complex(dtype)
        for result_layout in OptFftLayout
    ],
)
def test_inplace_view(
    framework,
    backend,
    shape,
    axes,
    slices,
    dtype,
    result_layout,
):
    shape = literal_eval(shape)
    axes = literal_eval(axes)

    if should_skip_3d_unsupported(shape, axes):
        pytest.skip("Pre 11.4.2 CTK does not support 3D batched FFT")

    slices = tuple(slice(*args) for args in literal_eval(slices))
    signal = get_random_input_data(framework, shape, dtype, backend, seed=105)
    signal_copy = copy_array(signal)
    signal_view = signal[slices]
    signal_view_strides = get_array_strides(signal_view)
    try:
        fft = nvmath.fft.fft(signal_view, axes=axes, options={"inplace": True, "result_layout": result_layout.value})
        assert_eq(get_array_strides(signal_view), signal_view_strides)
        assert_eq(get_array_strides(fft), signal_view_strides)
        assert_array_type(signal_view, framework, backend, dtype)
        ref_view = get_fft_ref(signal_copy[slices], axes=axes)
        assert_norm_close(signal_view, ref_view, axes=axes)
        signal_ref = copy_array(signal_copy)
        signal_ref[slices] = ref_view
        assert_norm_close(signal, signal_ref, axes=axes)
        ifft = nvmath.fft.ifft(signal_view, axes=axes, options={"inplace": True, "result_layout": result_layout.value})
        assert_eq(get_array_strides(signal_view), signal_view_strides)
        assert_eq(get_array_strides(ifft), signal_view_strides)
        signal_copy[slices] = get_scaled(signal_copy[slices], math.prod(signal_view.shape[a] for a in range(len(shape)) if a in axes))
        assert_norm_close(signal, signal_copy, axes=axes)
    except RuntimeError as e:
        assert "cannot be specified when copying to non-contiguous" in str(e)
        assert_eq(backend, Backend.cpu)
        assert_eq(framework, Framework.numpy)


@pytest.mark.parametrize(
    (
        "framework",
        "backend",
        "shape",
        "axes",
        "unfold_args",
        "dtype",
        "result_layout",
    ),
    [
        (
            framework,
            backend,
            repr(shape),
            repr(axes),
            repr(unfold_args),
            dtype,
            result_layout,
        )
        for framework in Framework.enabled()
        for backend in framework_backend_support[framework]
        for shape, axes, unfold_args in 
        [
            ((128,), (0,), (0, 8, 1)),
            ((128,), (0,), (0, 8, 7)),
            ((128,), (0,), (0, 8, 8)),
            ((128,), (0,), (0, 8, 16)),
            ((128,), (1,), (0, 8, 1)),
            ((128,), (1,), (0, 8, 7)),
            ((128,), (1,), (0, 8, 8)),
            ((128,), (1,), (0, 8, 16)),
            ((128,), (0, 1,), (0, 8, 1)),
            ((128,), (0, 1,), (0, 8, 7)),
            ((128,), (0, 1,), (0, 8, 8)),
            ((128,), (0, 1,), (0, 8, 16)),
            ((32, 32,), (0, 1, 2,), (0, 4, 2)),
            ((32, 32,), (0, 1, 2,), (0, 4, 4)),
            ((32, 32,), (0, 1, 2,), (0, 4, 8)),
            ((32, 32,), (0, 1, 2,), (1, 4, 2)),
            ((32, 32,), (0, 1, 2,), (1, 4, 4)),
            ((32, 32,), (0, 1, 2,), (1, 4, 8)),
        ]
        for dtype in framework_type_support[framework]
        if is_complex(dtype)
        for result_layout in OptFftLayout
    ],
)
def test_inplace_overlapping(
    framework,
    backend,
    shape,
    axes,
    unfold_args,
    dtype,
    result_layout,):
    shape = literal_eval(shape)
    axes = literal_eval(axes)
    unfold_args = literal_eval(unfold_args)
    unfold_dim, unfold_window_size, unfold_step = unfold_args
    signal = get_random_input_data(framework, shape, dtype, backend, seed=105)
    signal_view = unfold(signal, unfold_dim, unfold_window_size, unfold_step)
    signal_view_copy = copy_array(signal_view)

    if should_skip_3d_unsupported(shape, axes):
        pytest.skip("Pre 11.4.2 CTK does not support 3D batched FFT")
    
    if unfold_window_size > unfold_step:
        with pytest.raises(ValueError, match="overlaps in memory"):
            nvmath.fft.fft(signal_view, axes=axes, options={"inplace": True, "result_layout": result_layout.value})
        
        with pytest.raises(ValueError, match="overlaps in memory"):
            nvmath.fft.ifft(signal_view, axes=axes, options={"inplace": True, "result_layout": result_layout.value})
    else:
        try:
            nvmath.fft.fft(signal_view, axes=axes, options={"inplace": True, "result_layout": result_layout.value})
            assert_norm_close(signal_view, get_fft_ref(signal_view_copy, axes=axes), axes=axes)
        except RuntimeError as e:
            assert "cannot be specified when copying to non-contiguous" in str(e)
            assert_eq(backend, Backend.cpu)
            assert_eq(framework, Framework.numpy)


@pytest.mark.parametrize(
    (
        "framework",
        "backend",
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
            backend,
            repr(shape),
            repr(axes),
            stride,
            dtype,
            result_layout,
            inplace,
        )
        for framework in Framework.enabled()
        for backend in framework_backend_support[framework]
        for shape, axes, stride in
        [
            ((128, 1, 1), (1,), 1),
            ((128, 1, 1), (1,), 2),
            ((128, 1, 1), (1,), 4),
            ((1, 1, 128), (1,), 1),
            ((1, 1, 128), (1,), 2),
            ((1, 1, 128), (1,), 4),
            ((128, 1, 1), (1, 2,), 1),
            ((128, 1, 1), (1, 2,), 2),
            ((128, 1, 1), (1, 2,), 4),
            ((1, 1, 1), (1, 2,), 2),
            ((1, 1, 1), (0, 1,), 2),
            ((128, 1, 1, 1), (1, 2,), 1),
            ((128, 1, 1, 1), (1, 2,), 2),
            ((128, 1, 1, 1), (1, 2,), 4),
            ((128, 1, 1, 1), (1, 2, 3), 1),
            ((128, 1, 1, 1), (1, 2, 3), 2),
            ((128, 1, 1, 1), (1, 2, 3), 4),
        ]
        for dtype in framework_type_support[framework]
        if not is_half(dtype)
        for result_layout in OptFftLayout
        for inplace in [True, False]
        if not inplace or is_complex(dtype)
    ],
)
def test_repeated_strides_strided(
    framework,
    backend,
    shape,
    axes,
    stride,
    dtype,
    result_layout,
    inplace,
):
    shape = literal_eval(shape)
    axes = literal_eval(axes)

    if should_skip_3d_unsupported(shape, axes):
        pytest.skip("Pre 11.4.2 CTK does not support 3D batched FFT")

    vol = math.prod(shape) * stride
    signal = get_random_input_data(framework, (vol,), dtype, backend, seed=105)
    signal = signal[::stride].reshape(shape)
    signal_copy = copy_array(signal)
    fft_fn = nvmath.fft.fft if is_complex(dtype) else nvmath.fft.rfft
    try:
        fft = fft_fn(signal, axes=axes, options={"result_layout": result_layout.value, "inplace": inplace})
        assert_array_type(fft, framework, backend, get_fft_dtype(dtype))
        assert_norm_close(fft, get_fft_ref(signal_copy, axes=axes), axes=axes)
    except RuntimeError as e:
        assert "cannot be specified when copying to non-contiguous" in str(e)
        assert_eq(backend, Backend.cpu)
        assert_eq(framework, Framework.numpy)


@pytest.mark.parametrize(
    (
        "framework",
        "backend",
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
            backend,
            repr(shape),
            repr(strides),
            repr(axes),
            dtype,
            result_layout,
            inplace,
        )
        for framework in Framework.enabled()
        for backend in framework_backend_support[framework]
        for shape, strides, axes in
        [
            # non-tilable batch
            ((4, 4, 4, 4), (1, 1, 4, 4), (3, 1)),
            ((8, 8, 8, 8, 8), (1, 8, 8, 64, 64), (2, 3, 0)),
            # non-tialbe batch, correct output embedding
            ((4, 1, 4, 4), (1, 4, 4, 4), (0, 2,)),
            # wrong output embedding
            ((8, 4, 2, 4), (8, 2, 1, 8), (0, 1, 2)),
            ((5, 4, 8, 16), (8, 1, 4, 32), (1, 2, 3)),
            ((5, 4, 8, 16), (8, 2, 8, 64), (1, 2, 3)),
            ((5, 4, 8, 16), (16, 2, 16, 128), (1, 2, 3)),
            ((4, 8, 5), (1, 8, 1), (0, 1)),
            ((4, 8, 5), (4, 8, 4), (0, 1)),
            ((4, 8, 5), (4, 32, 4), (0, 1))
        ]
        for dtype in framework_type_support[framework]
        for result_layout in OptFftLayout
        for inplace in [True, False]
        if not inplace or is_complex(dtype)
    ],
)
def test_overlapping_non_tilable_output(
        framework,
        backend,
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

    if should_skip_3d_unsupported(shape, axes):
        pytest.skip("Pre 11.4.2 CTK does not support 3D batched FFT")

    max_offset = sum((extent - 1) * stride for extent, stride in zip(shape, strides))
    signal = get_random_input_data(framework, (max_offset + 1,), dtype, backend, seed=105)
    signal_view = as_strided(signal, shape, strides)
    signal_view_copy = copy_array(signal_view)
    fft_fn = nvmath.fft.fft if is_complex(dtype) else nvmath.fft.rfft
    try:
        fft = fft_fn(
            signal_view,
            axes=axes,
            options={"result_layout": result_layout.value, "inplace": inplace},
        )
    except nvmath.fft.UnsupportedLayoutError as e:
        data_transposed = get_permuted_copy(signal_view, e.permutation)
        assert is_decreasing(get_array_strides(data_transposed))
        res_transposed = fft_fn(
            data_transposed,
            axes=e.axes,
            options={"result_layout": result_layout.value, "inplace": inplace},
        )
        fft = get_permuted_copy(res_transposed, get_rev_perm(e.permutation))
    except ValueError as e:
        assert "overlaps in memory" in str(e)
        assert inplace
        return

    assert_array_type(fft, framework, backend, get_fft_dtype(dtype))
    assert_norm_close(fft, get_fft_ref(signal_view_copy, axes=axes), axes=axes)


@pytest.mark.parametrize(
    ("framework", "backend", "array_dim", "fft_dim", "dtype"),
    [
        (
            framework,
            backend,
            array_dim,
            fft_dim,
            rng.choice(framework_type_support[framework]),
        )
        for framework in Framework.enabled()
        for backend in framework_backend_support[framework]
        for array_dim in [10, 20, 32]
        for fft_dim in [1, 2, 3]
    ],
)
def test_high_dim_array(framework, backend, array_dim, fft_dim, dtype):
    if array_dim <= 20:
        shape = (2,) * array_dim
    else:
        shape = tuple(2 if i >= array_dim - fft_dim else 1 for i in range(array_dim))
    signal = get_random_input_data(framework, shape, dtype, backend, seed=444)
    axes = tuple(range(array_dim - fft_dim, array_dim))
    
    if should_skip_3d_unsupported(shape, axes):
        pytest.skip("Pre 11.4.2 CTK does not support 3D batched FFT")

    if is_complex(dtype):
        sample_fft = nvmath.fft.fft(signal, axes=axes)
    else:
        sample_fft = nvmath.fft.rfft(signal, axes=axes)
    fft_ref = get_fft_ref(signal, axes=axes)
    assert_norm_close(sample_fft, fft_ref, axes=axes)


@pytest.mark.parametrize(
    ("framework", "backend", "fft_dim", "dtype"),
    [
        (framework, backend, fft_dim, rng.choice(framework_type_support[framework]))
        for framework in Framework.enabled()
        for backend in framework_backend_support[framework]
        for fft_dim in [2, 3]
    ],
)
def test_unsupported_axes_gaps(framework, backend, fft_dim, dtype):
    shape = (32,) * (fft_dim + 1)
    signal = get_random_input_data(framework, shape, dtype, backend, seed=444)
    axes = (0, fft_dim)
    example_transpose = tuple(range(1, fft_dim)) + (0, fft_dim)
    fft_fn = nvmath.fft.fft if is_complex(dtype) else nvmath.fft.rfft
    with pytest.raises(
        nvmath.fft.UnsupportedLayoutError,
        match=f"create a transposed view using transpose{re.escape(str(example_transpose))}",
    ):
        fft_fn(signal, axes=axes)


@pytest.mark.parametrize(
    ("framework", "backend", "fft_dim", "negative", "dtype"),
    [
        (
            framework,
            backend,
            fft_dim,
            negative,
            rng.choice(framework_type_support[framework]),
        )
        for framework in Framework.enabled()
        for backend in framework_backend_support[framework]
        for fft_dim in [1, 2, 3]
        for negative in [True, False]
    ],
)
def test_unsupported_axes_out_of_range(framework, backend, fft_dim, negative, dtype):
    shape = (32,) * (fft_dim - 1)
    signal = get_random_input_data(framework, shape, dtype, backend, seed=444)
    if not negative:
        axes = tuple(range(fft_dim))
    else:
        axes = (-fft_dim - 1, -fft_dim)
    fft_fn = nvmath.fft.fft if is_complex(dtype) else nvmath.fft.rfft
    if len(shape) == 0:
        with pytest.raises(ValueError, match="FFT does not support scalars."):
            fft_fn(signal, axes=axes)
    else:
        with pytest.raises(
            ValueError,
            match=f"{re.escape(str(axes))} are out of bounds for a {fft_dim - 1}-D tensor",
        ):
            fft_fn(signal, axes=axes)


@pytest.mark.parametrize(
    ("framework", "backend", "fft_dim", "use_axes", "dtype"),
    [
        (
            framework,
            backend,
            fft_dim,
            use_axes,
            rng.choice(framework_type_support[framework]),
        )
        for framework in Framework.enabled()
        for backend in framework_backend_support[framework]
        for fft_dim in [4, 5]
        for use_axes in [False, True]
    ],
)
def test_unsupported_fft_dim(framework, backend, fft_dim, use_axes, dtype):
    shape = (2,) * fft_dim
    signal = get_random_input_data(framework, shape, dtype, backend, seed=444)

    fft_fn = nvmath.fft.fft if is_complex(dtype) else nvmath.fft.rfft
    if not use_axes:
        with pytest.raises(
            ValueError,
            match=f"FFTs in number of dimensions > 3 is not supported",
        ):
            fft_fn(signal)
    else:
        with pytest.raises(
            ValueError,
            match=f"Only upto 3D FFTs are currently supported.",
        ):
            fft_fn(signal, axes=tuple(range(fft_dim)))
