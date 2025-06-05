# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import contextlib
import random
import math
from ast import literal_eval

import pytest

try:
    import cupy as cp
except ImportError:
    cp = None

import nvmath
import nvmath.bindings

from .utils.common_axes import (
    ExecBackend,
    MemBackend,
    Framework,
    DType,
    OptFftLayout,
    Direction,
    ShapeKind,
    OptFftBlocking,
    OptFftInplace,
    OptFftType,
    AllowToFail,
    LtoCallback,
)
from .utils.axes_utils import (
    is_complex,
    get_dtype_from_array,
    get_framework_from_array,
    get_array_backend,
    get_fft_dtype,
    get_ifft_dtype,
    get_framework_module,
)
from .utils.support_matrix import (
    lto_callback_supperted_types,
    supported_backends,
    opt_fft_type_direction_support,
    opt_fft_type_input_type_support,
    inplace_opt_ftt_type_support,
)
from .utils.input_fixtures import (
    get_random_input_data,
    get_custom_stream,
    get_primes_up_to,
    init_assert_exec_backend_specified,
    # pytest fixture is used but not detected by linter because of strange syntax
    fx_last_operand_layout,  # noqa: F401
)
from .utils.check_helpers import (
    add_in_place,
    get_fft_ref,
    get_ifft_ref,
    get_scaled,
    get_norm,
    get_permuted,
    get_abs,
    get_array_element_strides,
    permute_copy_like,
    r2c_shape,
    use_stream,
    assert_norm_close,
    assert_array_type,
    assert_eq,
    get_array_device_id,
    unfold,
    get_raw_ptr,
    to_gpu,
    copy_array,
    as_type,
    has_only_small_factors,
    get_default_tolerance,
    free_framework_pools,
)

assert_exec_backend_specified = init_assert_exec_backend_specified()


def get_tolerance(a, shape_kind=None):
    dtype = get_dtype_from_array(a)
    framework = get_framework_from_array(a)
    mem_backend = get_array_backend(a)
    rtol, atol = get_default_tolerance(dtype, shape_kind, exec_backend=ExecBackend.cufft)
    # LTO EA was observed to have slightly off outputs for float32
    # The torch CPU fft has bigger difference as well
    if (nvmath.bindings.cufft.get_version() < 11300 and dtype in [DType.float32, DType.complex64]) or (
        framework == Framework.torch and mem_backend == MemBackend.cpu
    ):
        rtol *= 1.2
    return {"rtol": rtol, "atol": atol}


def assert_norm_close_check_constant(a, a_ref, rtol=None, atol=None, axes=None, shape_kind=None):
    """
    A number of tests in this module use scaling callbacks.
    This utility attempts to check if the mismatched output differs
    from the reference by a constant to facilitate catching cases
    when the root cause of the error is the callback being ignored.
    """
    if rtol is None and atol is None:
        _, rtol, atol = (tol := get_tolerance(a, shape_kind)), tol["rtol"], tol["atol"]
    try:
        assert_norm_close(a, a_ref, rtol, atol, axes, shape_kind)
    except AssertionError as e:
        if "are not norm-close" not in str(e):
            raise e
        a_norm = get_norm(a)
        a_ref_norm = get_norm(a_ref)
        factor = a_ref_norm / a_norm
        a_scaled = get_scaled(a, factor)
        try:
            assert_norm_close(a_scaled, a_ref, rtol, atol, axes, shape_kind)
        except AssertionError:
            raise e
        else:
            raise AssertionError(f"The outputs differ by a constant factor of {factor}") from e


def allow_to_fail_lto_ea_3d(e, shape, axes):
    if isinstance(e, ValueError) and "3D FFT with the last extent equal 1" in str(e):
        assert nvmath.bindings.cufft.get_version() < 11300
        fft_dim = len(axes) if axes is not None else len(shape)
        assert fft_dim == 3
        axes = axes or list(range(fft_dim))
        assert shape[axes[-1]] == 1
        assert sum(shape[a] == 1 for a in axes) == 1
        raise pytest.skip("cuFFT LTO EA 3D last extent 1 is not supported")


def allow_to_fail_compund_shape(e, shape, axes):
    if not has_only_small_factors(shape, axes):
        if nvmath.bindings.cufft.get_version() < 11300:
            if isinstance(e, ValueError) and "cuFFT LTO EA does not" in str(e):
                raise pytest.skip(f"NVMATH CHECK: Unsupported {shape} comprising primes larger than 127")
        else:
            if isinstance(e, nvmath.bindings.cufft.cuFFTError) and "CUFFT_NOT_SUPPORTED" in str(e):
                raise pytest.skip(f"CUFFT_UNSUPPORTED: Unsupported {shape} comprising primes larger than 127")
    raise


rng = random.Random(42)


def _has_numba():
    try:
        import numba  # noqa: F401

        return True
    except ModuleNotFoundError:
        return False


_has_dependencies = ExecBackend.cufft in supported_backends.exec and _has_numba()
_skip_cufft_jit_callback = not _has_dependencies or (
    nvmath.bindings._internal.cufft._inspect_function_pointer("__cufftXtSetJITCallback") == 0
)


def skip_unsupported_device(fn=None, dev_count=1, min_cc=70):
    def inner(fn):
        if not _has_dependencies:

            def test_skipped():
                pytest.skip("This test requires cufft, cupy, and numba")

            return test_skipped

        actual_dev_count = cp.cuda.runtime.getDeviceCount()

        if actual_dev_count < dev_count:

            def test_skipped():
                pytest.skip(f"Test requires at least {dev_count} gpus, got {actual_dev_count}")

            return test_skipped

        for d_id in range(dev_count):
            d = cp.cuda.Device(d_id)
            cc = d.compute_capability
            assert isinstance(cc, str) and len(cc) >= 2
            cc = int(cc)
            if cc < min_cc:

                def test_skipped():
                    pytest.skip(f"Test requires device {d_id} with comp cap at least {min_cc}, got {cc}")

                return test_skipped

        return fn

    if fn is None:
        return inner
    else:
        return inner(fn)


def skip_if_lto_unssuported(fn):
    def test_skipped():
        if not _has_dependencies:
            pytest.skip("No cufft, cupy, or numba was found")
        else:
            version = nvmath.bindings.cufft.get_version()
            pytest.skip(f"cuFFT ({version}) does not support LTO")

    if _skip_cufft_jit_callback:
        return test_skipped
    else:
        return fn


if _skip_cufft_jit_callback and _has_dependencies:

    @pytest.mark.parametrize(("callbacks",), [(callbacks,) for callbacks in LtoCallback])
    def test_error_unsupported(callbacks):
        signal = get_random_input_data(Framework.numpy, (16,), DType.complex64, MemBackend.cpu, seed=42)

        cb_kwargs = {
            cb: {"ltoir": b""}
            for cb, should_use in (
                ("prolog", callbacks.has_prolog()),
                ("epilog", callbacks.has_epilog()),
            )
            if should_use
        }

        version = nvmath.bindings.cufft.get_version()
        with pytest.raises(RuntimeError, match=f"cuFFT version {version} does not support LTO"):
            nvmath.fft.fft(signal, **cb_kwargs, execution="cuda")


@skip_if_lto_unssuported
@skip_unsupported_device
@pytest.mark.parametrize(
    (
        "shape_kind",
        "shape",
        "batch",
        "batch_size",
        "framework",
        "exec_backend",
        "mem_backend",
        "inplace",
        "dtype",
        "result_layout",
        "fft_callbacks",
        "ifft_callbacks",
    ),
    [
        (
            shape_kind,
            repr(shape),
            batch := rng.choice(["batch_none", "batch_left", "batch_right"]),
            1 if batch == "batch_none" else rng.choice([1, 3, 4, 5, 7]),
            framework := rng.choice(list(Framework.enabled())),
            ExecBackend.cufft,
            rng.choice(supported_backends.framework_mem[framework]),
            OptFftInplace(False if not is_complex(dtype) else rng.choice([False, True])),
            dtype,
            rng.choice(list(OptFftLayout)),
            rng.choice(list(LtoCallback)),
            rng.choice(list(LtoCallback)),
        )
        for shape_kind, shape in
        # --- cases with input shapes comprasing only <= 127 factors ---
        # 1D pow2, pow2357 sample
        [
            (ShapeKind.pow2, (1,)),
            (ShapeKind.pow2, (32,)),
            (ShapeKind.pow2357, (210,)),  # 2 * 3 * 5 * 7,
            # the fft shape factors into < 127 primes but
            # ifft shape has primes > 127,
            (ShapeKind.pow2, (512,)),
            (ShapeKind.pow2, (4096,)),
            (ShapeKind.pow2, (8192,)),
            (ShapeKind.pow2, (16384,)),
            (ShapeKind.pow2, (65536,)),
            (ShapeKind.pow2, (131072,)),
            (ShapeKind.pow2, (262144,)),
            # fft 44100=(2*3*5*7)**2 -> ifft 22051 that is prime
            (ShapeKind.pow2357, (44100,)),
        ]
        # 1D sample, prime sizes smaller than 2048
        + [(ShapeKind.prime, (p,)) for p in rng.sample([p for p in get_primes_up_to(2048) if p > 127], 15)]
        # 1D sample, random [2, 127 * 128] size * primes up to 127
        + [(ShapeKind.random, (rng.randint(1, 128) * p,)) for p in get_primes_up_to(127)]
        # 2D sample
        + [
            (ShapeKind.pow2, (1, 1)),
            (ShapeKind.pow2, (128, 1)),
            (ShapeKind.pow2, (1, 256)),
            (ShapeKind.pow2, (512, 128)),
            (ShapeKind.pow2, (256, 8192)),  # the ifft comprises primes > 127
            (ShapeKind.prime, (1951, 1091)),
            (ShapeKind.prime, (1951, 2039)),
            (ShapeKind.prime, (127, 181)),
            (ShapeKind.random, (9090, 13)),  # (2 * 9 * 5 * 101, 13)
            (ShapeKind.random, (512, 1063)),  # (512, prime)
            (ShapeKind.pow2357, (3150, 1470)),  # (2*3*3*5*5*7, 2*3*5*7*7)
            (ShapeKind.random, (4, 23226)),  # fft 2*3*7*7*79 -> ifft 2*5807
        ]
        # 3D sample
        + [
            (ShapeKind.pow2, (1, 1, 1)),
            (ShapeKind.pow2, (256, 1, 1)),
            (ShapeKind.pow2, (1, 128, 1)),
            (ShapeKind.pow2, (1, 1, 512)),
            (ShapeKind.pow2, (512, 2, 128)),
            (ShapeKind.pow2, (16, 8192, 16)),
            (ShapeKind.pow2357, (125, 343, 27)),  # (5**3, 7**3, 3**3)
            (ShapeKind.prime, (439, 83, 449)),
            (ShapeKind.prime, (127, 131, 137)),
            (ShapeKind.prime, (2017, 3, 2027)),
            (ShapeKind.random, (5, 3, 14993)),  # 11*29*47 -> 3*3*7*7*17
            (ShapeKind.random, (3, 1, 19998)),  # 2*3**2*11*101 -> 10**4
            (ShapeKind.random, (3, 1, 11613)),  # fft 3*7*7*79 -> ifft 5807
        ]
        # --- cases with input shapes comprasing some > 127 factors ---
        # 1D sample, compound shape with prime factor in [127, 200]
        + [(ShapeKind.random, (32 * p,)) for p in get_primes_up_to(200) if p > 127]
        # 1D sample, compound shape with prime factor in [1024, 1100]
        + [(ShapeKind.random, (rng.randint(4, 13) * p,)) for p in get_primes_up_to(1100) if p > 1024]
        # 1D sample, prime shape in [4096, 10000]
        + rng.sample(
            [(ShapeKind.prime, (p,)) for p in get_primes_up_to(10000) if p > 4096],
            10,
        )
        # ND samples, with prime factors > 127
        + [
            (ShapeKind.prime, (4111, 1103)),  # (prime, prime)
            (ShapeKind.prime, (11, 2797)),  # (prime, prime)
            (ShapeKind.random, (3067, 16)),  # (prime, 16)
            (ShapeKind.random, (7, 6163)),  # (7, 419 * 167)
            (ShapeKind.random, (13408, 32)),  # (419 * 32, 32)
            (ShapeKind.prime, (13, 11, 3001)),  # (prime, prime, prime)
            (ShapeKind.random, (8731, 16, 32)),  # (prime, 16, 32)
            (ShapeKind.prime, (5, 6701, 7)),  # (prime, prime, prime)
            (ShapeKind.random, (25, 9, 4813)),  # (25, 9, prime)
            (ShapeKind.prime, (2, 1, 5261)),  # (2, 1, prime)
        ]
        for dtype in [
            rng.choice([dt for dt in lto_callback_supperted_types if is_complex(dt)]),
            rng.choice([dt for dt in lto_callback_supperted_types if not is_complex(dt)]),
        ]
        if ExecBackend.cufft in supported_backends.exec
    ],
)
def test_operand_shape_fft_ifft(
    shape_kind,
    shape,
    batch,
    batch_size,
    framework,
    exec_backend,
    mem_backend,
    inplace,
    dtype,
    result_layout,
    fft_callbacks,
    ifft_callbacks,
):
    free_framework_pools(framework, mem_backend)

    shape = literal_eval(shape)
    axes = None if batch == "batch_none" else tuple(range(len(shape)))

    if batch == "batch_left":
        shape = (batch_size,) + shape
        axes = tuple(a + 1 for a in axes)
    elif batch == "batch_right":
        shape = shape + (batch_size,)
    else:
        assert batch == "batch_none"

    signal = get_random_input_data(framework, shape, dtype, mem_backend, seed=42)
    signal_copy = copy_array(signal) if inplace else signal

    def prolog_cb(data, offset, filter_data, unused):
        return data[offset] * 3

    def epilog_cb(data_out, offset, value, filter_data, unused):
        data_out[offset] = value * 5

    scaling = 1
    cb_kwargs = {}
    if fft_callbacks.has_prolog():
        prolog_ltoir = nvmath.fft.compile_prolog(prolog_cb, dtype.name, dtype.name)
        cb_kwargs["prolog"] = {"ltoir": prolog_ltoir}
        scaling *= 3

    if fft_callbacks.has_epilog():
        epilog_dtype = get_fft_dtype(dtype)
        epilog_ltoir = nvmath.fft.compile_epilog(epilog_cb, epilog_dtype.name, epilog_dtype.name)
        cb_kwargs["epilog"] = {"ltoir": epilog_ltoir}
        scaling *= 5
    # Test create_key() function
    try:
        key1 = nvmath.fft.FFT.create_key(
            signal,
            axes=axes,
            prolog=cb_kwargs["prolog"] if fft_callbacks.has_prolog() else None,
            epilog=cb_kwargs["epilog"] if fft_callbacks.has_epilog() else None,
        )
        assert key1 is not None
        key2 = nvmath.fft.FFT.create_key(
            signal,
            axes=axes,
            prolog=nvmath.fft.DeviceCallable(**cb_kwargs["prolog"]) if fft_callbacks.has_prolog() else None,
            epilog=nvmath.fft.DeviceCallable(**cb_kwargs["epilog"]) if fft_callbacks.has_epilog() else None,
        )
        assert key1 == key2
    except RuntimeError as e:
        if "The FFT CPU execution is not available" in str(e) and mem_backend == MemBackend.cpu:
            # Skip this check since create_key() function needs CPU FFT lib availability
            pass
        else:
            raise

    ref = get_fft_ref(get_scaled(signal, scaling), axes=axes)

    fft_fn = nvmath.fft.fft if is_complex(dtype) else nvmath.fft.rfft
    try:
        out = fft_fn(
            signal,
            axes=axes,
            execution=exec_backend.nvname,
            **cb_kwargs,
            options={
                "result_layout": result_layout.value,
                "inplace": inplace.value,
            },
        )
    except (nvmath.bindings.cufft.cuFFTError, ValueError) as e:
        allow_to_fail_compund_shape(e, shape, axes=axes)

    assert_norm_close_check_constant(out, ref, axes=axes, shape_kind=shape_kind)

    def iprolog_cb(data, offset, filter_data, unused):
        return data[offset] * 2

    def iepilog_cb(data_out, offset, value, filter_data, unused):
        data_out[offset] = value * 7

    icb_kwargs = {}
    iscaling = 1
    if ifft_callbacks.has_prolog():
        iprolog_dtype = get_fft_dtype(dtype)
        iprolog_ltoir = nvmath.fft.compile_prolog(iprolog_cb, iprolog_dtype.name, iprolog_dtype.name)
        icb_kwargs["prolog"] = {"ltoir": iprolog_ltoir}
        iscaling *= 2

    if ifft_callbacks.has_epilog():
        iepilog_ltoir = nvmath.fft.compile_epilog(iepilog_cb, dtype.name, dtype.name)
        icb_kwargs["epilog"] = {"ltoir": iepilog_ltoir}
        iscaling *= 7

    ifft_fn = nvmath.fft.ifft if is_complex(dtype) else nvmath.fft.irfft
    last_extent = shape[-1] if axes is None else shape[axes[-1]]

    try:
        iout = ifft_fn(
            out,
            axes=axes,
            execution=exec_backend.nvname,
            **icb_kwargs,
            options={
                "result_layout": result_layout.value,
                "inplace": inplace.value,
                "last_axis_parity": "odd" if last_extent % 2 == 1 else "even",
            },
        )
    except (nvmath.bindings.cufft.cuFFTError, ValueError) as e:
        allow_to_fail_compund_shape(e, shape, axes=axes)

    fft_scale = math.prod(shape[a] for a in (axes or range(len(shape))))
    signal_scaled = get_scaled(signal_copy, fft_scale * scaling * iscaling)
    assert_norm_close_check_constant(iout, signal_scaled, axes=axes, shape_kind=shape_kind)


@skip_if_lto_unssuported
@skip_unsupported_device
@pytest.mark.parametrize(
    (
        "shape",
        "axes",
        "framework",
        "exec_backend",
        "mem_backend",
        "dtype",
        "result_layout",
        "callbacks",
    ),
    [
        (
            repr(shape),
            repr(axes),
            framework := rng.choice(list(Framework.enabled())),
            ExecBackend.cufft,
            rng.choice(supported_backends.framework_mem[framework]),
            dtype,
            rng.choice(list(OptFftLayout)),
            callbacks,
        )
        for shape, axes in
        # test shapes such that `shape[last_axis]//2 + 1` has only < 127 factors
        # but the `shape` has some bigger ones
        [
            ((4971, 3), (0,)),
            ((4981, 5), (0,)),
            ((4983, 5), (0,)),
            ((3, 21910), (0, 1)),
            ((3, 21911), (0, 1)),
            ((3, 7, 4978), (2,)),
            ((3, 7, 4979), (2,)),
            ((3, 5, 21928), (1, 2)),
            ((1, 2, 21929), (0, 1, 2)),
        ]
        for dtype in lto_callback_supperted_types
        if not is_complex(dtype)
        for callbacks in LtoCallback
        if ExecBackend.cufft in supported_backends.exec
    ],
)
def test_operand_shape_ifft_c2r(
    shape,
    axes,
    framework,
    exec_backend,
    mem_backend,
    dtype,
    result_layout,
    callbacks,
):
    free_framework_pools(framework, mem_backend)

    shape = literal_eval(shape)
    axes = literal_eval(axes)

    signal = get_random_input_data(framework, shape, dtype, mem_backend, seed=42)
    fft_in = copy_array(get_fft_ref(signal, axes=axes))
    fft_dtype = get_fft_dtype(dtype)

    def prolog_cb(data, offset, filter_data, unused):
        return data[offset] * 3

    def epilog_cb(data_out, offset, value, filter_data, unused):
        data_out[offset] = value * 5

    scaling = 1
    cb_kwargs = {}
    if callbacks.has_prolog():
        prolog_ltoir = nvmath.fft.compile_prolog(prolog_cb, fft_dtype.name, fft_dtype.name)
        cb_kwargs["prolog"] = {"ltoir": prolog_ltoir}
        scaling *= 3

    if callbacks.has_epilog():
        epilog_ltoir = nvmath.fft.compile_epilog(epilog_cb, dtype.name, dtype.name)
        cb_kwargs["epilog"] = {"ltoir": epilog_ltoir}
        scaling *= 5

    ifft_fn = nvmath.fft.ifft if is_complex(dtype) else nvmath.fft.irfft
    try:
        out = ifft_fn(
            fft_in,
            axes=axes,
            execution=exec_backend.nvname,
            **cb_kwargs,
            options={
                "result_layout": result_layout.value,
                "last_axis_parity": "odd" if shape[axes[-1]] % 2 == 1 else "even",
            },
        )
    except (nvmath.bindings.cufft.cuFFTError, ValueError) as e:
        # for inverse transform allow to fail based on the output shape
        allow_to_fail_compund_shape(e, shape, axes=axes)

    fft_scale = math.prod(shape[a] for a in axes)
    assert_norm_close_check_constant(out, get_scaled(signal, scaling * fft_scale), axes=axes)


@skip_if_lto_unssuported
@skip_unsupported_device
@pytest.mark.parametrize(
    (
        "shape_base",
        "shape_slice_start",
        "shape_slice_size",
        "allow_to_fail",
        "batch",
        "batch_size",
        "framework",
        "exec_backend",
        "mem_backend",
        "dtype",
        "result_layout",
        "callbacks",
    ),
    [
        (
            repr(shape_base),
            repr(shape_slice_start),
            repr(shape_slice_size),
            AllowToFail(allow_to_fail),
            batch := rng.choice(["batch_none", "batch_left", "batch_right"]),
            1 if batch == "batch_none" else rng.choice([1, 2, 3]),
            rng.choice([f for f in Framework.enabled() if mem_backend in supported_backends.framework_mem[f]]),
            ExecBackend.cufft,
            mem_backend,
            dtype,
            rng.choice(list(OptFftLayout)),
            rng.choice(list(LtoCallback)),
        )
        # fmt: off
        for allow_to_fail, shape_base, shape_slice_start, shape_slice_size in [
            # Supported cases
            (False, (8971,), (0,), (2029,)),  # prime embedding, smaller < 2048 prime shape
            (False, (8971,), (4111,), (2029,)),  # prime base, prime offset, smaller < 2048 shape
            (False, (8971,), (4111,), (4096,)),  # prime base, prime offset, pow2 shape
            (False, (4129,), (0,), (1050,)),  # prime embedding, pow2357 shape
            (False, (2048, 4096), (1, 2), (109, 113)),  # pow2 base, offset, small prime shape, 2D
            (False, (4111, 4127), (0, 0), (8, 4096)),  # prime embed, pow2 shape, 2D
            (False, (173, 53, 4099), (0, 0, 0), (4, 8, 4096)),  # prime embedding, pow2 shape, 3D
            (False, (173, 4127, 53), (3, 2, 1), (8, 3379, 8)),  # prime base, offset, 109*31 shape, 3D
            (False, (5179, 17, 227), (0, 0, 0), (5120, 4, 4)),  # prime embedding, 10*512 shape, 3D
            # Unsupported shapes, should error out clearly
            (True, (8192,), (0,), (4132,)),  # pow2 embedding, 1033 * 4 shape
            (True, (7350,), (0,), (5003,)),  # pow2357 embedding, prime shape
            (True, (7350, 1024), (1, 1), (4381, 12)),  # pow2357, pow2 embedding, 337 * 13 shape
            (True, (1024, 7350), (3, 3), (12, 4381)),  # pow2357, pow2 embedding, 337 * 13 shape
            (True, (8, 7350, 8), (3, 3, 3), (1, 4381, 4)),  # pow2357, pow2 embedding, 337 * 13 shape
        ]
        # fmt: on
        for dtype in [
            DType.complex128,  # it is the "hardest" case, plus one more for coverage
            rng.choice([s for s in lto_callback_supperted_types if s != DType.complex128]),
        ]
        if ExecBackend.cufft in supported_backends.exec
        for mem_backend in MemBackend
    ],
)
def test_sliced_operand(
    shape_base,
    shape_slice_start,
    shape_slice_size,
    allow_to_fail,
    batch,
    batch_size,
    framework,
    exec_backend,
    mem_backend,
    dtype,
    result_layout,
    callbacks,
):
    free_framework_pools(framework, mem_backend)

    shape_base = literal_eval(shape_base)
    shape_slice_start = literal_eval(shape_slice_start)
    shape_slice_size = literal_eval(shape_slice_size)
    assert len(shape_base) == len(shape_slice_start) == len(shape_slice_size)

    axes = None if batch == "batch_none" else tuple(range(len(shape_base)))

    if batch == "batch_left":
        shape_base = (batch_size,) + shape_base
        shape_slice_start = (0,) + shape_slice_start
        shape_slice_size = (batch_size,) + shape_slice_size
        axes = tuple(a + 1 for a in axes)
    elif batch == "batch_right":
        shape_base = shape_base + (batch_size,)
        shape_slice_start = shape_slice_start + (0,)
        shape_slice_size = shape_slice_size + (batch_size,)
    else:
        assert batch == "batch_none"

    signal_base = get_random_input_data(framework, shape_base, dtype, mem_backend, seed=43)
    signal = signal_base[tuple(slice(s, s + e) for s, e in zip(shape_slice_start, shape_slice_size, strict=True))]

    def prolog_cb(data, offset, filter_data, unused):
        return data[offset] * 7

    def epilog_cb(data_out, offset, value, filter_data, unused):
        data_out[offset] = value * 3

    scaling = 1
    cb_kwargs = {}
    if callbacks.has_prolog():
        prolog_ltoir = nvmath.fft.compile_prolog(prolog_cb, dtype.name, dtype.name)
        cb_kwargs["prolog"] = {"ltoir": prolog_ltoir}
        scaling *= 7

    if callbacks.has_epilog():
        epilog_dtype = get_fft_dtype(dtype)
        epilog_ltoir = nvmath.fft.compile_epilog(epilog_cb, epilog_dtype.name, epilog_dtype.name)
        cb_kwargs["epilog"] = {"ltoir": epilog_ltoir}
        scaling *= 3

    ref = get_fft_ref(get_scaled(signal, scaling), axes=axes)

    fft_fn = nvmath.fft.fft if is_complex(dtype) else nvmath.fft.rfft
    try:
        out = fft_fn(
            signal,
            axes=axes,
            execution=exec_backend.nvname,
            **cb_kwargs,
            options={"result_layout": result_layout.value},
        )
    except (nvmath.bindings.cufft.cuFFTError, ValueError) as e:
        if (
            isinstance(e, nvmath.bindings.cufft.cuFFTError)
            and not is_complex(dtype)
            and any(e % 2 != 0 for e in shape_slice_start)
            and "CUFFT_INVALID_VALUE" in str(e)
        ):
            raise pytest.skip("The case is allowed to fail because of alignment limitations in R2C")
        elif allow_to_fail:
            allow_to_fail_compund_shape(e, shape_slice_size, axes=axes)
        else:
            raise

    assert_norm_close_check_constant(out, ref, axes=axes)


@skip_if_lto_unssuported
@skip_unsupported_device
@pytest.mark.parametrize(
    (
        "framework",
        "exec_backend",
        "mem_backend",
        "allow_to_fail",
        "base_shape",
        "axes",
        "unfold_args",
        "dtype",
        "result_layout",
        "callbacks",
    ),
    [
        (
            rng.choice([f for f in Framework.enabled() if MemBackend.cuda in supported_backends.framework_mem[f]]),
            ExecBackend.cufft,
            MemBackend.cuda,  # cpu -> gpu may make the layout dense, no point to check it here
            AllowToFail(allow_to_fail),
            repr(base_shape),
            repr(axes),
            repr(unfold_args),
            dtype,
            rng.choice(list(OptFftLayout)),
            rng.choice(list(LtoCallback)),
        )
        for allow_to_fail, base_shape, axes, unfold_args in [
            (False, (128,), (0,), (0, 8, 1)),
            (False, (128,), (0,), (0, 8, 7)),
            (False, (128,), (0,), (0, 8, 8)),
            (False, (128,), (0,), (0, 8, 9)),
            (False, (128,), (0,), (0, 8, 16)),
            (False, (8933,), (0,), (0, 5450, 1)),  # 109 * 50
            (False, (8933,), (0,), (0, 5450, 127)),  # 109 * 50
            (False, (128,), (1,), (0, 8, 1)),
            (False, (128,), (1,), (0, 8, 7)),
            (False, (128,), (1,), (0, 8, 8)),
            (False, (128,), (1,), (0, 8, 9)),
            (False, (128,), (1,), (0, 8, 16)),
            (False, (8933,), (1,), (0, 5450, 1)),
            (False, (8933,), (1,), (0, 5450, 127)),
            (False, (128,), (0, 1), (0, 8, 1)),
            (False, (128,), (0, 1), (0, 8, 7)),
            (False, (128,), (0, 1), (0, 8, 8)),
            (False, (128,), (0, 1), (0, 8, 9)),
            (False, (128,), (0, 1), (0, 8, 16)),
            (False, (8933,), (1, 0), (0, 5450, 1)),
            (False, (8933,), (0, 1), (0, 5450, 127)),
            (True, (8192,), (1,), (0, 4748, 1)),  # unfolded sample size 4 * 1187,
            (True, (8192,), (1,), (0, 4351, 1)),  # unfolded sample size 229 * 19
            (True, (8933,), (0, 1), (0, 4351, 1)),  # unfolded sample size 229 * 19
        ]
        for dtype in [
            DType.complex128,  # it is the "hardest" case, plus one more for coverage
            rng.choice([s for s in lto_callback_supperted_types if s != DType.complex128]),
        ]
        if ExecBackend.cufft in supported_backends.exec
    ],
)
def test_overlapping_stride_operand(
    framework,
    exec_backend,
    mem_backend,
    allow_to_fail,
    base_shape,
    axes,
    unfold_args,
    dtype,
    result_layout,
    callbacks,
):
    base_shape = literal_eval(base_shape)
    axes = literal_eval(axes)
    unfold_args = literal_eval(unfold_args)
    unfold_dim, unfold_window_size, unfold_step = unfold_args
    signal_base = get_random_input_data(framework, base_shape, dtype, mem_backend, seed=105)
    signal = unfold(signal_base, unfold_dim, unfold_window_size, unfold_step)

    def prolog_cb(data, offset, filter_data, unused):
        return data[offset] * 7

    def epilog_cb(data_out, offset, value, filter_data, unused):
        data_out[offset] = value * 3

    scaling = 1
    cb_kwargs = {}
    if callbacks.has_prolog():
        prolog_ltoir = nvmath.fft.compile_prolog(prolog_cb, dtype.name, dtype.name)
        cb_kwargs["prolog"] = {"ltoir": prolog_ltoir}
        scaling *= 7

    if callbacks.has_epilog():
        epilog_dtype = get_fft_dtype(dtype)
        epilog_ltoir = nvmath.fft.compile_epilog(epilog_cb, epilog_dtype.name, epilog_dtype.name)
        cb_kwargs["epilog"] = {"ltoir": epilog_ltoir}
        scaling *= 3

    ref = get_fft_ref(get_scaled(signal, scaling), axes=axes)

    fft_fn = nvmath.fft.fft if is_complex(dtype) else nvmath.fft.rfft

    try:
        out = fft_fn(
            signal,
            axes=axes,
            execution=exec_backend.nvname,
            **cb_kwargs,
            options={"result_layout": result_layout.value},
        )
    except (nvmath.bindings.cufft.cuFFTError, ValueError) as e:
        if allow_to_fail:
            allow_to_fail_compund_shape(e, signal.shape, axes=axes)
        else:
            raise

    assert_norm_close_check_constant(out, ref, axes=axes)


@skip_if_lto_unssuported
@skip_unsupported_device
@pytest.mark.parametrize(
    (
        "framework",
        "exec_backend",
        "mem_backend",
        "allow_to_fail",
        "base_shape",
        "base_axes",
        "permutation",
        "inplace",
        "fft_type",
        "direction",
        "dtype",
        "result_layout",
        "callbacks",
    ),
    [
        (
            rng.choice(avail_frameworks),
            ExecBackend.cufft,
            MemBackend.cuda,  # cpu -> gpu may make the layout dense, no point to check it here
            AllowToFail(allow_to_fail),
            repr(base_shape),
            repr(base_axes),
            repr(permutation),
            inplace,
            fft_type,
            rng.choice(opt_fft_type_direction_support[fft_type]),
            dtype,
            OptFftLayout.natural if inplace else rng.choice(list(OptFftLayout)),
            rng.choice(
                list(LtoCallback),
            ),
        )
        for avail_frameworks in [[f for f in Framework.enabled() if MemBackend.cuda in supported_backends.framework_mem[f]]]
        if avail_frameworks
        # fmt: off
        for allow_to_fail, base_shape, base_axes, permutation in [
            (False, (128, 1), (0,), (0, 1)),  # 1D batched, pow2
            (False, (128, 1), (0,), (1, 0)),  # 1D batched, pow2
            (False, (1, 128), (1,), (1, 0)),  # 1D batched, pow2
            (False, (128, 4), (0,), (1, 0)),  # 1D batched, pow2
            (False, (4, 128), (1,), (1, 0)),  # 1D batched, pow2
            (False, (16269, 1), (0,), (1, 0)),  # 1D batched, 17*29*11*3
            (False, (1, 16269), (1,), (1, 0)),  # 1D batched, 17*29*11*3
            (False, (16269, 3), (0,), (1, 0)),  # 1D batched, 17*29*11*3
            (False, (3, 16269), (1,), (1, 0)),  # 1D batched, 17*29*11*3
            (False, (1, 2039), (1,), (1, 0)),  # 1D batched, prime < 2048
            (False, (2039, 1), (1,), (1, 0)),  # 1D batched, prime < 2048
            (False, (7, 2039), (1,), (1, 0)),  # 1D batched, prime < 2048
            (False, (2039, 7), (1,), (1, 0)),  # 1D batched, prime < 2048
            (False, (2039, 1), (0, 1), (0, 1)),  # 2D no-batch, prime < 2048
            (False, (2039, 1), (0, 1), (1, 0)),  # 2D no-batch, prime < 2048
            (False, (2039, 7), (0, 1), (1, 0)),  # 2D no-batch, prime < 2048
            (False, (2005, 7, 1), (0, 1), (2, 1, 0)),  # 2D batched, 401 * 5
            (False, (2005, 7, 3), (0, 1), (2, 1, 0)),  # 2D batched, 401 * 5
            (False, (2005, 7, 1, 3), (0, 1), (3, 2, 1, 0)),  # 2D batched, 401 * 5
            (False, (2005, 1, 1, 3), (0, 1), (3, 2, 1, 0)),  # 2D batched, 401 * 5
            (False, (1, 2005, 2017), (1, 2), (1, 2, 0)),  # 2D batched, 401 * 5
            (False, (3, 2005, 2017), (1, 2), (1, 2, 0)),  # 2D batched, 401 * 5
            (False, (3, 1, 2005, 2017), (2, 3), (3, 2, 1, 0)),  # 2D batched, 401 * 5
            (False, (1, 3, 2005, 1), (2, 3), (3, 2, 1, 0)),  # 2D batched, 401 * 5
            (False, (1, 3, 1, 1, 2005), (3, 4), (3, 4, 2, 0, 1)),  # 2D batched, 401 * 5
            (False, (1, 3, 1, 1, 2005), (3, 4), (4, 3, 2, 0, 1)),  # 2D batched, 401 * 5
            (False, (1, 3, 1, 17, 13, 2017), (3, 4, 5), (4, 3, 5, 0, 1, 2)),  # 3D batched, 401 * 5
            (False, (2017, 1, 31, 3, 1), (0, 1, 2), (3, 4, 2, 1, 0)),  # 3D batch, repeat strides
            (True, (4952, 3), (0,), (1, 0)),  # 1D batched, 8 * 619
            (True, (3, 4952), (1,), (1, 0)),  # 1D batched, 8 * 619
            (True, (3, 4812, 2017), (1, 2), (2, 1, 0)),  # 2D batched, 401 * 12
            (True, (16, 1, 4812, 3, 1), (0, 1, 2), (3, 4, 2, 1, 0)),  # 3D batch, 401 * 12
        ]
        # fmt: on
        for inplace in OptFftInplace
        for fft_type in inplace_opt_ftt_type_support[inplace.value]
        for dtype in opt_fft_type_input_type_support[fft_type]
        if dtype in lto_callback_supperted_types
    ],
)
def test_permuted_stride_operand(
    fx_last_operand_layout,  # noqa: F811
    framework,
    exec_backend,
    mem_backend,
    allow_to_fail,
    base_shape,
    base_axes,
    permutation,
    inplace,
    fft_type,
    direction,
    dtype,
    result_layout,
    callbacks,
):
    free_framework_pools(framework, mem_backend)

    base_shape = literal_eval(base_shape)
    base_axes = literal_eval(base_axes)
    permutation = literal_eval(permutation)
    axes = tuple(permutation.index(a) for a in base_axes)
    assert len(base_shape) == len(permutation)

    if fft_type != OptFftType.c2r:
        signal_base = get_random_input_data(framework, base_shape, dtype, mem_backend, seed=105)
        signal = get_permuted(signal_base, permutation)
        signal_shape = tuple(base_shape[p] for p in permutation)
        if fft_type == OptFftType.c2c:
            output_shape = signal_shape
        else:
            output_shape = r2c_shape(signal_shape, axes)
    else:
        real_type = get_ifft_dtype(dtype, fft_type)
        assert not is_complex(real_type)
        signal_base = get_random_input_data(framework, base_shape, real_type, mem_backend, seed=105)
        signal_base = copy_array(get_fft_ref(signal_base, axes=base_axes))
        signal = get_permuted(signal_base, permutation)
        signal_shape = list(base_shape)
        signal_shape[base_axes[-1]] = signal_shape[base_axes[-1]] // 2 + 1
        signal_shape = tuple(signal_shape[p] for p in permutation)
        output_shape = tuple(base_shape[p] for p in permutation)

    signal_copy = copy_array(signal) if inplace else signal
    assert signal.shape == signal_shape
    last_axis_parity = "odd" if output_shape[axes[-1]] % 2 else "even"

    check_layouts, *_ = fx_last_operand_layout

    if fft_type != OptFftType.c2r:
        prolog_filter = get_random_input_data(framework, signal_shape, dtype, MemBackend.cuda, seed=243)
    else:
        # assure the required symmetry in the input
        prolog_filter = get_random_input_data(framework, output_shape, real_type, mem_backend, seed=243)
        prolog_filter = copy_array(get_fft_ref(prolog_filter, axes=axes))
    assert get_dtype_from_array(prolog_filter) == dtype
    assert prolog_filter.shape == signal_shape

    if direction == Direction.forward:
        epilog_dtype = get_fft_dtype(dtype)
    else:
        epilog_dtype = get_ifft_dtype(dtype, fft_type)
    epilog_filter = get_random_input_data(framework, output_shape, epilog_dtype, MemBackend.cuda, seed=143)

    def prolog_cb(data, offset, filter_data, unused):
        return data[offset] * filter_data[offset]

    def epilog_cb(data_out, offset, value, filter_data, unused):
        data_out[offset] = value * filter_data[offset] + 7

    cb_kwargs = {}
    if callbacks.has_prolog():
        prolog_ltoir = nvmath.fft.compile_prolog(prolog_cb, dtype.name, dtype.name)
        cb_kwargs["prolog"] = {"ltoir": prolog_ltoir}

    if callbacks.has_epilog():
        epilog_ltoir = nvmath.fft.compile_epilog(epilog_cb, epilog_dtype.name, epilog_dtype.name)
        cb_kwargs["epilog"] = {"ltoir": epilog_ltoir}

    with nvmath.fft.FFT(
        signal,
        axes=axes,
        execution=exec_backend.nvname,
        options={
            "fft_type": fft_type.value,
            "result_layout": result_layout.value,
            "inplace": inplace.value,
            "last_axis_parity": last_axis_parity,
        },
    ) as fft:
        if callbacks.has_prolog():
            signal_strides = get_array_element_strides(signal)
            prolog_strides = get_array_element_strides(prolog_filter)
            operand_shape, operand_strides = fft.get_input_layout()
            assert operand_shape == signal.shape
            # even for c2r internal copy should keep the strides here
            assert operand_strides == signal_strides
            assert prolog_filter.shape == signal.shape
            if prolog_strides != operand_strides:
                prolog_data = permute_copy_like(prolog_filter, operand_shape, operand_strides)
                assert get_array_element_strides(prolog_data) == operand_strides
            else:
                assert all(s == 1 for s in base_shape[1:])
                prolog_data = prolog_filter
            cb_kwargs["prolog"]["data"] = get_raw_ptr(prolog_data)

        if callbacks.has_epilog():
            epilog_strides = get_array_element_strides(epilog_filter)
            res_shape, res_strides = fft.get_output_layout()
            assert res_shape == epilog_filter.shape
            if res_strides != epilog_strides:
                epilog_data = permute_copy_like(epilog_filter, res_shape, res_strides)
                assert get_array_element_strides(epilog_data) == res_strides
            else:
                epilog_data = epilog_filter
            cb_kwargs["epilog"]["data"] = get_raw_ptr(epilog_data)

        try:
            fft.plan(**cb_kwargs)
        except (nvmath.bindings.cufft.cuFFTError, ValueError) as e:
            if not allow_to_fail:
                raise
            problem_shape = signal_shape if fft_type != OptFftType.c2r else output_shape
            allow_to_fail_compund_shape(e, problem_shape, axes=axes)

        out = fft.execute(direction=direction.value)
        check_layouts(
            exec_backend,
            mem_backend,
            axes,
            result_layout,
            fft_type,
            is_dense=True,
            inplace=inplace.value,
        )
        fft_ref = signal_copy
        if callbacks.has_prolog():
            fft_ref = as_type(fft_ref * prolog_filter, dtype)
        if direction == Direction.forward:
            fft_ref = get_fft_ref(fft_ref, axes)
        else:
            fft_ref = get_ifft_ref(
                fft_ref,
                axes,
                is_c2c=fft_type == OptFftType.c2c,
                last_axis_parity=last_axis_parity,
            )
        if callbacks.has_epilog():
            fft_ref = as_type(fft_ref * epilog_filter + 7, epilog_dtype)
        if inplace:
            assert signal is out
        assert_norm_close_check_constant(out, fft_ref, axes=axes)


def _operand_filter_dtype_shape_fft_ifft_case(
    dtype,
    prolog_filter_dtype,
    epilog_filter_dtype,
    shape,
    axes,
    framework,
    exec_backend,
    mem_backend,
    inplace,
    allow_to_fail,
    result_layout,
):
    free_framework_pools(framework, mem_backend)

    shape = literal_eval(shape)
    axes = literal_eval(axes)
    last_axis_parity = "odd" if shape[axes[-1]] % 2 else "even"

    signal = get_random_input_data(framework, shape, dtype, mem_backend, seed=101)
    prolog_filter = get_random_input_data(framework, shape, prolog_filter_dtype, mem_backend, seed=243)
    if is_complex(dtype):
        epilog_filter = get_random_input_data(framework, shape, epilog_filter_dtype, mem_backend, seed=143)
    else:
        # make sure the data we multiply in the forward epilog/
        # inverse prolog have the required hermitian symmetry
        epilog_real_dtype = (
            epilog_filter_dtype if not is_complex(epilog_filter_dtype) else get_ifft_dtype(epilog_filter_dtype, OptFftType.c2r)
        )
        epilog_filter_base = get_random_input_data(framework, shape, epilog_real_dtype, mem_backend, seed=143)
        # copy array to make sure it is dense
        epilog_filter_complex = copy_array(get_fft_ref(epilog_filter_base, axes=axes))
        if is_complex(epilog_filter_dtype):
            epilog_filter = epilog_filter_complex
        else:
            epilog_filter = get_abs(epilog_filter_complex)
        assert get_dtype_from_array(epilog_filter) == epilog_filter_dtype

    def get_prolog(operand_dtype, filter_dtype, scalar):
        if not is_complex(filter_dtype) or is_complex(operand_dtype):

            def cb(data, offset, filter_data, unused):
                return data[offset] * filter_data[offset] * scalar

            def ref(data, flt):
                return as_type(data * flt * scalar, operand_dtype)

        else:

            def cb(data, offset, filter_data, unused):
                return data[offset] * filter_data[offset].imag * filter_data[offset].real * scalar

            def ref(data, flt):
                module = get_framework_module(framework)
                real = module.real(flt)
                imag = module.imag(flt)
                return as_type(data * imag * real * scalar, operand_dtype)

        return cb, ref

    def get_epilog(op_output_dtype, filter_dtype, scalar):
        if not is_complex(filter_dtype) or is_complex(op_output_dtype):

            def cb(data, offset, element, filter_data, unused):
                data[offset] = element * filter_data[offset] * scalar

            def ref(data, flt):
                return as_type(data * flt * scalar, op_output_dtype)

        else:

            def cb(data, offset, element, filter_data, unused):
                data[offset] = scalar * element * filter_data[offset].real + filter_data[offset].imag

            def ref(data, flt):
                module = get_framework_module(framework)
                real = module.real(flt)
                imag = module.imag(flt)
                return as_type(scalar * data * real + imag, op_output_dtype)

        return cb, ref

    prolog_cb, prolog_cb_ref = get_prolog(dtype, prolog_filter_dtype, 1)
    epilog_cb, epilog_cb_ref = get_epilog(get_fft_dtype(dtype), epilog_filter_dtype, 3)
    ref = prolog_cb_ref(signal, prolog_filter)
    ref = get_fft_ref(ref, axes=axes)
    ref = epilog_cb_ref(ref, epilog_filter)

    iprolog_cb, iprolog_cb_ref = get_prolog(get_fft_dtype(dtype), epilog_filter_dtype, 2)
    iepilog_cb, iepilog_cb_ref = get_epilog(dtype, prolog_filter_dtype, 5)

    ifft_ref = iprolog_cb_ref(ref, epilog_filter)
    ifft_ref = get_ifft_ref(
        ifft_ref,
        axes=axes,
        is_c2c=is_complex(dtype),
        last_axis_parity=last_axis_parity,
    )
    ifft_ref = iepilog_cb_ref(ifft_ref, prolog_filter)

    prolog_ltoir = nvmath.fft.compile_prolog(prolog_cb, dtype.name, prolog_filter_dtype.name)
    epilog_ltoir = nvmath.fft.compile_epilog(epilog_cb, get_fft_dtype(dtype).name, epilog_filter_dtype.name)
    iprolog_ltoir = nvmath.fft.compile_prolog(iprolog_cb, get_fft_dtype(dtype).name, epilog_filter_dtype.name)
    iepilog_ltoir = nvmath.fft.compile_epilog(iepilog_cb, dtype.name, prolog_filter_dtype.name)

    prolog_filter_dev = prolog_filter if mem_backend == MemBackend.cuda else to_gpu(prolog_filter)
    epilog_filter_dev = epilog_filter if mem_backend == MemBackend.cuda else to_gpu(epilog_filter)
    assert get_dtype_from_array(prolog_filter_dev) == prolog_filter_dtype
    assert get_dtype_from_array(epilog_filter_dev) == epilog_filter_dtype

    with nvmath.fft.FFT(
        signal,
        axes=axes,
        execution=exec_backend.nvname,
        options={
            "fft_type": "C2C" if is_complex(dtype) else "R2C",
            "result_layout": result_layout.value,
            "inplace": inplace.value,
        },
    ) as f:
        # with optimized result layout, the epilog_filter_dev
        # strides may not match the output operand strides
        out_shape, out_strides = f.get_output_layout()
        assert_eq(out_shape, epilog_filter_dev.shape)
        if out_strides == get_array_element_strides(epilog_filter_dev):
            epilog_filter_dev_perm = epilog_filter_dev
        else:
            assert result_layout == OptFftLayout.optimized, (
                f"{result_layout}, {out_strides}, {get_array_element_strides(epilog_filter_dev)}"
            )
            epilog_filter_dev_perm = permute_copy_like(epilog_filter_dev, out_shape, out_strides)
        try:
            f.plan(
                prolog={"ltoir": prolog_ltoir, "data": get_raw_ptr(prolog_filter_dev)},
                epilog={
                    "ltoir": epilog_ltoir,
                    "data": get_raw_ptr(epilog_filter_dev_perm),
                },
            )
        except (nvmath.bindings.cufft.cuFFTError, ValueError) as e:
            allow_to_fail_lto_ea_3d(e, shape, axes)
            if not allow_to_fail:
                raise
            allow_to_fail_compund_shape(e, shape, axes=axes)

        fft_out = f.execute(direction=Direction.forward.value)
        assert_norm_close(fft_out, ref, axes=axes, **get_tolerance(signal))

    with nvmath.fft.FFT(
        fft_out,
        axes=axes,
        execution=exec_backend.nvname,
        options={
            "fft_type": "C2C" if is_complex(dtype) else "C2R",
            "result_layout": result_layout.value,
            "inplace": inplace.value,
            "last_axis_parity": last_axis_parity,
        },
    ) as f:
        in_shape, in_strides = f.get_input_layout()
        out_shape, out_strides = f.get_output_layout()
        if exec_backend.mem == mem_backend:
            assert_eq(in_strides, get_array_element_strides(fft_out))
        assert_eq(in_shape, epilog_filter_dev.shape)
        assert_eq(out_shape, prolog_filter_dev.shape)
        if in_strides == get_array_element_strides(epilog_filter_dev):
            epilog_filter_dev_perm = epilog_filter_dev
        else:
            assert_eq(result_layout, OptFftLayout.optimized)
            epilog_filter_dev_perm = permute_copy_like(epilog_filter_dev, in_shape, in_strides)
        if out_strides == get_array_element_strides(prolog_filter_dev):
            prolog_filter_dev_perm = prolog_filter_dev
        else:
            assert_eq(result_layout, OptFftLayout.optimized)
            prolog_filter_dev_perm = permute_copy_like(prolog_filter_dev, out_shape, out_strides)
        try:
            f.plan(
                prolog={
                    "ltoir": iprolog_ltoir,
                    "data": get_raw_ptr(epilog_filter_dev_perm),
                },
                epilog={
                    "ltoir": iepilog_ltoir,
                    "data": get_raw_ptr(prolog_filter_dev_perm),
                },
            )
        except (nvmath.bindings.cufft.cuFFTError, ValueError) as e:
            if not allow_to_fail:
                raise
            allow_to_fail_compund_shape(e, shape, axes=axes)

        ifft_out = f.execute(direction=Direction.inverse.value)
        assert_norm_close(ifft_out, ifft_ref, axes=axes, **get_tolerance(fft_out))


@skip_if_lto_unssuported
@skip_unsupported_device
@pytest.mark.parametrize(
    (
        "dtype",
        "prolog_filter_dtype",
        "epilog_filter_dtype",
        "shape",
        "axes",
        "framework",
        "exec_backend",
        "mem_backend",
        "result_layout",
    ),
    [
        (
            dtype,
            prolog_filter_dtype,
            epilog_filter_dtype,
            repr(shape),
            repr(axes),
            framework := rng.choice(list(Framework.enabled())),
            ExecBackend.cufft,
            rng.choice(list(supported_backends.framework_mem[framework])),
            OptFftLayout.optimized,
        )
        for dtype in lto_callback_supperted_types
        if ExecBackend.cufft in supported_backends.exec
        for prolog_filter_dtype in lto_callback_supperted_types
        for epilog_filter_dtype in lto_callback_supperted_types
        for shape, axes in [
            ((4, 32), (1,)),
            ((4, 32, 32), (1, 2)),
            ((2, 16, 16, 16), (1, 2, 3)),
        ]
    ],
)
def test_operand_and_filter_dtypes_fft_ifft(
    dtype,
    prolog_filter_dtype,
    epilog_filter_dtype,
    shape,
    axes,
    framework,
    exec_backend,
    mem_backend,
    result_layout,
):
    _operand_filter_dtype_shape_fft_ifft_case(
        dtype,
        prolog_filter_dtype,
        epilog_filter_dtype,
        shape,
        axes,
        framework,
        exec_backend,
        mem_backend,
        inplace=OptFftInplace.false,
        allow_to_fail=AllowToFail.false,
        result_layout=result_layout,
    )


@skip_if_lto_unssuported
@skip_unsupported_device
@pytest.mark.parametrize(
    (
        "shape",
        "axes",
        "dtype",
        "prolog_filter_dtype",
        "epilog_filter_dtype",
        "framework",
        "exec_backend",
        "mem_backend",
        "allow_to_fail",
        "inplace",
        "result_layout",
    ),
    [
        (
            repr(shape),
            repr(axes),
            rng.choice([dt for dt in lto_callback_supperted_types if not inplace or is_complex(dt)]),
            rng.choice(lto_callback_supperted_types),
            rng.choice(lto_callback_supperted_types),
            framework := rng.choice(list(Framework.enabled())),
            ExecBackend.cufft,
            rng.choice(list(supported_backends.framework_mem[framework])),
            AllowToFail(allow_to_fail),
            inplace,
            result_layout,
        )
        for allow_to_fail, shape, axes in [
            (False, (1,), (0,)),  # 1D, pow2 sample
            (False, (4,), (0,)),  # 1D, pow2 sample
            (False, (1, 4637), (0,)),  # 1D, pow2 sample, prime batch
            (False, (4637, 1), (1,)),  # 1D, pow2 sample, prime batch
            (False, (4, 4637), (0,)),  # 1D, pow2 sample, prime batch
            (False, (4637, 4), (1,)),  # 1D, pow2 sample, prime batch
            (False, (8192, 4637), (0,)),  # 1D pow2 sample, prime batch
            (False, (4637, 8192), (1,)),  # 1D pow2 sample, prime batch
            # 2D, ((2*3*5*7)**2, 49) shape
            (False, (44100, 49, 15), (0, 1)),
            (False, (49, 44100, 3, 1, 5), (0, 1)),  # 2D, repeated stride
            (False, (1, 17775, 3), (0, 1)),  # 2D, (4, 15*15*79)
            (False, (4, 17775, 3), (0, 1)),  # 2D, (4, 15*15*79)
            (False, (19991, 1, 4), (1, 2)),  # 2D, prime shape batch
            (False, (19991, 4, 4), (1, 2)),  # 2D, prime shape batch
            # 3D, prime batch, repeated stride
            (False, (4, 4, 1, 14969, 1), (0, 1, 2)),
            (False, (1, 17775, 1), (0, 1, 2)),
            (False, (2, 17775, 1), (0, 1, 2)),
            # 3D, prime batch, repeated stride
            (False, (14969, 1, 4, 1, 4), (2, 3, 4)),
            (True, (4099, 3), (0,)),  # 1D prime shape
            (True, (2, 17161, 2), (1, 2)),  # 2D shape (131*131, 2)
            # 3D, larger prime extent
            (True, (2053, 3, 1), (0, 1, 2)),
            # 3D, larger prime extent
            (True, (3, 2053, 1), (0, 1, 2)),
            # 3D, larger prime extent
            (True, (3, 1, 2053), (0, 1, 2)),
        ]
        for inplace in OptFftInplace
        for result_layout in OptFftLayout
        if ExecBackend.cufft in supported_backends.exec
    ],
)
def test_operand_and_filter_shapes_fft_ifft(
    shape,
    axes,
    dtype,
    prolog_filter_dtype,
    epilog_filter_dtype,
    framework,
    exec_backend,
    mem_backend,
    allow_to_fail,
    inplace,
    result_layout,
):
    _operand_filter_dtype_shape_fft_ifft_case(
        dtype,
        prolog_filter_dtype,
        epilog_filter_dtype,
        shape,
        axes,
        framework,
        exec_backend,
        mem_backend,
        inplace=inplace,
        allow_to_fail=allow_to_fail,
        result_layout=result_layout,
    )


@skip_if_lto_unssuported
@skip_unsupported_device
@pytest.mark.parametrize(
    (
        "framework",
        "exec_backend",
        "mem_backend",
        "dtype_0",
        "dtype_1",
        "shape_0",
        "axes_0",
        "shape_kind_0",
        "shape_1",
        "axes_1",
        "shape_kind_1",
        "callbacks_0",
        "callbacks_1",
    ),
    [
        (
            framework,
            ExecBackend.cufft,
            mem_backend,
            dtype_0,
            rng.choice(lto_callback_supperted_types),
            repr(shape_0),
            repr(axes_0),
            shape_kind_0,
            repr(shape_1),
            repr(axes_1),
            shape_kind_1,
            callbacks_0,
            rng.choice(list(LtoCallback)),
        )
        for dtype_0 in lto_callback_supperted_types
        for shape_0, axes_0, shape_kind_0, shape_1, axes_1, shape_kind_1 in [
            (
                (4200, 13),
                (0,),
                ShapeKind.pow2357,
                (7, 4199),
                (1,),
                ShapeKind.random,
            ),  # 2*2*2*3*5*5*7, 13*17*19
            (
                (420, 512, 3),
                (0, 1),
                ShapeKind.pow2357,
                (5, 4, 4307),
                (1, 2),
                ShapeKind.random,
            ),  # 4307=59*73
            (
                (2, 16, 16, 5),
                (0, 1, 2),
                ShapeKind.pow2,
                (3, 9, 49, 25),
                (1, 2, 3),
                ShapeKind.pow2357,
            ),
        ]
        for framework in Framework.enabled()
        if ExecBackend.cufft in supported_backends.exec
        if MemBackend.cuda in supported_backends.framework_mem[framework]
        for mem_backend in supported_backends.framework_mem[framework]
        for callbacks_0 in LtoCallback
    ],
)
def test_two_plans_different_cbs(
    framework,
    exec_backend,
    mem_backend,
    dtype_0,
    dtype_1,
    shape_0,
    axes_0,
    shape_kind_0,
    shape_1,
    axes_1,
    shape_kind_1,
    callbacks_0,
    callbacks_1,
):
    free_framework_pools(framework, mem_backend)

    shape_0 = literal_eval(shape_0)
    axes_0 = literal_eval(axes_0)
    shape_1 = literal_eval(shape_1)
    axes_1 = literal_eval(axes_1)
    fft_type_0 = "C2C" if is_complex(dtype_0) else "R2C"
    fft_type_1 = "C2C" if is_complex(dtype_1) else "R2C"
    epilog_shape_0 = shape_0 if is_complex(dtype_0) else r2c_shape(shape_0, axes=axes_0)
    epilog_shape_1 = shape_1 if is_complex(dtype_1) else r2c_shape(shape_1, axes=axes_1)

    signal_0 = get_random_input_data(framework, shape_0, dtype_0, mem_backend, seed=10)
    signal_1 = get_random_input_data(framework, shape_1, dtype_1, mem_backend, seed=13)

    epilog_dtype_0 = get_fft_dtype(dtype_0)
    epilog_dtype_1 = get_fft_dtype(dtype_1)
    filters = {
        "prolog_0": get_random_input_data(framework, shape_0, dtype_0, mem_backend, seed=101),
        "epilog_0": get_random_input_data(framework, epilog_shape_0, epilog_dtype_0, mem_backend, seed=102),
        "prolog_1": get_random_input_data(framework, shape_1, dtype_1, mem_backend, seed=103),
        "epilog_1": get_random_input_data(framework, epilog_shape_1, epilog_dtype_1, mem_backend, seed=104),
    }
    dev_filters = {k: (to_gpu(v) if mem_backend == MemBackend.cpu else v) for k, v in filters.items()}

    def prolog_cb_0(data, offset, filter_data, unused):
        return data[offset] * filter_data[offset] * 2

    def prolog_cb_1(data, offset, filter_data, unused):
        return data[offset] * filter_data[offset] * 3

    def epilog_cb_0(data_out, offset, value, filter_data, unused):
        data_out[offset] = value * filter_data[offset] * 5

    def epilog_cb_1(data_out, offset, value, filter_data, unused):
        data_out[offset] = value * filter_data[offset] * 7

    cb_kwargs_0, cb_kwargs_1 = {}, {}
    ref_0, ref_1 = signal_0, signal_1

    if callbacks_0.has_prolog():
        ltoir = nvmath.fft.compile_prolog(prolog_cb_0, dtype_0.name, dtype_0.name)
        cb_kwargs_0["prolog"] = {
            "ltoir": ltoir,
            "data": get_raw_ptr(dev_filters["prolog_0"]),
        }
        ref_0 = get_scaled(as_type(ref_0 * filters["prolog_0"], dtype_0), 2)

    ref_0 = get_fft_ref(ref_0, axes=axes_0)

    if callbacks_0.has_epilog():
        ltoir = nvmath.fft.compile_epilog(epilog_cb_0, epilog_dtype_0.name, epilog_dtype_0.name)
        cb_kwargs_0["epilog"] = {
            "ltoir": ltoir,
            "data": get_raw_ptr(dev_filters["epilog_0"]),
        }
        ref_0 = get_scaled(as_type(ref_0 * filters["epilog_0"], epilog_dtype_0), 5)

    if callbacks_1.has_prolog():
        ltoir = nvmath.fft.compile_prolog(prolog_cb_1, dtype_1.name, dtype_1.name)
        cb_kwargs_1["prolog"] = {
            "ltoir": ltoir,
            "data": get_raw_ptr(dev_filters["prolog_1"]),
        }
        ref_1 = get_scaled(as_type(ref_1 * filters["prolog_1"], dtype_1), 3)

    ref_1 = get_fft_ref(ref_1, axes=axes_1)

    if callbacks_1.has_epilog():
        ltoir = nvmath.fft.compile_epilog(epilog_cb_1, epilog_dtype_1.name, epilog_dtype_1.name)
        cb_kwargs_1["epilog"] = {
            "ltoir": ltoir,
            "data": get_raw_ptr(dev_filters["epilog_1"]),
        }
        ref_1 = get_scaled(as_type(ref_1 * filters["epilog_1"], epilog_dtype_1), 7)

    fft_0, fft_1 = None, None
    try:
        fft_0 = nvmath.fft.FFT(
            signal_0,
            axes=axes_0,
            execution=exec_backend.nvname,
            options={
                "fft_type": fft_type_0,
                "blocking": "auto",
                "result_layout": OptFftLayout.natural.value,
            },
        )
        fft_1 = nvmath.fft.FFT(
            signal_1,
            axes=axes_1,
            execution=exec_backend.nvname,
            options={
                "fft_type": fft_type_1,
                "blocking": "auto",
                "result_layout": OptFftLayout.natural.value,
            },
        )
        fft_0.plan(**cb_kwargs_0)
        fft_1.plan(**cb_kwargs_1)
        fft_0_out = fft_0.execute(direction=Direction.forward.value)
        fft_1_out = fft_1.execute(direction=Direction.forward.value)

    finally:
        if fft_1 is not None:
            fft_1.free()
        if fft_0 is not None:
            fft_0.free()

    assert_array_type(fft_0_out, framework, mem_backend, get_fft_dtype(dtype_0))
    assert_array_type(fft_1_out, framework, mem_backend, get_fft_dtype(dtype_1))

    assert_norm_close(fft_0_out, ref_0, **get_tolerance(signal_0, shape_kind=shape_kind_0))
    assert_norm_close(fft_1_out, ref_1, **get_tolerance(signal_1, shape_kind=shape_kind_1))


@skip_if_lto_unssuported
@skip_unsupported_device
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
        "callbacks",
    ),
    [
        (
            framework,
            ExecBackend.cufft,
            mem_backend,
            shape_kind,
            repr(shape),
            repr(axes),
            dtype,
            blocking,
            OptFftLayout.natural,
            rng.choice(list(LtoCallback)),
        )
        for framework in Framework.enabled()
        if ExecBackend.cufft in supported_backends.exec
        for mem_backend in MemBackend
        if mem_backend in supported_backends.framework_mem[framework]
        for shape_kind, shape, axes in [
            (ShapeKind.pow2, (5, 64, 64, 64), (1, 2, 3)),
            (ShapeKind.pow2, (64, 64, 64, 5), (0, 1, 2)),
            (ShapeKind.pow2, (128, 512, 17), (0, 1)),
            (ShapeKind.pow2, (17, 512, 128), (0, 1)),
            (ShapeKind.random, (7, 11811), (1,)),  # (7, 127 * 31 * 3)
            (ShapeKind.random, (11811, 7), (0,)),  # (7, 127 * 31 * 3)
        ]
        for dtype in lto_callback_supperted_types
        for blocking in OptFftBlocking
    ],
)
def test_custom_stream(
    framework,
    exec_backend,
    mem_backend,
    shape_kind,
    shape,
    axes,
    dtype,
    blocking,
    result_layout,
    callbacks,
):
    free_framework_pools(framework, mem_backend)

    shape = literal_eval(shape)
    axes = literal_eval(axes)
    s0 = get_custom_stream(framework)
    s1 = get_custom_stream(framework)
    fft_type = "C2C" if is_complex(dtype) else "R2C"
    epilog_dtype = get_fft_dtype(dtype)
    epilog_shape = shape if is_complex(dtype) else r2c_shape(shape, axes=axes)

    with use_stream(s0):
        signal = get_random_input_data(framework, shape, dtype, mem_backend, seed=44)
        p_filt = get_random_input_data(framework, shape, dtype, mem_backend, seed=45)
        e_filt = get_random_input_data(framework, epilog_shape, epilog_dtype, mem_backend, seed=46)
        p_filt_dev = p_filt if mem_backend == MemBackend.cuda else to_gpu(p_filt)
        e_filt_dev = e_filt if mem_backend == MemBackend.cuda else to_gpu(e_filt)

        scale = 4
        ref = signal
        if callbacks.has_prolog():
            ref = as_type(ref * p_filt, dtype)
            scale *= 2
        ref = get_fft_ref(ref, axes=axes)
        if callbacks.has_epilog():
            ref = as_type(ref * e_filt, epilog_dtype)
            scale *= 2
        ref = get_scaled(ref, scale)

    def prolog_cb(data, offset, filter_data, unused):
        return data[offset] * filter_data[offset]

    def epilog_cb(data_out, offset, value, filter_data, unused):
        data_out[offset] = value * filter_data[offset]

    cb_kwargs = {}

    if callbacks.has_prolog():
        ltoir = nvmath.fft.compile_prolog(prolog_cb, dtype.name, dtype.name)
        cb_kwargs["prolog"] = {
            "ltoir": ltoir,
            "data": get_raw_ptr(p_filt_dev),
        }

    if callbacks.has_epilog():
        ltoir = nvmath.fft.compile_epilog(epilog_cb, epilog_dtype.name, epilog_dtype.name)
        cb_kwargs["epilog"] = {
            "ltoir": ltoir,
            "data": get_raw_ptr(e_filt_dev),
        }

    with nvmath.fft.FFT(
        signal,
        axes=axes,
        execution=exec_backend.nvname,
        options={
            "fft_type": fft_type,
            "blocking": blocking.value,
            "result_layout": result_layout.value,
        },
        stream=s0,
    ) as f:
        f.plan(stream=s0, **cb_kwargs)

        with use_stream(s0):
            add_in_place(signal, signal)
            if mem_backend == MemBackend.cpu:
                f.reset_operand(signal, stream=s0)
            add_in_place(p_filt_dev, p_filt_dev)
            add_in_place(e_filt_dev, e_filt_dev)

        fft = f.execute(direction=Direction.forward.value, stream=s0)

        needs_sync = mem_backend == MemBackend.cuda and blocking == OptFftBlocking.auto
        with use_stream(s0 if needs_sync else s1):
            add_in_place(p_filt_dev, p_filt_dev)
            add_in_place(e_filt_dev, e_filt_dev)
            add_in_place(fft, fft)
            assert_array_type(fft, framework, mem_backend, get_fft_dtype(dtype))
            assert_norm_close(fft, ref, **get_tolerance(signal, shape_kind=shape_kind))


@skip_if_lto_unssuported
@skip_unsupported_device(dev_count=2)
@pytest.mark.parametrize(
    (
        "framework",
        "exec_backend",
        "mem_backend",
        "dtype",
        "shape",
        "axes",
        "callbacks",
    ),
    [
        (
            framework,
            ExecBackend.cufft,
            mem_backend,
            dtype,
            repr(shape),
            repr(axes),
            callbacks,
        )
        for dtype in lto_callback_supperted_types
        for shape, axes in [
            ((4096, 3), (0,)),
            ((3, 4699, 10), (1, 2)),  # 4699 = 37 * 127
            ((63, 64, 65, 3), (0, 1, 2)),
        ]
        for framework in Framework.enabled()
        if ExecBackend.cufft in supported_backends.exec
        for mem_backend in supported_backends.framework_mem[framework]
        for callbacks in LtoCallback
    ],
)
def test_another_device(framework, exec_backend, mem_backend, dtype, shape, axes, callbacks):
    device_id = 1
    device = cp.cuda.Device(device_id)
    cc = device.compute_capability

    device_ctx = device if mem_backend == MemBackend.cuda else contextlib.nullcontext()

    shape = literal_eval(shape)
    axes = literal_eval(axes)
    epilog_shape = shape if is_complex(dtype) else r2c_shape(shape, axes=axes)
    epilog_dtype = get_fft_dtype(dtype)

    in_device_id = None if mem_backend == MemBackend.cpu else device_id
    signal = get_random_input_data(framework, shape, dtype, mem_backend, seed=42, device_id=in_device_id)
    p_filt = get_random_input_data(framework, shape, dtype, mem_backend, seed=101, device_id=in_device_id)
    e_filt = get_random_input_data(
        framework,
        epilog_shape,
        epilog_dtype,
        mem_backend,
        seed=102,
        device_id=in_device_id,
    )
    p_filt_dev = p_filt if mem_backend == MemBackend.cuda else to_gpu(p_filt, device_id=device_id)
    e_filt_dev = e_filt if mem_backend == MemBackend.cuda else to_gpu(e_filt, device_id=device_id)

    with device_ctx:
        ref = signal
        if callbacks.has_prolog():
            ref = as_type(ref + p_filt, dtype)
        ref = get_fft_ref(ref, axes=axes)
        if callbacks.has_epilog():
            ref = as_type(ref * e_filt, epilog_dtype)

    def prolog_cb(data, offset, filter_data, unused):
        return data[offset] + filter_data[offset]

    def epilog_cb(data_out, offset, value, filter_data, unused):
        data_out[offset] = value * filter_data[offset]

    cb_kwargs = {}
    if callbacks.has_prolog():
        prolog_ltoir = nvmath.fft.compile_prolog(prolog_cb, dtype.name, dtype.name, compute_capability=cc)
        cb_kwargs["prolog"] = {
            "ltoir": prolog_ltoir,
            "data": get_raw_ptr(p_filt_dev),
        }

    if callbacks.has_epilog():
        epilog_ltoir = nvmath.fft.compile_epilog(epilog_cb, epilog_dtype.name, epilog_dtype.name, compute_capability=cc)
        cb_kwargs["epilog"] = {
            "ltoir": epilog_ltoir,
            "data": get_raw_ptr(e_filt_dev),
        }

    fft_fn = nvmath.fft.fft if is_complex(dtype) else nvmath.fft.rfft

    exec_options = {"name": exec_backend.nvname}
    if mem_backend == MemBackend.cpu:
        exec_options["device_id"] = device_id

    fft_out = fft_fn(
        signal,
        axes=axes,
        execution=exec_options,
        options={
            "result_layout": OptFftLayout.natural.value,
            "blocking": "auto",
        },
        **cb_kwargs,
    )

    assert_array_type(fft_out, framework, mem_backend, get_fft_dtype(dtype))
    if mem_backend == MemBackend.cuda:
        assert_eq(get_array_device_id(fft_out), device_id)

    with device_ctx:
        assert_norm_close(fft_out, ref, **get_tolerance(signal))


@skip_if_lto_unssuported
@skip_unsupported_device(dev_count=2)
@pytest.mark.parametrize(
    (
        "framework",
        "exec_backend",
        "mem_backend",
        "dtype",
        "shape_0",
        "axes_0",
        "shape_1",
        "axes_1",
        "callbacks_0",
        "callbacks_1",
    ),
    [
        (
            framework,
            ExecBackend.cufft,
            mem_backend,
            dtype,
            repr(shape_0),
            repr(axes_0),
            repr(shape_1),
            repr(axes_1),
            callbacks_0,
            rng.choice(list(LtoCallback)),
        )
        for framework in Framework.enabled()
        if ExecBackend.cufft in supported_backends.exec
        for mem_backend in supported_backends.framework_mem[framework]
        for dtype in lto_callback_supperted_types
        for shape_0, axes_0, shape_1, axes_1 in [
            ((4200, 13), (0,), (7, 4199), (1,)),  # 2*2*2*3*5*5*7, 13*17*19
            ((420, 512, 3), (0, 1), (5, 4, 4307), (1, 2)),  # 4307=59*73
            ((2, 16, 16, 5), (0, 1, 2), (3, 9, 49, 25), (1, 2, 3)),
        ]
        for callbacks_0 in LtoCallback
    ],
)
def test_two_devices(
    framework,
    exec_backend,
    mem_backend,
    dtype,
    shape_0,
    axes_0,
    shape_1,
    axes_1,
    callbacks_0,
    callbacks_1,
):
    free_framework_pools(framework, mem_backend)
    device_id_0, device_id_1 = 0, 1
    device_0, device_1 = tuple(cp.cuda.Device(did) for did in (device_id_0, device_id_1))
    cc_0, cc_1 = device_0.compute_capability, device_1.compute_capability

    device_ctx_0, device_ctx_1 = (
        (device_0, device_1) if mem_backend == MemBackend.cuda else (contextlib.nullcontext(), contextlib.nullcontext())
    )

    shape_0, shape_1 = literal_eval(shape_0), literal_eval(shape_1)
    axes_0, axes_1 = literal_eval(axes_0), literal_eval(axes_1)
    epilog_shape_0 = shape_0 if is_complex(dtype) else r2c_shape(shape_0, axes=axes_0)
    epilog_shape_1 = shape_1 if is_complex(dtype) else r2c_shape(shape_1, axes=axes_1)
    epilog_dtype = get_fft_dtype(dtype)

    def prolog_cb_0(data, offset, filter_data, unused):
        return data[offset] * filter_data[offset] * 2

    def prolog_cb_1(data, offset, filter_data, unused):
        return data[offset] * filter_data[offset] * 3

    def epilog_cb_0(data_out, offset, value, filter_data, unused):
        data_out[offset] = value * filter_data[offset] * 5

    def epilog_cb_1(data_out, offset, value, filter_data, unused):
        data_out[offset] = value * filter_data[offset] * 7

    in_device_id_0, in_device_id_1 = (None, None) if mem_backend == MemBackend.cpu else (device_id_0, device_id_1)
    signal_0 = get_random_input_data(framework, shape_0, dtype, mem_backend, seed=13, device_id=in_device_id_0)
    signal_1 = get_random_input_data(framework, shape_1, dtype, mem_backend, seed=21, device_id=in_device_id_1)

    pro_0 = get_random_input_data(
        framework,
        shape_0,
        dtype,
        mem_backend,
        seed=34,
        device_id=in_device_id_0,
    )
    ep_0 = get_random_input_data(
        framework,
        epilog_shape_0,
        epilog_dtype,
        mem_backend,
        seed=55,
        device_id=in_device_id_0,
    )
    pro_1 = get_random_input_data(
        framework,
        shape_1,
        dtype,
        mem_backend,
        seed=89,
        device_id=in_device_id_1,
    )
    ep_1 = get_random_input_data(
        framework,
        epilog_shape_1,
        epilog_dtype,
        mem_backend,
        seed=144,
        device_id=in_device_id_1,
    )

    if mem_backend == MemBackend.cuda:
        pro_0_dev, ep_0_dev, pro_1_dev, ep_1_dev = pro_0, ep_0, pro_1, ep_1
    else:
        pro_0_dev = to_gpu(pro_0, device_id=device_id_0)
        ep_0_dev = to_gpu(ep_0, device_id=device_id_0)
        pro_1_dev = to_gpu(pro_1, device_id=device_id_1)
        ep_1_dev = to_gpu(ep_1, device_id=device_id_1)

    cb_kwargs_0, cb_kwargs_1 = {}, {}
    for (
        cb_kwargs,
        (prolog_cb, epilog_cb),
        (pro_data_dev, ep_data_dev),
        callbacks,
        cc,
    ) in zip(
        [cb_kwargs_0, cb_kwargs_1],
        [(prolog_cb_0, epilog_cb_0), (prolog_cb_1, epilog_cb_1)],
        [(pro_0_dev, ep_0_dev), (pro_1_dev, ep_1_dev)],
        [callbacks_0, callbacks_1],
        [cc_0, cc_1],
        strict=True,
    ):
        if callbacks.has_prolog():
            ltoir = nvmath.fft.compile_prolog(prolog_cb, dtype.name, dtype.name, compute_capability=cc)
            cb_kwargs["prolog"] = {"ltoir": ltoir, "data": get_raw_ptr(pro_data_dev)}

        if callbacks.has_epilog():
            ltoir = nvmath.fft.compile_epilog(epilog_cb, epilog_dtype.name, epilog_dtype.name, compute_capability=cc)
            cb_kwargs["epilog"] = {"ltoir": ltoir, "data": get_raw_ptr(ep_data_dev)}

    refs = [signal_0, signal_1]
    for i, (
        axes,
        device_ctx,
        (pro_flt, ep_flt),
        callbacks,
        (pro_scale, ep_scale),
    ) in enumerate(
        zip(
            [axes_0, axes_1],
            [device_ctx_0, device_ctx_1],
            [(pro_0, ep_0), (pro_1, ep_1)],
            [callbacks_0, callbacks_1],
            [(2, 5), (3, 7)],
            strict=True,
        )
    ):
        with device_ctx:
            if callbacks.has_prolog():
                refs[i] = as_type(refs[i] * pro_flt * pro_scale, dtype)
            refs[i] = get_fft_ref(refs[i], axes=axes)
            if callbacks.has_epilog():
                refs[i] = as_type(refs[i] * ep_flt * ep_scale, epilog_dtype)

    exec_options_0 = {"name": exec_backend.nvname}
    if mem_backend == MemBackend.cpu:
        exec_options_0["device_id"] = device_id_0

    fft_0 = nvmath.fft.FFT(
        signal_0,
        axes=axes_0,
        execution=exec_options_0,
        options={
            "blocking": OptFftBlocking.auto.value,
            "result_layout": OptFftLayout.natural.value,
        },
    )

    exec_options_1 = {"name": exec_backend.nvname}
    if mem_backend == MemBackend.cpu:
        exec_options_1["device_id"] = device_id_1
    fft_1 = nvmath.fft.FFT(
        signal_1,
        axes=axes_1,
        execution=exec_options_1,
        options={
            "blocking": OptFftBlocking.auto.value,
            "result_layout": OptFftLayout.natural.value,
        },
    )
    fft_0.plan(**cb_kwargs_0)
    fft_1.plan(**cb_kwargs_1)
    fft_0_out = fft_0.execute(direction=Direction.forward.value)
    fft_1_out = fft_1.execute(direction=Direction.forward.value)

    assert_array_type(fft_0_out, framework, mem_backend, epilog_dtype)
    assert_array_type(fft_1_out, framework, mem_backend, epilog_dtype)
    if mem_backend == MemBackend.cuda:
        assert_eq(get_array_device_id(fft_0_out), device_id_0)
        assert_eq(get_array_device_id(fft_1_out), device_id_1)

    with device_ctx_0:
        assert_norm_close(fft_0_out, refs[0], **get_tolerance(signal_0))

    with device_ctx_1:
        assert_norm_close(fft_1_out, refs[1], **get_tolerance(signal_1))


@skip_if_lto_unssuported
@skip_unsupported_device
@pytest.mark.parametrize(
    (
        "prolog_epilog",
        "exec_backend",
        "fn_name",
    ),
    [
        (prolog_epilog, ExecBackend.cufft, fn_name)
        for prolog_epilog in (LtoCallback.prolog, LtoCallback.epilog)
        if ExecBackend.cufft in supported_backends.exec
        for fn_name in (
            "prolog_fn",
            "prolog_fn_no_ret",
            "epilog_fn",
            "epilog_fn_ret",
            "epilog_fn_ret_data",
            "data_only",
        )
        if (not prolog_epilog.has_prolog() or fn_name != "prolog_fn")
        and (not prolog_epilog.has_epilog() or fn_name != "epilog_fn")
    ],
)
def test_unsupported_callback_signature(prolog_epilog, exec_backend, fn_name):
    def prolog_fn(data, offset, flt, unused):
        return data[offset]

    def prolog_fn_no_ret(data, offset, flt, unused):
        data[offset] = flt[offset]

    def epilog_fn(data, offset, element, flt, unused):
        data[offset] = element

    def epilog_fn_ret(data, offset, element, flt, unused):
        return element

    def epilog_fn_ret_data(data, offset, element, flt, unused):
        return data[offset]

    def data_only(data, offset):
        return data[offset] * 3

    fns = {
        "prolog_fn": prolog_fn,
        "prolog_fn_no_ret": prolog_fn_no_ret,
        "epilog_fn": epilog_fn,
        "epilog_fn_ret": epilog_fn_ret,
        "epilog_fn_ret_data": epilog_fn_ret_data,
        "data_only": data_only,
    }
    with pytest.raises(Exception):
        if prolog_epilog == LtoCallback.prolog:
            nvmath.fft.compile_prolog(fns[fn_name], "complex64", "complex64")
        else:
            assert prolog_epilog == LtoCallback.epilog
            nvmath.fft.compile_epilog(fns[fn_name], "complex64", "complex64")


@skip_if_lto_unssuported
@skip_unsupported_device
@pytest.mark.parametrize(
    (
        "element",
        "exec_backend",
        "dtype",
        "callback",
    ),
    [
        (
            element,
            ExecBackend.cufft,
            dtype,
            callback,
        )
        for element in ("operand", "filter")
        if ExecBackend.cufft in supported_backends.exec
        for dtype in DType
        if dtype not in lto_callback_supperted_types
        for callback in (LtoCallback.prolog, LtoCallback.epilog)
    ],
)
def test_unsupported_type(element, exec_backend, dtype, callback):
    def prolog_fn(data, offset, flt, unused):
        return data[offset]

    def epilog_fn(data, offset, element, flt, unused):
        data[offset] = element

    if element == "operand":
        args = (dtype.name, "complex64")
        msg = "The specified operand data type"
    else:
        assert element == "filter"
        args = ("complex64", dtype.name)
        msg = "The specified user information"

    with pytest.raises(ValueError, match=msg):
        if callback == LtoCallback.prolog:
            nvmath.fft.compile_prolog(prolog_fn, *args)
        else:
            nvmath.fft.compile_epilog(epilog_fn, *args)


@skip_if_lto_unssuported
@skip_unsupported_device
@pytest.mark.parametrize(
    (
        "framework",
        "exec_backend",
        "mem_backend",
        "callback",
        "actual_dtype",
        "declared_dtype",
    ),
    [
        (
            framework,
            ExecBackend.cufft,
            rng.choice(supported_backends.framework_mem[framework]),
            callback,
            actual_dtype,
            rng.choice([dt for dt in lto_callback_supperted_types if dt != actual_dtype]),
        )
        for framework in Framework.enabled()
        if ExecBackend.cufft in supported_backends.exec
        for callback in (LtoCallback.prolog, LtoCallback.epilog)
        for actual_dtype in [dt for dt in lto_callback_supperted_types if callback == LtoCallback.prolog or is_complex(dt)]
    ],
)
def test_mismatched_operand_type(framework, exec_backend, mem_backend, callback, actual_dtype, declared_dtype):
    def prolog_fn(data, offset, flt, unused):
        return data[offset]

    def epilog_fn(data, offset, element, flt, unused):
        data[offset] = element

    cb_kwargs = {}
    if callback == LtoCallback.prolog:
        cb_kwargs["prolog"] = {"ltoir": nvmath.fft.compile_prolog(prolog_fn, declared_dtype.name, declared_dtype.name)}
    else:
        assert callback == LtoCallback.epilog
        cb_kwargs["epilog"] = {"ltoir": nvmath.fft.compile_epilog(epilog_fn, declared_dtype.name, declared_dtype.name)}

    shape = (16,)
    signal = get_random_input_data(framework, shape, actual_dtype, mem_backend, seed=101)
    fn = nvmath.fft.fft if is_complex(actual_dtype) else nvmath.fft.rfft
    with pytest.raises(nvmath.bindings.cufft.cuFFTError, match="CUFFT_INTERNAL_ERROR"):
        fn(signal, execution=exec_backend.nvname, **cb_kwargs)


@skip_if_lto_unssuported
@skip_unsupported_device
@pytest.mark.parametrize(
    (
        "framework",
        "exec_backend",
        "mem_backend",
        "callback",
        "dtype",
    ),
    [
        (
            framework,
            ExecBackend.fftw,
            mem_backend,
            callback,
            rng.choice(lto_callback_supperted_types),
        )
        for framework in Framework.enabled()
        if ExecBackend.fftw in supported_backends.exec
        for mem_backend in supported_backends.framework_mem[framework]
        for callback in (LtoCallback.prolog, LtoCallback.epilog)
    ],
)
def test_prolog_epilog_unsupported_exec(framework, exec_backend, mem_backend, callback, dtype):
    def prolog_fn(data, offset, flt, unused):
        return data[offset]

    def epilog_fn(data, offset, element, flt, unused):
        data[offset] = element

    cb_kwargs = {}
    if callback == LtoCallback.prolog:
        cb_kwargs["prolog"] = {"ltoir": nvmath.fft.compile_prolog(prolog_fn, dtype.name, dtype.name)}
    else:
        assert callback == LtoCallback.epilog
        cb_kwargs["epilog"] = {"ltoir": nvmath.fft.compile_epilog(epilog_fn, dtype.name, dtype.name)}

    shape = (16,)
    signal = get_random_input_data(framework, shape, dtype, mem_backend, seed=189)
    fn = nvmath.fft.fft if is_complex(dtype) else nvmath.fft.rfft

    with pytest.raises(
        ValueError,
        match="The 'prolog' and 'epilog' are not supported with CPU 'execution'",
    ):
        fn(signal, execution=exec_backend.nvname, **cb_kwargs)
