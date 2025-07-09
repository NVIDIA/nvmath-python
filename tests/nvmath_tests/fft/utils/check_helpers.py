# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

from itertools import accumulate
import math
import numpy as np

try:
    import cupy as cp

    CP_NDARRAY = cp.ndarray
except ImportError:
    cp = CP_NDARRAY = None

try:
    import torch
except ImportError:
    torch = None

import nvmath

from .common_axes import ExecBackend, MemBackend, Framework, DType, ShapeKind, OptFftType
from .axes_utils import (
    TORCH_TENSOR,
    get_framework_dtype,
    get_fft_dtype,
    get_ifft_dtype,
    is_complex,
    is_half,
    is_array,
    size_of,
    get_dtype_from_array,
    get_framework_from_array,
    get_framework_module,
    get_array_backend,
)

_torch_has_cuda = bool(torch and torch.cuda.is_available() and torch.cuda.device_count() > 0)


_cufft_version = None


def _get_cufft_version():
    from nvmath.bindings import cufft as _cufft  # type: ignore

    return _cufft.get_version()


def get_cufft_version():
    global _cufft_version
    if _cufft_version is None:
        _cufft_version = _get_cufft_version()
    return _cufft_version


def r2c_shape(shape, axes=None):
    out_shape = list(shape)
    last_fft_axis = (len(shape) - 1) if axes is None else axes[-1]
    out_shape[last_fft_axis] = shape[last_fft_axis] // 2 + 1
    return tuple(out_shape)


def slice_r2c(fft, axes=None):
    out_shape = r2c_shape(fft.shape, axes=axes)
    slices = tuple(slice(extent) for extent in out_shape)
    return fft[slices]


def is_pow_2(extent):
    return extent > 0 and (extent & (extent - 1) == 0)


def get_numpy_fft_ref(
    sample: np.ndarray,
    axes=None,
    norm=None,
):
    assert axes is None or len(axes)
    assert get_framework_from_array(sample) == Framework.numpy
    ref = np.fft.fftn(sample, axes=axes, norm=norm)
    dtype = get_dtype_from_array(sample)
    if not is_complex(dtype):
        ref = slice_r2c(ref, axes=axes)
    expected_ret_type = get_fft_dtype(dtype)
    actual_ret_dtype = get_dtype_from_array(ref)
    if expected_ret_type == actual_ret_dtype:
        return ref
    else:
        assert is_complex(actual_ret_dtype)
        assert size_of(actual_ret_dtype) >= size_of(expected_ret_type)
        return ref.astype(get_framework_dtype(Framework.numpy, expected_ret_type))


def get_cupy_fft_ref(
    sample: CP_NDARRAY,
    axes=None,
    norm=None,
):
    assert get_framework_from_array(sample) == Framework.cupy
    ref = cp.fft.fftn(sample, axes=axes, norm=norm)
    dtype = get_dtype_from_array(sample)
    if not is_complex(dtype):
        ref = slice_r2c(ref, axes=axes)
    assert_eq(ref.dtype, get_framework_dtype(Framework.cupy, get_fft_dtype(dtype)))
    return ref


def get_torch_fft_ref(
    sample: TORCH_TENSOR,
    axes=None,
    norm=None,
):
    assert get_framework_from_array(sample) == Framework.torch

    dtype = get_dtype_from_array(sample)
    mem_backend = get_array_backend(sample)
    device = sample.device

    # torch fft does not support f16/c32 cpu tensors
    # the gpu support is limited only to pows of 2
    if is_half(dtype):
        if _torch_has_cuda and mem_backend == MemBackend.cpu and all(is_pow_2(e) for e in sample.shape):
            in_sample = sample.to("cuda")
        else:
            assert dtype in [DType.float16, DType.complex32]
            in_sample = sample.type(torch.float32 if dtype == DType.float16 else torch.complex64)
    else:
        in_sample = sample

    # Workaround for bug in torch CPU fftn that leads to a crash
    # for larger 3D ffts with interleaved batch
    if (
        get_array_backend(in_sample) == MemBackend.cpu
        and axes is not None
        and len(axes) >= 3
        and len(sample.shape) > 3
        and max(axes) <= 2
    ):
        np_sample = np.array(in_sample)
        np_ref = get_numpy_fft_ref(np_sample, axes=axes, norm=norm)
        ref = torch.tensor(np_ref)
    else:
        ref = torch.fft.fftn(in_sample, dim=axes, norm=norm)
        if not is_complex(dtype):
            ref = slice_r2c(ref, axes=axes)

    if is_half(dtype):
        if _torch_has_cuda and mem_backend == MemBackend.cpu and all(is_pow_2(e) for e in sample.shape):
            ref = ref.to(device)
        else:
            assert_eq(ref.dtype, torch.complex64)
            ref = ref.type(torch.complex32)

    assert_eq(ref.dtype, get_framework_dtype(Framework.torch, get_fft_dtype(dtype)))
    assert_eq(mem_backend, get_array_backend(ref))
    assert_eq(ref.device, device)
    return ref


def get_fft_ref(
    sample: np.ndarray | CP_NDARRAY | TORCH_TENSOR,
    axes: list | None = None,
    norm=None,
):
    if axes is not None:
        ndim = sample.ndim
        axes = [axis % ndim for axis in axes]
    if get_framework_from_array(sample) == Framework.numpy:
        return get_numpy_fft_ref(sample, norm=norm, axes=axes)
    elif get_framework_from_array(sample) == Framework.cupy:
        return get_cupy_fft_ref(sample, norm=norm, axes=axes)
    elif get_framework_from_array(sample) == Framework.torch:
        return get_torch_fft_ref(sample, norm=norm, axes=axes)
    else:
        raise ValueError(f"Unknown framework {get_framework_from_array(sample)}")


def get_ifft_ref(
    sample: np.ndarray | CP_NDARRAY | TORCH_TENSOR,
    axes: list | None = None,
    is_c2c=True,
    last_axis_parity=None,
    norm="forward",
):
    framework = get_framework_from_array(sample)
    mem_backend = get_array_backend(sample)
    dtype = get_dtype_from_array(sample)
    out_dtype = get_ifft_dtype(dtype, fft_type=OptFftType.c2c if is_c2c else OptFftType.c2r)

    if is_c2c:
        out_shape, fft_out_shape = None, None
    else:
        assert last_axis_parity in ("odd", "even")
        shape = tuple(sample.shape)
        if axes is None:
            last_axis = len(shape) - 1
        else:
            assert isinstance(axes, tuple | list)
            assert len(axes)
            last_axis = axes[-1]
        out_last_axis_parity = (2 * shape[last_axis] - 2) if last_axis_parity == "even" else (2 * shape[last_axis] - 1)
        out_shape = tuple(e if a != last_axis else out_last_axis_parity for a, e in enumerate(shape))
        fft_out_shape = out_shape if axes is None else tuple(out_shape[a] for a in axes)

    if get_framework_from_array(sample) == Framework.numpy:
        fn = np.fft.ifftn if is_c2c else np.fft.irfftn
        ret = fn(sample, s=fft_out_shape, axes=axes, norm=norm)
        ret_dtype = get_dtype_from_array(ret)
        if out_dtype != ret_dtype:
            assert is_complex(ret_dtype) == is_complex(out_dtype), f"{ret_dtype} vs {out_dtype}"
            assert size_of(ret_dtype) >= size_of(out_dtype)
            ret = ret.astype(get_framework_dtype(Framework.numpy, out_dtype))
    elif get_framework_from_array(sample) == Framework.cupy:
        fn = cp.fft.ifftn if is_c2c else cp.fft.irfftn
        ret = fn(sample, s=fft_out_shape, axes=axes, norm=norm)
    elif get_framework_from_array(sample) == Framework.torch:
        # torch does not implement half precision fft
        # (or the support is partial)
        if dtype == DType.complex32:
            in_sample = sample.type(torch.complex64)
        else:
            in_sample = sample
        fn = torch.fft.ifftn if is_c2c else torch.fft.irfftn
        ret = fn(in_sample, s=fft_out_shape, norm=norm, dim=axes)
        if dtype == DType.complex32:
            ret_dtype = get_dtype_from_array(ret)
            expected_ret_dtype = DType.complex64 if is_c2c else DType.float32
            assert ret_dtype == expected_ret_dtype
            ret = ret.type(get_framework_dtype(Framework.torch, out_dtype))
    else:
        raise ValueError(f"Unknown framework {get_framework_from_array(sample)}")

    assert_array_type(ret, framework, mem_backend, out_dtype)
    return ret


def get_scaled(sample: np.ndarray | CP_NDARRAY | TORCH_TENSOR, scale):
    sample_scaled = sample * scale
    if sample_scaled.dtype != sample.dtype:
        sample_scaled = sample_scaled.astype(sample.dtype)
    return sample_scaled


def get_transposed(sample: np.ndarray | CP_NDARRAY | TORCH_TENSOR, d1: int, d2: int):
    framework = get_framework_from_array(sample)
    if framework == Framework.torch:
        return sample.transpose(d1, d2)
    else:
        swap = {d1: d2, d2: d1}
        transpose_axes = [swap.get(i, i) for i in range(sample.ndim)]
        return sample.transpose(transpose_axes)


def use_stream(stream):
    if isinstance(stream, cp.cuda.Stream):
        return stream
    elif isinstance(stream, torch.cuda.Stream):
        return torch.cuda.stream(stream)
    else:
        raise ValueError(f"Unknown stream type {type(stream)}")


def get_array_device_id(array) -> int:
    if CP_NDARRAY is not None and isinstance(array, CP_NDARRAY):
        return array.device.id
    elif TORCH_TENSOR is not None and isinstance(array, TORCH_TENSOR):
        return array.device.index
    else:
        raise ValueError(f"Unknown device array type {array}")


def get_raw_ptr(array) -> int:
    framework = get_framework_from_array(array)
    if framework == Framework.torch:
        return array.data_ptr()
    elif framework == Framework.numpy:
        return array.ctypes.data
    else:
        assert framework == Framework.cupy
        return array.data.ptr


def to_gpu(array, device_id: int | None = None):
    framework = get_framework_from_array(array)
    if framework == Framework.torch:
        device = "cuda" if device_id is None else f"cuda:{device_id}"
        return array.to(device)
    elif framework == Framework.numpy:
        if device_id is None:
            return cp.array(array)
        else:
            with cp.cuda.Device(device_id):
                return cp.array(array)
    else:
        assert framework == Framework.cupy
        assert device_id is None
        return cp.asnumpy(array)


def as_type(array, dtype: DType):
    if get_dtype_from_array(array) == dtype:
        return array
    framework = get_framework_from_array(array)
    if framework == Framework.torch:
        ret = array.type(get_framework_dtype(framework, dtype))
    else:
        ret = array.astype(get_framework_dtype(framework, dtype))
    ret_dtype = get_dtype_from_array(ret)
    assert ret_dtype == dtype, f"{ret_dtype} vs {dtype}"
    return ret


def record_event(stream):
    if isinstance(stream, cp.cuda.Stream):
        return stream.record()
    elif isinstance(stream, torch.cuda.Stream):
        return stream.record_event()
    else:
        raise ValueError(f"Unknown stream type {type(stream)}")


def assert_all_close(a, b, rtol, atol):
    assert type(a) is type(b), f"{type(a)}!= {type(b)}"
    if isinstance(a, np.ndarray):
        return np.testing.assert_allclose(a, b, rtol=rtol, atol=atol)
    elif CP_NDARRAY is not None and isinstance(a, CP_NDARRAY):
        return cp.testing.assert_allclose(a, b, rtol=rtol, atol=atol)
    elif TORCH_TENSOR is not None and isinstance(a, TORCH_TENSOR):
        return torch.testing.assert_close(a, b, rtol=rtol, atol=atol)
    else:
        raise ValueError(f"Unknown array type {a}")


def get_ifft_c2r_options(out_type, last_axis_parity):
    if is_complex(out_type):
        return {}
    return {
        "fft_type": "C2R",
        "last_axis_parity": "odd" if last_axis_parity % 2 else "even",
    }


def get_default_tolerance(dtype: DType, shape_kind: ShapeKind | None, exec_backend: ExecBackend | None):
    module = cp or np
    dtype_bytes_num = size_of(dtype)
    if is_complex(dtype):
        dtype_bytes_num //= 2
    if dtype_bytes_num == 2:
        return 1e-2, module.finfo(module.float16).eps
    elif dtype_bytes_num == 4:
        atol = module.finfo(module.float32).eps
        if exec_backend == ExecBackend.fftw:
            return 4e-6, atol
        else:
            return 1e-6, atol
    elif dtype_bytes_num == 8:
        atol = module.finfo(module.float64).eps
        if shape_kind is None or shape_kind in (ShapeKind.pow2, ShapeKind.pow2357):
            return 1e-14, atol
        # The differences between the cufft and the ref for double precision
        # tensors seem to be bigger if shape comprises larger primes,
        # especially in older CTKs
        if exec_backend == ExecBackend.cufft and get_cufft_version() >= 10702:  # shipped with CTK 11.7
            return 3e-14, atol
        else:
            return 1e-12, atol
    else:
        raise ValueError(f"Unexpected dtype {dtype}")


def get_array_strides(a: np.ndarray | CP_NDARRAY | TORCH_TENSOR):
    nd_types = tuple(t for t in (np.ndarray, CP_NDARRAY) if t is not None)
    if isinstance(a, nd_types):
        return a.strides
    else:
        return a.stride()


def get_array_element_strides(a: np.ndarray | CP_NDARRAY | TORCH_TENSOR):
    strides = get_array_strides(a)
    if get_framework_from_array(a) not in (Framework.numpy, Framework.cupy):
        return tuple(strides)
    assert all(s % a.itemsize == 0 for s in strides)
    return tuple(s // a.itemsize for s in strides)


def get_dense_strides(shape):
    return list(accumulate([1] + list(shape), lambda x, y: x * y))[:-1]


def is_decreasing(values):
    if len(values) <= 1:
        return True
    previous, *values = values
    for v in values:
        if previous < v:
            return False
        previous = v
    return True


def get_norm(a, axes=None):
    framework = get_framework_from_array(a)
    if isinstance(axes, list):
        axes = tuple(axes)

    # Use wider type for half-precision tensors
    # Older torch versions don't support vector norm
    # on halfs.
    dtype = get_dtype_from_array(a)
    if dtype == DType.complex32:
        a = a.type(torch.complex64)
    elif dtype == DType.float16:
        a = a.type(torch.float32)

    if framework == Framework.numpy:
        if not isinstance(axes, tuple) or len(axes) <= 2:
            # without ord specified it is vector l2 norm (even for 2d)
            return np.linalg.norm(a, axis=axes)
        else:
            a = np.abs(a)
            return np.sqrt(np.sum(a * a, axis=axes))
    elif framework == Framework.cupy:
        if not isinstance(axes, tuple) or len(axes) <= 2:
            return np.linalg.norm(a, axis=axes)
        else:
            a = cp.abs(a)
            return cp.sqrt(cp.sum(a * a, axis=axes))
    elif framework == Framework.torch:
        return torch.linalg.vector_norm(a, dim=axes)
    else:
        raise ValueError(f"Unknown framework {framework}")


def array_all(a):
    if not is_array(a):
        return a
    framework = get_framework_from_array(a)
    framework_module = get_framework_module(framework)
    return framework_module.all(a)


def get_abs(a):
    framework = get_framework_from_array(a)
    framework_module = get_framework_module(framework)
    return framework_module.abs(a)


def get_decreasing_stride_order(shape, strides):
    d = len(shape)
    assert len(strides) == d
    return tuple(i for _, _, i in sorted(zip(strides, shape, range(d), strict=True), reverse=True))


def permute_copy_like(a, ref_shape, ref_strides):
    assert a.shape == ref_shape, f"{a.shape} vs {ref_shape}"
    axis_order = get_decreasing_stride_order(ref_shape, ref_strides)
    b = get_permuted_copy(a, axis_order)
    ret = get_permuted(b, get_rev_perm(axis_order))
    ret_dtype = get_dtype_from_array(ret)
    a_dtype = get_dtype_from_array(a)
    assert ret_dtype == a_dtype, f"{ret.dtype} vs {a_dtype}"
    assert ret.shape == ref_shape, f"{ret.shape} vs {ref_shape}"
    ret_strides = get_array_element_strides(ret)
    assert ret_strides == ref_strides, f"{ret_strides} vs {ref_strides}"
    return ret


def copy_array(a):
    if get_framework_from_array(a) != Framework.torch:
        return a.copy()
    else:
        # torch contiguous or copy does not always
        # enforce truly contiguous stride
        t = torch.empty(a.shape, dtype=a.dtype, device=a.device)
        t.copy_(a, non_blocking=True)
        return t


def get_permuted(sample: np.ndarray | CP_NDARRAY | TORCH_TENSOR, permutation):
    framework = get_framework_from_array(sample)
    if framework == Framework.torch:
        return torch.permute(sample, permutation)
    else:
        return sample.transpose(permutation)


def get_permuted_copy(sample: np.ndarray | CP_NDARRAY | TORCH_TENSOR, permutation):
    framework = get_framework_from_array(sample)
    if framework == Framework.torch:
        return torch.permute_copy(sample, permutation)
    else:
        return sample.transpose(permutation).copy()


def add_in_place(sample, addend):
    framework = get_framework_from_array(sample)
    if framework in [Framework.cupy, Framework.numpy]:
        return get_framework_module(framework).add(sample, addend, out=sample)
    else:
        assert framework is Framework.torch
        return sample.add_(addend)


def free_cupy_pool():
    if cp is not None:
        cp.get_default_memory_pool().free_all_blocks()


def free_torch_pool():
    if torch is not None:
        torch.cuda.empty_cache()


def free_framework_pools(framework, mem_backend):
    if mem_backend != MemBackend.cuda:
        free_cupy_pool()
        free_torch_pool()
    elif framework != Framework.cupy:
        free_cupy_pool()
    elif framework != Framework.torch:
        free_torch_pool()


def get_rev_perm(permutation):
    return tuple(np.argsort(permutation))


def unfold(array: np.ndarray | CP_NDARRAY | TORCH_TENSOR, dim, window_size, step):
    framework = get_framework_from_array(array)
    if framework == Framework.torch:
        return array.unfold(dim, window_size, step)
    else:
        assert framework in [Framework.numpy, Framework.cupy]
        module = get_framework_module(framework)
        shape = array.shape
        strides = array.strides
        assert 0 <= dim <= len(shape)
        assert shape[dim] >= window_size
        new_extent_size = (shape[dim] - window_size) // step + 1
        new_shape = list(shape)
        new_shape[dim] = new_extent_size
        new_shape = tuple(new_shape) + (window_size,)
        new_strides = list(strides)
        new_strides[dim] = strides[dim] * step
        new_strides = tuple(new_strides) + (strides[dim],)
        return module.lib.stride_tricks.as_strided(array, new_shape, new_strides)


def as_strided(array: np.ndarray | CP_NDARRAY | TORCH_TENSOR, shape, strides):
    framework = get_framework_from_array(array)
    if framework == Framework.torch:
        return torch.as_strided(array, shape, strides)
    else:
        assert framework in [Framework.numpy, Framework.cupy]
        module = get_framework_module(framework)
        item_size = array.itemsize
        strides = tuple(stride * item_size for stride in strides)
        return module.lib.stride_tricks.as_strided(array, shape, strides)


def unravel_index(idx, a):
    if TORCH_TENSOR is not None and isinstance(idx, TORCH_TENSOR):
        idx = idx.cpu() if idx.is_cuda else idx
    return np.unravel_index(idx, a.shape)


def arg_max(a):
    framework = get_framework_from_array(a)
    framework_module = get_framework_module(framework)
    return framework_module.argmax(a)


def assert_norm_close(
    a,
    a_ref,
    rtol=None,
    atol=None,
    axes=None,
    shape_kind=None,
    exec_backend=None,
):
    assert a.shape == a_ref.shape, f"{a.shape} != {a_ref.shape}"
    assert get_framework_from_array(a) == get_framework_from_array(a_ref), (
        f"{get_framework_from_array(a)} != {get_framework_from_array(a_ref)}"
    )
    dtype = get_dtype_from_array(a)
    ref_dtype = get_dtype_from_array(a_ref)
    assert dtype == ref_dtype, f"{dtype} != {ref_dtype}"
    if rtol is None or atol is None:
        rtol_default, atol_default = get_default_tolerance(dtype, shape_kind, exec_backend)
        rtol = rtol if rtol is not None else rtol_default
        atol = atol if atol is not None else atol_default
    dist = get_norm(a - a_ref, axes=axes)
    ref_mag = get_norm(a_ref, axes=axes)
    is_correct = dist < rtol * ref_mag + atol
    if not array_all(is_correct):
        if not is_array(dist):
            # assuming scalar
            offending_dist, offending_ref = dist, ref_mag
        else:
            offending_idx = arg_max(dist - rtol * ref_mag + atol)
            offending_coords = unravel_index(offending_idx, dist)
            offending_dist = dist[offending_coords]
            offending_ref = ref_mag[offending_coords]
        raise AssertionError(
            f"Tensors are not norm-close. The {a} and {a_ref} differ. \n"
            f"The assertion ``||a - a_ref||2 < rtol * ||a_ref|| + atol`` failed, with "
            f"||a-a_ref||2 = {offending_dist}, ||a_ref||2 = {offending_ref}, "
            f"rtol = {rtol}, atol = {atol} (max allowed ||a-a_ref||2 = {offending_ref * rtol + atol})."
        )


def assert_array_type(a: np.ndarray | CP_NDARRAY | TORCH_TENSOR, framework, mem_backend, dtype):
    assert_eq(get_framework_from_array(a), framework)
    assert_eq(get_array_backend(a), mem_backend)
    assert_eq(get_framework_dtype(framework, dtype), a.dtype)


def assert_array_equal(a, ref):
    framework = get_framework_from_array(a)
    if framework == Framework.numpy:
        np.testing.assert_array_equal(a, ref)
    elif framework == Framework.cupy:
        cp.testing.assert_array_equal(a, ref)
    else:
        assert framework == Framework.torch
        assert torch.equal(a, ref), f"{a} != {ref}"


def assert_eq(value, value_ref):
    assert value == value_ref, f"{value} != {value_ref}"


def intercept_xt_exec_device_id(monkeypatch):
    from nvmath.bindings import cufft  # type: ignore

    ctx = {"current_device_xt_exec": None}
    actual_xt_exec = cufft.xt_exec

    def _xt_exec(*args, **kwargs):
        nonlocal ctx
        ctx["current_device_xt_exec"] = cp.cuda.runtime.getDevice()
        return actual_xt_exec(*args, **kwargs)

    monkeypatch.setattr(cufft, "xt_exec", _xt_exec)

    return ctx


def intercept_device_id(monkeypatch, *calls):
    device_ids = {name: None for _, name in calls}

    def intercept(module, name):
        actual_method = getattr(module, name)

        def wrapper(*args, **kwargs):
            nonlocal device_ids
            device_ids[name] = cp.cuda.runtime.getDevice()
            return actual_method(*args, **kwargs)

        monkeypatch.setattr(module, name, wrapper)

    for module, name in calls:
        intercept(module, name)

    return device_ids


def intercept_default_allocations(monkeypatch):
    allocations = {"raw": 0, "cupy": 0, "torch": 0}
    from nvmath.internal.tensor_wrapper import maybe_register_package
    from nvmath.memory import _MEMORY_MANAGER

    def get_memalloc_wrapper(manager, alloc_key):
        actual = manager.memalloc_async

        def wrapper(self, *args, **kwargs):
            allocations[alloc_key] += 1
            return actual(self, *args, **kwargs)

        return wrapper

    managers = [
        (_MEMORY_MANAGER["_raw"], "raw"),
    ]

    if cp is not None:
        maybe_register_package("cupy")

        managers += [
            (_MEMORY_MANAGER["cupy"], "cupy"),
        ]

    if torch is not None:
        maybe_register_package("torch")

        managers += [
            (_MEMORY_MANAGER["torch"], "torch"),
        ]

    for manager, alloc_key in managers:
        monkeypatch.setattr(manager, "memalloc_async", get_memalloc_wrapper(manager, alloc_key))

    return allocations


def check_layout_fallback(data, axes, cb):
    try:
        cb(data, axes)
    except nvmath.fft.UnsupportedLayoutError as e:
        # Note, there is a catch with repeated-strides tensors in torch:
        # using the `torch.permute` -> `copy`/`contiguous` is not enough,
        # e.g. (64, 1) : (1, 1) --permute((1, 0))--> (1, 64) : (1, 1)
        # --clone/contiguous--> (1, 64) : (1, 1) <- the second stride remains 1!
        # User needs to call torch.permute_copy or
        # t = torch.permute(t, ) -> t.view(t.shape) -> t.copy/contiguous
        data_transposed = get_permuted_copy(data, e.permutation)
        assert is_decreasing(get_array_strides(data_transposed))
        res_transposed = cb(data_transposed, e.axes)
        return get_permuted_copy(res_transposed, get_rev_perm(e.permutation))
    else:
        raise AssertionError("The call was expected to raise unsupported layout error")


def should_skip_3d_unsupported(exec_backend, shape, axes=None):
    assert isinstance(exec_backend, ExecBackend), f"{exec_backend}"
    return (
        exec_backend == ExecBackend.cufft
        and get_cufft_version() < 10502
        and axes is not None
        and len(axes) == 3
        and math.prod(shape[a] for a in range(len(shape)) if a not in axes) > 1
    )


def extent_comprises_only_small_factors(extent):
    # fast track for powers of 2 (and zero)
    if extent & (extent - 1) == 0:
        return True

    for k in range(2, 128):
        while extent % k == 0:
            extent //= k
    return extent == 1


def has_only_small_factors(shape, axes=None):
    return all(extent_comprises_only_small_factors(extent) for a, extent in enumerate(shape) if axes is None or a in axes)
