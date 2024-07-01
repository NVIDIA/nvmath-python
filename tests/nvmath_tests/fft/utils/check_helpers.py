# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

from typing import Union, Optional, List
from itertools import accumulate
import math
import numpy as np
import cupy as cp
try:
    import torch
except:
    torch = None

import nvmath

from .common_axes import Framework, DType, ShapeKind
from .axes_utils import (
    TORCH_TENSOR,
    get_framework_dtype,
    get_fft_dtype,
    is_complex,
    is_array,
    size_of,
    get_dtype_from_array,
    get_framework_from_array,
    get_framework_module,
    get_array_backend,
)

_cufft_version = None

def _get_cufft_version():
    from nvmath.bindings import cufft as _cufft
    return _cufft.get_version()


def get_cufft_version():
    global _cufft_version
    if _cufft_version is None:
        _cufft_version = _get_cufft_version()
    return _cufft_version


def slice_r2c(fft, axes=None):
    ndim = fft.ndim
    last_fft_axis = ndim - 1 if axes is None else axes[-1]
    last_slice = slice(None, fft.shape[last_fft_axis] // 2 + 1)
    slices = tuple(
        slice(None) if i != last_fft_axis else last_slice for i in range(ndim)
    )
    return fft[slices]


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
    sample: cp.ndarray,
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
    # torch does not implement half precision fft
    if dtype == DType.complex32:
        in_sample = sample.type(torch.complex64)
    elif dtype == DType.float16:
        in_sample = sample.type(torch.float32)
    else:
        in_sample = sample

    ref = torch.fft.fftn(in_sample, dim=axes, norm=norm)

    if not is_complex(dtype):
        ref = slice_r2c(ref, axes=axes)

    if dtype in [DType.complex32, DType.float16]:
        assert_eq(ref.dtype, torch.complex64)
        ref = ref.type(torch.complex32)

    assert_eq(ref.dtype, get_framework_dtype(Framework.torch, get_fft_dtype(dtype)))
    return ref


def get_fft_ref(
    sample: Union[np.ndarray, cp.ndarray, TORCH_TENSOR],
    axes: Optional[List] = None,
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


def get_scaled(sample: Union[np.ndarray, cp.ndarray, TORCH_TENSOR], scale):
    sample_scaled = sample * scale
    if sample_scaled.dtype != sample.dtype:
        sample_scaled = sample_scaled.astype(sample.dtype)
    return sample_scaled


def get_transposed(
    sample: Union[np.ndarray, cp.ndarray, TORCH_TENSOR], d1: int, d2: int
):
    framework = get_framework_from_array(sample)
    if framework == Framework.torch:
        return sample.transpose(d1, d2)
    else:
        swap = {d1: d2, d2: d1}
        transpose_axes = [i if i not in swap else swap[i] for i in range(sample.ndim)]
        return sample.transpose(transpose_axes)


def use_stream(stream):
    if isinstance(stream, cp.cuda.Stream):
        return stream
    elif isinstance(stream, torch.cuda.Stream):
        return torch.cuda.stream(stream)
    else:
        raise ValueError(f"Unknown stream type {type(stream)}")


def get_array_device_id(array: Union[cp.ndarray, TORCH_TENSOR]) -> int:
    if isinstance(array, cp.ndarray):
        return array.device.id
    elif isinstance(array, TORCH_TENSOR):
        return array.device.index
    else:
        raise ValueError(f"Unknown device array type {array}")


def record_event(stream):
    if isinstance(stream, cp.cuda.Stream):
        return stream.record()
    elif isinstance(stream, torch.cuda.Stream):
        return stream.record_event()
    else:
        raise ValueError(f"Unknown stream type {type(stream)}")


def assert_all_close(a, b, rtol, atol):
    assert type(a) == type(b), f"{type(a)}!= {type(b)}"
    if isinstance(a, np.ndarray):
        return np.testing.assert_allclose(a, b, rtol=rtol, atol=atol)
    elif isinstance(a, cp.ndarray):
        return cp.testing.assert_allclose(a, b, rtol=rtol, atol=atol)
    elif isinstance(a, TORCH_TENSOR):
        return torch.testing.assert_close(a, b, rtol=rtol, atol=atol)
    else:
        raise ValueError(f"Unknown array type {a}")


def get_ifft_c2r_options(out_type, last_axis_size):
    if is_complex(out_type):
        return {}
    return {
        "fft_type": "C2R",
        "last_axis_size": "odd" if last_axis_size % 2 else "even",
    }


def get_default_tolerance(dtype: DType, shape_kind : Optional[ShapeKind]):
    dtype_bytes_num = size_of(dtype)
    if is_complex(dtype):
        dtype_bytes_num //= 2
    if dtype_bytes_num == 2:
        return 1e-2, cp.finfo(cp.float16).eps
    elif dtype_bytes_num == 4:
        return 1e-6, cp.finfo(cp.float32).eps
    elif dtype_bytes_num == 8:
        atol = cp.finfo(cp.float64).eps
        if shape_kind is None or shape_kind in (ShapeKind.pow2, ShapeKind.pow2357):
            return 1e-14, atol
        # The differences between the cufft and the ref for double precision
        # tensors seem to be bigger if shape comprises larger primes,
        # especially in older CTKs
        cufft_version = get_cufft_version()
        if cufft_version >= 10702:  # shipped with CTK 11.7
            return 3e-14, atol
        else:
            return 1e-12, atol
    else:
        raise ValueError(f"Unexpected dtype {dtype}")


def get_array_strides(a: Union[np.ndarray, cp.ndarray, TORCH_TENSOR]):
    if isinstance(a, (np.ndarray, cp.ndarray)):
        return a.strides
    else:
        return a.stride()


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


def copy_array(a):
    if get_framework_from_array(a) != Framework.torch:
        return a.copy()
    else:
        if not a.is_contiguous():
            return a.contiguous()
        else:
            return a.clone()
    

def get_permuted(
    sample: Union[np.ndarray, cp.ndarray, TORCH_TENSOR], permutation
):
    framework = get_framework_from_array(sample)
    if framework == Framework.torch:
        return torch.permute(sample, permutation)
    else:
        return sample.transpose(permutation)


def get_permuted_copy(
    sample: Union[np.ndarray, cp.ndarray, TORCH_TENSOR], permutation
):
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
    import cupy
    cupy.get_default_memory_pool().free_all_blocks()


def get_rev_perm(permutation):
    return tuple(np.argsort(permutation))


def unfold(array : Union[np.ndarray, cp.ndarray, TORCH_TENSOR], dim, window_size, step):
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


def as_strided(array : Union[np.ndarray, cp.ndarray, TORCH_TENSOR], shape, strides):
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


def assert_norm_close(a, a_ref, rtol=None, atol=None, axes=None, shape_kind=None):
    assert a.shape == a_ref.shape, f"{a.shape} != {a_ref.shape}"
    assert get_framework_from_array(a) == get_framework_from_array(
        a_ref
    ), f"{get_framework_from_array(a)} != {get_framework_from_array(a_ref)}"
    dtype = get_dtype_from_array(a)
    ref_dtype = get_dtype_from_array(a_ref)
    assert dtype == ref_dtype, f"{dtype} != {ref_dtype}"
    if rtol is None or atol is None:
        rtol_default, atol_default = get_default_tolerance(dtype, shape_kind)
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


def assert_array_type(
    a: Union[np.ndarray, cp.ndarray, TORCH_TENSOR], framework, backend, dtype
):
    assert_eq(get_framework_from_array(a), framework)
    assert_eq(get_array_backend(a), backend)
    assert_eq(get_framework_dtype(framework, dtype), a.dtype)


def assert_eq(value, value_ref):
    assert value == value_ref, f"{value} != {value_ref}"


def intercept_xt_exec_device_id(monkeypatch):
    from nvmath.bindings import cufft

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

    from nvmath.memory import (
        _RawCUDAMemoryManager,
        _CupyCUDAMemoryManager,
        _TorchCUDAMemoryManager,
    )

    def get_memalloc_wrapper(manager, alloc_key):
        actual = manager.memalloc

        def wrapper(self, size):
            allocations[alloc_key] += 1
            return actual(self, size)

        return wrapper

    for manager, alloc_key in (
        (_RawCUDAMemoryManager, "raw"),
        (_CupyCUDAMemoryManager, "cupy"),
        (_TorchCUDAMemoryManager, "torch"),
    ):
        monkeypatch.setattr(
            manager, "memalloc", get_memalloc_wrapper(manager, alloc_key)
        )

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
        assert False, "The call was expected to raise unsupported layout error"


def should_skip_3d_unsupported(shape, axes=None):
    return (
        get_cufft_version() < 10502
        and axes is not None
        and len(axes) == 3
        and math.prod(shape[a] for a in range(len(shape)) if a not in axes) > 1
    )
