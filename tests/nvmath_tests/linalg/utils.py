# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

try:
    import torch
except:
    torch = None

import pytest

try:
    import cupy
except ModuleNotFoundError:
    cupy = None

import datetime
import math
import re

import numpy as np

import nvmath
from nvmath.linalg._internal.utils import axis_order_in_memory, calculate_strides


def sample_matrix(framework, dtype, shape, use_cuda, min=-5, max=5):
    """
    Generates a sample matrix with random contents.
    """

    if framework == "numpy/cupy":
        framework = "cupy" if use_cuda else "numpy"

    if framework == "torch":
        if torch is None:
            pytest.skip("pytorch not present")
        dtype = getattr(torch, dtype)
        r = ((max - min) * torch.rand(shape) + min).type(dtype)
        return r.cuda() if use_cuda else r
    elif framework == "cupy":
        if cupy is None:
            pytest.skip("cupy not installed")
        if not use_cuda:
            raise NotImplementedError("CPU tensors not supported by cupy")
        if dtype == "bfloat16":
            raise NotImplementedError("bfloat16 not supported by cupy")
        r = (10 * cupy.random.rand(*shape) - 5).astype(dtype)
        return r
    elif framework == "numpy":
        if use_cuda:
            raise NotImplementedError("GPU tensors not supported by numpy")
        if dtype == "bfloat16":
            raise NotImplementedError("bfloat16 not supported by numpy")
        r = (10 * np.random.rand(*shape) - 5).astype(dtype)
        return r


def sample_float_tensor(shape, use_cuda=True):
    return sample_matrix("numpy/cupy", "float32", shape, use_cuda)


def to_numpy(tensor):
    """
    Converts whatever is provided to a numpy ndarray.
    """
    if torch is not None and isinstance(tensor, torch.Tensor):
        if tensor.device:
            tensor = tensor.cpu()
        if tensor.dtype in (torch.bfloat16,):
            tensor = tensor.type(torch.float64)
        return tensor.numpy()
    elif cupy is not None and isinstance(tensor, cupy.ndarray):
        return cupy.asnumpy(tensor)
    elif isinstance(tensor, np.ndarray):
        return tensor
    elif isinstance(tensor, np.number):
        return np.array(tensor)
    else:
        msg = f"Cannot convert to numpy from {type(tensor)}"
        raise AssertionError(msg)


def get_framework(tensor):
    if torch is not None and isinstance(tensor, torch.Tensor):
        return torch
    elif cupy is not None and isinstance(tensor, cupy.ndarray):
        return cupy
    elif isinstance(tensor, np.ndarray):
        return np
    else:
        msg = f"framework of {type(tensor)} is unknown"
        raise AssertionError(msg)


def get_machine_eps(value):
    eps = np.finfo(to_numpy(value).dtype).eps
    if torch is not None and value.dtype == torch.bfloat16:
        eps = 2**-6
    return eps


def get_absolute_tolerance(value):
    return get_machine_eps(value) ** 0.5


def get_relative_tolerance(value):
    return max(1e-5, get_machine_eps(value) ** 0.5)


def compare_tensors(result, reference, atol=None, rtol=None):
    if atol is None:
        atol = get_absolute_tolerance(result)
    if rtol is None:
        rtol = get_relative_tolerance(result)
    return np.allclose(to_numpy(result), to_numpy(reference), atol=atol, rtol=rtol)


def assert_tensors_equal(result, reference, atol=None, rtol=None):
    """
    Checks if result is close to the provided numpy reference.
    """
    assert result is not reference, "same object passed as `result` and `reference`!"
    ok = compare_tensors(result, reference, atol=atol, rtol=rtol)
    if not ok:
        print(f"Absdiff: {np.max(np.abs(to_numpy(result) - to_numpy(reference)))}")
        print(f"Reldiff: {np.max(np.abs(to_numpy(result) - to_numpy(reference)) / (np.abs(to_numpy(reference)) + 0.000001))}")
        print("Result:\n", result)
        print("Reference:\n", reference)
    assert ok


def is_torch_available():
    return torch is not None


def random_torch_complex(shape, use_cuda, transposed=False):
    if transposed:
        shape = tuple(reversed(shape))
    real = sample_matrix("torch", "float32", shape, use_cuda)
    imag = sample_matrix("torch", "float32", shape, use_cuda)
    result = real + 1j * imag
    return result.T if transposed else result


def skip_if_cublas_before(version, message="Unsupported cublas version."):
    if not version or nvmath.bindings.cublasLt.get_version() < version:
        pytest.skip(message)
        return True
    return False


# Setting the seed once per day allows randomness, but helps with reproducibility.
matmul_with_random_autotune_rng = np.random.default_rng(seed=abs(hash(datetime.date.today())))


def matmul_with_random_autotune(*args, p=0.25, **kwargs):
    """
    Executes matmul, using autotuning with probability p.
    """
    constructor_kwargs = ("c", "alpha", "beta", "qualifiers", "options", "stream", "quantization_scales")
    plan_kwargs = ("preferences", "epilog", "epilog_inputs", "algorithms", "stream")
    execute_kwargs = ("stream",)
    with nvmath.linalg.advanced.Matmul(*args, **{k: kwargs[k] for k in constructor_kwargs if k in kwargs}) as mm:
        mm.plan(**{k: kwargs[k] for k in plan_kwargs if k in kwargs})
        if matmul_with_random_autotune_rng.random() < p:
            mm.autotune(iterations=3)
        return mm.execute(**{k: kwargs[k] for k in execute_kwargs if k in kwargs})


class allow_cublas_unsupported:
    def __init__(self, *, allow_invalid_value=True, unsupported_before=None, message="Unsupported cublas version."):
        if allow_invalid_value:
            self.regex = r"\(?(CUBLAS_STATUS_)?(NOT_SUPPORTED|INVALID_VALUE)\)?"
        else:
            self.regex = r"\(?(CUBLAS_STATUS_)?NOT_SUPPORTED\)?"
        self.unsupported_before = unsupported_before
        self.message = message

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type is nvmath.bindings.cublasLt.cuBLASLtError and re.search(self.regex, str(exc_value)):
            return skip_if_cublas_before(self.unsupported_before, self.message)
        return False


def pad_and_slice(x: "torch.Tensor", row_pad_elems: int = 16, col_pad_elems: int = 16) -> "torch.Tensor":
    """
    Embed a tensor in a larger allocation with padding on all four sides,
    then return the inner slice.

    The result has the same shape and data as the input, but lives inside a
    bigger buffer -- so it has padded strides and a non-zero storage_offset,
    as if the caller had sliced a sub-region from a larger tensor.
    Works with any dtype and preserves the row-major / column-major
    layout of the input.

    Preconditions:
        - ``x`` is a :class:`torch.Tensor` with ``ndim >= 2``.
        - ``x`` is either row-major (``stride[-1] == 1``) or column-major
          (``stride[-2] == 1``).
        - ``row_pad_elems >= 0`` and ``col_pad_elems >= 0``.

    Args:
        x: Torch tensor (>= 2D). For D >= 3 the tensor is assumed batched
           (leading dimensions are batch dimensions).
        row_pad_elems: Non-negative number of elements of padding above and
                       below each matrix (default 16). The total byte padding
                       is computed as row_pad_elems * element_size.
                       For FP4 operands the resulting storage offset must preserve
                       cuBLASLt pointer alignment (16 bytes).
        col_pad_elems: Non-negative number of elements of padding to the
                       left and right of each matrix (default 16). The total byte
                       padding is computed as col_pad_elems * element_size.

    Returns:
        Tensor with the same logical shape as x, viewed as a slice of a larger
        allocation (non-zero storage_offset, padded strides).
    """
    if torch is None:
        raise RuntimeError("pad_and_slice requires PyTorch")
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"Expected a torch.Tensor, got {type(x).__name__}")
    if x.ndim < 2:
        raise ValueError(f"Expected at least 2D tensor, got {x.ndim}D")
    if x.stride(-1) != 1 and x.stride(-2) != 1:
        raise ValueError(f"Expected row-major (stride[-1]==1) or column-major (stride[-2]==1) layout, got strides {x.stride()}")
    if row_pad_elems < 0:
        raise ValueError(f"row_pad_elems must be non-negative, got {row_pad_elems}")
    if col_pad_elems < 0:
        raise ValueError(f"col_pad_elems must be non-negative, got {col_pad_elems}")

    # Safe even when torch lacks FP4 support: if float4_e2m1fn_x2 doesn't
    # exist, _fp4_dtype is None, is_fp4 is always False, and the FP4 code
    # paths are never reached.  No tensor can have an FP4 dtype on a torch
    # build that doesn't define it.
    _fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)
    is_fp4 = _fp4_dtype is not None and x.dtype == _fp4_dtype
    work_dtype = torch.uint8 if is_fp4 else x.dtype
    x_work = x.view(work_dtype) if is_fp4 else x

    device = x.device
    batch_shape, r, c = x.shape[:-2], x.shape[-2], x.shape[-1]
    padded_num_rows = r + 2 * row_pad_elems
    padded_num_cols = c + 2 * col_pad_elems

    axis_order = axis_order_in_memory(strides=x.stride())
    buf_shape = (*batch_shape, padded_num_rows, padded_num_cols)
    buf_strides = calculate_strides(shape=buf_shape, axis_order=axis_order)
    buf_flat = torch.empty(math.prod(buf_shape), dtype=work_dtype, device=device)
    buf = torch.as_strided(buf_flat, size=buf_shape, stride=buf_strides)
    result = buf[..., row_pad_elems : row_pad_elems + r, col_pad_elems : col_pad_elems + c]
    result[...] = x_work

    if is_fp4:
        # result is uint8; we need to return FP4.  A simple
        # result.view(_fp4_dtype) fails for column-major slices
        # (stride[-1] != 1), so reinterpret the flat buffer as FP4
        # and carve out the same region via as_strided.
        # Since uint8 and FP4 are both 1 byte per element, the
        # strides and offsets are identical.
        buf_fp4 = buf_flat.view(_fp4_dtype)
        result = torch.as_strided(
            buf_fp4,
            size=result.size(),
            stride=result.stride(),
            storage_offset=result.storage_offset(),
        )
        if result.data_ptr() % 16 != 0:
            raise ValueError(
                f"Padded slice is not 16-byte aligned (data_ptr=0x{result.data_ptr():x}). "
                f"Adjust row_pad_elems ({row_pad_elems}) and col_pad_elems ({col_pad_elems}) "
                f"so that the storage offset is a multiple of 16."
            )

    return result
