# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
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
    pytest.skip("cupy is required for matmul tests", allow_module_level=True)

import numpy as np

import nvmath
import re


def sample_matrix(framework, dtype, shape, use_cuda):
    """
    Generates a sample matrix with random contents.
    """

    if framework == "numpy/cupy":
        framework = "cupy" if use_cuda else "numpy"

    if framework == "torch":
        if torch is None:
            pytest.skip("pytorch not present")
        dtype = getattr(torch, dtype)
        r = (10 * torch.rand(shape) - 5).type(dtype)
        return r.cuda() if use_cuda else r
    elif framework == "cupy":
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
    elif isinstance(tensor, cupy.ndarray):
        return cupy.asnumpy(tensor)
    elif isinstance(tensor, np.ndarray):
        return tensor
    else:
        raise AssertionError()


def get_tolerance(value):
    eps = np.finfo(to_numpy(value).dtype).eps
    if torch is not None and value.dtype == torch.bfloat16:
        eps = 2**-6
    return eps**0.5


def compare_tensors(result, reference):
    return np.allclose(to_numpy(result), to_numpy(reference), atol=get_tolerance(result))


def assert_tensors_equal(result, reference):
    """
    Checks if result is close to the provided numpy reference.
    """
    ok = compare_tensors(result, reference)
    if not ok:
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


class allow_cublas_unsupported:
    def __init__(self, *, allow_invalid_value=True, unsupported_before=None, message="Unsupported cublas version."):
        if allow_invalid_value:
            self.regex = r"\(CUBLAS_STATUS_(NOT_SUPPORTED|INVALID_VALUE)\)"
        else:
            self.regex = r"\(CUBLAS_STATUS_NOT_SUPPORTED\)"
        self.unsupported_before = unsupported_before
        self.message = message

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type is nvmath.bindings.cublasLt.cuBLASLtError:
            if re.search(self.regex, str(exc_value)):
                return skip_if_cublas_before(self.unsupported_before, self.message)
        return False
