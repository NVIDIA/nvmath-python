# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

from nvmath.bindings import cublas
from nvmath.linalg.advanced import matmul, Matmul, MatmulOptions
import logging
import nvmath
import pytest

from .utils import assert_tensors_equal, sample_matrix, is_torch_available

try:
    import cupy_backends.cuda
except ModuleNotFoundError:
    pytest.skip("cupy is required for matmul tests", allow_module_level=True)


try:
    import torch
except:
    torch = None

"""
This set of tests checks Matmul's options
"""


def check_matmul_with_options(size, options, use_cuda=False, dtype="float32", atol=None, rtol=None):
    a = b = sample_matrix("numpy/cupy" if dtype != "bfloat16" else "torch", dtype, (size, size), use_cuda)
    is_complex = "_C_" in str(options.scale_type) or (options.compute_type is None and "complex" in dtype)
    alpha = 0.42 + 0.24j if is_complex else 0.42
    result = matmul(a, b, alpha=alpha, options=options)
    assert_tensors_equal(result, alpha * (a @ b), atol=atol, rtol=rtol)
    return result


ct = cublas.ComputeType
st = nvmath.CudaDataType


@pytest.mark.parametrize(
    "dtype,compute_type,scale_type",
    (
        # None specified
        ("bfloat16", None, None),
        ("float16", None, None),
        ("float32", None, None),
        ("float64", None, None),
        ("complex64", None, None),
        ("complex128", None, None),
        # Only compute type specified
        ("float16", ct.COMPUTE_16F, None),
        ("float16", ct.COMPUTE_16F_PEDANTIC, None),
        ("float16", ct.COMPUTE_32F, None),
        ("float32", ct.COMPUTE_32F, None),
        ("bfloat16", ct.COMPUTE_32F_PEDANTIC, None),
        ("complex64", ct.COMPUTE_32F, None),
        ("float16", ct.COMPUTE_32F_PEDANTIC, None),
        ("float32", ct.COMPUTE_32F_PEDANTIC, None),
        ("bfloat16", ct.COMPUTE_32F_PEDANTIC, None),
        ("complex64", ct.COMPUTE_32F_PEDANTIC, None),
        ("float32", ct.COMPUTE_32F_FAST_16F, None),
        ("float32", ct.COMPUTE_32F_FAST_16BF, None),
        ("float32", ct.COMPUTE_32F_FAST_TF32, None),
        ("float64", ct.COMPUTE_64F, None),
        ("float64", ct.COMPUTE_64F_PEDANTIC, None),
        ("complex128", ct.COMPUTE_64F, None),
        ("complex128", ct.COMPUTE_64F_PEDANTIC, None),
        # Only scale type specified
        ("float16", None, st.CUDA_R_16F),
        ("float16", None, st.CUDA_R_32F),
        ("bfloat16", None, st.CUDA_R_32F),
        ("float32", None, st.CUDA_R_32F),
        ("complex64", None, st.CUDA_C_32F),
        ("float32", None, st.CUDA_R_32F),
        ("float64", None, st.CUDA_R_64F),
        ("complex128", None, st.CUDA_C_64F),
        # Both compute and scale type specified
        ("float16", ct.COMPUTE_16F, st.CUDA_R_16F),
        ("float16", ct.COMPUTE_16F_PEDANTIC, st.CUDA_R_16F),
        ("float16", ct.COMPUTE_32F, st.CUDA_R_32F),
        ("bfloat16", ct.COMPUTE_32F, st.CUDA_R_32F),
        ("float32", ct.COMPUTE_32F, st.CUDA_R_32F),
        ("complex64", ct.COMPUTE_32F, st.CUDA_C_32F),
        ("float16", ct.COMPUTE_32F_PEDANTIC, st.CUDA_R_32F),
        ("bfloat16", ct.COMPUTE_32F_PEDANTIC, st.CUDA_R_32F),
        ("float32", ct.COMPUTE_32F_PEDANTIC, st.CUDA_R_32F),
        ("complex64", ct.COMPUTE_32F_PEDANTIC, st.CUDA_C_32F),
        ("float32", ct.COMPUTE_32F_FAST_16F, st.CUDA_R_32F),
        ("float32", ct.COMPUTE_32F_FAST_16BF, st.CUDA_R_32F),
        ("float32", ct.COMPUTE_32F_FAST_TF32, st.CUDA_R_32F),
        ("float64", ct.COMPUTE_64F, st.CUDA_R_64F),
        ("float64", ct.COMPUTE_64F_PEDANTIC, st.CUDA_R_64F),
        ("complex128", ct.COMPUTE_64F, st.CUDA_C_64F),
        ("complex128", ct.COMPUTE_64F_PEDANTIC, st.CUDA_C_64F),
    ),
)
def test_compute_scale_type(dtype, compute_type, scale_type):
    check_matmul_with_options(
        2,
        MatmulOptions(compute_type=compute_type, scale_type=scale_type),
        dtype=dtype,
        use_cuda=True,
        atol=0.1,
        rtol=None,
    )


@pytest.mark.parametrize(
    "dtype,compute_type,scale_type",
    (
        ("float16", ct.COMPUTE_32F, st.CUDA_R_16F),
        ("float32", ct.COMPUTE_16F, st.CUDA_R_32F),
        ("float64", ct.COMPUTE_64F, st.CUDA_R_32F),
        ("complex64", ct.COMPUTE_32F_PEDANTIC, st.CUDA_R_32F),
        ("float64", ct.COMPUTE_32F_FAST_16F, st.CUDA_R_32F),
        ("float16", ct.COMPUTE_32F_FAST_16BF, st.CUDA_R_32F),
    ),
)
def test_unsupported_compute_scale_type(dtype, compute_type, scale_type):
    with pytest.raises(Exception, match="not supported|INVALID_VALUE|NOT_SUPPORTED"):
        check_matmul_with_options(
            2,
            MatmulOptions(compute_type=compute_type, scale_type=scale_type),
            dtype=dtype,
            use_cuda=True,
        )


@pytest.mark.parametrize(
    "memory_limit, expected_result",
    (
        (8, 8),
        (0.5, 500),
        (1.0, 1000),
        (1, 1),
        ("0.01", ValueError),
        ("8 b", 8),
        ("100%", 1000),
        ("1gib", 1024**3),
        ("2mib", 2 * 1024**2),
        ("3kib", 3 * 1024),
        ("4 GiB", 4 * 1024**3),
        ("5 MiB", 5 * 1024**2),
        ("6 KiB", 6 * 1024),
        ("1gb", 1000**3),
        ("2mb", 2 * 1000**2),
        ("3kb", 3 * 1000),
        ("4 GB", 4 * 1000**3),
        ("5 MB", 5 * 1000**2),
        ("6 KB", 6 * 1000),
        ("6e2 KB", 600 * 1000),
        ("1e-1 Kb", 100),
        ("0.1 Kb", 100),
        ("123 megabytes", ValueError),
        (-1, ValueError),
        (-0.1, ValueError),
        ("-1%", ValueError),
        ("-1gib", ValueError),
        (
            "-1",
            (
                ValueError,
                "The memory limit must be specified in one of the following forms",
            ),
        ),
    ),
)
def test_memory_limit_parsing(memory_limit, expected_result):
    """
    Tests if various forms of memory limits are parsed correctly.
    """
    if isinstance(expected_result, int):
        assert expected_result == nvmath.internal.utils._get_memory_limit(memory_limit, 1_000)
    else:
        if isinstance(expected_result, tuple):
            exception, pattern = expected_result
        else:
            exception, pattern = expected_result, None

        with pytest.raises(exception, match=pattern):
            nvmath.internal.utils._get_memory_limit(memory_limit, 1_000)


def test_memory_limit():
    """
    Tests if specifying a memory limit doesn't break anything
    """
    options = MatmulOptions()
    options.memory_limit = 0.9
    check_matmul_with_options(10, options)


def test_memory_limit_filtering():
    """
    Tests if some algorithms are filtered with memory limit set.
    """
    a = b = sample_matrix("numpy/cupy", "float32", (1000, 1000), True)

    def get_memory_requirements(algos):
        return [alg.algorithm.workspace_size for alg in algos]

    all_memory = get_memory_requirements(Matmul(a, b).plan())

    filtered = get_memory_requirements(Matmul(a, b, options=MatmulOptions(memory_limit="1 b")).plan())

    assert max(filtered) < max(all_memory)


def test_logger():
    """
    Tests if specifying a custom logger works as expected.
    """
    import logging
    from io import StringIO

    log_stream = StringIO()
    logger = logging.Logger("test_logger", level=logging.DEBUG)
    logger.addHandler(logging.StreamHandler(log_stream))
    options = MatmulOptions(logger=logger)
    check_matmul_with_options(10, options)
    assert len(log_stream.getvalue()) > 0


def test_allocator():
    """
    Tests if manually specifying an allocator works
    """
    if not is_torch_available():
        pytest.skip("no pytorch")

    from nvmath.memory import _TorchCUDAMemoryManager

    allocator = _TorchCUDAMemoryManager(0, logging.getLogger())
    options = MatmulOptions(allocator=allocator)
    check_matmul_with_options(10, options)


def test_different_allocator():
    """
    Tests if matmul of torch tensors can be performed with cupy allocator
    """
    from nvmath.memory import _CupyCUDAMemoryManager

    allocator = _CupyCUDAMemoryManager(0, logging.getLogger())
    options = MatmulOptions(allocator=allocator)
    check_matmul_with_options(10, options)


def test_custom_allocator():
    """
    Checks if custom allocator is actually used
    """
    if not is_torch_available():
        pytest.skip("no pytorch")

    from nvmath.memory import _TorchCUDAMemoryManager

    class MockAllocator(_TorchCUDAMemoryManager):
        def __init__(self, device_id, logger):
            super().__init__(device_id, logger)
            self.counter = 0

        def memalloc(self, size, *args, **kwargs):
            print("ALLOC", size)
            self.counter += 1
            return super().memalloc(size, *args, **kwargs)

    allocator = MockAllocator(0, logging.getLogger())
    options = MatmulOptions(allocator=allocator)
    check_matmul_with_options(10, options)
    assert allocator.counter >= 0


def test_invalid_allocator():
    """
    Tests if reasonable error is produced when an invalid allocator is specified
    """
    with pytest.raises(TypeError):
        MatmulOptions(allocator="Hello, I'm a real allocator!")


def test_uninstantiated_allocator():
    """
    Tests if reasonable error is produced when an allocator class is provided instead of an
    instance
    """
    if not is_torch_available():
        pytest.skip("no pytorch")

    from nvmath.memory import _TorchCUDAMemoryManager

    try:
        # This may not fail if allocator won't be used
        options = MatmulOptions(allocator=_TorchCUDAMemoryManager)
        check_matmul_with_options(10, options)
    except TypeError:
        pass


def test_device_id():
    """
    Tests if specifying a device id works as expected.
    """
    options = MatmulOptions(device_id=0)
    check_matmul_with_options(10, options, use_cuda=False)


def test_invalid_device_id():
    """
    Tests if specifying negative device id raises an error
    """
    options = MatmulOptions(device_id=-1)
    with pytest.raises((RuntimeError, cupy_backends.cuda.api.runtime.CUDARuntimeError, ValueError), match="device"):
        check_matmul_with_options(10, options)
