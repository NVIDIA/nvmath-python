# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import nvmath
from nvmath.bindings import cublas
from nvmath.linalg.advanced import matmul, Matmul, MatmulOptions
from .utils import *
import pytest
import logging

try:
    import torch
except:
    torch = None

"""
This set of tests checks Matmul's options
"""


def check_matmul_with_options(size, options, use_cuda=False, dtype="float32"):
    a = b = sample_matrix("numpy/cupy", dtype, (size, size), use_cuda)
    result = matmul(a, b, alpha=0.42, options=options)
    assert_tensors_equal(result, 0.42 * (a @ b))
    return result


@pytest.mark.parametrize(
    "dtype,compute_type",
    (
        ("float32", cublas.ComputeType.COMPUTE_32F),
        ("float64", cublas.ComputeType.COMPUTE_64F),
        ("float16", cublas.ComputeType.COMPUTE_32F),
    ),
)
def test_compute_type(dtype, compute_type):
    check_matmul_with_options(
        5,
        MatmulOptions(compute_type=compute_type),
        dtype=dtype,
        use_cuda=True,
    )


@pytest.mark.parametrize(
    "dtype,scale_type",
    (
        ("float32", nvmath.CudaDataType.CUDA_R_32F),
        ("float64", nvmath.CudaDataType.CUDA_R_64F),
        ("float16", nvmath.CudaDataType.CUDA_R_32F),
    ),
)
def test_scale_type(dtype, scale_type):
    check_matmul_with_options(
        5,
        MatmulOptions(scale_type=scale_type),
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

    class MockDevice:
        def __init__(self, memory):
            self.mem_info = (None, memory)

    device = MockDevice(1000)
    if isinstance(expected_result, int):
        assert expected_result == nvmath._internal.utils.get_memory_limit(
            memory_limit, device
        )
    else:
        if isinstance(expected_result, tuple):
            exception, pattern = expected_result
        else:
            exception, pattern = expected_result, None

        with pytest.raises(exception, match=pattern):
            nvmath._internal.utils.get_memory_limit(memory_limit, device)


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
        return [int(alg.algorithm["workspace_size"]) for alg in algos]

    all_memory = get_memory_requirements(Matmul(a, b).plan())

    filtered = get_memory_requirements(
        Matmul(a, b, options=MatmulOptions(memory_limit="1 b")).plan()
    )

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

        def memalloc(self, size):
            print("ALLOC", size)
            self.counter += 1
            return super().memalloc(size)

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
    Tests if reasonable error is produced when an allocator class is provided instead of an instance
    """
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
    with pytest.raises(RuntimeError):
        check_matmul_with_options(10, options)
