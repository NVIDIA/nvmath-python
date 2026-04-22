# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
Test suite for reset_operand_unchecked() method
"""

import math

import pytest

try:
    import cupy as cp
except ImportError:
    cp = None

try:
    import torch
except ImportError:
    torch = None

import nvmath

from .utils.check_helpers import (
    assert_norm_close,
    copy_array,
    get_fft_ref,
)
from .utils.common_axes import (
    DType,
    Framework,
    MemBackend,
)
from .utils.input_fixtures import (
    get_random_input_data,
    init_assert_exec_backend_specified,
)
from .utils.support_matrix import (
    framework_exec_type_support,
    supported_backends,
)

# Enforce execution backend specification in tests
assert_exec_backend_specified = init_assert_exec_backend_specified()


@pytest.mark.parametrize(
    (
        "framework",
        "exec_backend",
        "mem_backend",
        "dtype",
    ),
    [
        (
            framework,
            exec_backend,
            exec_backend.mem,
            dtype,
        )
        for framework in Framework.enabled()
        for exec_backend in supported_backends.exec
        if exec_backend.mem in supported_backends.framework_mem[framework]
        for dtype in framework_exec_type_support[framework][exec_backend]
    ],
)
def test_swap_operand_many_iterations(
    seeder,
    framework,
    exec_backend,
    mem_backend,
    dtype,
):
    """
    Test reset_operand_unchecked with many sequential swaps.
    This is kind of a "sanity check" to ensure that the method works correctly.
    """
    shape = (64, 64)
    num_iterations = 10

    # Create initial operand and FFT instance
    operand = get_random_input_data(framework, shape, dtype, mem_backend)
    fft = nvmath.fft.FFT(operand, execution=exec_backend.nvname)

    with fft:
        fft.plan()

        # Perform many iterations with different operands
        for i in range(num_iterations):
            # Execute with current operand
            result = fft.execute()

            # Verify result is correct
            ref = get_fft_ref(operand)
            assert_norm_close(result, ref, exec_backend=exec_backend)

            # Generate new operand for next iteration
            if i < num_iterations - 1:
                operand = get_random_input_data(framework, shape, dtype, mem_backend)
                fft.reset_operand_unchecked(operand)


@pytest.mark.parametrize(
    ("framework", "dtype", "shape", "axes"),
    [
        (framework, dtype, shape, axes)
        for framework in Framework.enabled()
        if framework in (Framework.cupy, Framework.torch)  # Only CUDA-capable frameworks for this test
        for dtype in [DType.complex64, DType.complex128]
        for shape in [(32, 32), (64, 64), (128, 64)]
        for axes in [None, (0, 1), (0,), (1,)]
    ],
)
def test_swap_operand_vs_reset_operand_cuda(seeder, framework, dtype, shape, axes):
    """
    Test reset_operand_unchecked vs reset_operand on CUDA.
    This tests the CUDA memory + CUDA execution where operands
    are in theory simply swapped without copying.
    """
    # Generate test operands
    operand1 = get_random_input_data(framework, shape, dtype, MemBackend.cuda)
    operand2 = get_random_input_data(framework, shape, dtype, MemBackend.cuda)
    operand3 = get_random_input_data(framework, shape, dtype, MemBackend.cuda)

    # Test path 1: Using reset_operand (validated)
    fft1 = nvmath.fft.FFT(operand1, axes=axes, execution="cuda")
    with fft1:
        fft1.plan()
        result1a = copy_array(fft1.execute())

        fft1.reset_operand(operand2)
        result1b = copy_array(fft1.execute())

        fft1.reset_operand(operand3)
        result1c = copy_array(fft1.execute())

    # Test path 2: Using reset_operand_unchecked (unchecked)
    fft2 = nvmath.fft.FFT(operand1, axes=axes, execution="cuda")
    with fft2:
        fft2.plan()
        result2a = copy_array(fft2.execute())

        fft2.reset_operand_unchecked(operand2)
        result2b = copy_array(fft2.execute())

        fft2.reset_operand_unchecked(operand3)
        result2c = copy_array(fft2.execute())

    # Verify results match
    assert_norm_close(result1a, result2a, exec_backend="cuda")
    assert_norm_close(result1b, result2b, exec_backend="cuda")
    assert_norm_close(result1c, result2c, exec_backend="cuda")

    # Verify correctness against reference
    assert_norm_close(result2a, get_fft_ref(operand1, axes=axes), exec_backend="cuda")
    assert_norm_close(result2b, get_fft_ref(operand2, axes=axes), exec_backend="cuda")
    assert_norm_close(result2c, get_fft_ref(operand3, axes=axes), exec_backend="cuda")


@pytest.mark.parametrize(
    ("framework", "shape"),
    [
        (framework, shape)
        for framework in Framework.enabled()
        if framework in (Framework.cupy, Framework.torch)  # Only CUDA-capable frameworks for this test
        for shape in [(32, 32), (64, 64), (128, 64)]
    ],
)
def test_swap_operand_c2r_scenario(seeder, framework, shape):
    """
    Test reset_operand_unchecked with C2R transform.

    C2R requires special handling because it uses an auxiliary buffer to prevent
    the input from being overwritten by the FFT operation. This tests that
    reset_operand_unchecked correctly handles the copy to the auxiliary buffer.
    """
    dtype = DType.complex64

    operand1 = get_random_input_data(framework, shape, dtype, MemBackend.cuda)
    operand2 = get_random_input_data(framework, shape, dtype, MemBackend.cuda)

    # Create FFT with C2R option
    options = nvmath.fft.FFTOptions(fft_type="C2R")

    # Test with reset_operand
    fft1 = nvmath.fft.FFT(operand1, options=options, execution="cuda")
    with fft1:
        fft1.plan()
        result1a = copy_array(fft1.execute())

        fft1.reset_operand(operand2)
        result1b = copy_array(fft1.execute())

    # Test with reset_operand_unchecked
    fft2 = nvmath.fft.FFT(operand1, options=options, execution="cuda")
    with fft2:
        fft2.plan()
        result2a = copy_array(fft2.execute())

        fft2.reset_operand_unchecked(operand2)
        result2b = copy_array(fft2.execute())

    # Verify both methods produce identical results
    assert_norm_close(result1a, result2a, exec_backend="cuda")
    assert_norm_close(result1b, result2b, exec_backend="cuda")


@pytest.mark.parametrize(
    ("shape", "dtype"),
    [(shape, dtype) for shape in [(32, 32), (64, 64)] for dtype in [DType.complex64, DType.complex128]],
)
def test_swap_operand_cpu_operand_cuda_execution(seeder, shape, dtype):
    """
    Test CPU memory + CUDA execution.

    When operands are on CPU but execution is on CUDA, the operand must be
    copied to GPU. This tests that reset_operand_unchecked correctly
    handles this cross-device transfer.
    """
    framework = Framework.numpy
    mem_backend = MemBackend.cpu

    # Create operands on CPU (NumPy arrays)
    operand1 = get_random_input_data(framework, shape, dtype, mem_backend)
    operand2 = get_random_input_data(framework, shape, dtype, mem_backend)

    # Test with reset_operand
    fft1 = nvmath.fft.FFT(operand1, execution="cuda")
    with fft1:
        fft1.plan()
        result1a = copy_array(fft1.execute())

        fft1.reset_operand(operand2)
        result1b = copy_array(fft1.execute())

    # Test with reset_operand_unchecked
    fft2 = nvmath.fft.FFT(operand1, execution="cuda")
    with fft2:
        fft2.plan()
        result2a = copy_array(fft2.execute())

        fft2.reset_operand_unchecked(operand2)
        result2b = copy_array(fft2.execute())

    # Verify results match
    assert_norm_close(result1a, result2a, exec_backend="cuda")
    assert_norm_close(result1b, result2b, exec_backend="cuda")

    # Verify correctness
    assert_norm_close(result2a, get_fft_ref(operand1), exec_backend="cuda")
    assert_norm_close(result2b, get_fft_ref(operand2), exec_backend="cuda")


@pytest.mark.parametrize(
    (
        "framework",
        "exec_backend",
        "mem_backend",
        "dtype",
        "inplace",
    ),
    [
        (
            framework,
            exec_backend,
            exec_backend.mem,
            dtype,
            inplace,
        )
        for framework in Framework.enabled()
        for exec_backend in supported_backends.exec
        if exec_backend.mem in supported_backends.framework_mem[framework]
        for dtype in [DType.complex64]
        for inplace in [True, False]
    ],
)
def test_swap_operand_inplace_vs_out_of_place(
    seeder,
    framework,
    exec_backend,
    mem_backend,
    dtype,
    inplace,
):
    """
    Test reset_operand_unchecked with both in-place and out-of-place FFTs.
    """
    shape = (64, 64)

    # Generate test operands
    # Note: We need separate copies for inplace mode since execute() modifies the input
    operand1_for_path1 = get_random_input_data(framework, shape, dtype, mem_backend)
    operand1_for_path2 = copy_array(operand1_for_path1)
    operand2_for_path1 = get_random_input_data(framework, shape, dtype, mem_backend)
    operand2_for_path2 = copy_array(operand2_for_path1)

    # Test with reset_operand
    fft1 = nvmath.fft.FFT(operand1_for_path1, execution=exec_backend.nvname, options=nvmath.fft.FFTOptions(inplace=inplace))
    with fft1:
        fft1.plan()
        result1a = copy_array(fft1.execute())

        fft1.reset_operand(operand2_for_path1)
        result1b = copy_array(fft1.execute())

    # Test with reset_operand_unchecked
    fft2 = nvmath.fft.FFT(operand1_for_path2, execution=exec_backend.nvname, options=nvmath.fft.FFTOptions(inplace=inplace))
    with fft2:
        fft2.plan()
        result2a = copy_array(fft2.execute())

        fft2.reset_operand_unchecked(operand2_for_path2)
        result2b = copy_array(fft2.execute())

    # Verify results match
    assert_norm_close(result1a, result2a, exec_backend=exec_backend)
    assert_norm_close(result1b, result2b, exec_backend=exec_backend)


@pytest.mark.parametrize(
    (
        "framework",
        "exec_backend",
        "mem_backend",
        "dtype",
    ),
    [
        (
            framework,
            exec_backend,
            exec_backend.mem,
            dtype,
        )
        for framework in Framework.enabled()
        for exec_backend in supported_backends.exec
        if exec_backend.mem in supported_backends.framework_mem[framework]
        for dtype in [DType.complex64]
    ],
)
def test_swap_operand_with_forward_inverse(
    seeder,
    framework,
    exec_backend,
    mem_backend,
    dtype,
):
    """
    Test reset_operand_unchecked with forward and inverse transforms.

    Verifies that swapping operands works correctly when alternating between
    forward and inverse FFT directions.
    """
    shape = (64, 64)
    axes = (0, 1)
    scale = math.prod(shape[a] for a in axes)

    signal = get_random_input_data(framework, shape, dtype, mem_backend)

    fft = nvmath.fft.FFT(signal, axes=axes, execution=exec_backend.nvname)
    with fft:
        fft.plan()

        # Forward transform
        result_fwd = copy_array(fft.execute(direction="forward"))
        assert_norm_close(result_fwd, get_fft_ref(signal, axes=axes), exec_backend=exec_backend)

        # Swap to the forward result for inverse
        fft.reset_operand_unchecked(result_fwd)

        # Inverse transform
        result_inv = copy_array(fft.execute(direction="inverse"))

        # Should recover original signal (scaled)
        signal_recovered = result_inv / scale
        assert_norm_close(signal_recovered, signal, exec_backend=exec_backend)
