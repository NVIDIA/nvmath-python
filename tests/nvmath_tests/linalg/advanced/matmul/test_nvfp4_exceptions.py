# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
Tests for NVFP4 matmul exceptions.
"""

import pytest

from .fp4_utils import NVFP4_SKIP_REASON, NVFP4_SUPPORTED

if not NVFP4_SUPPORTED:
    pytest.skip(NVFP4_SKIP_REASON, allow_module_level=True)

import torch

from nvmath.bindings import cublas
from nvmath.linalg.advanced import matmul

from .fp4_utils import create_uniform_fp4_scales


def _create_zero_fp4_matrix_a(m, k, device="cuda"):
    a_bytes = torch.zeros((m, k // 2), dtype=torch.uint8, device=device)
    return a_bytes.view(torch.float4_e2m1fn_x2)


def _create_zero_fp4_matrix_b(k, n, device="cuda"):
    # Column-major: strides (1, K//2) so columns are contiguous
    b_bytes = torch.zeros((k // 2) * n, dtype=torch.uint8, device=device)
    b_fp4 = b_bytes.view(torch.float4_e2m1fn_x2)
    return torch.as_strided(b_fp4, size=(k // 2, n), stride=(1, k // 2))


@pytest.mark.parametrize("atype,btype", [("fp4", "float32"), ("float32", "fp4")])
def test_mixed_fp4_non_fp4_operands(atype, btype):
    m, n, k = 128, 128, 64

    # Create one FP4 and one float32 operand
    if atype == "fp4":
        a = _create_zero_fp4_matrix_a(m, k)
    else:
        a = torch.randn(m, k, dtype=torch.float32, device="cuda")

    if btype == "fp4":
        b = _create_zero_fp4_matrix_b(k, n)
    else:
        b = torch.randn(k, n, dtype=torch.float32, device="cuda")

    # Mixed FP4/non-FP4 is rejected by the dtype combination check
    with pytest.raises(ValueError, match="Mixed FP4/non-FP4 A/B operands are not supported"):
        matmul(a, b)


@pytest.mark.parametrize("block_scaling_value", [False, None])
def test_fp4_requires_block_scaling(block_scaling_value):
    m, n, k = 128, 128, 64

    a = _create_zero_fp4_matrix_a(m, k)
    b = _create_zero_fp4_matrix_b(k, n)
    a_scale = create_uniform_fp4_scales(a, scale_value=1.0)
    b_scale = create_uniform_fp4_scales(b, scale_value=1.0)
    scales = {"a": a_scale, "b": b_scale}

    if block_scaling_value is None:
        options = {}  # Default (block_scaling=False)
    else:
        options = {"block_scaling": block_scaling_value}

    with pytest.raises(ValueError, match="block_scaling=True is required"):
        matmul(a, b, quantization_scales=scales, options=options)


def test_fp4_a_must_be_row_major():
    """Test that FP4 operand A with column-major strides raises ValueError."""
    m, n, k = 128, 128, 64

    # Create A with correct shape (M, K//2) but column-major strides
    # Row-major has strides (K//2, 1), column-major has strides (1, M)
    a_bytes = torch.zeros((m, k // 2), dtype=torch.uint8, device="cuda")
    a_wrong = torch.as_strided(
        a_bytes.view(torch.float4_e2m1fn_x2).flatten(),
        size=(m, k // 2),
        stride=(1, m),  # Column-major strides (wrong for A)
    )

    b = _create_zero_fp4_matrix_b(k, n)

    with pytest.raises(ValueError, match=r"FP4 operand A must be.*row-major"):
        matmul(a_wrong, b)


def test_fp4_b_must_be_column_major():
    m, n, k = 128, 128, 64

    a = _create_zero_fp4_matrix_a(m, k)

    # Create B in row-major layout (wrong for FP4)
    # Correct B is column-major (K//2, N) with strides (1, K//2)
    # Wrong B is row-major (K//2, N) with strides (N, 1)
    b_bytes = torch.zeros((k // 2, n), dtype=torch.uint8, device="cuda")
    b_wrong = b_bytes.view(torch.float4_e2m1fn_x2)  # Row-major by default

    # Create valid scales
    a_scale = create_uniform_fp4_scales(a, scale_value=1.0)
    b_correct = _create_zero_fp4_matrix_b(k, n)
    b_scale = create_uniform_fp4_scales(b_correct, scale_value=1.0)
    scales = {"a": a_scale, "b": b_scale}
    options = {"block_scaling": True}

    with pytest.raises(ValueError, match=r"FP4 operand B must be.*column-major"):
        matmul(a, b_wrong, quantization_scales=scales, options=options)


@pytest.mark.parametrize("scale_dtype", [torch.uint8, torch.float32, torch.int8])
def test_fp4_scale_dtype_validation(scale_dtype):
    m, n, k = 128, 128, 64

    a = _create_zero_fp4_matrix_a(m, k)
    b = _create_zero_fp4_matrix_b(k, n)
    # Create scales with wrong dtype
    num_a_scales = (m * k) // 16
    num_b_scales = (k * n) // 16
    a_scale = torch.ones(num_a_scales, dtype=scale_dtype, device="cuda")
    b_scale = torch.ones(num_b_scales, dtype=scale_dtype, device="cuda")

    scales = {"a": a_scale, "b": b_scale}
    options = {"block_scaling": True}

    with pytest.raises(ValueError, match="must be float8_e4m3fn tensor"):
        matmul(a, b, quantization_scales=scales, options=options)


@pytest.mark.parametrize(
    "m,n,k,expected_error",
    [
        (64, 128, 64, "M=64 must be divisible by 128"),  # M not divisible by 128
        (128, 64, 64, "N=64 must be divisible by 128"),  # N not divisible by 128
        (128, 128, 32, "K=32 must be divisible by 64"),  # K not divisible by 64
    ],
)
def test_fp4_dimension_requirements(m, n, k, expected_error):
    a = _create_zero_fp4_matrix_a(m, k)
    b = _create_zero_fp4_matrix_b(k, n)

    # dimension validation happens early in __init__ before scale validation,
    # so we don't need correctly-sized scales
    with pytest.raises(ValueError, match=expected_error):
        matmul(a, b)


def test_fp4_invalid_compute_type():
    m, n, k = 128, 128, 64
    a = _create_zero_fp4_matrix_a(m, k)
    b = _create_zero_fp4_matrix_b(k, n)
    a_scale = create_uniform_fp4_scales(a, scale_value=1.0)
    b_scale = create_uniform_fp4_scales(b, scale_value=1.0)
    scales = {"a": a_scale, "b": b_scale}
    options = {
        "block_scaling": True,
        "compute_type": cublas.ComputeType.COMPUTE_64F,  # Invalid for FP4
    }

    with pytest.raises(ValueError, match="are not supported"):
        matmul(a, b, quantization_scales=scales, options=options)
