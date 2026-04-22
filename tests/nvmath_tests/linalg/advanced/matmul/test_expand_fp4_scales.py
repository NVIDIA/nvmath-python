# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
Tests for expand_fp4_scales_to_matrix function.
"""

import pytest

from .fp4_utils import HAS_TORCH_2_9

if not HAS_TORCH_2_9:
    pytest.skip("Torch >= 2.9 required for float8_e4m3fn support", allow_module_level=True)

import torch

from nvmath.linalg.advanced.helpers.matmul import (
    expand_block_scale,
    get_block_scale_offset,
)


@pytest.mark.parametrize("scales_device", ["cpu", "cuda"])
@pytest.mark.parametrize("target_device", [None, "cpu", "cuda"])
@pytest.mark.parametrize("output_dtype", ["smallest", "float32"])
def test_expand_single_tile_linear_scales(scales_device, target_device, output_dtype):
    """
    Test expand_block_scale with a single 128x4 tile.
    This test verifies the NVFP4 scale expansion pattern for a 128x64 matrix.
    Input: 1D scales array [s0, s1, s2, ..., s511]  (512 scales)
    Output: 128x64 matrix where each scale is replicated across 16 columns.

    The cuBLASLt tiling pattern: the 128 rows are divided into 4 blocks
    of 32 rows each, and the 512 scales form a 32x16 layout where
    each block occupies 4 adjacent columns.
    Expected indexing formula (from cuBLASLt spec):
        For position (row, col):
        - col_group = col // 16  (which 16-element group: 0, 1, 2, or 3)
        - scale_index = (row % 32) * 16 + (row // 32) * 4 + col_group
    """
    m, n = 128, 64
    num_scales = (m * n) // 16  # 512 scales

    # Create unique float values that are representable in float8_e4m3fn.
    # We use small positive values (0.5, 1.0, 1.5, 2.0, 2.5, ...)
    scales_float = torch.arange(num_scales, dtype=torch.float32, device=scales_device) * 0.5 + 0.5
    scales_fp8 = scales_float.to(torch.float8_e4m3fn)

    if output_dtype != "smallest":
        expected_output_dtype = output_dtype = getattr(torch, output_dtype)
    else:
        expected_output_dtype = torch.float8_e4m3fn
    scales_expanded = expand_block_scale(scales_fp8, (m, n), "NVFP4", axis=-1, device=target_device, output_dtype=output_dtype)
    assert scales_expanded.shape == (m, n)
    assert scales_expanded.dtype == expected_output_dtype
    expected_device = target_device if target_device is not None else scales_device
    assert scales_expanded.device.type == expected_device

    # Convert scales to float for comparison
    scales_fp8_as_float = scales_fp8.float()

    # Verify each position has the correct scale using explicit formula
    # cuBLASLt tiling: scale_index = (row % 32) * 16 + (row // 32) * 4 + col_group
    for row in range(m):
        for col in range(n):
            # Compute which 16-element group this column belongs to
            col_group = col // 16

            # Explicit formula from cuBLASLt spec (manually computed, not using
            # any helper functions to avoid circular testing)
            expected_idx = (row % 32) * 16 + (row // 32) * 4 + col_group

            # Cross-validate: verify get_block_scale_offset matches explicit formula
            helper_idx = get_block_scale_offset((row, col_group), (m, n), "NVFP4", axis=-1)
            assert expected_idx == helper_idx, (
                f"Position ({row}, {col}): explicit formula gives index {expected_idx}, "
                f"but get_block_scale_offset gives {helper_idx}"
            )

            expected_scale = scales_fp8_as_float[expected_idx].item()
            actual_scale = scales_expanded[row, col].item()
            assert abs(actual_scale - expected_scale) < 1e-6, (
                f"Position ({row}, {col}): expected scale from index {expected_idx} "
                f"({expected_scale:.6f}), got {actual_scale:.6f}"
            )


@pytest.mark.parametrize("scales_device", ["cpu", "cuda"])
@pytest.mark.parametrize("target_device", [None, "cpu", "cuda"])
@pytest.mark.parametrize(
    "m,n",
    [
        (128, 128),  # 1x2 tiles
        (256, 64),  # 2x1 tiles
        (256, 128),  # 2x2 tiles
        (128, 256),  # 1x4 tiles
        (512, 256),  # 4x4 tiles
    ],
)
def test_expand_multi_tile_exhaustive(m, n, scales_device, target_device):
    """
    Verify expand_block_scale with different matrix sizes.
    """
    num_scales = (m * n) // 16

    # Build uint8 scales that avoid the two float8_e4m3fn NaN bytes (0x7F, 0xFF)
    # so the comparison can use a simple torch.equal without NaN headaches.
    valid_bytes = torch.tensor([i for i in range(256) if i not in (0x7F, 0xFF)], dtype=torch.uint8)
    scales = valid_bytes[torch.arange(num_scales) % len(valid_bytes)].to(scales_device)

    assert not scales.view(torch.float8_e4m3fn).float().isnan().any(), "test setup: scales contain NaN"

    result = expand_block_scale(scales, (m, n), "NVFP4", axis=-1, device=target_device, output_dtype=torch.float32)

    assert result.shape == (m, n)
    assert result.dtype == torch.float32
    expected_device = target_device if target_device is not None else scales_device
    assert result.device.type == expected_device

    # Vectorized reference: raw cuBLASLt tile formula in torch ops.
    # https://docs.nvidia.com/cuda/cublas/index.html#d-block-scaling-factors-layout
    row, col = torch.meshgrid(torch.arange(m), torch.arange(n), indexing="ij")
    group = col // 16
    tile_offset = ((group // 4) * 4 + (row // 128) * (n // 16)) * 128
    intra = (row % 32) * 16 + ((row % 128) // 32) * 4 + group % 4
    expected_idx = tile_offset + intra

    helpers_idx = get_block_scale_offset((row, group), (m, n), "NVFP4", axis=-1)
    assert torch.equal(expected_idx, helpers_idx)

    scales_on_device = scales.to(result.device)
    expected = scales_on_device.view(torch.float8_e4m3fn).float()[expected_idx]

    assert torch.equal(result, expected), (
        f"Mismatch for ({m}, {n}): {(result != expected).sum().item()} / {result.numel()} elements differ"
    )


def test_expand_uint8_matches_fp8():
    """
    Verify that uint8 and float8_e4m3fn inputs produce identical results.

    The function accepts both dtypes; uint8 is internally reinterpreted
    via view(torch.float8_e4m3fn).  This test ensures the two paths agree.
    """
    m, n = 128, 64
    num_scales = (m * n) // 16

    scales_fp8 = torch.arange(num_scales, dtype=torch.float32) * 0.5 + 0.5
    scales_fp8 = scales_fp8.to(torch.float8_e4m3fn)
    scales_uint8 = scales_fp8.view(torch.uint8)

    result_fp8 = expand_block_scale(scales_fp8, (m, n), "NVFP4", axis=-1)
    result_uint8 = expand_block_scale(scales_uint8, (m, n), "NVFP4", axis=-1)

    assert result_fp8.dtype == torch.float8_e4m3fn
    assert result_uint8.dtype == torch.float8_e4m3fn
    assert torch.equal(result_fp8, result_uint8)


@pytest.mark.parametrize(
    "scales,operand_shape,axis,device,error_type,match",
    [
        # non-Tensor input
        ([1, 2, 3], (128, 64), -1, None, TypeError, "torch.Tensor"),
        # wrong ndim (2D instead of 1D)
        (torch.ones(512, 2, dtype=torch.uint8), (128, 64), -1, None, ValueError, "1D"),
        # wrong dtype
        (torch.ones(512, dtype=torch.float32), (128, 64), -1, None, TypeError, "float8_e4m3fn or torch.uint8"),
        # invalid device string
        (torch.ones(512, dtype=torch.uint8), (128, 64), -1, "tpu", ValueError, "device"),
        # unblocked extent not a multiple of 128 (axis=-1 -> unblocked is axis -2)
        (torch.ones(512, dtype=torch.uint8), (100, 64), -1, None, ValueError, "multiple of 128"),
        # blocked extent not a multiple of 64 (axis=-1 -> blocked is axis -1)
        (torch.ones(512, dtype=torch.uint8), (128, 100), -1, None, ValueError, "multiple of 64"),
        # wrong number of scale elements
        (torch.ones(999, dtype=torch.uint8), (128, 64), -1, None, ValueError, "elements"),
    ],
)
def test_expand_exceptions(scales, operand_shape, axis, device, error_type, match):
    """
    Verify that all validation branches raise the expected errors.
    """
    with pytest.raises(error_type, match=match):
        expand_block_scale(scales, operand_shape, "NVFP4", axis=axis, device=device)
