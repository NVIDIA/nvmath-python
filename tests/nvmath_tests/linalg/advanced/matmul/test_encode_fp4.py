# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for quantize_to_fp4 function."""

import pytest

from .fp4_utils import HAS_TORCH_2_9

if not HAS_TORCH_2_9:
    pytest.skip("Torch >= 2.9 required for float4_e2m1fn_x2 support", allow_module_level=True)

import torch

from nvmath.linalg.advanced.helpers.matmul import (
    _FP4_DECODE_VALUES,
    quantize_to_fp4,
    unpack_fp4,
)

# FP4 E2M1 representable values (same order as bits 0x0..0xF)
FP4_VALUES = list(_FP4_DECODE_VALUES)


def _round_to_fp4(x: torch.Tensor) -> torch.Tensor:
    """
    Round each element of x to the nearest FP4 representable value.

    Tie-breaking: if two values are equally close, choose the one with
    smaller absolute value (matches torch.bucketize with right=False).
    """
    fp4 = torch.tensor(FP4_VALUES, dtype=torch.float32, device=x.device)
    dist = torch.abs(x.unsqueeze(-1) - fp4)
    dmin = dist.min(dim=-1, keepdim=True).values
    tied = dist == dmin
    score = torch.where(tied, torch.abs(fp4), torch.tensor(float("inf"), device=x.device))
    return fp4[score.argmin(dim=-1)]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_encode_fp4_1d(device):
    """Test 1D vector encoding."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    k = 64
    truth = torch.zeros(k, dtype=torch.float32, device=device)
    truth[0] = 1.0
    truth[1] = 2.0
    truth[10] = 3.0
    truth[11] = -1.5
    truth[62] = 4.0
    truth[63] = -2.0

    encoded = quantize_to_fp4(truth, axis=-1)
    decoded = unpack_fp4(encoded, axis=-1)

    assert encoded.dtype == torch.float4_e2m1fn_x2
    assert encoded.shape == (k // 2,)
    assert encoded.stride() == (1,)
    assert decoded.shape == (k,)
    assert torch.allclose(decoded, truth)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_encode_fp4_1d_rounding(device):
    """Test 1D encoding with all FP4 values perturbed by random epsilon
    to verify rounding within each owning cell."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    fp4_magnitudes = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    boundaries = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]

    rng = torch.Generator().manual_seed(123)
    n_samples = 50

    inputs = []
    expected = []
    for i, mag in enumerate(fp4_magnitudes):
        lo = boundaries[i - 1] if i > 0 else 0.0
        hi = boundaries[i] if i < len(boundaries) else 6.0

        eps = torch.rand(n_samples, generator=rng) * (hi - lo)
        perturbed = lo + eps
        inputs.extend(perturbed.tolist())
        expected.extend([mag] * n_samples)

        if mag != 0.0:
            inputs.extend((-perturbed).tolist())
            expected.extend([-mag] * n_samples)

    if len(inputs) % 2 != 0:
        inputs.append(0.0)
        expected.append(0.0)

    k = len(inputs)
    truth = torch.tensor(inputs, dtype=torch.float32, device=device)
    exp = torch.tensor(expected, dtype=torch.float32, device=device)

    encoded = quantize_to_fp4(truth, axis=-1)
    decoded = unpack_fp4(encoded, axis=-1)

    assert encoded.dtype == torch.float4_e2m1fn_x2
    assert encoded.shape == (k // 2,)
    assert encoded.stride() == (1,)
    assert decoded.shape == (k,)
    assert torch.allclose(decoded, exp), f"max diff: {(decoded - exp).abs().max().item()}"


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_encode_fp4_2d_rowwise(device):
    """Test 2D matrix with row-wise packing."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    m, k = 128, 64
    truth = torch.zeros((m, k), dtype=torch.float32, device=device)
    truth[0, 0] = 1.0
    truth[0, 1] = 2.0
    truth[5, 10] = 3.0
    truth[5, 11] = -1.5
    truth[127, 62] = 4.0
    truth[127, 63] = -2.0

    encoded = quantize_to_fp4(truth, axis=-1)
    decoded = unpack_fp4(encoded, axis=-1)

    assert encoded.dtype == torch.float4_e2m1fn_x2
    assert encoded.shape == (m, k // 2)
    assert encoded.stride() == (k // 2, 1)
    assert decoded.shape == (m, k)
    assert torch.allclose(decoded, truth)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_encode_fp4_2d_columnwise(device):
    """Test 2D matrix with column-wise packing."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    k, n = 64, 256
    truth = torch.zeros((k, n), dtype=torch.float32, device=device)
    truth[0, 0] = 1.5
    truth[1, 0] = -3.0
    truth[10, 5] = 2.0
    truth[11, 5] = 0.5
    truth[62, 255] = 6.0
    truth[63, 255] = -6.0

    encoded = quantize_to_fp4(truth, axis=-2)
    decoded = unpack_fp4(encoded, axis=-2)

    assert encoded.dtype == torch.float4_e2m1fn_x2
    assert encoded.shape == (k // 2, n)
    assert encoded.stride() == (1, k // 2)
    assert decoded.shape == (k, n)
    assert torch.allclose(decoded, truth)


def test_encode_fp4_negative_zero():
    """Negative zero must round-trip through FP4 with its sign bit preserved."""
    x = torch.tensor([0.0, -0.0, 1.0, -0.0], dtype=torch.float32)
    encoded = quantize_to_fp4(x, axis=-1)
    decoded = unpack_fp4(encoded, axis=-1)
    assert torch.equal(decoded, x)
    assert not torch.signbit(decoded[0]), "+0.0 sign bit must be clear"
    assert torch.signbit(decoded[1]), "-0.0 sign bit must be set"
    assert not torch.signbit(decoded[2]), "+1.0 sign bit must be clear"
    assert torch.signbit(decoded[3]), "-0.0 sign bit must be set"


def test_encode_fp4_hardcoded_non_representable():
    """
    Hardcoded non-representable values across the full FP4 range,
    with expected quantized output.
    """
    #              input  → expected
    # cell [0, 0.25)       → 0.0
    # cell [0.25, 0.75)    → 0.5
    # cell [0.75, 1.25)    → 1.0
    # cell [1.25, 1.75)    → 1.5
    # cell [1.75, 2.5)     → 2.0
    # cell [2.5, 3.5)      → 3.0
    # cell [3.5, 5.0)      → 4.0
    # cell [5.0, Infinity) → 6.0
    inputs = [0.1, 0.3, 0.9, 1.4, 2.1, 2.8, 4.5, 5.5, -0.1, -0.3, -0.9, -1.4, -2.1, -2.8, -4.5, -5.5]
    expected = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
    x = torch.tensor(inputs, dtype=torch.float32)
    exp = torch.tensor(expected, dtype=torch.float32)
    encoded = quantize_to_fp4(x, axis=-1)
    decoded = unpack_fp4(encoded, axis=-1)
    assert torch.allclose(decoded, exp)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_encode_decode_roundtrip_random_2d(device):
    """Encode random 2D matrix, decode, and compare with expected quantized values."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    torch.manual_seed(42)
    m, k = 32, 64
    x = torch.rand(m, k, dtype=torch.float32, device=device) * 12 - 6

    encoded = quantize_to_fp4(x, axis=-1)
    decoded = unpack_fp4(encoded, axis=-1)

    expected = _round_to_fp4(x)
    assert torch.allclose(decoded, expected)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_encode_decode_columnwise_k2(device):
    """
    Round-trip for column-wise packing with K=2 (packed dim reduces to 1).
    Regression test for the stride-ambiguity case: when K=2 and axis=-2,
    the packed tensor has shape (1, N) with strides (1, 1).
    """
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    k, n = 2, 8
    truth = torch.zeros((k, n), dtype=torch.float32, device=device)
    truth[0, 0] = 1.0
    truth[1, 0] = 2.0
    truth[0, 7] = 3.0
    truth[1, 7] = -1.5

    encoded = quantize_to_fp4(truth, axis=-2)
    assert encoded.shape == (1, n)

    decoded = unpack_fp4(encoded, axis=-2)
    assert decoded.shape == (k, n)
    assert torch.allclose(decoded, truth)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("batch_shape", [(2,), (2, 3)])
def test_encode_fp4_nd_rowwise(device, batch_shape):
    """Test N-D tensor with row-wise packing (axis=-1)."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    m, k = 32, 64
    shape = (*batch_shape, m, k)
    truth = torch.zeros(shape, dtype=torch.float32, device=device)
    truth[..., 0, 0] = 1.0
    truth[..., 0, 1] = 2.0
    truth[..., 5, 10] = 3.0
    truth[..., 5, 11] = -1.5

    encoded = quantize_to_fp4(truth, axis=-1)
    decoded = unpack_fp4(encoded, axis=-1)

    assert encoded.dtype == torch.float4_e2m1fn_x2
    assert encoded.shape == (*batch_shape, m, k // 2)
    assert encoded.stride()[-1] == 1, "Last stride should be 1 (row-major)"
    assert decoded.shape == shape
    assert torch.allclose(decoded, truth)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("batch_shape", [(2,), (2, 3)])
def test_encode_fp4_nd_columnwise(device, batch_shape):
    """Test N-D tensor with column-wise packing (axis=-2)."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    k, n = 64, 32
    shape = (*batch_shape, k, n)
    truth = torch.zeros(shape, dtype=torch.float32, device=device)
    truth[..., 0, 0] = 1.5
    truth[..., 1, 0] = -3.0
    truth[..., 10, 5] = 2.0
    truth[..., 11, 5] = 0.5

    encoded = quantize_to_fp4(truth, axis=-2)
    decoded = unpack_fp4(encoded, axis=-2)

    assert encoded.dtype == torch.float4_e2m1fn_x2
    assert encoded.shape == (*batch_shape, k // 2, n)
    assert encoded.stride()[-2] == 1, "Second-to-last stride should be 1 (column-major)"
    assert decoded.shape == shape
    assert torch.allclose(decoded, truth)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("axis", [-1, -2])
def test_encode_decode_roundtrip_random_3d(device, axis):
    """Encode random 3D tensor, decode, and compare with expected quantized values."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    torch.manual_seed(42)
    batch, m, k = 3, 16, 32
    x = torch.rand(batch, m, k, dtype=torch.float32, device=device) * 12 - 6

    encoded = quantize_to_fp4(x, axis=axis)
    decoded = unpack_fp4(encoded, axis=axis)

    expected = _round_to_fp4(x)
    assert torch.allclose(decoded, expected)


def test_encode_fp4_odd_dim_1d_raises():
    x = torch.zeros(7, dtype=torch.float32)
    with pytest.raises(ValueError, match="Packed dimension must be even"):
        quantize_to_fp4(x, axis=-1)


def test_encode_fp4_odd_dim_2d_rowwise_raises():
    x = torch.zeros((4, 7), dtype=torch.float32)
    with pytest.raises(ValueError, match="Packed dimension must be even"):
        quantize_to_fp4(x, axis=-1)


def test_encode_fp4_odd_dim_2d_columnwise_raises():
    x = torch.zeros((7, 4), dtype=torch.float32)
    with pytest.raises(ValueError, match="Packed dimension must be even"):
        quantize_to_fp4(x, axis=-2)


def test_encode_fp4_wrong_dtype_raises():
    x = torch.zeros(4, dtype=torch.float16)
    with pytest.raises(ValueError, match="x must be float32"):
        quantize_to_fp4(x, axis=-1)


def test_encode_fp4_wrong_type_raises():
    with pytest.raises(TypeError, match=r"x must be a torch\.Tensor"):
        quantize_to_fp4([1.0, 2.0], axis=-1)


def test_encode_fp4_invalid_axis_raises():
    x = torch.zeros((4, 8), dtype=torch.float32)
    with pytest.raises(ValueError, match="axis must be -1 or -2"):
        quantize_to_fp4(x, axis=0)


def test_encode_fp4_1d_axis_minus2_raises():
    x = torch.zeros(4, dtype=torch.float32)
    with pytest.raises(ValueError, match="axis must be -1 for 1D tensors"):
        quantize_to_fp4(x, axis=-2)
