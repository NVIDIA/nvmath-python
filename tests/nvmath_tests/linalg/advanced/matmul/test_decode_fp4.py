# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for unpack_fp4 function."""

import numpy as np
import pytest

from .fp4_utils import HAS_TORCH_2_9

if not HAS_TORCH_2_9:
    pytest.skip("Torch >= 2.9 required for float4_e2m1fn_x2 support", allow_module_level=True)

import torch

from nvmath.linalg.advanced.helpers.matmul import unpack_fp4

_FP4_VALUE_TO_CODE = {
    0.0: 0x0,
    0.5: 0x1,
    1.0: 0x2,
    1.5: 0x3,
    2.0: 0x4,
    3.0: 0x5,
    4.0: 0x6,
    6.0: 0x7,
    -0.5: 0x9,
    -1.0: 0xA,
    -1.5: 0xB,
    -2.0: 0xC,
    -3.0: 0xD,
    -4.0: 0xE,
    -6.0: 0xF,
}


def _assert_representable_in_fp4(value):
    if value not in _FP4_VALUE_TO_CODE:
        raise ValueError(f"Value {value} not representable in FP4")


def _pack_two_fp4_values_into_byte(val1, val2):
    """
    Pack two FP4 values into a single byte.

    FP4 values are 4-bit, so two values fit in one byte:
    - val1 goes into the low bits (bits 0-3)
    - val2 goes into the high bits (bits 4-7)

    Args:
        val1: First float32 value (goes into low bits)
        val2: Second float32 value (goes into high bits)

    Returns:
        uint8 byte with packed FP4 values
    """
    low_bits = _FP4_VALUE_TO_CODE[val1] & 0xF
    high_bits = (_FP4_VALUE_TO_CODE[val2] & 0xF) << 4
    return low_bits | high_bits


def _encode_fp4_rowwise_packing(float_matrix):
    """Pack consecutive elements within each row: (M, K) -> (M, K//2)."""
    m, k = float_matrix.shape
    assert k % 2 == 0

    fp4_bytes = np.zeros((m, k // 2), dtype=np.uint8)
    for i in range(m):
        for j in range(k // 2):
            val1 = float(float_matrix[i, j * 2])
            val2 = float(float_matrix[i, j * 2 + 1])
            _assert_representable_in_fp4(val1)
            _assert_representable_in_fp4(val2)
            fp4_bytes[i, j] = _pack_two_fp4_values_into_byte(val1, val2)

    fp4_tensor = torch.from_numpy(fp4_bytes).view(torch.float4_e2m1fn_x2)
    assert fp4_tensor.stride() == (k // 2, 1), f"Expected stride ({k // 2}, 1) for row-wise packing"
    return fp4_tensor


def _encode_fp4_columnwise_packing(float_matrix):
    """Pack consecutive elements within each column: (K, N) -> (K//2, N)."""
    k, n = float_matrix.shape
    assert k % 2 == 0

    fp4_bytes = np.zeros((k // 2, n), dtype=np.uint8, order="F")
    for i in range(k // 2):
        for j in range(n):
            val1 = float(float_matrix[i * 2, j])
            val2 = float(float_matrix[i * 2 + 1, j])
            _assert_representable_in_fp4(val1)
            _assert_representable_in_fp4(val2)
            fp4_bytes[i, j] = _pack_two_fp4_values_into_byte(val1, val2)

    fp4_tensor = torch.from_numpy(fp4_bytes).view(torch.float4_e2m1fn_x2)
    assert fp4_tensor.stride() == (1, k // 2), f"Expected stride (1, {k // 2}) for column-wise packing"
    return fp4_tensor


def _encode_fp4_1d(float_vector):
    """Pack consecutive elements in 1D vector: (K,) -> (K//2,)."""
    k = float_vector.shape[0]
    assert k % 2 == 0

    fp4_bytes = np.zeros(k // 2, dtype=np.uint8)
    for i in range(k // 2):
        val1 = float(float_vector[i * 2])
        val2 = float(float_vector[i * 2 + 1])
        _assert_representable_in_fp4(val1)
        _assert_representable_in_fp4(val2)
        fp4_bytes[i] = _pack_two_fp4_values_into_byte(val1, val2)

    fp4_tensor = torch.from_numpy(fp4_bytes).view(torch.float4_e2m1fn_x2)
    assert fp4_tensor.stride() == (1,), "Expected stride (1,) for 1D tensor"
    return fp4_tensor


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_decode_fp4_1d(device):
    """Test 1D vector decoding."""
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

    fp4_encoded = _encode_fp4_1d(truth.cpu()).to(device)
    decoded = unpack_fp4(fp4_encoded, axis=-1)

    assert decoded.device.type == device
    assert decoded.shape == (k,)
    assert decoded.stride() == (1,)
    assert torch.equal(decoded, truth)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_decode_fp4_2d_rowwise(device):
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

    fp4_encoded = _encode_fp4_rowwise_packing(truth.cpu()).to(device)
    decoded = unpack_fp4(fp4_encoded, axis=-1)

    assert decoded.shape == (m, k)
    assert decoded.device == fp4_encoded.device
    assert decoded.stride() == (k, 1), f"Expected stride ({k}, 1) for row-wise packed output"
    assert torch.equal(decoded, truth)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_decode_fp4_2d_columnwise(device):
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

    fp4_encoded = _encode_fp4_columnwise_packing(truth.cpu()).to(device)
    decoded = unpack_fp4(fp4_encoded, axis=-2)

    assert decoded.shape == (k, n)
    assert decoded.device == fp4_encoded.device
    assert decoded.stride() == (1, k), f"Expected stride (1, {k}) for column-wise packed output"
    assert torch.equal(decoded, truth)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_decode_fp4_3d_rowwise(device):
    """Test 3D tensor with single batch dimension and row-wise packing."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    batch, m, k = 4, 128, 64
    truth = torch.zeros((batch, m, k), dtype=torch.float32, device=device)

    truth[0, 0, 0] = 1.0
    truth[0, 0, 1] = 2.0
    truth[1, 10, 20] = 3.0
    truth[1, 10, 21] = -1.0
    truth[2, 50, 30] = 4.0
    truth[2, 50, 31] = 1.5
    truth[3, 127, 62] = -2.0
    truth[3, 127, 63] = 0.5

    fp4_bytes_list = []
    for b in range(batch):
        fp4_bytes_list.append(_encode_fp4_rowwise_packing(truth[b].cpu()).view(torch.uint8))
    fp4_bytes = torch.stack(fp4_bytes_list).to(device)
    fp4_encoded = fp4_bytes.view(torch.float4_e2m1fn_x2)

    decoded = unpack_fp4(fp4_encoded, axis=-1)

    assert decoded.shape == (batch, m, k)
    assert decoded.device == fp4_encoded.device
    assert decoded.stride() == (m * k, k, 1), "Expected row-wise stride pattern"
    assert torch.equal(decoded, truth)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_decode_fp4_4d_rowwise(device):
    """Test 4D tensor with two batch dimensions and row-wise packing."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    b1, b2, m, k = 2, 3, 128, 64
    truth = torch.zeros((b1, b2, m, k), dtype=torch.float32, device=device)

    truth[0, 0, 0, 0] = 1.0
    truth[0, 0, 0, 1] = -1.0
    truth[0, 1, 10, 20] = 2.0
    truth[0, 1, 10, 21] = 3.0
    truth[0, 2, 50, 40] = -4.0
    truth[0, 2, 50, 41] = 4.0
    truth[1, 0, 100, 50] = 1.5
    truth[1, 0, 100, 51] = -1.5
    truth[1, 1, 127, 62] = 6.0
    truth[1, 2, 127, 63] = -6.0

    fp4_bytes_list = []
    for i in range(b1):
        for j in range(b2):
            fp4_bytes_list.append(_encode_fp4_rowwise_packing(truth[i, j].cpu()).view(torch.uint8))
    fp4_bytes = torch.stack(fp4_bytes_list).reshape(b1, b2, m, k // 2).to(device)
    fp4_encoded = fp4_bytes.view(torch.float4_e2m1fn_x2)

    decoded = unpack_fp4(fp4_encoded, axis=-1)

    assert decoded.shape == (b1, b2, m, k)
    assert decoded.device == fp4_encoded.device
    assert decoded.stride() == (b2 * m * k, m * k, k, 1), "Expected row-wise stride pattern"
    assert torch.equal(decoded, truth)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_decode_fp4_4d_columnwise(device):
    """Test 4D tensor with two batch dimensions and column-wise packing."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    b1, b2, k, n = 2, 3, 64, 256
    truth = torch.zeros((b1, b2, k, n), dtype=torch.float32, device=device)

    truth[0, 0, 0, 0] = 1.0
    truth[0, 0, 1, 0] = -1.0
    truth[0, 1, 10, 20] = 2.0
    truth[0, 1, 11, 20] = 3.0
    truth[1, 0, 50, 100] = -2.0
    truth[1, 0, 51, 100] = 0.5
    truth[1, 2, 62, 255] = 4.0
    truth[1, 2, 63, 255] = -3.0

    k_half = k // 2
    # Create batched column-major tensor manually.
    storage_size = b1 * b2 * k_half * n
    fp4_storage = torch.zeros(storage_size, dtype=torch.uint8, device="cpu")
    fp4_bytes_batched = torch.as_strided(fp4_storage, size=(b1, b2, k_half, n), stride=(b2 * k_half * n, k_half * n, 1, k_half))

    for i in range(b1):
        for j in range(b2):
            single_fp4 = _encode_fp4_columnwise_packing(truth[i, j].cpu()).view(torch.uint8)
            fp4_bytes_batched[i, j] = single_fp4

    fp4_encoded = fp4_bytes_batched.to(device).view(torch.float4_e2m1fn_x2)

    decoded = unpack_fp4(fp4_encoded, axis=-2)

    assert decoded.shape == (b1, b2, k, n)
    assert decoded.device == fp4_encoded.device
    assert decoded.stride() == (b2 * k * n, k * n, 1, k), "Expected column-wise stride pattern"
    assert torch.equal(decoded, truth)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
