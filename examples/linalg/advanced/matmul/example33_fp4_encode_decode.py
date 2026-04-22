# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example demonstrates the two helper functions quantize_to_fp4
and unpack_fp4.

FP4 E2M1 packs two 4-bit values per byte.
Representable FP4 values: 0, +/-0.5, +/-1, +/-1.5, +/-2, +/-3, +/-4, +/-6.
Other values are quantized to the nearest representable value.

Requires torch >= 2.9 for float4_e2m1fn_x2 support.
"""

import torch

from nvmath.linalg.advanced.helpers.matmul import (
    quantize_to_fp4,
    unpack_fp4,
)

device = "cuda" if torch.cuda.is_available() else "cpu"

# --- 1D: encode and decode ---
x_1d = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32, device=device)
x_fp4 = quantize_to_fp4(x_1d, axis=-1)
x_decoded = unpack_fp4(x_fp4, axis=-1)
# After encoding, the shape along the packed axis is halved (two 4-bit values per byte).
# Shape (K,) -> packed (K//2,)
print(f"1D round-trip: {x_1d.tolist()} -> encode -> decode -> {x_decoded.tolist()}")
print(f"   shapes:  {tuple(x_1d.shape)} -> {tuple(x_fp4.shape)} -> {tuple(x_decoded.shape)}")
print(f"   strides: {tuple(x_1d.stride())} -> {tuple(x_fp4.stride())} -> {tuple(x_decoded.stride())}")
assert torch.equal(x_1d, x_decoded)
print()

# --- Quantization: non-representable values round to nearest ---
y_1d = torch.tensor([2.3, 5.7], dtype=torch.float32, device=device)
y_fp4 = quantize_to_fp4(y_1d, axis=-1)
y_decoded = unpack_fp4(y_fp4, axis=-1)
print(f"Quantization: {y_1d.tolist()} -> {y_decoded.tolist()}  (2.3->2.0, 5.7->6.0)")
print(f"   shapes:  {tuple(y_1d.shape)} -> {tuple(y_fp4.shape)} -> {tuple(y_decoded.shape)}")
print(f"   strides: {tuple(y_1d.stride())} -> {tuple(y_fp4.stride())} -> {tuple(y_decoded.stride())}")
assert not torch.equal(y_1d, y_decoded), "Non-representable values should differ after round-trip"
print()

# --- 2D row-wise ---
a_float = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32, device=device)
a_fp4 = quantize_to_fp4(a_float, axis=-1)
a_decoded = unpack_fp4(a_fp4, axis=-1)
print(f"2D row-wise (A layout): {a_float.tolist()} -> encode -> decode -> {a_decoded.tolist()}")
# Last dimension (K) is packed; after encoding the packed axis is halved.
print(f"   shapes:  {tuple(a_float.shape)} -> {tuple(a_fp4.shape)} -> {tuple(a_decoded.shape)}")
# Last stride = 1.
print(f"   strides: {tuple(a_float.stride())} -> {tuple(a_fp4.stride())} -> {tuple(a_decoded.stride())}")
assert torch.equal(a_float, a_decoded)
print()

# --- 2D column-wise ---
b_float = torch.tensor([[1.0, 3.0], [2.0, 4.0], [1.5, 2.0], [0.5, 6.0]], dtype=torch.float32, device=device)
b_fp4 = quantize_to_fp4(b_float, axis=-2)
b_decoded = unpack_fp4(b_fp4, axis=-2)
print(f"2D column-wise (B layout): {b_float.tolist()} -> encode -> decode -> {b_decoded.tolist()}")
# Second-to-last dimension (K) is packed; after encoding the packed axis is halved.
print(f"   shapes:  {tuple(b_float.shape)} -> {tuple(b_fp4.shape)} -> {tuple(b_decoded.shape)}")
# Second-to-last stride = 1.
print(f"   strides: {tuple(b_float.stride())} -> {tuple(b_fp4.stride())} -> {tuple(b_decoded.stride())}")
assert torch.equal(b_float, b_decoded)
print()

# --- 3D row-wise (batched) ---
# quantize_to_fp4 preserves leading batch dimensions; each matrix is packed independently.
c_float = torch.tensor(
    [[[1.0, 2.0, 3.0, 4.0], [0.5, 1.5, 2.0, 6.0]], [[4.0, 3.0, 2.0, 1.0], [6.0, 2.0, 1.5, 0.5]]],
    dtype=torch.float32,
    device=device,
)  # shape (2, 2, 4)
c_fp4 = quantize_to_fp4(c_float, axis=-1)
c_decoded = unpack_fp4(c_fp4, axis=-1)
# Last dimension is packed
print(f"3D row-wise (batched): shape {tuple(c_float.shape)} -> {tuple(c_fp4.shape)} -> {tuple(c_decoded.shape)}")
print(f"   strides: {tuple(c_float.stride())} -> {tuple(c_fp4.stride())} -> {tuple(c_decoded.stride())}")
assert torch.equal(c_float, c_decoded)
print()

print("All encode/decode examples passed.")
