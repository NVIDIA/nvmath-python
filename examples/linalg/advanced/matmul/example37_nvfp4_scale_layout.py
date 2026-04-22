# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example demonstrates specifying scales for batched NVFP4 matmul on 4D tensors.
The last two dimensions of the input tensors are matrices used for matmul,
while the first two dimensions are batch dimensions.

The NVFP4 scales are specified as corresponding 4D tensors, with shapes
taking into account the blocking of 16 consecutive elements.
"""

import math

import numpy as np
import torch

import nvmath
from nvmath.linalg.advanced.helpers.matmul import (
    expand_block_scale,
    get_block_scale_offset,
    quantize_to_fp4,
    to_block_scale,
    unpack_fp4,
)

batch_shape = 2, 2
m, k, n = 640, 192, 384
device = "cuda"

# A matrix: logical shape batch_shape + (m, k).
a_shape = batch_shape + (m, k)
a_float = torch.ones(a_shape, device=device, dtype=torch.float32)
# Quantize to FP4 using row-wise packing since for A this is the contracting dimension.
a_fp4 = quantize_to_fp4(a_float, axis=-1)  # batch_shape + (m, k//2)

# B matrix: logical shape batch_shape + (k, n).
b_shape = batch_shape + (k, n)
b_float = torch.ones(b_shape, device=device, dtype=torch.float32)
# Quantize to FP4 using column-wise packing since for B this is the contracting dimension.
b_fp4 = quantize_to_fp4(b_float, axis=-2)  # batch_shape + (k//2, n)

# Build scale tensors for operands A and B.
# NVFP4 matmul accepts scales as 1D tensor, with elements laid out
# in certain tiled layout expected by cuBLAS.
# Here, we will initialize 4D (2 batch dims + 2D matrix dims) scale tensors,
# with shapes corresponding to the logical shapes of operands A and B
# (accounting for blocking by 16 elements)
# and use helper to copy the scales to the desired 1D tensor layout.

# A scales: A has shape batch_shape + (m, k).
# For A, the blocked dimension is -1, i.e. every consecutive 16 elements in A rows
# will have the same scale. Thus, the unique scales tensor has shape
# batch_shape + (m, k // 16).
a_scale_matrix = torch.ones(
    batch_shape + (m, k // 16),
    dtype=torch.float8_e4m3fn,
    device=device,
)
# We initialized A scales to 1., now let's adjust some of them.
for sample_idx in range(math.prod(batch_shape)):
    sample_nd_idx = np.unravel_index(sample_idx, batch_shape)
    # Second row of each sample in A will be scaled by sample_idx + 1
    a_scale_matrix[sample_nd_idx][1, :] = sample_idx + 1

# Finally, we convert the 4D scale tensor to the layout expected by cuBLAS.
a_scale = to_block_scale(a_scale_matrix, a_fp4, "NVFP4")
assert a_scale.ndim == 1
assert a_scale.shape[0] == a_scale_matrix.nelement()
assert a_scale.shape[0] == math.prod(a_shape) // 16
# The helper accepts A operand (``a_fp4``) to infer shape and blocked axis from it;
# ``block_scaling_format`` must still be passed explicitly.
assert torch.equal(a_scale, to_block_scale(a_scale_matrix, a_shape, "NVFP4", axis=-1))
# Please note, the conversion is not the same as just flattening the scales tensor.
assert not torch.equal(a_scale, a_scale_matrix.reshape(-1))

# Now, let's create scales for B operand.
# For B, the blocked dimension is -2, i.e. every consecutive 16 elements in B columns
# will have the same scale. Thus, the unique scales tensor has shape
# batch_shape + (k // 16, n).
b_scale_matrix = torch.ones(batch_shape + (k // 16, n), dtype=torch.float8_e4m3fn, device=device)
for sample_idx in range(math.prod(batch_shape)):
    sample_nd_idx = np.unravel_index(sample_idx, batch_shape)
    # Second column of each sample in B will be scaled by 1/(sample_idx + 1)
    b_scale_matrix[sample_nd_idx][:, 1] = 1 / (sample_idx + 1)
# We can preallocate the flat scale tensor
b_scale = torch.empty(math.prod(b_shape) // 16, dtype=torch.float8_e4m3fn, device=device)
to_block_scale(b_scale_matrix, b_fp4, "NVFP4", out=b_scale)
assert torch.equal(b_scale, to_block_scale(b_scale_matrix, b_shape, "NVFP4", axis=-2))

# Execute batched NVFP4 matmul.
result_fp32 = nvmath.linalg.advanced.matmul(
    a_fp4,
    b_fp4,
    quantization_scales={"a": a_scale, "b": b_scale},
    options={"result_type": nvmath.CudaDataType.CUDA_R_32F, "block_scaling": True},
)

output_shape = batch_shape + (m, n)
assert result_fp32.shape == output_shape
# Verify results.
print("Result:", result_fp32)
expected = torch.full(batch_shape + (m, n), k, device=device, dtype=torch.float32)
for sample_idx in range(math.prod(batch_shape)):
    sample_nd_idx = np.unravel_index(sample_idx, batch_shape)
    expected[sample_nd_idx][1, :] *= sample_idx + 1
    expected[sample_nd_idx][:, 1] /= sample_idx + 1

torch.testing.assert_close(result_fp32, expected, rtol=0.1, atol=1e-6)

# Now, let's run the same matmul, but request narrow-precision output.
# The return value is a tuple (result, aux) where aux["d_out_scale"] contains
# the scales used for output blocked scaling.
result_fp4, aux = nvmath.linalg.advanced.matmul(
    a_fp4,
    b_fp4,
    quantization_scales={"a": a_scale, "b": b_scale},
    options={"result_type": nvmath.CudaDataType.CUDA_R_4F_E2M1, "block_scaling": True},
)

d_out_scale = aux["d_out_scale"]
# Output scales are blocked in rows (axis=-1)
assert d_out_scale.ndim == 1
assert d_out_scale.shape[0] == math.prod(output_shape) // 16

# We reshape and broadcast the scales to match result operand.
d_out_scale_expanded = expand_block_scale(d_out_scale, result_fp4, "NVFP4")
print(
    f"D_out scale expanding:\n"
    f"shape: {d_out_scale.shape} -> {d_out_scale_expanded.shape}\n"
    f"dtype: {d_out_scale.dtype} -> {d_out_scale_expanded.dtype}"
)
assert d_out_scale_expanded.shape == output_shape

# Note, expand_block_scale is faster equivalent to the following:
indices = [torch.arange(s) for s in output_shape]
# convert output element indices to block indices
# for the blocked dimension (axis=-1)
indices[-1] //= 16
# convert each element index to block scale offset
offsets = get_block_scale_offset(torch.meshgrid(*indices, indexing="ij"), result_fp4, "NVFP4")
# get the scales and convert to float for comparison
d_out_scale_indexed = d_out_scale[offsets]
assert torch.equal(d_out_scale_indexed, d_out_scale_expanded)

# And unpack the fp4 results.
result_unpacked = unpack_fp4(result_fp4, axis=-1)
print(
    f"Result unpacking:\n"
    f"shape: {result_fp4.shape} -> {result_unpacked.shape},\n"
    f"dtype: {result_fp4.dtype} -> {result_unpacked.dtype}"
)
assert tuple(result_unpacked.shape) == output_shape

# We convert the out scale from float8_e4m3fn to float32,
# as torch does not support fp32 * fp8_e4m3fn
# elementwise multiplication directly.
result_unpacked *= d_out_scale_expanded.float()

print(f"Result unpacked: {result_unpacked}")
torch.testing.assert_close(result_unpacked, result_fp32, rtol=0.1, atol=1e-6)
