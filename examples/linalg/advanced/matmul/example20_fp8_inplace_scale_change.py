# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example demonstrates how quantization scales passed as GPU tensors can be modified
in-place without needing to call reset_operands().

FP8 is only supported with cuBLAS 12.8 or newer and on devices with compute
capability 8.9 or higher.
"""

import torch

import nvmath

# Prepare sample input data
m, n, k = 128, 256, 16
a = torch.ones(m, k, device="cuda", dtype=torch.float8_e4m3fn)
b = torch.ones(n, k, device="cuda", dtype=torch.float8_e5m2).T
print(f"A = \n{a}")
print(f"\nB = \n{b}")

# Create 1D single-element float32 GPU tensors to hold the quantization scales.
# These will be modified in-place later.
scales = {
    "a": torch.full((1,), 3, dtype=torch.float32, device="cuda"),
    "b": torch.full((1,), 2, dtype=torch.float32, device="cuda"),
    "d": torch.full((1,), 1, dtype=torch.float32, device="cuda"),
}

with nvmath.linalg.advanced.Matmul(
    a, b, quantization_scales=scales, options={"result_type": nvmath.CudaDataType.CUDA_R_8F_E5M2}
) as mm:
    # Plan the multiplication
    mm.plan()

    # Execute the multiplication and print the result
    result = mm.execute()
    print(
        f"\nA (A scale: {scales['a'].item()}) @ B (B scale: {scales['b'].item()}) = (D scale: {scales['d'].item()}) \n{result}"
    )

    # Modify the quantization scales for A and D in-place
    scales["a"][:] = 2
    scales["d"][:] = 0.25

    # Execute the multiplication again with the new quantization scales and print the result
    result2 = mm.execute()
    print(
        f"\nA (A scale: {scales['a'].item()}) @ B (B scale: {scales['b'].item()}) = (D scale: {scales['d'].item()}) \n{result2}"
    )
