# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example demonstrates how the reset_operands method of a Matmul object can be used to
change both the operands and their quantization scales.

FP8 is only supported with cuBLAS 12.8 or newer and on devices with compute
capability 8.9 or higher.
"""

import torch

import nvmath

# Prepare sample input data
m, n, k = 128, 256, 16
a = torch.ones(m, k, device="cuda", dtype=torch.float8_e5m2)
b = torch.ones(n, k, device="cuda", dtype=torch.float8_e4m3fn).T
print(f"A = \n{a}")
print(f"\nB = \n{b}")

scales = {"a": 3, "b": 2, "d": 1}

with nvmath.linalg.advanced.Matmul(
    a, b, quantization_scales=scales, options={"result_type": nvmath.CudaDataType.CUDA_R_8F_E5M2}
) as mm:
    # Plan the multiplication
    mm.plan()

    # Execute the multiplication and print the result
    result = mm.execute()
    print(f"\nA (A scale: {scales['a']}) @ B (B scale: {scales['b']}) = (D scale: {scales['d']}) \n{result}")

    # Replace A with a matrix filled with 128 and adjust A and D scales.
    # Note that since no new scale for B is specified, it will remain unchanged.
    new_a = torch.full((m, k), 128, device="cuda").type(torch.float8_e5m2)
    print(f"\nnew A = \n{new_a}")
    new_a_scale = 1
    new_d_scale = 0.01
    mm.reset_operands(a=new_a, quantization_scales={"a": new_a_scale, "d": new_d_scale})

    # Execute the multiplication again and print the new result
    result2 = mm.execute()
    print(f"\nA (A scale: {new_a_scale}) @ B (B scale: {scales['b']}) = (D scale: {new_d_scale}) \n{result2}")
