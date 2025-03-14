# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example demonstrates how different data types can be used in FP8 multiplication.

Two kinds of FP8 are supported: float8_e4m3fn and float8_e5m2, which have 4 and 5 bits for
the exponent respectively. We support e4m3*e4m3, e4m3*e5m2, and e5m2*e4m3 operations.

In FP8 operations, the MatmulOptions.result_type option can be used to specify the desired
output type. For the full list of supported type combinations, please visit the cuBLAS
documentation at https://docs.nvidia.com/cuda/cublas/#cublasltmatmul.

FP8 is only supported with cuBLAS 12.8 or newer and on devices with compute
capability 8.9 or higher.
"""

import torch

import nvmath

# Prepare sample input data. Note that A and B are FP8 numbers of different types.
m, n, k = 64, 32, 48
a = torch.rand(m, k, device="cuda").type(torch.float8_e5m2)
b = torch.rand(n, k, device="cuda").type(torch.float8_e4m3fn).T

# Perform the multiplication, requesting FP32 output. Note that a scale for the result (D
# is not specified because it is not FP8.
result_fp32 = nvmath.linalg.advanced.matmul(
    a, b, quantization_scales={"a": 1, "b": 1}, options={"result_type": nvmath.CudaDataType.CUDA_R_32F}
)

# Perform the multiplication, requesting FP16 output.
result_fp16 = nvmath.linalg.advanced.matmul(
    a, b, quantization_scales={"a": 1, "b": 1}, options={"result_type": nvmath.CudaDataType.CUDA_R_16F}
)

# Now, request FP8 (e4m3fn) output. We set the scale for D to 1 for simplicity - with small
# values in A and B, we won't exceed the range of the type anyway.
result_fp8_e4m3fn = nvmath.linalg.advanced.matmul(
    a, b, quantization_scales={"a": 1, "b": 1, "d": 1}, options={"result_type": nvmath.CudaDataType.CUDA_R_8F_E4M3}
)

# Finally, request FP8 (e5m2) output.
result_fp8_e5m2fn = nvmath.linalg.advanced.matmul(
    a, b, quantization_scales={"a": 1, "b": 1, "d": 1}, options={"result_type": nvmath.CudaDataType.CUDA_R_8F_E5M2}
)


# Print mean relative error for each of the types
def mean_relative_error_vs_fp32(x):
    reference = result_fp32.cpu()
    actual = x.cpu().type(torch.float32)
    return ((reference - actual).abs() / reference.abs()).mean()


print(f"{result_fp32.dtype=}.")
print(
    f"{result_fp16.dtype=}. The mean relative error to the FP32 reference is {mean_relative_error_vs_fp32(result_fp16):.07f}."
)
print(
    f"{result_fp8_e4m3fn.dtype=}. The mean relative error to the FP32 reference is "
    f"{mean_relative_error_vs_fp32(result_fp8_e4m3fn):.07f}."
)
print(
    f"{result_fp8_e5m2fn.dtype=}. The mean relative error to the FP32 reference is "
    f"{mean_relative_error_vs_fp32(result_fp8_e5m2fn):.07f}."
)
