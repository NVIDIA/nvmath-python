# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example demonstrates how to obtain the maximum absolute value (amax) in the result,
computed before quantization.

In previous examples, quantization scales were set manually to appropriate values. Amax can
be used to automatically set proper scales in FP8 operations, as it indicates how much the
result needs to be scaled to fit into the dynamic range of the result type. In this example,
we first compute the result without scaling, then use amax to compute the correct scale, and
repeat the multiplication. While this approach is inefficient, it demonstrates the concept.
For a more practical example, see the `fp8_delayed_scaling` example, which uses amax from
previous iterations to choose scales for subsequent multiplications.

FP8 is only supported with cuBLAS 12.8 or newer and on devices with compute
capability 8.9 or higher.
"""

import torch

import nvmath

# Fill the input tensors with random numbers from (0, 30).
m, n, k = 128, 128, 16
a = (torch.rand(m, k, device="cuda") * 30).type(torch.float8_e4m3fn)
b = (torch.rand(n, k, device="cuda") * 30).type(torch.float8_e4m3fn).T

# To request amax, set `result_amax` option to True.
options = {"result_amax": True}

# When result_amax is set, a tuple containing the actual result and the auxiliary outputs
# will be returned instead of just the result.
result, aux = nvmath.linalg.advanced.matmul(a, b, quantization_scales={"a": 1, "b": 1, "d": 1}, options=options)

# With all quantization scales set to 1, most of the elements are clamped to the maximum
# value:
print("Result is:")
print(result)

# Amax will be present in the auxiliary outputs dictionary as "result_amax".
print(f"Matmul returned the result and the auxiliary outputs of type {type(aux)}: {aux}")

# Compute the proper scale by dividing the maximum representable value by amax.
max_representable_value = 448
amax = aux["result_amax"].item()
d_scale = max_representable_value / amax
print(f"d_scale = max_representable_value / amax = {max_representable_value} / {amax:.5f} = {d_scale:.5f}")

# Repeat the computation, this time using the proper scale for D.
result2 = nvmath.linalg.advanced.matmul(a, b, quantization_scales={"a": 1, "b": 1, "d": d_scale})
print(f"Result (with D scale set to {d_scale:.5f}) is:")
print(result2)
