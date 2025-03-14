# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example demonstrates using GELU_AUX epilog with FP8 outputs.

For GELU_AUX epilog, when A and B are e4m3fn, you can request the auxiliary output to
be returned as FP8. To request FP8 auxiliary output, set epilog.aux_type to an FP8 type
in MatmulPlanPreferences.

You can specify the scale for this auxiliary output by passing the scale
as "epilog_aux_scale" input in `epilog_inputs`. Additionally, you can request amax to be
computed for this output by setting `epilog.aux_amax=True` in MatmulPlanPreferences.

Note that FP8 auxiliary outputs are supported only for particular epilogs and type
combinations. For more details on the supported configurations, please see the cuBLAS
documentation.

FP8 is only supported with cuBLAS 12.8 or newer and on devices with compute
capability 8.9 or higher.
"""

import torch

import nvmath

m, n, k = 64, 64, 64
a = (torch.randn(m, k, device="cuda") - 0.5).type(torch.float8_e4m3fn)
b = (torch.randn(n, k, device="cuda") - 0.5).type(torch.float8_e4m3fn).T

scales = {"a": 1, "b": 1, "d": 1}

# Specify quantization scale to use for auxiliary epilog output
epilog_inputs = {"aux_quantization_scale": 0.1}

# Instead of a Dict, you may instantiate MatmulPlanPreferences object.
preferences = {
    "epilog": {
        "aux_type": nvmath.CudaDataType.CUDA_R_8F_E4M3,
        "aux_amax": True,
    }
}

# Execute the operation. Note that we pass `preferences` argument.
result, aux = nvmath.linalg.advanced.matmul(
    a,
    b,
    epilog=nvmath.linalg.advanced.MatmulEpilog.GELU_AUX,
    epilog_inputs=epilog_inputs,
    preferences=preferences,
    quantization_scales=scales,
)

# Print the result.
print("Result:")
print(result)
print()

# Print the auxiliary values returned. There should be "gelu_aux" (scaled by 0.1) and
# "gelu_aux_amax" containing the maximum absolute value before scaling (amax).
assert set(aux.keys()) == {"gelu_aux", "gelu_aux_amax"}
print(f"Auxiliary outputs are {set(aux.keys())}:")
print(aux)
print()

print(f"Note that gelu_aux is an FP8 tensor: {aux['gelu_aux'].dtype=}")
print(f"Also, amax has been returned: {aux['gelu_aux_amax']=}")
print(f"Also, amax has been returned: {aux['gelu_aux_amax']=}")
