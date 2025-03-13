# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example demonstrates how to implement a simple delayed scaling algorithm. We use the
amax value from the previous iteration to set the scale for the next iteration. In a more
advanced setup, an average amax from N previous iterations could be used as well. In each
iteration, we multiply two normally-distributed matrices A and B and add matrix C to the
result.

FP8 is only supported with cuBLAS 12.8 or newer and on devices with compute
capability 8.9 or higher.
"""

import torch

import nvmath

m, n, k = 256, 256, 256
a = torch.zeros(m, k, device="cuda", dtype=torch.float8_e4m3fn)
b = torch.zeros(n, k, device="cuda", dtype=torch.float8_e4m3fn).T
c = torch.zeros(m, n, device="cuda", dtype=torch.float16)


def regenerate_inputs():
    a[:] = torch.randn(a.shape, device="cuda") * 10
    b[:] = torch.randn(b.shape, device="cuda") * 10
    c[:] = torch.randn(c.shape, device="cuda") * 10
    return a, b, c


# Keep D scale in a GPU tensor instead of a Python float to allow in-place changes
dscale = torch.ones((1,), dtype=torch.float32, device="cuda")
scales = {"a": 1, "b": 1, "d": dscale}

# Request FP8 output and AMAX calculation
options = {"result_type": nvmath.CudaDataType.CUDA_R_8F_E4M3, "result_amax": True}

with nvmath.linalg.advanced.Matmul(a, b, c=c, beta=1, quantization_scales=scales, options=options) as mm:
    mm.plan()

    for iteration in range(10):
        # Populate a, b, and c with fresh random data
        regenerate_inputs()

        # Execute the matrix multiplication
        result, aux = mm.execute()
        amax = aux["result_amax"]

        # Calculate the percentage of clamped values
        max_representable_value = 448
        clamped_percent = (
            100 * ((result == max_representable_value) | (result == -max_representable_value)).sum().item() / result.nelement()
        )

        # Print a report. Note that the percentage of clamped values will rapidly decrease
        print(
            f"Iteration {iteration} with dscale={dscale.item():05f}: "
            f"amax={amax.item():.2f}, {clamped_percent:.02f}% of values were clamped to the max value."
        )

        # Update D scale for the next iteration
        dscale[:] = max_representable_value / amax
