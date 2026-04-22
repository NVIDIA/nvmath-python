# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example demonstrates automatic output scaling in MXFP8 mode.
When using MXFP8, D is automatically scaled during the matmul operation and the scale used
is returned as "d_out_scale". This scale can be used as input for subsequent matrix
multiplications (see mxfp8_chaining example) or applied to the result using a helper
function.

To use MXFP8, set the `block_scaling` option to True.

The layout of MXFP8 scales is complex. To simplify working with them, we provide helper
functions in `nvmath.distributed.linalg.advanced.helpers.matmul`. For more advanced
operations on MXFP8 scales, please refer to the cuBLAS documentation.

MXFP8 is only supported with compute capability 10.0 or higher.

$ mpiexec -n 4 python example19_mxfp8_d_out.py
"""

import numpy as np
import torch
from mpi4py import MPI

import nvmath.distributed
from nvmath.distributed.distribution import Slab
from nvmath.distributed.linalg.advanced import matrix_qualifiers_dtype

# Initialize nvmath.distributed.
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nranks = comm.Get_size()
device_id = rank % torch.cuda.device_count()
# cuBLASMp requires NCCL communication backends.
nvmath.distributed.initialize(device_id, comm, backends=["nccl"])

# Prepare sample input data. Note that N, M and K (for local GEMMs)
# must be divisible by 128 for MXFP8.
m, n, k = 512, 256, 512

row_wise_distribution = Slab.X
col_wise_distribution = Slab.Y

# FP8 operations require TN input layout.

with torch.cuda.device(device_id):
    a = torch.zeros(*row_wise_distribution.shape(rank, (m, k)), device="cuda").type(torch.float8_e4m3fn)
    b = torch.rand(*row_wise_distribution.shape(rank, (n, k)), device="cuda").type(torch.float8_e4m3fn)

# Get a transposed view to obtain column-major memory layout. Note that this
# also changes the distribution of a and b (see example01 for more information).
a = a.T  # a is now (k, m) with col_wise_distribution
b = b.T  # b is now (k, n) with col_wise_distribution

# Distributions for A, B, and result matrix D
distributions = [col_wise_distribution, col_wise_distribution, row_wise_distribution]

# Create matrix A with values increasing by column to demonstrate scaling with different
# magnitudes (each column will have progressively larger values)
a[:] = torch.arange(m // nranks * rank, m // nranks * (rank + 1))[None, :]
print("Matrix A:")
print(a)
print()

print("Matrix B:")
print(b)
print()

# Prepare quantization scales for A and B using the create_mxfp8_scale helper.
# Note: We don't set a scale for D since MXFP8 automatically scales the result to fit
# within the output type's dynamic range.
scales = {
    "a": nvmath.distributed.linalg.advanced.helpers.matmul.create_mxfp8_scale(a, -6),  # 2^-6 = 0.015625
    "b": nvmath.distributed.linalg.advanced.helpers.matmul.create_mxfp8_scale(b, 0),  # 2^0 = 1
}

# Enable block scaling
options = {"block_scaling": True}

qualifiers = np.zeros((3,), dtype=matrix_qualifiers_dtype)
qualifiers[0]["is_transpose"] = True

# Perform the multiplication. The result is a tuple containing (result, aux).
# The aux dictionary contains "d_out_scale" - the scale used for the result.
result, aux = nvmath.distributed.linalg.advanced.matmul(
    a,
    b,
    distributions=distributions,
    qualifiers=qualifiers,
    quantization_scales=scales,
    options=options,
)

# Display results
print("Result (each block scaled to fit within float8_e4m3fn range):")
# Printing the tensor synchronizes on the default CUDA stream.
print(result)
print()

# Examine the D_OUT quantization scales
if rank == 0:
    print(f"Auxiliary output contains these keys: {list(aux.keys())}")
print(
    f"D scale tensor shape: {aux['d_out_scale'].shape}, type: {aux['d_out_scale'].dtype}. "
    f"Contains {len(aux['d_out_scale'].unique())} unique scale factors."
)

# Apply the scale to get the actual result. Note: This helper function is for demonstration
# purposes and may use significant memory. For production use, set result_type to a
# non-FP8 type instead.
actual_result = nvmath.distributed.linalg.advanced.helpers.matmul.apply_mxfp8_scale(result, aux["d_out_scale"])
print("Final result (with quantization scales applied):")
print(actual_result)
