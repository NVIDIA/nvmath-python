# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example demonstrates basic distributed matrix multiplication of FP8 tensors using
MXFP8 (microscaled FP8) quantization scales.

Key differences from FP8:
- MXFP8 scales are applied to each 32-element block of the tensors, rather than using a
  single tensor-wide scaling factor. This allows more fine-grained control over scaling
  and improves the accuracy of MXFP8 operations.
- MXFP8 scales are uint8 numbers in exponent-only format, representing values of the form
  2^n, where n is an integer between -127 and 128.
- In MXFP8 mode, if D is FP8, it is scaled automatically during the matmul operation and
  the quantization scales used are returned as "d_out_scale". This is covered in the next
  example.

To use MXFP8, set the `block_scaling` option to True.

The layout of the quantization scales is relatively complex. To facilitate working with
MXFP8, we provide helper functions in `nvmath.distributed.linalg.advanced.helpers.matmul`.

FP8 is only supported with compute capability 10.0 or higher.

$ mpiexec -n 4 python example18_mxfp8.py
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
m, n, k = 512, 512, 512

row_wise_distribution = Slab.X
col_wise_distribution = Slab.Y

# FP8 operations require TN input layout.

with torch.cuda.device(device_id):
    a = torch.zeros(*row_wise_distribution.shape(rank, (m, k)), dtype=torch.float8_e4m3fn, device="cuda")
    # B is filled with ones.
    b = torch.ones(*row_wise_distribution.shape(rank, (n, k)), dtype=torch.float8_e4m3fn, device="cuda")

# Get a transposed view to obtain column-major memory layout. Note that this
# also changes the distribution of a and b (see example01 for more information).
a = a.T  # a is now (k, m) with col_wise_distribution
b = b.T  # b is now (k, n) with col_wise_distribution

# Distributions for A, B, and result matrix D
distributions = [col_wise_distribution, col_wise_distribution, row_wise_distribution]

# Initialize A as global identity matrix.
with torch.cuda.device(device_id):
    i = rank * (m // nranks)
    j = i + (m // nranks)
    a[i:j, :] = torch.eye(m // nranks, device="cuda", dtype=torch.float8_e4m3fn)

# Prepare quantization scales for A and B using the `create_mxfp8_scale` helper.
# While MXFP8 allows different scales for different blocks in A and B,
# this helper creates uniform scaling across all blocks.
# For more advanced scale configurations, see the cuBLAS documentation and
# the `get_mxfp8_scale_offset` helper.
scales = {
    "a": nvmath.distributed.linalg.advanced.helpers.matmul.create_mxfp8_scale(a, -1),  # 2^-1 = 0.5
    "b": nvmath.distributed.linalg.advanced.helpers.matmul.create_mxfp8_scale(b, 3),  # 2^3 = 8
}

qualifiers = np.zeros((3,), dtype=matrix_qualifiers_dtype)
qualifiers[0]["is_transpose"] = True

# Enable block scaling by setting the `block_scaling` option to True. For simplicity, we
# request FP16 output. For FP8 output scaling, see the mxfp8_d_out_scale example.
options = {"block_scaling": True, "result_type": nvmath.CudaDataType.CUDA_R_16F}

# Perform the multiplication.
result = nvmath.distributed.linalg.advanced.matmul(
    a,
    b,
    distributions=distributions,
    qualifiers=qualifiers,
    quantization_scales=scales,
    options=options,
)

# Compute reference result without scaling
reference = nvmath.distributed.linalg.advanced.matmul(
    a.type(torch.float16),
    b.type(torch.float16),
    distributions=distributions,
    qualifiers=qualifiers,
)
if rank == 0:
    # Printing the tensor synchronizes on the default CUDA stream.
    print(f"Reference result (without scaling):\n{reference}")

    # Print the result with scaling applied
    print(f"Result with scaling (A scaled by 0.5, B scaled by 8):\n{result}")
