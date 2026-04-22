# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example demonstrates how different data types can be used in FP8 multiplication.

Two kinds of FP8 are supported: float8_e4m3fn and float8_e5m2, which have 4 and 5 bits for
the exponent respectively. We support e4m3*e4m3, e4m3*e5m2, and e5m2*e4m3 operations.

In FP8 operations, the MatmulOptions.result_type option can be used to specify the desired
output type. For the full list of supported type combinations, please visit the cuBLASMp
documentation at https://docs.nvidia.com/cuda/cublasmp/usage/functions.html#cublasmpmatmul.

FP8 is only supported with compute capability 8.9 or higher.

$ mpiexec -n 4 python example11_fp8_types.py
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
# cuBLASMp requires NCCL communication backend.
nvmath.distributed.initialize(device_id, comm, backends=["nccl"])

# Prepare sample input data. Note that A and B are FP8 numbers of different types.
m, n, k = 128, 64, 96

row_wise_distribution = Slab.X
col_wise_distribution = Slab.Y

# FP8 operations require TN input layout.

with torch.cuda.device(device_id):
    a = torch.rand(*row_wise_distribution.shape(rank, (m, k)), device="cuda").type(torch.float8_e5m2)
    b = torch.rand(*row_wise_distribution.shape(rank, (n, k)), device="cuda").type(torch.float8_e4m3fn)

# Get a transposed view to obtain column-major memory layout. Note that this
# also changes the distribution of a and b (see example01 for more information).
a = a.T  # a is now (k, m) with col_wise_distribution
b = b.T  # b is now (k, n) with col_wise_distribution

# Distributions for A, B, and result matrix D
distributions = [col_wise_distribution, col_wise_distribution, row_wise_distribution]

qualifiers = np.zeros((3,), dtype=matrix_qualifiers_dtype)
qualifiers[0]["is_transpose"] = True  # a is transposed

# Perform the multiplication, requesting FP32 output. Note that a scale for the result (D)
# is not specified because it is not FP8.
result_fp32 = nvmath.distributed.linalg.advanced.matmul(
    a,
    b,
    distributions=distributions,
    qualifiers=qualifiers,
    quantization_scales={"a": 1, "b": 1},
    options={"result_type": nvmath.CudaDataType.CUDA_R_32F},
)

# Perform the multiplication, requesting FP16 output.
result_fp16 = nvmath.distributed.linalg.advanced.matmul(
    a,
    b,
    distributions=distributions,
    qualifiers=qualifiers,
    quantization_scales={"a": 1, "b": 1},
    options={"result_type": nvmath.CudaDataType.CUDA_R_16F},
)

# Now, request FP8 (e4m3fn) output. We set the scale for D to 1 for simplicity - with small
# values in A and B we won't exceed the range of the type anyway.
result_fp8_e4m3fn = nvmath.distributed.linalg.advanced.matmul(
    a,
    b,
    distributions=distributions,
    qualifiers=qualifiers,
    quantization_scales={"a": 1, "b": 1, "d": 1},
    options={"result_type": nvmath.CudaDataType.CUDA_R_8F_E4M3},
)

# Finally, request FP8 (e5m2) output.
result_fp8_e5m2 = nvmath.distributed.linalg.advanced.matmul(
    a,
    b,
    distributions=distributions,
    qualifiers=qualifiers,
    quantization_scales={"a": 1, "b": 1, "d": 1},
    options={"result_type": nvmath.CudaDataType.CUDA_R_8F_E5M2},
)


# Print local mean relative error for each of the types
def mean_relative_error_vs_fp32(x):
    reference = result_fp32.cpu()
    actual = x.cpu().type(torch.float32)
    return ((reference - actual).abs() / reference.abs()).mean()


if rank == 0:
    print(f"{result_fp32.dtype=}.")
    # Execution with GPU operands is non-blocking by default. mean_relative_error_vs_fp32
    # brings the tensors to CPU and so it implicitly synchronizes on the CUDA default
    # stream.
    print(
        f"{result_fp16.dtype=}. The local mean relative error to the FP32 reference is "
        f"{mean_relative_error_vs_fp32(result_fp16):.07f}."
    )
    print(
        f"{result_fp8_e4m3fn.dtype=}. The local mean relative error to the FP32 reference is "
        f"{mean_relative_error_vs_fp32(result_fp8_e4m3fn):.07f}."
    )
    print(
        f"{result_fp8_e5m2.dtype=}. The local mean relative error to the FP32 reference is "
        f"{mean_relative_error_vs_fp32(result_fp8_e5m2):.07f}."
    )
