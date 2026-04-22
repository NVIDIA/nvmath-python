# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
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

FP8 is only supported with compute capability 8.9 or higher.

$ mpiexec -n 4 python example17_fp8_epilog_aux.py
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

m, n, k = 128, 128, 128

row_wise_distribution = Slab.X
col_wise_distribution = Slab.Y

# FP8 operations require TN input layout.

with torch.cuda.device(device_id):
    a = (torch.randn(*row_wise_distribution.shape(rank, (m, k)), device="cuda") - 0.5).type(torch.float8_e4m3fn)
    b = (torch.randn(*row_wise_distribution.shape(rank, (n, k)), device="cuda") - 0.5).type(torch.float8_e4m3fn)

# Get a transposed view to obtain column-major memory layout. Note that this
# also changes the distribution of a and b (see example01 for more information).
a = a.T  # a is now (k, m) with col_wise_distribution
b = b.T  # b is now (k, n) with col_wise_distribution

# Distributions for A, B, and result matrix D
distributions = [col_wise_distribution, col_wise_distribution, row_wise_distribution]

qualifiers = np.zeros((3,), dtype=matrix_qualifiers_dtype)
qualifiers[0]["is_transpose"] = True

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
result, aux = nvmath.distributed.linalg.advanced.matmul(
    a,
    b,
    distributions=distributions,
    qualifiers=qualifiers,
    epilog=nvmath.distributed.linalg.advanced.MatmulEpilog.GELU_AUX,
    epilog_inputs=epilog_inputs,
    preferences=preferences,
    quantization_scales=scales,
)

# Print the result.
if rank == 0:
    print("Result:")
    # Printing the tensor synchronizes on the default CUDA stream.
    print(result)
    print()

# Print the auxiliary values returned. There should be "gelu_aux" (scaled by 0.1) and
# "gelu_aux_amax" containing the maximum absolute value before scaling (amax).
assert set(aux.keys()) == {"gelu_aux", "gelu_aux_amax"}
if rank == 0:
    print(f"Auxiliary outputs are {set(aux.keys())}:")
    print(aux)
    print()

    print(f"Note that gelu_aux is an FP8 tensor: {aux['gelu_aux'].dtype=}")
    print(f"Also, amax has been returned: {aux['gelu_aux_amax']=}")
