# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
An example illustrating how to specify execution options including memory limit
and the execution device for a binary tensor contraction operation.

The input execution options can be provided in the following ways:
- As a string ('cuda')
- As a :class:`ExecutionCUDA` object
- As a dictionary containing the parameters for the :class:`ExecutionCUDA` constructor

The inputs as well as the result are NumPy ndarrays.
"""

import numpy as np

import cuda.core.experimental as ccx

from nvmath.tensor import ExecutionCUDA, binary_contraction


a = np.random.rand(4, 4, 12, 12)
b = np.random.rand(12, 12, 8, 8)


# By default, the execution is set to "cuda" with device_id = 0
result = binary_contraction("ijkl,klmn->ijmn", a, b, execution="cuda")

assert np.allclose(result, np.einsum("ijkl,klmn->ijmn", a, b))

# Execution can also be provided as an ExecutionCUDA object
num_devices = ccx.system.num_devices

for device_id in range(num_devices):
    execution = ExecutionCUDA(device_id=device_id)
    result = binary_contraction("ijkl,klmn->ijmn", a, b, execution=execution)
    assert np.allclose(result, np.einsum("ijkl,klmn->ijmn", a, b))

# Additionally, execution can be provided as a dictionary
# and the name key must be set to 'cuda'
execution = {"name": "cuda", "device_id": num_devices - 1}
result = binary_contraction("ijkl,klmn->ijmn", a, b, execution=execution)
assert np.allclose(result, np.einsum("ijkl,klmn->ijmn", a, b))
