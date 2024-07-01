# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
Algorithm utilities.
"""

__all__ = ['algorithm_dtype']


import numpy as np

algorithm_dtype = np.dtype([('algorithm', np.uint64, (8,)), ('workspace_size', np.uint64), ('status', np.int32), ('waves_count', np.float32), ('reserved', np.int32, (4,))])
