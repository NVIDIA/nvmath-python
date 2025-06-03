# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

np_float16x2 = np.dtype([("x", np.float16), ("y", np.float16)], align=True)
np_float16x4 = np.dtype([("x", np.float16), ("y", np.float16), ("z", np.float16), ("w", np.float16)], align=True)

REAL_NP_TYPES: list = [np.float16, np.float32, np.float64]
INT_NP_TYPES: list = [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64]
