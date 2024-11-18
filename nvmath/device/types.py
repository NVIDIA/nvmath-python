# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

np_float16x2 = np.dtype([("x", np.float16), ("y", np.float16)], align=True)
np_float16x4 = np.dtype([("x", np.float16), ("y", np.float16), ("z", np.float16), ("w", np.float16)], align=True)

REAL_NP_TYPES = [np.float16, np.float32, np.float64]
