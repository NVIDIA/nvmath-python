# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

from .ndbuffer cimport NDBuffer


cpdef NDBuffer empty_numpy_like(NDBuffer other, object axis_order=*)
cpdef NDBuffer wrap_numpy_array(object array)
cpdef NDBuffer wrap_cupy_array(object array)
cpdef str is_c_or_f(object shape, object strides)
