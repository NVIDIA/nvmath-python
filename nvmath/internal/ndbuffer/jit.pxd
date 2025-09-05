# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport intptr_t

cdef intptr_t get_kernel(str kernel_code, str kernel_name, int device_id, str includes_key, object logger=*) except -1
cpdef discover_includes(list include_dirs)
cpdef bint register_includes(str includes_key, list include_names, list includes)
cpdef get_includes(str includes_key)
cpdef _invalidate_kernel_cache()
