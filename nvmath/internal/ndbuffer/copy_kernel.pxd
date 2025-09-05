# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

cimport cython
from libc.stdint cimport intptr_t
from .data_layout cimport Layout

cdef int launch_copy_kernel(Layout dst, Layout src, intptr_t dst_ptr, intptr_t src_ptr, int device_id, intptr_t stream_ptr, object logger=*) except -1 nogil
