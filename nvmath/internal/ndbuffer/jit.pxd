# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport intptr_t

cdef intptr_t get_kernel(str kernel_code, str kernel_name, int device_id, include_path, object logger=*) except? 0
