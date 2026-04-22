# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

cimport cython
from libc.stdint cimport int64_t, intptr_t, uint64_t

cpdef get_device_memory_resource(int device_id)
cpdef allocate_from_mr(mr, int64_t size, stream, int device_id, logger=*)
cpdef free_reserved_memory(bint sync=*)
