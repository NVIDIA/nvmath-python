# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated across versions from 11.0.3 to 12.8.0. Do not modify it directly.

from ..cycusolver cimport *


###############################################################################
# Wrapper functions
###############################################################################

cdef cusolverStatus_t _cusolverGetProperty(libraryPropertyType type, int* value) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusolverStatus_t _cusolverGetVersion(int* version) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil
