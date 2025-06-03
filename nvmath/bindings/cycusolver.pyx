# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated across versions from 11.0.3 to 12.8.0. Do not modify it directly.

from ._internal cimport cusolver as _cusolver


###############################################################################
# Wrapper functions
###############################################################################

cdef cusolverStatus_t cusolverGetProperty(libraryPropertyType type, int* value) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cusolver._cusolverGetProperty(type, value)


cdef cusolverStatus_t cusolverGetVersion(int* version) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cusolver._cusolverGetVersion(version)
