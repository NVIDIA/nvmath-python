# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated across versions from 11.0.3 to 12.8.0. Do not modify it directly.

cimport cython

from libc.stdint cimport intptr_t

from .cycusolver cimport *


###############################################################################
# Types
###############################################################################



ctypedef cudaStream_t Stream
ctypedef cudaDataType DataType
ctypedef libraryPropertyType_t LibraryPropertyType


###############################################################################
# Enum
###############################################################################

ctypedef cusolverStatus_t _Status
ctypedef cusolverEigType_t _EigType
ctypedef cusolverEigMode_t _EigMode
ctypedef cusolverEigRange_t _EigRange
ctypedef cusolverNorm_t _Norm
ctypedef cusolverIRSRefinement_t _IRSRefinement
ctypedef cusolverPrecType_t _PrecType
ctypedef cusolverAlgMode_t _AlgMode
ctypedef cusolverStorevMode_t _StorevMode
ctypedef cusolverDirectMode_t _DirectMode
ctypedef cusolverDeterministicMode_t _DeterministicMode


###############################################################################
# Functions
###############################################################################

cpdef int get_property(int type) except? -1
cpdef int get_version() except? -1


###############################################################################
# Error handling
###############################################################################

cdef class cuSOLVERError(Exception): pass


@cython.profile(False)
cpdef inline check_status(int status):
    if status != 0:
        raise cuSOLVERError(status)
