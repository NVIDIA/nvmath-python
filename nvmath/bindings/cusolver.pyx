# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated across versions from 11.0.3 to 12.6.2. Do not modify it directly.

cimport cython  # NOQA

from enum import IntEnum as _IntEnum


###############################################################################
# Enum
###############################################################################

class Status(_IntEnum):
    """See `cusolverStatus_t`."""
    SUCCESS = CUSOLVER_STATUS_SUCCESS
    NOT_INITIALIZED = CUSOLVER_STATUS_NOT_INITIALIZED
    ALLOC_FAILED = CUSOLVER_STATUS_ALLOC_FAILED
    INVALID_VALUE = CUSOLVER_STATUS_INVALID_VALUE
    ARCH_MISMATCH = CUSOLVER_STATUS_ARCH_MISMATCH
    MAPPING_ERROR = CUSOLVER_STATUS_MAPPING_ERROR
    EXECUTION_FAILED = CUSOLVER_STATUS_EXECUTION_FAILED
    INTERNAL_ERROR = CUSOLVER_STATUS_INTERNAL_ERROR
    MATRIX_TYPE_NOT_SUPPORTED = CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED
    NOT_SUPPORTED = CUSOLVER_STATUS_NOT_SUPPORTED
    ZERO_PIVOT = CUSOLVER_STATUS_ZERO_PIVOT
    INVALID_LICENSE = CUSOLVER_STATUS_INVALID_LICENSE
    IRS_PARAMS_NOT_INITIALIZED = CUSOLVER_STATUS_IRS_PARAMS_NOT_INITIALIZED
    IRS_PARAMS_INVALID = CUSOLVER_STATUS_IRS_PARAMS_INVALID
    IRS_PARAMS_INVALID_PREC = CUSOLVER_STATUS_IRS_PARAMS_INVALID_PREC
    IRS_PARAMS_INVALID_REFINE = CUSOLVER_STATUS_IRS_PARAMS_INVALID_REFINE
    IRS_PARAMS_INVALID_MAXITER = CUSOLVER_STATUS_IRS_PARAMS_INVALID_MAXITER
    IRS_INTERNAL_ERROR = CUSOLVER_STATUS_IRS_INTERNAL_ERROR
    IRS_NOT_SUPPORTED = CUSOLVER_STATUS_IRS_NOT_SUPPORTED
    IRS_OUT_OF_RANGE = CUSOLVER_STATUS_IRS_OUT_OF_RANGE
    IRS_NRHS_NOT_SUPPORTED_FOR_REFINE_GMRES = CUSOLVER_STATUS_IRS_NRHS_NOT_SUPPORTED_FOR_REFINE_GMRES
    IRS_INFOS_NOT_INITIALIZED = CUSOLVER_STATUS_IRS_INFOS_NOT_INITIALIZED
    IRS_INFOS_NOT_DESTROYED = CUSOLVER_STATUS_IRS_INFOS_NOT_DESTROYED
    IRS_MATRIX_SINGULAR = CUSOLVER_STATUS_IRS_MATRIX_SINGULAR
    INVALID_WORKSPACE = CUSOLVER_STATUS_INVALID_WORKSPACE

class EigType(_IntEnum):
    """See `cusolverEigType_t`."""
    TYPE_1 = CUSOLVER_EIG_TYPE_1
    TYPE_2 = CUSOLVER_EIG_TYPE_2
    TYPE_3 = CUSOLVER_EIG_TYPE_3

class EigMode(_IntEnum):
    """See `cusolverEigMode_t`."""
    NOVECTOR = CUSOLVER_EIG_MODE_NOVECTOR
    VECTOR = CUSOLVER_EIG_MODE_VECTOR

class EigRange(_IntEnum):
    """See `cusolverEigRange_t`."""
    ALL = CUSOLVER_EIG_RANGE_ALL
    I = CUSOLVER_EIG_RANGE_I
    V = CUSOLVER_EIG_RANGE_V

class Norm(_IntEnum):
    """See `cusolverNorm_t`."""
    INF_NORM = CUSOLVER_INF_NORM
    MAX_NORM = CUSOLVER_MAX_NORM
    ONE_NORM = CUSOLVER_ONE_NORM
    FRO_NORM = CUSOLVER_FRO_NORM

class IRSRefinement(_IntEnum):
    """See `cusolverIRSRefinement_t`."""
    IRS_REFINE_NOT_SET = CUSOLVER_IRS_REFINE_NOT_SET
    IRS_REFINE_NONE = CUSOLVER_IRS_REFINE_NONE
    IRS_REFINE_CLASSICAL = CUSOLVER_IRS_REFINE_CLASSICAL
    IRS_REFINE_CLASSICAL_GMRES = CUSOLVER_IRS_REFINE_CLASSICAL_GMRES
    IRS_REFINE_GMRES = CUSOLVER_IRS_REFINE_GMRES
    IRS_REFINE_GMRES_GMRES = CUSOLVER_IRS_REFINE_GMRES_GMRES
    IRS_REFINE_GMRES_NOPCOND = CUSOLVER_IRS_REFINE_GMRES_NOPCOND
    PREC_DD = CUSOLVER_PREC_DD
    PREC_SS = CUSOLVER_PREC_SS
    PREC_SHT = CUSOLVER_PREC_SHT

class PrecType(_IntEnum):
    """See `cusolverPrecType_t`."""
    R_8I = CUSOLVER_R_8I
    R_8U = CUSOLVER_R_8U
    R_64F = CUSOLVER_R_64F
    R_32F = CUSOLVER_R_32F
    R_16F = CUSOLVER_R_16F
    R_16BF = CUSOLVER_R_16BF
    R_TF32 = CUSOLVER_R_TF32
    R_AP = CUSOLVER_R_AP
    C_8I = CUSOLVER_C_8I
    C_8U = CUSOLVER_C_8U
    C_64F = CUSOLVER_C_64F
    C_32F = CUSOLVER_C_32F
    C_16F = CUSOLVER_C_16F
    C_16BF = CUSOLVER_C_16BF
    C_TF32 = CUSOLVER_C_TF32
    C_AP = CUSOLVER_C_AP

class AlgMode(_IntEnum):
    """See `cusolverAlgMode_t`."""
    ALG_0 = CUSOLVER_ALG_0
    ALG_1 = CUSOLVER_ALG_1
    ALG_2 = CUSOLVER_ALG_2

class StorevMode(_IntEnum):
    """See `cusolverStorevMode_t`."""
    COLUMNWISE = CUBLAS_STOREV_COLUMNWISE
    ROWWISE = CUBLAS_STOREV_ROWWISE

class DirectMode(_IntEnum):
    """See `cusolverDirectMode_t`."""
    FORWARD = CUBLAS_DIRECT_FORWARD
    BACKWARD = CUBLAS_DIRECT_BACKWARD

class DeterministicMode(_IntEnum):
    """See `cusolverDeterministicMode_t`."""
    DETERMINISTIC_RESULTS = CUSOLVER_DETERMINISTIC_RESULTS
    ALLOW_NON_DETERMINISTIC_RESULTS = CUSOLVER_ALLOW_NON_DETERMINISTIC_RESULTS


###############################################################################
# Error handling
###############################################################################

cdef class cuSOLVERError(Exception):

    def __init__(self, status):
        self.status = status
        s = Status(status)
        cdef str err = f"{s.name} ({s.value})"
        super(cuSOLVERError, self).__init__(err)

    def __reduce__(self):
        return (type(self), (self.status,))


###############################################################################
# Wrapper functions
###############################################################################

cpdef int get_property(int type) except? -1:
    """See `cusolverGetProperty`."""
    cdef int value
    with nogil:
        status = cusolverGetProperty(<LibraryPropertyType>type, &value)
    check_status(status)
    return value


cpdef int get_version() except? -1:
    """See `cusolverGetVersion`."""
    cdef int version
    with nogil:
        status = cusolverGetVersion(&version)
    check_status(status)
    return version
