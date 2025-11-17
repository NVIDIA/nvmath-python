# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated across versions from 0.5.0 to 0.6.0. Do not modify it directly.

from libc.stdint cimport intptr_t

from .cycublasMp cimport *


###############################################################################
# Types
###############################################################################

ctypedef ncclComm_t ncclComm
ctypedef cublasMpHandle_t Handle
ctypedef cublasMpGrid_t Grid
ctypedef cublasMpMatrixDescriptor_t MatrixDescriptor
ctypedef cublasMpMatmulDescriptor_t MatmulDescriptor
ctypedef cublasMpLoggerCallback_t LoggerCallback

ctypedef cudaStream_t Stream
ctypedef cudaDataType DataType
ctypedef libraryPropertyType_t LibraryPropertyType


###############################################################################
# Enum
###############################################################################

ctypedef cublasOperation_t _Operation
ctypedef cublasComputeType_t _ComputeType
ctypedef cublasMpStatus_t _Status
ctypedef cublasMpGridLayout_t _GridLayout
ctypedef cublasMpMatmulDescriptorAttribute_t _MatmulDescriptorAttribute
ctypedef cublasMpMatmulAlgoType_t _MatmulAlgoType
ctypedef cublasMpMatmulEpilogue_t _MatmulEpilogue
ctypedef cublasMpMatmulMatrixScale_t _MatmulMatrixScale


###############################################################################
# Functions
###############################################################################

cpdef intptr_t create(intptr_t stream) except? 0
cpdef destroy(intptr_t handle)
cpdef stream_set(intptr_t handle, intptr_t stream)
cpdef int get_version() except? 0
cpdef intptr_t grid_create(int64_t nprow, int64_t npcol, int layout, intptr_t comm) except? 0
cpdef grid_destroy(intptr_t grid)
cpdef intptr_t matrix_descriptor_create(int64_t m, int64_t n, int64_t mb, int64_t nb, int64_t rsrc, int64_t csrc, int64_t lld, int type, intptr_t grid) except? 0
cpdef matrix_descriptor_destroy(intptr_t desc)
cpdef intptr_t matmul_descriptor_create(int compute_type) except? 0
cpdef matmul_descriptor_destroy(intptr_t matmul_desc)
cpdef get_matmul_descriptor_attribute_dtype(int attr)
cpdef matmul_descriptor_attribute_set(intptr_t matmul_desc, int attr, intptr_t buf, size_t size_in_bytes)
cpdef matmul_descriptor_attribute_get(intptr_t matmul_desc, int attr, intptr_t buf, size_t size_in_bytes, intptr_t size_written)
cpdef tuple matmul_buffer_size(intptr_t handle, intptr_t matmul_desc, int64_t m, int64_t n, int64_t k, intptr_t alpha, intptr_t a, int64_t ia, int64_t ja, intptr_t desc_a, intptr_t b, int64_t ib, int64_t jb, intptr_t desc_b, intptr_t beta, intptr_t c, int64_t ic, int64_t jc, intptr_t desc_c, intptr_t d, int64_t id, int64_t jd, intptr_t desc_d)
cpdef matmul(intptr_t handle, intptr_t matmul_desc, int64_t m, int64_t n, int64_t k, intptr_t alpha, intptr_t a, int64_t ia, int64_t ja, intptr_t desc_a, intptr_t b, int64_t ib, int64_t jb, intptr_t desc_b, intptr_t beta, intptr_t c, int64_t ic, int64_t jc, intptr_t desc_c, intptr_t d, int64_t id, int64_t jd, intptr_t desc_d, intptr_t d_work, size_t workspace_size_in_bytes_on_device, intptr_t h_work, size_t workspace_size_in_bytes_on_host)
cpdef int64_t numroc(int64_t n, int64_t nb, uint32_t iproc, uint32_t isrcproc, uint32_t nprocs) except? -1
