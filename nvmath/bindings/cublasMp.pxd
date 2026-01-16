# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated across versions from 0.6.0 to 0.7.0. Do not modify it directly.

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

ctypedef cublasMpStatus_t _Status
ctypedef cublasMpGridLayout_t _GridLayout
ctypedef cublasMpMatmulDescriptorAttribute_t _MatmulDescriptorAttribute
ctypedef cublasMpMatmulAlgoType_t _MatmulAlgoType
ctypedef cublasMpMatmulEpilogue_t _MatmulEpilogue
ctypedef cublasMpMatmulMatrixScale_t _MatmulMatrixScale
ctypedef cublasMpEmulationStrategy_t _EmulationStrategy


###############################################################################
# Functions
###############################################################################

cpdef intptr_t create(intptr_t stream) except? 0
cpdef destroy(intptr_t handle)
cpdef stream_set(intptr_t handle, intptr_t stream)
cpdef intptr_t stream_get(intptr_t handle) except? 0
cpdef int get_version() except? 0
cpdef intptr_t grid_create(int64_t nprow, int64_t npcol, int layout, intptr_t comm) except? 0
cpdef grid_destroy(intptr_t grid)
cpdef intptr_t matrix_descriptor_create(int64_t m, int64_t n, int64_t mb, int64_t nb, int64_t rsrc, int64_t csrc, int64_t lld, int type, intptr_t grid) except? 0
cpdef matrix_descriptor_destroy(intptr_t desc)
cpdef matrix_descriptor_init(int64_t m, int64_t n, int64_t mb, int64_t nb, int64_t rsrc, int64_t csrc, int64_t lld, int type, intptr_t grid, intptr_t desc)
cpdef intptr_t matmul_descriptor_create(int compute_type) except? 0
cpdef matmul_descriptor_destroy(intptr_t matmul_desc)
cpdef matmul_descriptor_init(intptr_t matmul_desc, int compute_type)
cpdef get_matmul_descriptor_attribute_dtype(int attr)
cpdef matmul_descriptor_attribute_set(intptr_t matmul_desc, int attr, intptr_t buf, size_t size_in_bytes)
cpdef matmul_descriptor_attribute_get(intptr_t matmul_desc, int attr, intptr_t buf, size_t size_in_bytes, intptr_t size_written)
cpdef tuple trsm_buffer_size(intptr_t handle, int side, int uplo, int trans, int diag, int64_t m, int64_t n, intptr_t alpha, intptr_t a, int64_t ia, int64_t ja, intptr_t desc_a, intptr_t b, int64_t ib, int64_t jb, intptr_t desc_b, int compute_type)
cpdef trsm(intptr_t handle, int side, int uplo, int trans, int diag, int64_t m, int64_t n, intptr_t alpha, intptr_t a, int64_t ia, int64_t ja, intptr_t desc_a, intptr_t b, int64_t ib, int64_t jb, intptr_t desc_b, int compute_type, intptr_t d_work, size_t workspace_size_in_bytes_on_device, intptr_t h_work, size_t workspace_size_in_bytes_on_host)
cpdef tuple gemm_buffer_size(intptr_t handle, int trans_a, int trans_b, int64_t m, int64_t n, int64_t k, intptr_t alpha, intptr_t a, int64_t ia, int64_t ja, intptr_t desc_a, intptr_t b, int64_t ib, int64_t jb, intptr_t desc_b, intptr_t beta, intptr_t c, int64_t ic, int64_t jc, intptr_t desc_c, int compute_type)
cpdef gemm(intptr_t handle, int trans_a, int trans_b, int64_t m, int64_t n, int64_t k, intptr_t alpha, intptr_t a, int64_t ia, int64_t ja, intptr_t desc_a, intptr_t b, int64_t ib, int64_t jb, intptr_t desc_b, intptr_t beta, intptr_t c, int64_t ic, int64_t jc, intptr_t desc_c, int compute_type, intptr_t d_work, size_t workspace_size_in_bytes_on_device, intptr_t h_work, size_t workspace_size_in_bytes_on_host)
cpdef tuple matmul_buffer_size(intptr_t handle, intptr_t matmul_desc, int64_t m, int64_t n, int64_t k, intptr_t alpha, intptr_t a, int64_t ia, int64_t ja, intptr_t desc_a, intptr_t b, int64_t ib, int64_t jb, intptr_t desc_b, intptr_t beta, intptr_t c, int64_t ic, int64_t jc, intptr_t desc_c, intptr_t d, int64_t id, int64_t jd, intptr_t desc_d)
cpdef matmul(intptr_t handle, intptr_t matmul_desc, int64_t m, int64_t n, int64_t k, intptr_t alpha, intptr_t a, int64_t ia, int64_t ja, intptr_t desc_a, intptr_t b, int64_t ib, int64_t jb, intptr_t desc_b, intptr_t beta, intptr_t c, int64_t ic, int64_t jc, intptr_t desc_c, intptr_t d, int64_t id, int64_t jd, intptr_t desc_d, intptr_t d_work, size_t workspace_size_in_bytes_on_device, intptr_t h_work, size_t workspace_size_in_bytes_on_host)
cpdef tuple syrk_buffer_size(intptr_t handle, int uplo, int trans, int64_t n, int64_t k, intptr_t alpha, intptr_t a, int64_t ia, int64_t ja, intptr_t desc_a, intptr_t beta, intptr_t c, int64_t ic, int64_t jc, intptr_t desc_c, int compute_type)
cpdef syrk(intptr_t handle, int uplo, int trans, int64_t n, int64_t k, intptr_t alpha, intptr_t a, int64_t ia, int64_t ja, intptr_t desc_a, intptr_t beta, intptr_t c, int64_t ic, int64_t jc, intptr_t desc_c, int compute_type, intptr_t d_work, size_t workspace_size_in_bytes_on_device, intptr_t h_work, size_t workspace_size_in_bytes_on_host)
cpdef int64_t numroc(int64_t n, int64_t nb, uint32_t iproc, uint32_t isrcproc, uint32_t nprocs) except? -1
cpdef tuple gemr2d_buffer_size(intptr_t handle, int64_t m, int64_t n, intptr_t a, int64_t ia, int64_t ja, intptr_t desc_a, intptr_t b, int64_t ib, int64_t jb, intptr_t desc_b, intptr_t global_comm)
cpdef gemr2d(intptr_t handle, int64_t m, int64_t n, intptr_t a, int64_t ia, int64_t ja, intptr_t desc_a, intptr_t b, int64_t ib, int64_t jb, intptr_t desc_b, intptr_t d_work, size_t workspace_size_in_bytes_on_device, intptr_t h_work, size_t workspace_size_in_bytes_on_host, intptr_t global_comm)
cpdef tuple trmr2d_buffer_size(intptr_t handle, int uplo, int diag, int64_t m, int64_t n, intptr_t a, int64_t ia, int64_t ja, intptr_t desc_a, intptr_t b, int64_t ib, int64_t jb, intptr_t desc_b, intptr_t global_comm)
cpdef trmr2d(intptr_t handle, int uplo, int diag, int64_t m, int64_t n, intptr_t a, int64_t ia, int64_t ja, intptr_t desc_a, intptr_t b, int64_t ib, int64_t jb, intptr_t desc_b, intptr_t d_work, size_t workspace_size_in_bytes_on_device, intptr_t h_work, size_t workspace_size_in_bytes_on_host, intptr_t global_comm)
cpdef tuple geadd_buffer_size(intptr_t handle, int trans, int64_t m, int64_t n, intptr_t alpha, intptr_t a, int64_t ia, int64_t ja, intptr_t desc_a, intptr_t beta, intptr_t c, int64_t ic, int64_t jc, intptr_t desc_c)
cpdef geadd(intptr_t handle, int trans, int64_t m, int64_t n, intptr_t alpha, intptr_t a, int64_t ia, int64_t ja, intptr_t desc_a, intptr_t beta, intptr_t c, int64_t ic, int64_t jc, intptr_t desc_c, intptr_t d_work, size_t workspace_size_in_bytes_on_device, intptr_t h_work, size_t workspace_size_in_bytes_on_host)
cpdef tuple tradd_buffer_size(intptr_t handle, int uplo, int trans, int64_t m, int64_t n, intptr_t alpha, intptr_t a, int64_t ia, int64_t ja, intptr_t desc_a, intptr_t beta, intptr_t c, int64_t ic, int64_t jc, intptr_t desc_c)
cpdef tradd(intptr_t handle, int uplo, int trans, int64_t m, int64_t n, intptr_t alpha, intptr_t a, int64_t ia, int64_t ja, intptr_t desc_a, intptr_t beta, intptr_t c, int64_t ic, int64_t jc, intptr_t desc_c, intptr_t d_work, size_t workspace_size_in_bytes_on_device, intptr_t h_work, size_t workspace_size_in_bytes_on_host)
cpdef set_emulation_strategy(intptr_t handle, int emulation_strategy)
cpdef int get_emulation_strategy(intptr_t handle) except? -1
