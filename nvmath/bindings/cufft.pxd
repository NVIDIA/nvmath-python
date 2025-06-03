# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated across versions from 11.0.3 to 12.8.0. Do not modify it directly.

from libc.stdint cimport intptr_t

from .cycufft cimport *


###############################################################################
# Types
###############################################################################

ctypedef cufftXt1dFactors Xt1dFactors
ctypedef cufftCallbackLoadC CallbackLoadC
ctypedef cufftCallbackLoadZ CallbackLoadZ
ctypedef cufftCallbackLoadR CallbackLoadR
ctypedef cufftCallbackLoadD CallbackLoadD
ctypedef cufftJITCallbackLoadC JITCallbackLoadC
ctypedef cufftJITCallbackLoadZ JITCallbackLoadZ
ctypedef cufftJITCallbackLoadR JITCallbackLoadR
ctypedef cufftJITCallbackLoadD JITCallbackLoadD
ctypedef cufftCallbackStoreR CallbackStoreR
ctypedef cufftJITCallbackStoreR JITCallbackStoreR
ctypedef cufftCallbackStoreD CallbackStoreD
ctypedef cufftJITCallbackStoreD JITCallbackStoreD
ctypedef cufftCallbackStoreC CallbackStoreC
ctypedef cufftJITCallbackStoreC JITCallbackStoreC
ctypedef cufftCallbackStoreZ CallbackStoreZ
ctypedef cufftJITCallbackStoreZ JITCallbackStoreZ

ctypedef cudaStream_t Stream
ctypedef cudaDataType DataType
ctypedef libraryPropertyType_t LibraryPropertyType


###############################################################################
# Enum
###############################################################################

ctypedef libFormat_t _LibFormat
ctypedef cufftResult _Result
ctypedef cufftType _Type
ctypedef cufftCompatibility _Compatibility
ctypedef cufftXtSubFormat _XtSubFormat
ctypedef cufftXtCopyType _XtCopyType
ctypedef cufftXtQueryType _XtQueryType
ctypedef cufftXtWorkAreaPolicy _XtWorkAreaPolicy
ctypedef cufftXtCallbackType _XtCallbackType
ctypedef cufftProperty _Property


###############################################################################
# Functions
###############################################################################

cpdef plan1d(intptr_t plan, int nx, int type, int batch)
cpdef plan2d(intptr_t plan, int nx, int ny, int type)
cpdef plan3d(intptr_t plan, int nx, int ny, int nz, int type)
cpdef plan_many(intptr_t plan, int rank, n, inembed, int istride, int idist, onembed, int ostride, int odist, int type, int batch)
cpdef size_t make_plan1d(int plan, int nx, int type, int batch) except? -1
cpdef size_t make_plan2d(int plan, int nx, int ny, int type) except? -1
cpdef size_t make_plan3d(int plan, int nx, int ny, int nz, int type) except? -1
cpdef size_t make_plan_many(int plan, int rank, n, inembed, int istride, int idist, onembed, int ostride, int odist, int type, int batch) except? -1
cpdef size_t make_plan_many64(int plan, int rank, n, inembed, long long int istride, long long int idist, onembed, long long int ostride, long long int odist, int type, long long int batch) except? -1
cpdef size_t get_size_many64(int plan, int rank, n, inembed, long long int istride, long long int idist, onembed, long long int ostride, long long int odist, int type, long long int batch) except? -1
cpdef size_t estimate1d(int nx, int type, int batch) except? -1
cpdef size_t estimate2d(int nx, int ny, int type) except? -1
cpdef size_t estimate3d(int nx, int ny, int nz, int type) except? -1
cpdef size_t estimate_many(int rank, n, inembed, int istride, int idist, onembed, int ostride, int odist, int type, int batch) except? -1
cpdef int create() except? -1
cpdef size_t get_size1d(int handle, int nx, int type, int batch) except? -1
cpdef size_t get_size2d(int handle, int nx, int ny, int type) except? -1
cpdef size_t get_size3d(int handle, int nx, int ny, int nz, int type) except? -1
cpdef size_t get_size_many(int handle, int rank, intptr_t n, intptr_t inembed, int istride, int idist, intptr_t onembed, int ostride, int odist, int type, int batch) except? -1
cpdef size_t get_size(int handle) except? -1
cpdef set_work_area(int plan, intptr_t work_area)
cpdef set_auto_allocation(int plan, int auto_allocate)
cpdef exec_c2c(int plan, intptr_t idata, intptr_t odata, int direction)
cpdef exec_r2c(int plan, intptr_t idata, intptr_t odata)
cpdef exec_c2r(int plan, intptr_t idata, intptr_t odata)
cpdef exec_z2z(int plan, intptr_t idata, intptr_t odata, int direction)
cpdef exec_d2z(int plan, intptr_t idata, intptr_t odata)
cpdef exec_z2d(int plan, intptr_t idata, intptr_t odata)
cpdef set_stream(int plan, intptr_t stream)
cpdef destroy(int plan)
cpdef int get_version() except? -1
cpdef int get_property(int type) except? -1
cpdef xt_set_gpus(int handle, int n_gpus, which_gpus)
cpdef intptr_t xt_malloc(int plan, int format) except? -1
cpdef xt_memcpy(int plan, intptr_t dst_pointer, intptr_t src_pointer, int type)
cpdef xt_free(intptr_t descriptor)
cpdef xt_set_work_area(int plan, intptr_t work_area)
cpdef xt_exec_descriptor_c2c(int plan, intptr_t input, intptr_t output, int direction)
cpdef xt_exec_descriptor_r2c(int plan, intptr_t input, intptr_t output)
cpdef xt_exec_descriptor_c2r(int plan, intptr_t input, intptr_t output)
cpdef xt_exec_descriptor_z2z(int plan, intptr_t input, intptr_t output, int direction)
cpdef xt_exec_descriptor_d2z(int plan, intptr_t input, intptr_t output)
cpdef xt_exec_descriptor_z2d(int plan, intptr_t input, intptr_t output)
cpdef xt_query_plan(int plan, intptr_t query_struct, int query_type)
cpdef xt_clear_callback(int plan, int cb_type)
cpdef xt_set_callback_shared_size(int plan, int cb_type, size_t shared_size)
cpdef size_t xt_make_plan_many(int plan, int rank, n, inembed, long long int istride, long long int idist, int inputtype, onembed, long long int ostride, long long int odist, int outputtype, long long int batch, int executiontype) except? 0
cpdef size_t xt_get_size_many(int plan, int rank, n, inembed, long long int istride, long long int idist, int inputtype, onembed, long long int ostride, long long int odist, int outputtype, long long int batch, int executiontype) except? 0
cpdef xt_exec(int plan, intptr_t input, intptr_t output, int direction)
cpdef xt_exec_descriptor(int plan, intptr_t input, intptr_t output, int direction)
cpdef xt_set_work_area_policy(int plan, int policy, intptr_t work_size)
cpdef xt_set_jit_callback(int plan, lto_callback_fatbin, size_t lto_callback_fatbin_size, int type, caller_info)
cpdef xt_set_subformat_default(int plan, int subformat_forward, int subformat_inverse)
cpdef set_plan_property_int64(int plan, int property, long long int input_value_int)
cpdef long long int get_plan_property_int64(int plan, int property) except? -1
cpdef reset_plan_property(int plan, int property)
