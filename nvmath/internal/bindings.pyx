# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

cimport cython
cimport cuda.bindings.cydriver as cdriver
from cuda.bindings.cydriver cimport (
    CUresult, CUstream, CUdeviceptr, CUmodule,
    CUfunction, CUdevice_attribute,
    CUmemPool_attribute, CUmemoryPool,
)

import cuda.bindings.driver as driver


_CUresult_enum = driver.CUresult


class CudaError(RuntimeError):
    def __init__(self, result):
        self.error_code = result
        super().__init__(f"{_CUresult_enum(result).name}")


class CudaOutOfMemoryError(CudaError):
    def __init__(self):
        super().__init__(CUresult.CUDA_ERROR_OUT_OF_MEMORY)


cdef int check_driver_error(CUresult result) except -1 nogil:
    if result == CUresult.CUDA_SUCCESS:
        return 0
    elif result == CUresult.CUDA_ERROR_OUT_OF_MEMORY:
        raise CudaOutOfMemoryError()
    else:
        raise CudaError(result)


cpdef int stream_sync(intptr_t stream) except -1 nogil:
    return check_driver_error(
        cdriver.cuStreamSynchronize(<CUstream>stream)
    )


cpdef int memcpy_async(intptr_t dst_ptr, intptr_t src_ptr, int64_t size, intptr_t stream) except -1 nogil:
    return check_driver_error(
        cdriver.cuMemcpyAsync(
            <CUdeviceptr>dst_ptr,
            <CUdeviceptr>src_ptr,
            size,
            <CUstream>stream
        )
    )


cpdef intptr_t get_device_current_memory_pool(int device_id) except? 0 nogil:
    cdef CUmemoryPool pool
    check_driver_error(cdriver.cuDeviceGetMemPool(&pool, device_id))
    return <intptr_t>pool


cpdef int set_memory_pool_release_threshold(intptr_t pool_ptr, uint64_t threshold) except -1 nogil:
    check_driver_error(cdriver.cuMemPoolSetAttribute(<CUmemoryPool>pool_ptr, CUmemPool_attribute.CU_MEMPOOL_ATTR_RELEASE_THRESHOLD, &threshold))
    return 0

cpdef uint64_t get_memory_pool_release_threshold(intptr_t pool_ptr) except? -1 nogil:
    cdef uint64_t value
    check_driver_error(cdriver.cuMemPoolGetAttribute(<CUmemoryPool>pool_ptr, CUmemPool_attribute.CU_MEMPOOL_ATTR_RELEASE_THRESHOLD, &value))
    return value

cpdef uint64_t get_memory_pool_reserved_memory_size(intptr_t pool_ptr) except? -1 nogil:
    cdef uint64_t value
    check_driver_error(cdriver.cuMemPoolGetAttribute(<CUmemoryPool>pool_ptr, CUmemPool_attribute.CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT, &value))
    return value


cpdef uint64_t get_memory_pool_used_memory_size(intptr_t pool_ptr) except? -1 nogil:
    cdef uint64_t value
    check_driver_error(cdriver.cuMemPoolGetAttribute(<CUmemoryPool>pool_ptr, CUmemPool_attribute.CU_MEMPOOL_ATTR_USED_MEM_CURRENT, &value))
    return value


cpdef int free_memory_pool_reserved_memory(intptr_t pool_ptr) except -1 nogil:
    check_driver_error(cdriver.cuMemPoolTrimTo(<CUmemoryPool>pool_ptr, 0))
    return 0


cpdef intptr_t mem_alloc_async(int64_t size, intptr_t stream_handle) except? -1 nogil:
    cdef CUdeviceptr dptr
    check_driver_error(cdriver.cuMemAllocAsync(&dptr, size, <CUstream>stream_handle))
    return <intptr_t>dptr


cpdef int mem_free_async(intptr_t dptr, intptr_t stream_handle) except -1 nogil:
    check_driver_error(cdriver.cuMemFreeAsync(<CUdeviceptr>dptr, <CUstream>stream_handle))
    return 0


cpdef int launch_kernel(intptr_t f, intptr_t kernel_params, Dim3 grid_dim, Dim3 block_dim, unsigned int shared_mem_bytes, intptr_t stream_handle) except -1 nogil:
    check_driver_error(
        cdriver.cuLaunchKernel(
            <CUfunction>f,
            grid_dim.x,
            grid_dim.y,
            grid_dim.z,
            block_dim.x,
            block_dim.y,
            block_dim.z,
            shared_mem_bytes,
            <CUstream>stream_handle,
            <void**>kernel_params,
            NULL
        )
    )
    return 0


cdef int get_cc(int &major, int &minor, int device_id) except? -1 nogil:
    check_driver_error(cdriver.cuDeviceGetAttribute(&major, CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device_id))
    check_driver_error(cdriver.cuDeviceGetAttribute(&minor, CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device_id))
    return 0


cpdef intptr_t get_function_from_module(intptr_t module, const char *name) except? 0 nogil:
    cdef CUfunction f
    check_driver_error(cdriver.cuModuleGetFunction(&f, <CUmodule>module, name))
    return <intptr_t>f
