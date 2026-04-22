# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

from cuda.bindings.driver import (
    CUresult,
    cuStreamSynchronize,
    cuMemcpyAsync,
    CUmemPool_attribute,
    cuMemPoolGetAttribute,
    cuMemPoolTrimTo,
    cuLaunchKernel,
)


class CudaError(RuntimeError):
    def __init__(self, error_code):
        self.error_code = error_code
        super().__init__(f"{CUresult(error_code).name}")


cdef inline check_driver_error(result):
    if result != CUresult.CUDA_SUCCESS:
        raise CudaError(result)


def handle_return(tuple result):
    check_driver_error(result[0])
    cdef int out_len = len(result)
    if out_len == 1:
        return
    elif out_len == 2:
        return result[1]
    else:
        return result[1:]


cpdef stream_sync(intptr_t stream):
   handle_return(
       cuStreamSynchronize(stream)
   )


cpdef memcpy_async(intptr_t dst_ptr, intptr_t src_ptr, int64_t size, intptr_t stream):
   handle_return(
       cuMemcpyAsync(
           dst_ptr,
           src_ptr,
           size,
           stream
       )
   )


cpdef uint64_t get_memory_pool_reserved_memory_size(pool) except? -1:
    cdef ret = handle_return(
        cuMemPoolGetAttribute(
            pool, CUmemPool_attribute.CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT
        )
    )
    return ret


cpdef uint64_t get_memory_pool_used_memory_size(pool) except? -1:
    cdef ret = handle_return(
        cuMemPoolGetAttribute(
            pool, CUmemPool_attribute.CU_MEMPOOL_ATTR_USED_MEM_CURRENT
        )
    )
    return ret


cpdef free_memory_pool_reserved_memory(pool):
   handle_return(
        cuMemPoolTrimTo(pool, 0)
    )


cpdef launch_kernel(intptr_t f, intptr_t kernel_params, int gx, int gy, int gz, int bx, int by, int bz, unsigned int shared_mem_bytes, intptr_t stream_handle):
    return handle_return(
        cuLaunchKernel(
            f,
            gx,
            gy,
            gz,
            bx,
            by,
            bz,
            shared_mem_bytes,
            stream_handle,
            kernel_params,
            0
        )
    )
