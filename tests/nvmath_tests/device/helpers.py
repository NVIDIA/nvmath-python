# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import time

import numpy as np
from cuda import cudart, nvrtc, cuda
import cupy

from nvmath.device import CodeType, ComputeCapability


def CHECK_CUDART(err):
    if err != cudart.cudaError_t.cudaSuccess:
        err2, str = cudart.cudaGetErrorString(err)
        raise RuntimeError(f"CUDArt Error: {str} ({err})")


def CHECK_CUDA(err):
    if err != cuda.CUresult.CUDA_SUCCESS:
        err2, str = cuda.cuGetErrorName(err)
        raise RuntimeError(f"CUDA Error: {str} ({err})")


def CHECK_NVRTC(err, prog):
    if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        err, logsize = nvrtc.nvrtcGetProgramLogSize(prog)
        log = b" " * logsize
        err = nvrtc.nvrtcGetProgramLog(prog, log)
        raise RuntimeError(f"NVRTC error: {log.decode('ascii')}")


def set_device():
    (err,) = cudart.cudaSetDevice(0)
    CHECK_CUDART(err)
    err, prop = cudart.cudaGetDeviceProperties(0)
    CHECK_CUDART(err)
    # TODO: dx does not support platforms > arch90 for now and version is capped
    # at 9.0
    if (prop.major, prop.minor) > (9, 0):
        return (9, 0)
    return (prop.major, prop.minor)


def random_complex(shape, real_dtype, module=np):
    return module.random.randn(*shape).astype(real_dtype) + 1.0j * module.random.randn(*shape).astype(real_dtype)


def random_real(shape, real_dtype, module=np):
    return module.random.randn(*shape).astype(real_dtype)


def time_this(name, fun, *args, **kwargs):
    start = time.time()
    out = fun(*args, **kwargs)
    end = time.time()
    print(f"{name} finished in {end - start} sec.")
    return out


_TOLERANCE = {np.float16: 1e-2, np.float32: 1e-6, np.float64: 1e-14}


def show_FFT_traits(FFT):
    print(f"FFT.precision =            {FFT.precision}")
    print(f"FFT.value_type =           {FFT.value_type}")
    print(f"FFT.input_type =           {FFT.input_type}")
    print(f"FFT.output_type =          {FFT.output_type}")
    print(f"FFT.storage_size =         {FFT.storage_size}")
    print(f"FFT.shared_memory_size =   {FFT.shared_memory_size}")
    print(f"FFT.ffts_per_block =       {FFT.ffts_per_block}")
    print(f"FFT.code =                 {FFT.files}")
    print(f"FFT.stride =               {FFT.stride}")
    print(f"FFT.size =                 {FFT.size}")
    print(f"FFT.elements_per_thread =  {FFT.elements_per_thread}")
    print(f"FFT.block_dim =            {FFT.block_dim}")
    print(f"FFT.requires_workspace =   {FFT.requires_workspace}")
    print(f"FFT.workspace_size =       {FFT.workspace_size}")


def show_MM_traits(MM):
    print(f"MM.size =                  {MM.size}")
    print(f"MM.files =                 {MM.files}")
    print(f"MM.transpose_mode =        {MM.transpose_mode}")
    print(f"MM.value_type =            {MM.value_type}")
    print(f"MM.input_type =            {MM.input_type}")
    print(f"MM.output_type =           {MM.output_type}")
    print(f"MM.a_dim =                 {MM.a_dim}")
    print(f"MM.b_dim =                 {MM.b_dim}")
    print(f"MM.c_dim =                 {MM.c_dim}")
    print(f"MM.a_size =                {MM.a_size}")
    print(f"MM.b_size =                {MM.b_size}")
    print(f"MM.c_size =                {MM.c_size}")
    print(f"MM.leading_dimension =     {MM.leading_dimension}")
    print(f"MM.shared_memory_size =    {MM.shared_memory_size}")
    print(f"MM.block_dim =             {MM.block_dim}")
    print(f"MM.max_threads_per_block = {MM.max_threads_per_block}")


def l2error(test, ref, module=np):
    return module.linalg.norm(ref - test) / module.linalg.norm(test)


def convert_to_cuda_array(cupy_array):
    # Allocate
    size_bytes = np.prod(cupy_array.shape) * cupy_array.itemsize
    err, cuda_array = cuda.cuMemAlloc(size_bytes)
    CHECK_CUDA(err)
    # Copy
    (err,) = cuda.cuMemcpyDtoD(cuda_array, cupy_array.__cuda_array_interface__["data"][0], size_bytes)
    CHECK_CUDA(err)
    return cuda_array


def make_cuda_array(np_array):
    # Allocate
    size_bytes = np.prod(np_array.shape) * np_array.itemsize
    err, cuda_array = cuda.cuMemAlloc(size_bytes)
    CHECK_CUDA(err)
    # Copy
    (err,) = cuda.cuMemcpyHtoD(cuda_array, np_array.ctypes.data, size_bytes)
    CHECK_CUDA(err)
    return cuda_array


def copy_to_numpy(cuda_array, np_array):
    size_bytes = np.prod(np_array.shape) * np_array.itemsize
    (err,) = cuda.cuMemcpyDtoH(np_array.ctypes.data, cuda_array, size_bytes)
    CHECK_CUDA(err)


def copy_to_cupy(cuda_array, cupy_array):
    size_bytes = np.prod(cupy_array.shape) * cupy_array.itemsize
    (err,) = cuda.cuMemcpyDtoD(cupy_array.__cuda_array_interface__["data"][0], cuda_array, size_bytes)
    CHECK_CUDA(err)


def free_array(cuda_array):
    (err,) = cuda.cuMemFree(cuda_array)
    CHECK_CUDA(err)


def make_args(*args):
    args = [np.array([int(arg)], dtype=np.uint64) for arg in args]
    args = np.array([arg.ctypes.data for arg in args], dtype=np.uint64)
    return args


def get_unsigned(module, name):
    err, ptr, numbytes = cuda.cuModuleGetGlobal(module, name.encode(encoding="ascii"))
    value = np.array([0], dtype=np.uint32)
    (err,) = cuda.cuMemcpyDtoH(value.ctypes.data, ptr, value.itemsize)
    return value[0]


def time_check_cupy(fun, reference, ncycles, *args):
    args = [(cupy.array(arg) if isinstance(arg, np.ndarray | np.generic) else arg) for arg in args]
    start, stop = cupy.cuda.Event(), cupy.cuda.Event()
    out = fun(*args)

    start.record(None)
    for _ in range(ncycles):
        out = fun(*args)
    stop.record(None)
    stop.synchronize()

    t_cupy_ms = cupy.cuda.get_elapsed_time(start, stop) / ncycles

    error = l2error(test=out, ref=reference)

    assert error < _TOLERANCE[np.float32]

    return {"time_ms": t_cupy_ms}


def fp16x2_to_complex64(data):
    return data[..., ::2] + 1.0j * data[..., 1::2]


def complex64_to_fp16x2(data):
    shape = (*data.shape[:-1], data.shape[-1] * 2)
    output = np.zeros(shape=shape, dtype=np.float16)
    output[..., 0::2] = data.real
    output[..., 1::2] = data.imag
    return output


# Return smallest n such that
# n >= a / b
# n % b == 0
def smallest_multiple(a, b):
    return ((a + b - 1) // b) * b


SM70 = CodeType("lto", ComputeCapability(7, 0))
SM72 = CodeType("lto", ComputeCapability(7, 2))
SM75 = CodeType("lto", ComputeCapability(7, 5))
SM80 = CodeType("lto", ComputeCapability(8, 0))
SM86 = CodeType("lto", ComputeCapability(8, 6))
SM89 = CodeType("lto", ComputeCapability(8, 9))
SM90 = CodeType("lto", ComputeCapability(9, 0))
