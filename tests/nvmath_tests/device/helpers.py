# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import contextlib
import os
import psutil
import time

import numpy as np
from cuda.bindings import runtime as cudart, nvrtc, driver as cudadrv
import cupy
import pytest

from nvmath.device import CodeType, ComputeCapability
from nvmath.device.common_cuda import MAX_SUPPORTED_CC, get_default_code_type
from nvmath._utils import get_nvrtc_version


def CHECK_CUDART(err):
    if err != cudart.cudaError_t.cudaSuccess:
        err2, str = cudart.cudaGetErrorString(err)
        raise RuntimeError(f"CUDArt Error: {str} ({err})")


def CHECK_CUDA(err):
    if err != cudadrv.CUresult.CUDA_SUCCESS:
        err2, str = cudadrv.cuGetErrorName(err)
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
    if (prop.major, prop.minor) > MAX_SUPPORTED_CC:
        return MAX_SUPPORTED_CC
    return (prop.major, prop.minor)


def random_complex(shape, real_dtype, order="C", module=np) -> np.ndarray:
    return random_real(shape, real_dtype, order, module=module) + 1.0j * random_real(shape, real_dtype, order, module=module)


def random_real(shape, real_dtype, order="C", module=np) -> np.ndarray:
    return module.random.randn(np.prod(shape)).astype(real_dtype).reshape(shape, order=order)


def random_int(shape, int_dtype):
    """
    Generate random integers in the range [-2, 2) for signed integers and [0, 4)
    for unsigned integers.
    """
    min_val, max_val = 0, 4
    if issubclass(int_dtype, np.signedinteger):
        min_val, max_val = -2, 2
    return np.random.randint(min_val, max_val, size=shape, dtype=int_dtype)


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
    print(f"FFT.workspace_size =       {FFT.workspace_size}")


def show_MM_traits(MM):
    print(f"MM.size =                  {MM.size}")
    print(f"MM.files =                 {MM.files}")
    print(f"MM.transpose_mode =        {MM.transpose_mode}")
    print(f"MM.a_value_type =          {MM.a_value_type}")
    print(f"MM.b_value_type =          {MM.b_value_type}")
    print(f"MM.c_value_type =          {MM.c_value_type}")
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
    err, cuda_array = cudadrv.cuMemAlloc(size_bytes)
    CHECK_CUDA(err)
    # Copy
    (err,) = cudadrv.cuMemcpyDtoD(cuda_array, cupy_array.__cuda_array_interface__["data"][0], size_bytes)
    CHECK_CUDA(err)
    return cuda_array


def make_cuda_array(np_array):
    # Allocate
    size_bytes = np.prod(np_array.shape) * np_array.itemsize
    err, cuda_array = cudadrv.cuMemAlloc(size_bytes)
    CHECK_CUDA(err)
    # Copy
    (err,) = cudadrv.cuMemcpyHtoD(cuda_array, np_array.ctypes.data, size_bytes)
    CHECK_CUDA(err)
    return cuda_array


def copy_to_numpy(cuda_array, np_array):
    size_bytes = np.prod(np_array.shape) * np_array.itemsize
    (err,) = cudadrv.cuMemcpyDtoH(np_array.ctypes.data, cuda_array, size_bytes)
    CHECK_CUDA(err)


def copy_to_cupy(cuda_array, cupy_array):
    size_bytes = np.prod(cupy_array.shape) * cupy_array.itemsize
    (err,) = cudadrv.cuMemcpyDtoD(cupy_array.__cuda_array_interface__["data"][0], cuda_array, size_bytes)
    CHECK_CUDA(err)


def free_array(cuda_array):
    (err,) = cudadrv.cuMemFree(cuda_array)
    CHECK_CUDA(err)


def make_args(*args):
    args = [np.array([int(arg)], dtype=np.uint64) for arg in args]
    args = np.array([arg.ctypes.data for arg in args], dtype=np.uint64)
    return args


def get_unsigned(module, name):
    err, ptr, numbytes = cudadrv.cuModuleGetGlobal(module, name.encode(encoding="ascii"))
    value = np.array([0], dtype=np.uint32)
    (err,) = cudadrv.cuMemcpyDtoH(value.ctypes.data, ptr, value.itemsize)
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


def skip_nvbug_5218000(precision, sm=None, ctk=None, size=(1, 1, 1), dynamic_ld=False):
    not_affected_itemsize = 4  # itemsize less than 4 bytes are affected
    if isinstance(precision, tuple):
        if all(np.dtype(p).itemsize >= not_affected_itemsize for p in precision[:2]):
            return
    else:
        if np.dtype(precision).itemsize >= not_affected_itemsize:
            return

    if tuple(s % 16 for s in size) == (0, 0, 0):
        return

    if dynamic_ld:
        return

    if ctk is None:
        ctk = get_nvrtc_version()
    if ctk < (12, 8, 0) or ctk > (12, 9, 41):  # before 12.8u0 or after 12.9u1
        return

    if sm is None:
        sm = get_default_code_type()
    if sm.cc != SM80.cc and sm.cc < SM90.cc:
        return

    pytest.skip("Skipping test due to NVBug 5218000.")


def skip_unsupported_sm(sm=None):
    """Skip tests for unsupported SM versions by nvrtc."""
    if isinstance(sm, CodeType):
        cc = sm.cc
    elif isinstance(sm, ComputeCapability):
        cc = sm
    elif sm is None:
        cc = get_default_code_type().cc
    else:
        raise TypeError(f"Unsupported argument type: {type(sm)}")
    err, supported_archs = nvrtc.nvrtcGetSupportedArchs()
    assert err == nvrtc.nvrtcResult.NVRTC_SUCCESS
    if cc.integer / 10 not in supported_archs:
        err, major, minor = nvrtc.nvrtcVersion()
        assert err == nvrtc.nvrtcResult.NVRTC_SUCCESS
        pytest.skip(f"nvrtc version {major}.{minor} does not support compute capability {cc}")


SM70 = CodeType("lto", ComputeCapability(7, 0))
SM72 = CodeType("lto", ComputeCapability(7, 2))
SM75 = CodeType("lto", ComputeCapability(7, 5))
SM80 = CodeType("lto", ComputeCapability(8, 0))
SM86 = CodeType("lto", ComputeCapability(8, 6))
SM89 = CodeType("lto", ComputeCapability(8, 9))
SM90 = CodeType("lto", ComputeCapability(9, 0))
SM100 = CodeType("lto", ComputeCapability(10, 0))
SM101 = CodeType("lto", ComputeCapability(10, 1))
SM103 = CodeType("lto", ComputeCapability(10, 3))
SM120 = CodeType("lto", ComputeCapability(12, 0))
SM121 = CodeType("lto", ComputeCapability(12, 1))


class AssertFilesClosed(contextlib.AbstractContextManager):
    """A context which asserts that the number of open files has not changed."""

    def __init__(self):
        super().__init__()
        self.process: psutil.Process
        self.before_count = 0

    def __enter__(self, *args, **kwargs):
        self.process = psutil.Process(os.getpid())
        self.before_count = len(self.process.open_files())

    def __exit__(self, *args, **kwargs):
        after_count = len(self.process.open_files())
        assert after_count == self.before_count, f"The number of open files changed from {self.before_count} to {after_count}"
        for file in self.process.open_files():
            assert "ltoir" not in file.path, f"{file.path} is still open"
