# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

from cuda import cuda
from cuda import nvrtc

from nvmath.device.common_mathdx import CUDA_HOME as _CUDA_HOME
from nvmath.device.common_mathdx import MATHDX_HOME as _MATHDX_HOME

from .helpers import CHECK_CUDA, CHECK_NVRTC, make_args, get_unsigned


def run_and_time(kernel, grid_dim, block_dim, shared_memory_size, ncycles, *args):

    # Prepare void** args
    args = make_args(*args)

    err, stream = cuda.cuStreamCreate(0)
    CHECK_CUDA(err)
    err, start = cuda.cuEventCreate(0)
    CHECK_CUDA(err)
    err, stop = cuda.cuEventCreate(0)
    CHECK_CUDA(err)

    err, = cuda.cuCtxSynchronize()
    CHECK_CUDA(err)

    err, = cuda.cuEventRecord(start, stream)
    for _ in range(ncycles):
        err, = cuda.cuLaunchKernel(
            kernel,
            grid_dim[0],              # grid x dim
            grid_dim[1],              # grid y dim
            grid_dim[2],              # grid z dim
            block_dim[0],             # block x dim
            block_dim[1],             # block y dim
            block_dim[2],             # block z dim
            shared_memory_size,       # dynamic shared memory
            stream,                   # stream
            args.ctypes.data,         # kernel arguments
            0,                        # extra (ignore)
        )
        CHECK_CUDA(err)
    err, = cuda.cuEventRecord(stop, stream)
    CHECK_CUDA(err)
    err, = cuda.cuStreamSynchronize(stream)
    CHECK_CUDA(err)
    err, time_ms = cuda.cuEventElapsedTime(start, stop)
    CHECK_CUDA(err)
    time_ms  = time_ms / ncycles

    err, = cuda.cuStreamDestroy(stream)
    CHECK_CUDA(err)
    err, = cuda.cuEventDestroy(start)
    CHECK_CUDA(err)
    err, = cuda.cuEventDestroy(stop)
    CHECK_CUDA(err)

    return time_ms

def compile_cpp_kernel(cpp, sm, mangled):

    print(f"compile_cpp_kernel CUDA_HOME = {_CUDA_HOME}, _MATHDX_HOME = {_MATHDX_HOME}")

    opts = [b"--std=c++17", \
            b"--device-as-default-execution-space", \
            b"-DCUFFTDX_DETAIL_USE_CUDA_STL=1"] + \
           [bytes(f"--include-path={h}/include", encoding='ascii') for h in _CUDA_HOME] + \
           [bytes(f"--include-path={_MATHDX_HOME}/include/", encoding='ascii'), \
            bytes(f"--include-path={_MATHDX_HOME}/include/cufftdx", encoding='ascii'), \
            bytes(f"--include-path={_MATHDX_HOME}/include/cublasdx/include", encoding='ascii'), \
            bytes(f"--include-path={_MATHDX_HOME}/external/cutlass/include/", encoding='ascii'), \
            bytes(f"--gpu-architecture=sm_{sm[0] * 10 + sm[1]}", encoding='ascii')]

    err, prog = nvrtc.nvrtcCreateProgram(str.encode(cpp), b"code.cu", 0, [], [])
    assert(err == nvrtc.nvrtcResult.NVRTC_SUCCESS)

    err, = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
    CHECK_NVRTC(err, prog)

    err, cubinSize = nvrtc.nvrtcGetCUBINSize(prog)
    CHECK_NVRTC(err, prog)

    cubin = b" " * cubinSize
    err, = nvrtc.nvrtcGetCUBIN(prog, cubin)
    CHECK_NVRTC(err, prog)

    err, = nvrtc.nvrtcDestroyProgram(prog)
    CHECK_NVRTC(err, prog)

    # To figure out mangled name:
    # with open("output.cubin", "wb") as f:
    #     print("cubin written to output.cubin")
    #     f.write(cubin)
    print(f"compile_cpp_kernel Mangled name = {mangled}, cubin size = {len(cubin)} B")

    #
    # Generate a CUmodule and CUfunction
    #
    err, module = cuda.cuModuleLoadData(cubin)
    CHECK_CUDA(err)
    err, kernel = cuda.cuModuleGetFunction(module, mangled.encode(encoding='ascii'))
    CHECK_CUDA(err)

    shared_memory_size = get_unsigned(module, "shared_memory_size")
    print(f"compile_cpp_kernel shared_memory_size = {shared_memory_size}")

    return (module, kernel, shared_memory_size)
