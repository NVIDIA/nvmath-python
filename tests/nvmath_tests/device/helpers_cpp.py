# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import os
from cuda.bindings import runtime as cudart, nvrtc, driver as cudadrv

from nvmath._utils import PLATFORM_LINUX, PLATFORM_WIN
from nvmath.device.common_mathdx import CUDA_HOME as _CUDA_HOME
from importlib.metadata import files, PackageNotFoundError

from .helpers import CHECK_CUDA, CHECK_CUDART, CHECK_NVRTC, make_args, get_unsigned

_MATHDX_HOME = None
_CUTLASS_HOME = None


def run_and_time(kernel, grid_dim, block_dim, shared_memory_size, ncycles, *args):
    # Prepare void** args
    args = make_args(*args)

    err, stream = cudadrv.cuStreamCreate(0)
    CHECK_CUDA(err)
    err, start = cudadrv.cuEventCreate(0)
    CHECK_CUDA(err)
    err, stop = cudadrv.cuEventCreate(0)
    CHECK_CUDA(err)

    (err,) = cudadrv.cuCtxSynchronize()
    CHECK_CUDA(err)

    (err,) = cudadrv.cuEventRecord(start, stream)
    for _ in range(ncycles):
        (err,) = cudadrv.cuLaunchKernel(
            kernel,
            grid_dim[0],  # grid x dim
            grid_dim[1],  # grid y dim
            grid_dim[2],  # grid z dim
            block_dim[0],  # block x dim
            block_dim[1],  # block y dim
            block_dim[2],  # block z dim
            shared_memory_size,  # dynamic shared memory
            stream,  # stream
            args.ctypes.data,  # kernel arguments
            0,  # extra (ignore)
        )
        CHECK_CUDA(err)
    (err,) = cudadrv.cuEventRecord(stop, stream)
    CHECK_CUDA(err)
    (err,) = cudadrv.cuStreamSynchronize(stream)
    CHECK_CUDA(err)
    err, time_ms = cudadrv.cuEventElapsedTime(start, stop)
    CHECK_CUDA(err)
    time_ms = time_ms / ncycles

    (err,) = cudadrv.cuStreamDestroy(stream)
    CHECK_CUDA(err)
    (err,) = cudadrv.cuEventDestroy(start)
    CHECK_CUDA(err)
    (err,) = cudadrv.cuEventDestroy(stop)
    CHECK_CUDA(err)

    return time_ms


def compile_cpp_kernel(cpp, mangled):
    print(f"compile_cpp_kernel CUDA_HOME = {_CUDA_HOME}, MATHDX_HOME = {_MATHDX_HOME}")

    err, prop = cudart.cudaGetDeviceProperties(0)
    CHECK_CUDART(err)
    sm = (prop.major, prop.minor)

    opts = (
        [b"--std=c++17", b"--device-as-default-execution-space", b"-DCUFFTDX_DETAIL_USE_CUDA_STL=1"]
        + [bytes(f"--include-path={h}/include", encoding="ascii") for h in _CUDA_HOME]
        + [bytes(f"--include-path={h}/include/cccl", encoding="ascii") for h in _CUDA_HOME]
        + [
            bytes(f"--include-path={_MATHDX_HOME}/include", encoding="ascii"),
            bytes(f"--include-path={_MATHDX_HOME}/include/cufftdx", encoding="ascii"),
            bytes(f"--include-path={_MATHDX_HOME}/include/cublasdx/include", encoding="ascii"),
            bytes(f"--include-path={_CUTLASS_HOME}/include", encoding="ascii"),
            bytes(f"--gpu-architecture=sm_{sm[0] * 10 + sm[1]}", encoding="ascii"),
        ]
    )

    err, prog = nvrtc.nvrtcCreateProgram(str.encode(cpp), b"code.cu", 0, [], [])
    assert err == nvrtc.nvrtcResult.NVRTC_SUCCESS

    (err,) = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
    CHECK_NVRTC(err, prog)

    err, cubinSize = nvrtc.nvrtcGetCUBINSize(prog)
    CHECK_NVRTC(err, prog)

    cubin = b" " * cubinSize
    (err,) = nvrtc.nvrtcGetCUBIN(prog, cubin)
    CHECK_NVRTC(err, prog)

    (err,) = nvrtc.nvrtcDestroyProgram(prog)
    CHECK_NVRTC(err, prog)

    # To figure out mangled name:
    # with open("output.cubin", "wb") as f:
    #     print("cubin written to output.cubin")
    #     f.write(cubin)
    print(f"compile_cpp_kernel Mangled name = {mangled}, cubin size = {len(cubin)} B")

    #
    # Generate a CUmodule and CUfunction
    #
    err, module = cudadrv.cuModuleLoadData(cubin)
    CHECK_CUDA(err)
    err, kernel = cudadrv.cuModuleGetFunction(module, mangled.encode(encoding="ascii"))
    CHECK_CUDA(err)

    shared_memory_size = get_unsigned(module, "shared_memory_size")
    print(f"compile_cpp_kernel shared_memory_size = {shared_memory_size}")

    return (module, kernel, shared_memory_size)


def check_mathdx_home():
    # Find mathDx headers
    global _MATHDX_HOME

    # Try wheel
    try:
        _MATHDX_HOME = files("nvidia-mathdx")
    except PackageNotFoundError:
        pass
    else:
        # use cufftdx.hpp as a proxy
        _MATHDX_HOME = [f for f in _MATHDX_HOME if "cufftdx.hpp" in str(f)][0]
        _MATHDX_HOME = os.path.join(os.path.dirname(_MATHDX_HOME.locate()), "..")
        return

    # Try conda
    if "CONDA_PREFIX" in os.environ:
        if PLATFORM_LINUX:
            conda_include = os.path.join(os.environ["CONDA_PREFIX"], "include")
        elif PLATFORM_WIN:
            conda_include = os.path.join(os.environ["CONDA_PREFIX"], "Library", "include")
        if os.path.isfile(os.path.join(conda_include, "cufftdx.hpp")):
            _MATHDX_HOME = os.path.join(conda_include, "..")
            return

    # Try local
    if "MATHDX_HOME" not in os.environ:
        raise RuntimeError(
            "mathDx headers not found. Depending on how you install nvmath-python and other CUDA packages, "
            "you may need to perform one of the steps below:\n"
            "   - pip install nvidia-mathdx\n"
            "   - conda install -c conda-forge mathdx\n"
            "   - export MATHDX_HOME=/path/to/mathdx"
        )
    else:
        _MATHDX_HOME = os.environ["MATHDX_HOME"]


def check_cutlass_home():
    # Find CUTLASS headers
    global _CUTLASS_HOME

    # Try bundle
    if os.path.isdir(os.path.join(_MATHDX_HOME, "external", "cutlass")):
        _CUTLASS_HOME = os.path.join(_MATHDX_HOME, "external", "cutlass")
        return

    # Try wheel
    try:
        _CUTLASS_HOME = files("nvidia-cutlass")
    except PackageNotFoundError:
        pass
    else:
        # use cutlass.h as a proxy
        _CUTLASS_HOME = [f for f in _CUTLASS_HOME if "cutlass.h" in str(f)][0]
        _CUTLASS_HOME = os.path.join(os.path.dirname(_CUTLASS_HOME.locate()), "../..")
        return

    # Try conda
    if "CONDA_PREFIX" in os.environ:
        if PLATFORM_LINUX:
            conda_include = os.path.join(os.environ["CONDA_PREFIX"], "include")
        elif PLATFORM_WIN:
            conda_include = os.path.join(os.environ["CONDA_PREFIX"], "Library", "include")
        if os.path.isfile(os.path.join(conda_include, "cutlass", "cutlass.h")):
            _CUTLASS_HOME = os.path.join(conda_include, "..")
            return

    # Try local
    if "CUTLASS_HOME" not in os.environ:
        raise RuntimeError(
            "CUTLASS headers not found. Depending on how you install nvmath-python and other CUDA packages, "
            "you may need to perform one of the steps below:\n"
            "   - pip install nvidia-cutlass\n"
            "   - conda install -c conda-forge cutlass\n"
            "   - export CUTLASS_HOME=/path/to/cutlass"
        )
    else:
        _CUTLASS_HOME = os.environ["CUTLASS_HOME"]


check_mathdx_home()
check_cutlass_home()
