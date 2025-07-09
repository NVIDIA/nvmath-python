# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import functools
import logging

from cuda.bindings import nvrtc

from .caching import disk_cache
from .common import check_in
from .common_cuda import ISAVersion
from .common_mathdx import CUDA_HOME


def CHECK_NVRTC(err, prog):
    if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        err, logsize = nvrtc.nvrtcGetProgramLogSize(prog)
        log = b" " * logsize
        err = nvrtc.nvrtcGetProgramLog(prog, log)
        raise RuntimeError(f"NVRTC error: {log.decode('ascii')}")


# cpp is the C++ source code
# cc is an instance of ComputeCapability
# rdc is true or false
# code is lto or ptx
# @cache
@functools.lru_cache(maxsize=32)  # Always enabled
@disk_cache  # Optional, see caching.py
def compile_impl(cpp, cc, rdc, code, cuda_home, nvrtc_path, nvrtc_version):
    logging.debug(f"Compiling with CUDA_HOME={cuda_home}, and NVRTC {nvrtc_version}")

    check_in("rdc", rdc, [True, False])
    check_in("code", code, ["lto", "ptx"])

    opts = (
        [b"--std=c++17", b"--device-as-default-execution-space", b"-DCUFFTDX_DETAIL_USE_CUDA_STL=1"]
        + [bytes(f"--include-path={h}/include", encoding="ascii") for h in cuda_home]
        + [
            bytes(f"--gpu-architecture=compute_{cc.major * 10 + cc.minor}", encoding="ascii"),
        ]
    )
    if rdc:
        opts += [b"--relocatable-device-code=true"]

    if code == "lto":
        opts += [b"-dlto"]

    # Create program
    err, prog = nvrtc.nvrtcCreateProgram(str.encode(cpp), b"code.cu", 0, [], [])
    if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        raise RuntimeError(f"nvrtcCreateProgram error: {err}")

    (err,) = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
    CHECK_NVRTC(err, prog)

    if code == "lto":
        err, ltoSize = nvrtc.nvrtcGetLTOIRSize(prog)
        CHECK_NVRTC(err, prog)

        lto = b" " * ltoSize
        (err,) = nvrtc.nvrtcGetLTOIR(prog, lto)
        CHECK_NVRTC(err, prog)

        (err,) = nvrtc.nvrtcDestroyProgram(prog)
        CHECK_NVRTC(err, prog)

        return lto

    elif code == "ptx":
        err, ptxSize = nvrtc.nvrtcGetPTXSize(prog)
        CHECK_NVRTC(err, prog)

        ptx = b" " * ptxSize
        (err,) = nvrtc.nvrtcGetPTX(prog, ptx)
        CHECK_NVRTC(err, prog)

        (err,) = nvrtc.nvrtcDestroyProgram(prog)
        CHECK_NVRTC(err, prog)

        return ptx.decode("ascii")


def compile(**kwargs):
    err, major, minor = nvrtc.nvrtcVersion()
    if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        raise RuntimeError(f"nvrtcVersion error: {err}")
    nvrtc_version = ISAVersion(major, minor)
    return nvrtc_version, compile_impl(
        **kwargs,
        cuda_home=CUDA_HOME,
        nvrtc_path=nvrtc.__file__,
        nvrtc_version=nvrtc_version,
    )
