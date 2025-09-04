# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

from cuda.bindings import nvrtc
import cuda.bindings.driver as driver


def check_nvrtc_error(err, prog):
    if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        if prog is None:
            raise RuntimeError(f"NVRTC error: {nvrtc.nvrtcResult(err).name}")
        else:
            log_err, logsize = nvrtc.nvrtcGetProgramLogSize(prog)
            if log_err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
                raise RuntimeError(f"NVRTC error: {nvrtc.nvrtcResult(err).name}. No logs available.")
            log = b" " * logsize
            (log_err,) = nvrtc.nvrtcGetProgramLog(prog, log)
            if log_err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
                raise RuntimeError(f"NVRTC error: {nvrtc.nvrtcResult(err).name}. No logs available.")
            raise RuntimeError(f"NVRTC error: {nvrtc.nvrtcResult(err).name}. Compilation log: \n{log.decode('ascii')}")


def check_cuda_error(err):
    if err != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"CUDA Error: {driver.CUresult(err).name}")


class CompiledCode:
    def __init__(self, data, size):
        self.data = data
        self.size = size

    def load(self):
        """
        It is caller responsibility to assure correct device context is set.
        """
        err, module = driver.cuModuleLoadData(self.data)
        check_cuda_error(err)
        return module


class CompileHelper:
    def __init__(self, include_names, includes, cc):
        self.include_names = include_names
        self.includes = includes
        self.num_headers = len(self.include_names)
        assert self.num_headers == len(self.includes)
        self.source_name = b"code.cu"
        err, self.nvrtc_version_major, self.nvrtc_version_minor = nvrtc.nvrtcVersion()
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError(f"nvrtcVersion error: {err}")
        self.cc = cc
        major, minor = cc
        self.arch_opt = bytes(f"--gpu-architecture=sm_{major}{minor}", "ascii")
        self.opts = [
            b"--fmad=true",
            self.arch_opt,
            b"--std=c++17",
            b"-default-device",
        ]

    def compile(self, code, logger=None):
        if logger is not None:
            logger.debug(f"Compiling kernel to 'cubin' with options: {self.opts}")

        # Create program
        err, prog = nvrtc.nvrtcCreateProgram(str.encode(code), b"code.cu", self.num_headers, self.includes, self.include_names)
        check_nvrtc_error(err, None)

        try:
            (err,) = nvrtc.nvrtcCompileProgram(prog, len(self.opts), self.opts)
            check_nvrtc_error(err, prog)

            err, data_size = nvrtc.nvrtcGetCUBINSize(prog)
            check_nvrtc_error(err, prog)
            data = b" " * data_size
            (err,) = nvrtc.nvrtcGetCUBIN(prog, data)
            check_nvrtc_error(err, prog)

            return CompiledCode(data, data_size)
        finally:
            (err,) = nvrtc.nvrtcDestroyProgram(prog)
            check_nvrtc_error(err, None)
