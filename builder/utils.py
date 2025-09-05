# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import os
import sys

from setuptools.command.build_ext import build_ext as _build_ext


def detect_cuda_paths():
    # The search logic below is designed to support all use cases:
    # - build against pip wheels (skipped via --no-build-isolation)
    # - build against custom CUDA location (CUDA_PATH or CUDA_HOME)
    # - build against default Linux CUDA Location
    # The bindings can be built against any recent CUDA version, and we don't really
    # need any version/library detection here. But, we do need basic CUDA driver/runtime
    # headers, and in the wheel case they are scattered in two wheels. When build
    # isolation is on, the build prefix is added to sys.path, but this is the only
    # implementation detail that we rely on.
    potential_build_prefixes = (
        [os.path.join(p, "nvidia/cuda_runtime") for p in sys.path]
        + [os.path.join(p, "nvidia/cuda_nvcc") for p in sys.path]
        # internal/bindings depends on cuda_bindings cydriver,
        # which introduces dependency on cudaProfiler.h
        + [os.path.join(p, "nvidia/cuda_profiler_api") for p in sys.path]
        + [os.environ.get("CUDA_PATH", os.environ.get("CUDA_HOME", "")), "/usr/local/cuda"]
    )
    cuda_paths = []

    def check_path(header):
        for prefix in potential_build_prefixes:
            cuda_h = os.path.join(prefix, "include", header)
            if os.path.isfile(cuda_h):
                if prefix not in cuda_paths:
                    cuda_paths.append(prefix)
                break
        else:
            raise RuntimeError(f"{header} not found")

    check_path("cuda.h")
    check_path("crt/host_defines.h")
    # internal/bindings depends on cuda_bindings cydriver,
    # which introduces dependency on cudaProfiler.h
    check_path("cudaProfiler.h")
    return cuda_paths


def decide_lib_name(ext_name):
    # TODO: move the record of the supported lib list elsewhere?
    for lib in ("cublas", "cusolver", "cufftMp", "cufft", "cusparse", "curand", "nvpl", "nvshmem", "mathdx", "cudss"):
        if lib in ext_name:
            return lib
    else:
        return None


class build_ext(_build_ext):
    def __init__(self, *args, **kwargs):
        self._nvmath_cuda_paths = detect_cuda_paths()
        print("\n" + "*" * 80)
        for p in self._nvmath_cuda_paths:
            print("CUDA path(s):", p)
        print("*" * 80 + "\n")
        super().__init__(*args, **kwargs)

    def _prep_includes_libs_rpaths(self, lib_name):
        """
        Set cuda_incl_dir and extra_linker_flags.
        """
        cuda_incl_dir = [os.path.join(p, "include") for p in self._nvmath_cuda_paths]
        extra_linker_flags = []

        site_packages = ["$ORIGIN/../../.."]
        if self.editable_mode:
            import site

            site_packages = site.getsitepackages()
        else:
            # strip binaries to remove debug symbols which significantly
            # increase wheel size
            extra_linker_flags = ["-Wl,--strip-all"]
        nvpl_site_packages = [f"{p}/.." for p in site_packages]

        # Note: soname = library major version
        # We need to be able to search for cuBLAS/cuSOLVER/... at run time, in case they
        # are installed via pip wheels.
        # The rpaths must be adjusted given the following full-wheel installation:
        # - $ORIGIN:          site-packages/nvmath/bindings/_internal/
        # - cublas:           site-packages/nvidia/cublas/lib/
        # - cusolver:         site-packages/nvidia/cusolver/lib/
        # -   ...                             ...
        if lib_name is None:
            return cuda_incl_dir, extra_linker_flags

        ldflag = "-Wl,--disable-new-dtags"
        if lib_name == "nvpl":
            # 1. the nvpl bindings land in
            # site-packages/nvmath/bindings/nvpl/_internal/ as opposed to other
            # packages that have their bindings in
            # site-packages/nvmath/bindings/_internal/, so we need one extra
            # `..` to get into `site-packages` and then the lib_name=nvpl is not
            # in nvidia dir but directly in the site-packages.
            # 2. mkl lib is placed directly in the python `lib` directory, not
            # in python{ver}/site-packages
            rpath = ":".join([f"{pth}/{lib_name}/lib:{pth}/../../" for pth in nvpl_site_packages])
            ldflag += f",-rpath,{rpath}"
        extra_linker_flags.append(ldflag)

        return cuda_incl_dir, extra_linker_flags

    def build_extension(self, ext):
        lib_name = decide_lib_name(ext.name)
        ext.include_dirs, ext.extra_link_args = self._prep_includes_libs_rpaths(lib_name)
        if ext.name.endswith("cusparse"):
            # too noisy
            ext.define_macros = [
                ("DISABLE_CUSPARSE_DEPRECATED", None),
            ]

        super().build_extension(ext)

    def build_extensions(self):
        self.parallel = os.cpu_count()
        super().build_extensions()
