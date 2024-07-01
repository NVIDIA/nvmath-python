# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

#
# Monkey-patching of pynvjitlink and Numba to support LTO code generation and linking.
#
# This is only necessary with Numba < 0.60 (ie 0.59) and require pynvjitlink 0.1.14.
# Numba 0.60+ should make this obsolete.
#

import ctypes
import os
import pathlib
from importlib import metadata

import numba
from numba.cuda.codegen import CUDACodeLibrary
from numba.cuda.cudadrv.driver import Linker, LinkerError, FILE_EXTENSION_MAP
from numba.cuda.cudadrv import devices
from numba.cuda.cudadrv import libs
from numba.cuda.cudadrv.libs import get_cudalib
import numba.cuda.cudadrv.nvvm as nvvm
import pynvjitlink
from pynvjitlink import NvJitLinkError
from pynvjitlink.patch import new_patched_linker, PatchedLinker
import pynvjitlink.patch

from nvmath import _utils


#
# pynvjitlink patches
#

def add_file(self, path, kind):
    try:
        with open(path, "rb") as f:
            data = f.read()
    except FileNotFoundError:
        raise LinkerError(f"{path} not found")

    name = pathlib.Path(path).name
    if kind == 'ltoir':
        fn = self._linker.add_ltoir
    elif kind == FILE_EXTENSION_MAP["cubin"]:
        fn = self._linker.add_cubin
    elif kind == FILE_EXTENSION_MAP["fatbin"]:
        fn = self._linker.add_fatbin
    elif kind == FILE_EXTENSION_MAP["a"]:
        fn = self._linker.add_library
    elif kind == FILE_EXTENSION_MAP["ptx"]:
        return self.add_ptx(data, name)
    elif kind == FILE_EXTENSION_MAP["o"]:
        fn = self._linker.add_object
    else:
        raise LinkerError(f"Don't know how to link {kind}")

    try:
        fn(data, name)
    except NvJitLinkError as e:
        raise LinkerError from e


#
# Numba patches
#

def add_file_guess_ext(self, path):
    """Add a file to the link, guessing its type from its extension."""

    ext = os.path.splitext(path)[1][1:]
    if ext == '':
        raise RuntimeError("Don't know how to link file with no extension")
    elif ext == 'cu':
        self.add_cu_file(path)
    else:
        if ext == 'ltoir':
            kind = 'ltoir'
        else:
            kind = FILE_EXTENSION_MAP.get(ext, None)
            if kind is None:
                raise RuntimeError("Don't know how to link file with extension "
                                f".{ext}")

        self.add_file(path, kind)


def get_ltoir(self, cc):

    arch = nvvm.get_arch_option(*cc)
    options = self._nvvm_options.copy()
    options['arch'] = arch
    options['gen-lto'] = None

    irs = self.llvm_strs
    return nvvm.llvm_to_ptx(irs, **options) # llvm_to_ptx can produce LTOIR too


def get_cubin(self, cc=None):
    if cc is None:
        ctx = devices.get_context()
        device = ctx.device
        cc = device.compute_capability

    cubin = self._cubin_cache.get(cc, None)
    if cubin:
        return cubin

    linker = new_patched_linker(max_registers=self._max_registers, cc=cc, lto=True)

    ltoir = self.get_ltoir(cc=cc)
    linker.add_ltoir(ltoir)
    for path in self._linking_files:
        linker.add_file_guess_ext(path)
    if self.needs_cudadevrt:
        linker.add_file_guess_ext(get_cudalib('cudadevrt', static=True))

    cubin = linker.complete()
    self._cubin_cache[cc] = cubin
    self._linkerinfo_cache[cc] = linker.info_log

    return cubin


def __nvvm_new__(cls):
    with nvvm._nvvm_lock:
        # was: __INSTANCE, changed to _NVVM__INSTANCE due to name mangling...
        if cls._NVVM__INSTANCE is None:
            cls._NVVM__INSTANCE = inst = object.__new__(cls)
            try:
                # was: inst.driver = open_cudalib('nvvm')
                inst.driver = _utils._nvvm_obj[0]
            except OSError as e:
                cls._NVVM__INSTANCE = None
                errmsg = ("libNVVM cannot be found. Do `conda install "
                          "cudatoolkit`:\n%s")
                raise nvvm.NvvmSupportError(errmsg % e)

            # Find & populate functions
            for name, proto in inst._PROTOTYPES.items():
                func = getattr(inst.driver, name)
                func.restype = proto[0]
                func.argtypes = proto[1:]
                setattr(inst, name, func)
    
    return cls._NVVM__INSTANCE


def __nvvm_init__(self):
    ir_versions = self.get_ir_version()
    self._majorIR = ir_versions[0]
    self._minorIR = ir_versions[1]
    self._majorDbg = ir_versions[2]
    self._minorDbg = ir_versions[3]
    # don't overwrite self._supported_ccs!


#
# Monkey patching
#

def patch_codegen():

    # Check Numba version
    required_numba_ver = (0, 59)
    numba_ver = numba.version_info.short
    if numba_ver != required_numba_ver:
        raise RuntimeError(f"numba version {required_numba_ver} is required, but got {numba.__version__} (aka {numba_ver})")

    # Check pynvjitlink version
    required_pynvjitlink_ver = '0.1.14'
    try:
        pynvjitlink_ver = metadata.version("pynvjitlink-cu12")
    except metadata.PackageNotFoundError:
        # for conda or local dev
        pynvjitlink_ver = metadata.version("pynvjitlink")
    if pynvjitlink_ver != required_pynvjitlink_ver:
        raise RuntimeError(f"pynvjitlink version {required_pynvjitlink_ver} is required, but got {pynvjitlink_ver}")

    # Add LTO compilation support to Numba
    CUDACodeLibrary.get_cubin = get_cubin
    CUDACodeLibrary.get_ltoir = get_ltoir

    # Add LTO linking support to Numba
    Linker.add_file_guess_ext = add_file_guess_ext

    # Add .ltoir file type support to pynvjitlink
    PatchedLinker.add_file = add_file

    # Add new LTO-IR linker to Numba (from pynvjitlink)
    pynvjitlink.patch.patch_numba_linker()

    # Patch Numba to support wheels
    # Patch NVVM object
    _utils.force_loading_nvvm()
    nvvm.NVVM.__new__ = __nvvm_new__
    nvvm.NVVM.__init__ = __nvvm_init__
    n = nvvm.NVVM()  # this is a singleton
    n._supported_ccs = ((7, 0), (7, 2), (7, 5),
                        (8, 0), (8, 6), (8, 7), (8, 9),
                        (9, 0))

    # Patch libdevice object
    if _utils._nvvm_obj[0]._name.startswith(("libnvvm", "nvvm64")):
        # libnvvm found in sys path (ex: LD_LIBRARY_PATH), fall back to Numba's way
        # way of locating it
        from numba.cuda.cudadrv.libs import get_libdevice
        libdevice_path = get_libdevice()
        # custom CUDA path is a corner case
        if libdevice_path is None:
            raise RuntimeError("cannot locate libdevice, perhaps you need to set "
                               "CUDA_HOME? Please follow Numba's instruction at:\n"
                               "https://numba.readthedocs.io/en/stable/cuda/overview.html#setting-cuda-installation-path")
    else:
        # maybe it's pip or conda
        libdevice_path = os.path.join(os.path.dirname(_utils._nvvm_obj[0]._name),
                                      "../libdevice/libdevice.10.bc")
    assert os.path.isfile(libdevice_path), f"{libdevice_path=}"
    with open(libdevice_path, "rb") as f:
        nvvm.LibDevice._cache_ = f.read()

    # No need to patch Numba's NVRTC class since we use ours,
    # but we do need to force-load NVRTC.
    # Note that our device apis only support CUDA 12+.
    _utils.force_loading_nvrtc("12")
