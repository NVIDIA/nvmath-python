# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import os
import glob
import threading
import pathlib

from libc.stdint cimport intptr_t
from ..bindings cimport get_cc, get_function_from_module

from .nvrtc_helper import CompileHelper

thread_local = threading.local()

# In multithreaded environment we share compiled code and modules between threads.
# Each thread has its own cache with loaded kernels, but on a cache miss,
# we resort to the shared cache guarded with _kernel_lock.
_kernel_lock = threading.Lock()
_kernel_code_cache = {}  # cc -> kernel_code -> CompiledCode
_kernel_module_cache = {}  # device_id -> kernel_code -> KernelModule


cdef class KernelModule:
    cdef readonly object module
    cdef readonly intptr_t function_ptr

    def __init__(self, object module, intptr_t function_ptr):
        self.module = module
        self.function_ptr = function_ptr


cdef int _query_device_cc(int device_id) except? -1 nogil:
    cdef int major = 0
    cdef int minor = 0
    get_cc(major, minor, device_id)
    return major * 10 + minor


cdef int _get_device_cc(int device_id):
    # device_id -> cc
    if not hasattr(thread_local, "device_ccs"):
        thread_local.device_ccs = {}
    cdef dict _device_ccs = thread_local.device_ccs
    cc = _device_ccs.get(device_id)
    if cc is None:
        cc = _query_device_cc(device_id)
        _device_ccs[device_id] = cc
    return cc


cdef _get_compile_helper(int cc, str includes_key):
    # cc -> include_key -> CompileHelper
    if not hasattr(thread_local, "compile_helpers"):
        thread_local.compile_helpers = {}
    cdef dict _compile_helpers = thread_local.compile_helpers
    cc_compile_helpers = _compile_helpers.get(cc)
    if cc_compile_helpers is None:
        cc_compile_helpers = {}
        _compile_helpers[cc] = cc_compile_helpers
    compile_helper = cc_compile_helpers.get(includes_key)
    if compile_helper is None:
        include_names, includes = get_includes(includes_key)
        major, minor = cc // 10, cc % 10
        compile_helper = CompileHelper(include_names, includes, (major, minor))
        cc_compile_helpers[includes_key] = compile_helper
    return compile_helper


cpdef discover_includes(list include_dirs):
    """
    Helper function to read headers from a list of directories.
    The `include_dirs` must be a list of (base_dir, dir) tuples. Each dir
    is traversed (not recursively) to find header files (.h). A name of a header
    is formed by stripping `base_dir` from the path of any single header file in dir.
    """
    include_names, includes = [], []
    for include_dir_base, include_dir in include_dirs:
        for filename in glob.glob(os.path.join(include_dir, "*.h")):
            with open(filename, "rb") as f:
                includes.append(f.read())
            header_rel_path = os.path.relpath(filename, include_dir_base)
            header_rel_path = pathlib.PurePath(header_rel_path).as_posix()
            include_names.append(header_rel_path.encode())
    return include_names, includes


cpdef bint register_includes(str includes_key, list include_names, list includes):
    """
    Register includes for a given key. Doing so once for a lifetime of the thread
    is (slightly) more efficient than re-reading headers for each kernel compilation.
    NOTE, each thread has its own cache of includes, so this function MUST be called
    by all threads that use the kernel cache.
    """
    if not hasattr(thread_local, "includes"):
        thread_local.includes = {}
    cdef dict _includes = thread_local.includes
    if includes_key in _includes:
        return False
    _includes[includes_key] = (tuple(include_names), tuple(includes))
    return True


cpdef get_includes(str includes_key):
    if not hasattr(thread_local, "includes"):
        thread_local.includes = {}
    cdef dict _includes = thread_local.includes
    return _includes[includes_key]


cpdef _invalidate_kernel_cache():
    """
    WARNING: this is internal utility meant for testing.
    In multithreaded environment this function MUST be
    called by all threads that use the kernel cache.
    """
    thread_local.kernel_ptr_cache = {}
    with _kernel_lock:
        for device_id in _kernel_module_cache:
            _kernel_module_cache[device_id].clear()
        for cc in _kernel_code_cache:
            _kernel_code_cache[cc].clear()
    return 0


cdef _get_compiled_code(str kernel_code, str kernel_name, int device_id, str includes_key, object logger=None):
    """
    Returns compiled code, either from cache or compiled from scratch.
    The function MUST be called while holding the _kernel_lock.
    """
    cc = _get_device_cc(device_id)
    cc_cache = _kernel_code_cache.get(cc)
    if cc_cache is None:
        cc_cache = {}
        _kernel_code_cache[cc] = cc_cache
    compiled = cc_cache.get(kernel_code)
    if compiled is None:
        if logger is not None:
            logger.debug(f"Compiling kernel {kernel_name} for device {device_id} (cc={cc}).\n{kernel_code}")
        compile_helper = _get_compile_helper(cc, includes_key)
        compiled = compile_helper.compile(kernel_code, logger)
        cc_cache[kernel_code] = compiled
    elif logger is not None:
        logger.debug(f"Using cached compiled kernel {kernel_name} for device {device_id} (cc={cc}).\n{kernel_code}")
    return compiled


cdef intptr_t _get_kernel(str kernel_code, str kernel_name, int device_id, str includes_key, object logger=None) except -1:
    """
    Returns compiled and loaded module, either from cache or compiled from scratch.
    The function MUST be called while holding the _kernel_lock.
    """
    device_cache = _kernel_module_cache.get(device_id)
    if device_cache is None:
        device_cache = {}
        _kernel_module_cache[device_id] = device_cache

    kernel_module = device_cache.get(kernel_code)
    if kernel_module is None:
        compiled_code = _get_compiled_code(kernel_code, kernel_name, device_id, includes_key, logger)
        module = compiled_code.load()
        kernel_module = KernelModule(module, get_function_from_module(int(module), kernel_name.encode()))
        device_cache[kernel_code] = kernel_module
        if logger is not None:
            logger.debug(f"Stored kernel {kernel_name} ({kernel_module.function_ptr}) for device {device_id} in global cache.\n{kernel_code}")
    elif logger is not None:
        logger.debug(f"Loaded kernel {kernel_name} ({kernel_module.function_ptr}) for device {device_id} from global cache.\n{kernel_code}")
    return kernel_module.function_ptr



cdef intptr_t get_kernel(str kernel_code, str kernel_name, int device_id, str includes_key, object logger=None) except -1:
    """
    Returns a pointer to the kernel function for a given kernel code and device id. The kernel will be compiled
    and loaded into device memory first time this function is called. Subsequent calls with the same kernel
    code and device id return a pointer to the cached kernel function. Note, that kernel name and includes
    are not used in the cached lookup, it is caller responsibility to ensure that kernel name and includes
    do not change between calls.

    In multithreaded environment, each thread has its own cache with pointers to the loaded
    modules, if the cache is not populated, the shared cache guarded with _kernel_lock is used.
    """
    if not hasattr(thread_local, "kernel_ptr_cache"):
        thread_local.kernel_ptr_cache = {}
    cdef dict _kernel_ptr_cache = thread_local.kernel_ptr_cache
    device_cache = _kernel_ptr_cache.get(device_id)
    if device_cache is None:
        device_cache = {}
        _kernel_ptr_cache[device_id] = device_cache

    kernel_ptr = device_cache.get(kernel_code)
    if kernel_ptr is None:
        with _kernel_lock:
            kernel_ptr = _get_kernel(kernel_code, kernel_name, device_id, includes_key, logger)
            device_cache[kernel_code] = kernel_ptr
    elif logger is not None:
        logger.debug(f"Loaded kernel {kernel_name} ({kernel_ptr}) for device {device_id} from thread local cache.\n{kernel_code}")
    return kernel_ptr
